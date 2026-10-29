/*
 * CoarsenedGraphView.cpp
 *
 *  Implementation of memory-efficient coarsened graph view
 */

#include <networkit/auxiliary/Log.hpp>
#include <networkit/auxiliary/Timer.hpp>
#include <networkit/coarsening/CoarsenedGraphView.hpp>

#include <algorithm>
#include <cstdlib>
#include <omp.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <tlx/unused.hpp>

namespace NetworKit {

namespace {
// Phase-scoped scratch for aggregation: reused across calls within a move/refine phase,
// released via releaseThreadScratch() at each phase boundary so a high-water mark from one
// coarsening level cannot leak into the next.
thread_local std::vector<std::pair<node, edgeweight>> t_staged;

// Hash accumulator for heavy supernodes, sized by distinct neighbours.
thread_local std::unordered_map<node, edgeweight> t_accMap;

constexpr size_t kDefaultStageCap = size_t{1} << 16;

size_t eagerThresholdFromEnv() {
    if (const char *env = std::getenv("NETWORKIT_LEIDEN_EAGER_THRESHOLD")) {
        try {
            const unsigned long long parsed = std::stoull(env);
            return static_cast<size_t>(parsed);
        } catch (...) {
            // Keep default if parsing fails.
        }
    }
    return 2048;
}

// Staged entries above which aggregation switches to hashing (0 = always).
size_t stageCapFromEnv() {
    static const size_t cap = [] {
        if (const char *env = std::getenv("ICEBUG_VIEW_STAGE_CAP")) {
            try {
                return static_cast<size_t>(std::stoull(env));
            } catch (...) {
            }
        }
        return kDefaultStageCap;
    }();
    return cap;
}

} // namespace

void CoarsenedGraphView::releaseThreadScratch() {
    t_staged.clear();
    t_staged.shrink_to_fit();
    std::unordered_map<node, edgeweight>().swap(t_accMap);
}

CoarsenedGraphView::CoarsenedGraphView(const Graph &originalGraph, const Partition &partition)
    : originalGraph(originalGraph) {

    // Compact the partition to ensure contiguous supernode IDs
    Partition compactPartition = partition;
    compactPartition.compact();
    numSupernodes = compactPartition.upperBound();

    // Create node mapping
    nodeMapping.resize(originalGraph.upperNodeIdBound());
    supernodeToOriginal.resize(numSupernodes);

    originalGraph.forNodes([&](node u) {
        node supernode = compactPartition[u];
        nodeMapping[u] = supernode;
        supernodeToOriginal[supernode].push_back(u);
    });
    initCache();

    TRACE("Created CoarsenedGraphView with ", numSupernodes, " supernodes from ",
          originalGraph.numberOfNodes(), " original nodes");
}

CoarsenedGraphView::CoarsenedGraphView(const CoarsenedGraphView &baseView,
                                       const Partition &partition)
    : originalGraph(baseView.originalGraph) {

    Partition compactPartition = partition;
    compactPartition.compact();
    numSupernodes = compactPartition.upperBound();

    nodeMapping.resize(originalGraph.upperNodeIdBound());
    supernodeToOriginal.resize(numSupernodes);

    originalGraph.forNodes([&](node originalNode) {
        const node baseSupernode = baseView.nodeMapping[originalNode];
        const node supernode = compactPartition[baseSupernode];
        nodeMapping[originalNode] = supernode;
        supernodeToOriginal[supernode].push_back(originalNode);
    });
    initCache();

    TRACE("Created layered CoarsenedGraphView with ", numSupernodes, " supernodes from ",
          baseView.numberOfNodes(), " base supernodes");
}

void CoarsenedGraphView::initCache() {
    eagerNeighborThreshold = eagerThresholdFromEnv();
    neighborCache_.resize(numSupernodes);
    neighborCached_.assign(numSupernodes, 0);
    cacheMutexes_ = std::vector<std::mutex>(numSupernodes);
}

count CoarsenedGraphView::numberOfCachedNeighborhoods() const {
    count cached = 0;
    for (char flag : neighborCached_)
        cached += flag ? 1 : 0;
    return cached;
}

size_t CoarsenedGraphView::aggregationWork(node supernode) const {
    size_t work = 0;
    for (node originalNode : supernodeToOriginal[supernode])
        work += static_cast<size_t>(originalGraph.degree(originalNode));
    return work;
}

count CoarsenedGraphView::numberOfEdges() const {
    count edges = 0;
    for (node u = 0; u < numberOfNodes(); ++u) {
        const auto neighbors = computeNeighbors(u);
        for (const auto &entry : neighbors) {
            if (u <= entry.first) { // Count each edge only once
                edges++;
            }
        }
    }
    return edges;
}

count CoarsenedGraphView::degree(node supernode) const {
    if (!hasNode(supernode))
        return 0;
    return computeNeighbors(supernode).size();
}

count CoarsenedGraphView::numberOfSelfLoops() const {
    count selfLoops = 0;
    for (node u = 0; u < numSupernodes; ++u) {
        for (const auto &entry : computeNeighbors(u)) {
            if (entry.first == u) { // aggregated entries carry positive weight only
                ++selfLoops;
                break;
            }
        }
    }
    return selfLoops;
}

const std::vector<node> &CoarsenedGraphView::getOriginalNodes(node supernode) const {
    if (!hasNode(supernode)) {
        static const std::vector<node> empty;
        return empty;
    }
    return supernodeToOriginal[supernode];
}

std::vector<std::pair<node, edgeweight>>
CoarsenedGraphView::aggregateNeighbors(node supernode, size_t workEstimate, size_t &workOut) const {
    // Flat aggregation (InducedSubgraphView-style): stage (supernode, weight) pairs into the
    // phase-scoped thread-local buffer, sort, then combine runs. This avoids per-call
    // std::unordered_map hashing and its per-node bucket allocations; the only allocation
    // left is the returned vector. Sort order also makes the output deterministic.
    // Call releaseThreadScratch() at phase boundaries to shrink after every phase.
    // Supernodes above the stage cap are hashed, so duplicated edges are removed before being
    // cached.
    workOut = 0;
    if (workEstimate > stageCapFromEnv())
        return aggregateNeighborsBounded(supernode, workOut);

    t_staged.clear();

    // No locks needed here - supernodeToOriginal and nodeMapping are read-only after
    // construction. Iterate through all original nodes in this supernode.
    for (node originalNode : supernodeToOriginal[supernode]) {
        // Iterate through neighbors of each original node
        originalGraph.forNeighborsOf(originalNode, [&](node originalNeighbor, edgeweight weight) {
            node neighborSupernode = nodeMapping[originalNeighbor];
            /*
             * An undirected edge sits in the adjacency of both endpoints, so an edge inside
             * this supernode would be aggregated twice. Count it once, from the higher
             * endpoint, mirroring ParallelPartitionCoarsening's aggregation.
             */
            if (neighborSupernode == supernode && originalNode < originalNeighbor)
                return;
            t_staged.emplace_back(neighborSupernode, weight);
        });
    }

    if (t_staged.empty())
        return {};

    std::sort(t_staged.begin(), t_staged.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    workOut = t_staged.size();

    // De-duplicate t_staged
    size_t k = 0;
    for (size_t i = 0; i < t_staged.size();) {
        const node cur = t_staged[i].first;
        edgeweight acc = 0.0;
        for (; i < t_staged.size() && t_staged[i].first == cur; ++i)
            acc += t_staged[i].second;
        if (acc > 0.0) // Only include edges with positive weight
            t_staged[k++] = {cur, acc};
    }
    return {t_staged.begin(), t_staged.begin() + k};
}

std::vector<std::pair<node, edgeweight>>
CoarsenedGraphView::aggregateNeighborsBounded(node supernode, size_t &workOut) const {
    // Hash-map based neighbor aggregation
    t_accMap.clear();
    size_t staged = 0;
    for (node originalNode : supernodeToOriginal[supernode]) {
        originalGraph.forNeighborsOf(originalNode, [&](node originalNeighbor, edgeweight weight) {
            const node neighborSupernode = nodeMapping[originalNeighbor];
            if (neighborSupernode == supernode && originalNode < originalNeighbor)
                return;
            ++staged;
            t_accMap[neighborSupernode] += weight;
        });
    }
    workOut = staged;

    std::vector<std::pair<node, edgeweight>> neighbors;
    neighbors.reserve(t_accMap.size());
    for (const auto &kv : t_accMap)
        if (kv.second > 0.0) // Only include edges with positive weight
            neighbors.emplace_back(kv.first, kv.second);
    t_accMap.clear();
    std::sort(neighbors.begin(), neighbors.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    return neighbors;
}

std::vector<std::pair<node, edgeweight>>
CoarsenedGraphView::computeNeighbors(node supernode) const {
    if (!hasNode(supernode))
        return {};
    {
        std::lock_guard<std::mutex> guard(cacheMutexes_[supernode]);
        if (neighborCached_[supernode])
            return neighborCache_[supernode];
    }
    // Uncached: aggregate without holding the lock, then publish under it. A loser of the
    // race recomputes the identical vector and discards it. The work estimate is hoisted
    // here so the hot path pays the O(members) scan once.
    const size_t workEstimate = aggregationWork(supernode);
    size_t work = 0;
    auto neighbors = aggregateNeighbors(supernode, workEstimate, work);
    // Lazy promotion: hot supernodes pay aggregation on every visit, so keep the result.
    if (work >= eagerNeighborThreshold) {
        std::lock_guard<std::mutex> guard(cacheMutexes_[supernode]);
        if (!neighborCached_[supernode]) {
            neighborCache_[supernode] = neighbors;
            neighborCached_[supernode] = 1;
        }
    }
    return neighbors;
}

void CoarsenedGraphView::ensureEagerCache() {
    // Eager path: materialize hot supernodes up front, in parallel. The work estimate is
    // O(members) per supernode from already-resident mapping storage. Each entry has its
    // own mutex, so supernodes proceed independently.
    if (eagerNeighborThreshold == none) {
        return; // caching disabled
    }
#pragma omp parallel for schedule(dynamic)
    for (omp_index s = 0; s < static_cast<omp_index>(numSupernodes); ++s) {
        const node supernode = static_cast<node>(s);
        // Hoisted work estimate: O(members) scan once per supernode, reused for eligibility
        // and the stage-cap branch. Mapping storage is read-only, so computing it outside
        // the entry lock is safe. Aggregation itself runs unlocked (t_staged is
        // thread-local) and publishes under the lock.
        const size_t workEstimate = aggregationWork(supernode);
        bool eligible = false;
        {
            std::lock_guard<std::mutex> guard(cacheMutexes_[supernode]);
            eligible = !neighborCached_[supernode] && workEstimate >= eagerNeighborThreshold;
        }
        if (!eligible)
            continue;
        size_t work = 0;
        auto neighbors = aggregateNeighbors(supernode, workEstimate, work);
        tlx::unused(work);
        std::lock_guard<std::mutex> guard(cacheMutexes_[supernode]);
        if (!neighborCached_[supernode]) {
            neighborCache_[supernode] = std::move(neighbors);
            neighborCached_[supernode] = 1;
        }
    }
}

} /* namespace NetworKit */
