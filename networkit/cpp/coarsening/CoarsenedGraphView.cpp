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
#include <utility>
#include <vector>
#include <tlx/unused.hpp>

namespace NetworKit {

namespace {
// Phase-scoped scratch for aggregation: reused across calls within a move/refine phase,
// released via releaseThreadScratch() at each phase boundary so a high-water mark from one
// coarsening level cannot leak into the next.
thread_local std::vector<std::pair<node, edgeweight>> t_staged;

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
} // namespace

void CoarsenedGraphView::releaseThreadScratch() {
    t_staged.clear();
    t_staged.shrink_to_fit();
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
CoarsenedGraphView::aggregateNeighbors(node supernode, size_t &workOut) const {
    // Flat aggregation (InducedSubgraphView-style): stage (supernode, weight) pairs into the
    // phase-scoped thread-local buffer, sort, then combine runs. This avoids per-call
    // std::unordered_map hashing and its per-node bucket allocations; the only allocation
    // left is the returned vector. Sort order also makes the output deterministic.
    // Call releaseThreadScratch() at phase boundaries to shrink after every phase.
    t_staged.clear();
    workOut = 0;

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

    // Combine runs of equal supernodes into the output vector.
    std::vector<std::pair<node, edgeweight>> neighbors;
    neighbors.reserve(t_staged.size());
    node cur = t_staged[0].first;
    edgeweight acc = 0.0;
    for (const auto &entry : t_staged) {
        if (entry.first != cur) {
            if (acc > 0.0)
                neighbors.emplace_back(cur, acc);
            cur = entry.first;
            acc = 0.0;
        }
        acc += entry.second;
    }
    if (acc > 0.0) // Only include edges with positive weight
        neighbors.emplace_back(cur, acc);

    workOut = t_staged.size();
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
    // race recomputes the identical vector and discards it.
    size_t work = 0;
    auto neighbors = aggregateNeighbors(supernode, work);
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
        // Eligibility check under the entry lock; aggregation itself runs unlocked
        // (t_staged is thread-local) and publishes under the lock.
        bool eligible = false;
        {
            std::lock_guard<std::mutex> guard(cacheMutexes_[supernode]);
            eligible =
                !neighborCached_[supernode] && aggregationWork(supernode) >= eagerNeighborThreshold;
        }
        if (!eligible)
            continue;
        size_t work = 0;
        auto neighbors = aggregateNeighbors(supernode, work);
        tlx::unused(work);
        std::lock_guard<std::mutex> guard(cacheMutexes_[supernode]);
        if (!neighborCached_[supernode]) {
            neighborCache_[supernode] = std::move(neighbors);
            neighborCached_[supernode] = 1;
        }
    }
}

} /* namespace NetworKit */
