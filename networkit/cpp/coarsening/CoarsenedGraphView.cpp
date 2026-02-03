/*
 * CoarsenedGraphView.cpp
 *
 *  Implementation of memory-efficient coarsened graph view
 */

#include <networkit/auxiliary/Log.hpp>
#include <networkit/auxiliary/Timer.hpp>
#include <networkit/coarsening/CoarsenedGraphView.hpp>

namespace NetworKit {

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

    TRACE("Created CoarsenedGraphView with ", numSupernodes, " supernodes from ",
          originalGraph.numberOfNodes(), " original nodes");
}

count CoarsenedGraphView::numberOfEdges() const {
    count edges = 0;
    for (node u = 0; u < numberOfNodes(); ++u) {
        const auto &neighbors = getNeighbors(u);
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
    return getNeighbors(supernode).size();
}

edgeweight CoarsenedGraphView::weightedDegree(node supernode, bool countSelfLoops) const {
    if (!hasNode(supernode))
        return 0.0;

    const auto &neighbors = getNeighbors(supernode);

    edgeweight totalWeight = 0.0;
    for (const auto &entry : neighbors) {
        if (entry.first == supernode) {
            if (countSelfLoops) {
                totalWeight += entry.second;
            }
        } else {
            totalWeight += entry.second;
        }
    }
    return totalWeight;
}

bool CoarsenedGraphView::hasEdge(node u, node v) const {
    if (!hasNode(u) || !hasNode(v))
        return false;

    const auto &neighbors = getNeighbors(u);
    for (const auto &entry : neighbors) {
        if (entry.first == v) {
            return true;
        }
    }
    return false;
}

edgeweight CoarsenedGraphView::weight(node u, node v) const {
    if (!hasNode(u) || !hasNode(v))
        return 0.0;

    const auto &neighbors = getNeighbors(u);
    for (const auto &entry : neighbors) {
        if (entry.first == v) {
            return entry.second;
        }
    }
    return 0.0;
}

const std::vector<node> &CoarsenedGraphView::getOriginalNodes(node supernode) const {
    if (!hasNode(supernode)) {
        static const std::vector<node> empty;
        return empty;
    }
    return supernodeToOriginal[supernode];
}

std::vector<std::pair<node, edgeweight>>
CoarsenedGraphView::computeNeighbors(node supernode) const {
    std::unordered_map<node, edgeweight> aggregatedWeights;

    // No locks needed here - supernodeToOriginal and nodeMapping are read-only after construction
    // Iterate through all original nodes in this supernode
    for (node originalNode : supernodeToOriginal[supernode]) {
        // Iterate through neighbors of each original node
        originalGraph.forNeighborsOf(originalNode, [&](node originalNeighbor, edgeweight weight) {
            node neighborSupernode = nodeMapping[originalNeighbor];
            // Aggregate weights to the same supernode
            aggregatedWeights[neighborSupernode] += weight;
        });
    }

    // Convert to vector format
    std::vector<std::pair<node, edgeweight>> neighbors;
    neighbors.reserve(aggregatedWeights.size());

    for (const auto &entry : aggregatedWeights) {
        if (entry.second > 0.0) { // Only include edges with positive weight
            neighbors.emplace_back(entry.first, entry.second);
        }
    }

    return neighbors;
}

const std::vector<std::pair<node, edgeweight>> &
CoarsenedGraphView::getNeighbors(node supernode) const {
    // Double-checked locking pattern to minimize lock contention
    // First check without lock (fast path for cached entries)
    {
        std::lock_guard<std::mutex> lock(cacheMutex);
        auto it = neighborCache.find(supernode);
        if (it != neighborCache.end()) {
            return it->second;
        }
    }

    // Compute neighbors WITHOUT holding the lock (slow path)
    // This is safe because computeNeighbors() only reads immutable data
    auto result = computeNeighbors(supernode);

    // Now acquire lock and insert the result
    std::lock_guard<std::mutex> lock(cacheMutex);
    // Check again in case another thread computed it while we were computing
    auto it = neighborCache.find(supernode);
    if (it != neighborCache.end()) {
        // Another thread beat us to it, use their result
        return it->second;
    }
    // We're the first to compute it, insert our result
    auto [insertIt, inserted] = neighborCache.emplace(supernode, std::move(result));
    return insertIt->second;
}

} /* namespace NetworKit */
