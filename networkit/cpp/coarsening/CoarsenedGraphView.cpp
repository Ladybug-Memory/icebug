/*
 * CoarsenedGraphView.cpp
 *
 *  Implementation of memory-efficient coarsened graph view
 */

#include <networkit/auxiliary/Log.hpp>
#include <networkit/auxiliary/Timer.hpp>
#include <networkit/coarsening/CoarsenedGraphView.hpp>
#include <algorithm>

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

edgeweight CoarsenedGraphView::weightedDegree(node supernode, bool countSelfLoopsTwice) const {
    if (!hasNode(supernode))
        return 0.0;

    const auto &neighbors = getNeighbors(supernode);

    edgeweight totalWeight = 0.0;
    for (const auto &entry : neighbors) {
        if (entry.first == supernode) {
            totalWeight += entry.second;
            if (countSelfLoopsTwice) {
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
    // Hash-map based aggregation dominated runtime on large instances.
    // Collect then sort/fold to reduce hashing and allocator pressure.
    count incidentEdgeUpperBound = 0;
    for (node originalNode : supernodeToOriginal[supernode]) {
        incidentEdgeUpperBound += originalGraph.degree(originalNode);
    }

    std::vector<std::pair<node, edgeweight>> rawNeighbors;
    rawNeighbors.reserve(incidentEdgeUpperBound);

    // No locks needed here - supernodeToOriginal and nodeMapping are read-only after construction
    // Iterate through all original nodes in this supernode
    for (node originalNode : supernodeToOriginal[supernode]) {
        // Iterate through neighbors of each original node
        originalGraph.forNeighborsOf(originalNode, [&](node originalNeighbor, edgeweight weight) {
            node neighborSupernode = nodeMapping[originalNeighbor];
            // For internal edges, aggregate each undirected edge only once.
            // This mirrors ParallelPartitionCoarsening semantics.
            if (neighborSupernode == supernode && originalNode < originalNeighbor) {
                return;
            }
            rawNeighbors.emplace_back(neighborSupernode, weight);
        });
    }

    if (rawNeighbors.empty()) {
        return {};
    }

    std::sort(rawNeighbors.begin(), rawNeighbors.end(),
              [](const std::pair<node, edgeweight> &a, const std::pair<node, edgeweight> &b) {
                  return a.first < b.first;
              });

    // Fold adjacent entries with same neighbor supernode.
    std::vector<std::pair<node, edgeweight>> neighbors;
    neighbors.reserve(rawNeighbors.size());
    node currentNeighbor = rawNeighbors[0].first;
    edgeweight currentWeight = rawNeighbors[0].second;
    for (size_t i = 1; i < rawNeighbors.size(); ++i) {
        if (rawNeighbors[i].first == currentNeighbor) {
            currentWeight += rawNeighbors[i].second;
        } else {
            if (currentWeight > 0.0) {
                neighbors.emplace_back(currentNeighbor, currentWeight);
            }
            currentNeighbor = rawNeighbors[i].first;
            currentWeight = rawNeighbors[i].second;
        }
    }
    if (currentWeight > 0.0) {
        neighbors.emplace_back(currentNeighbor, currentWeight);
    }

    return neighbors;
}

} /* namespace NetworKit */
