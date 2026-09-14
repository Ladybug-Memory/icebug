/*
 * CoarsenedGraphView.cpp
 *
 *  Implementation of memory-efficient coarsened graph view
 */

#include <networkit/auxiliary/Log.hpp>
#include <networkit/auxiliary/Timer.hpp>
#include <networkit/coarsening/CoarsenedGraphView.hpp>

#include <algorithm>
#include <utility>
#include <vector>

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

    TRACE("Created layered CoarsenedGraphView with ", numSupernodes, " supernodes from ",
          baseView.numberOfNodes(), " base supernodes");
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
CoarsenedGraphView::computeNeighbors(node supernode) const {
    // Flat aggregation (InducedSubgraphView-style): stage (supernode, weight) pairs into a
    // reusable thread-local buffer, sort, then combine runs. This avoids per-call
    // std::unordered_map hashing and its per-node bucket allocations; the only allocation
    // left is the returned vector. Sort order also makes the output deterministic.
    thread_local std::vector<std::pair<node, edgeweight>> staged;
    staged.clear();

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
            staged.emplace_back(neighborSupernode, weight);
        });
    }

    if (staged.empty())
        return {};

    std::sort(staged.begin(), staged.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    // Combine runs of equal supernodes into the output vector.
    std::vector<std::pair<node, edgeweight>> neighbors;
    neighbors.reserve(staged.size());
    node cur = staged[0].first;
    edgeweight acc = 0.0;
    for (const auto &entry : staged) {
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

    return neighbors;
}

} /* namespace NetworKit */
