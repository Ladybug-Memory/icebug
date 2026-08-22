/*
 * CoarsenedGraphView.cpp
 *
 *  Implementation of memory-efficient coarsened graph view
 */

#include <networkit/auxiliary/Log.hpp>
#include <networkit/auxiliary/Timer.hpp>
#include <networkit/coarsening/CoarsenedGraphView.hpp>

#include <unordered_map>

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
    std::unordered_map<node, edgeweight> aggregatedWeights;

    // No locks needed here - supernodeToOriginal and nodeMapping are read-only after construction
    // Iterate through all original nodes in this supernode
    for (node originalNode : supernodeToOriginal[supernode]) {
        // Iterate through neighbors of each original node
        originalGraph.forNeighborsOf(originalNode, [&](node originalNeighbor, edgeweight weight) {
            node neighborSupernode = nodeMapping[originalNeighbor];
            /*
             * An undirected edge sits in the adjacency of both endpoints, so an edge inside this
             * supernode would be aggregated twice. Count it once, from the higher endpoint,
             * mirroring ParallelPartitionCoarsening's aggregation.
             */
            if (neighborSupernode == supernode && originalNode < originalNeighbor)
                return;
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

} /* namespace NetworKit */
