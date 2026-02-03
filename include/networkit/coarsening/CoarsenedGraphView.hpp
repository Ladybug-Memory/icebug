/*
 * CoarsenedGraphView.hpp
 *
 *  Memory-efficient layered graph view for coarsening operations
 */

#ifndef NETWORKIT_COARSENING_COARSENED_GRAPH_VIEW_HPP_
#define NETWORKIT_COARSENING_COARSENED_GRAPH_VIEW_HPP_

#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <networkit/Globals.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/structures/Partition.hpp>

namespace NetworKit {

/**
 * @ingroup coarsening
 * Memory-efficient view of a coarsened graph that avoids creating new graph structures.
 * This class provides a graph-like interface over the original CSR graph by maintaining
 * only the node mapping information, computing edges on-demand.
 */
class CoarsenedGraphView {
public:
    /**
     * Construct a coarsened graph view from the original graph and partition.
     * @param originalGraph The original CSR graph
     * @param partition Partition defining how nodes are grouped into supernodes
     */
    CoarsenedGraphView(const Graph &originalGraph, const Partition &partition);

    /**
     * Get the number of nodes (supernodes) in the coarsened view
     */
    count numberOfNodes() const { return numSupernodes; }

    /**
     * Get the number of edges in the coarsened view
     */
    count numberOfEdges() const;

    /**
     * Check if a supernode exists
     */
    bool hasNode(node supernode) const { return supernode < numSupernodes; }

    /**
     * Get the degree of a supernode (number of adjacent supernodes)
     */
    count degree(node supernode) const;

    /**
     * Get the weighted degree of a supernode
     */
    edgeweight weightedDegree(node supernode, bool countSelfLoops = false) const;

    /**
     * Check if there's an edge between two supernodes
     */
    bool hasEdge(node u, node v) const;

    /**
     * Get the weight of an edge between two supernodes
     */
    edgeweight weight(node u, node v) const;

    /**
     * Iterate over neighbors of a supernode
     * @param supernode The supernode to iterate neighbors for
     * @param handle Function to call for each neighbor: void(node neighbor, edgeweight weight)
     */
    template <typename Lambda>
    void forNeighborsOf(node supernode, Lambda handle) const {
        if (!hasNode(supernode))
            return;

        auto neighbors = getNeighbors(supernode);
        for (const auto &entry : neighbors) {
            handle(entry.first, entry.second);
        }
    }

    /**
     * Iterate over all edges in the coarsened view
     * @param handle Function to call for each edge: void(node u, node v, edgeweight weight)
     */
    template <typename Lambda>
    void forEdges(Lambda handle) const {
        for (node u = 0; u < numberOfNodes(); ++u) {
            forNeighborsOf(u, [&](node v, edgeweight w) {
                if (u <= v) { // Only report each edge once for undirected graphs
                    handle(u, v, w);
                }
            });
        }
    }

    /**
     * Parallel iteration over nodes
     */
    template <typename Lambda>
    void parallelForNodes(Lambda handle) const {
#pragma omp parallel for
        for (omp_index u = 0; u < static_cast<omp_index>(numberOfNodes()); ++u) {
            handle(static_cast<node>(u));
        }
    }

    /**
     * Get the mapping from original nodes to supernodes
     */
    const std::vector<node> &getNodeMapping() const { return nodeMapping; }

    /**
     * Get original nodes that belong to a supernode
     */
    const std::vector<node> &getOriginalNodes(node supernode) const;

    /**
     * Check if the graph is weighted (always true for coarsened views)
     */
    bool isWeighted() const { return true; }

    /**
     * Check if the graph is directed (always false for current implementation)
     */
    bool isDirected() const { return false; }

    /**
     * Get upper bound for node IDs
     */
    node upperNodeIdBound() const { return numSupernodes; }

private:
    const Graph &originalGraph;
    std::vector<node> nodeMapping;                      // original_node -> supernode
    std::vector<std::vector<node>> supernodeToOriginal; // supernode -> [original_nodes]
    count numSupernodes;

    // Cache for computed neighborhoods to avoid recomputation
    mutable std::unordered_map<node, std::vector<std::pair<node, edgeweight>>> neighborCache;
    mutable std::mutex cacheMutex; // Mutex to protect neighborCache access

    /**
     * Compute neighbors of a supernode on-demand
     */
    std::vector<std::pair<node, edgeweight>> computeNeighbors(node supernode) const;

    /**
     * Get neighbors of a supernode (cached)
     */
    const std::vector<std::pair<node, edgeweight>> &getNeighbors(node supernode) const;
};

} /* namespace NetworKit */

#endif // NETWORKIT_COARSENING_COARSENED_GRAPH_VIEW_HPP_
