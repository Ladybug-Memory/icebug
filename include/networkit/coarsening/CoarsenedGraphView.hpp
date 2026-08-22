/*
 * CoarsenedGraphView.hpp
 *
 *  Memory-efficient layered graph view for coarsening operations
 */

#ifndef NETWORKIT_COARSENING_COARSENED_GRAPH_VIEW_HPP_
#define NETWORKIT_COARSENING_COARSENED_GRAPH_VIEW_HPP_

#include <utility>
#include <vector>

#include <networkit/Globals.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/graph/GraphConcepts.hpp>
#include <networkit/graph/GraphIterationOps.hpp>
#include <networkit/structures/Partition.hpp>

namespace NetworKit {

/**
 * @ingroup coarsening
 * Memory-efficient zero-copy view of a coarsened graph that avoids creating new graph structures.
 * This class provides a GraphLike interface over the original CSR graph by maintaining only the
 * node mapping information, computing edges on-demand.
 *
 * The view implements the GraphLike concept: it declares the primitives from
 * <networkit/graph/GraphConcepts.hpp> and inherits the whole @c for* family, lookups and weight
 * aggregates as free operations, re-exposed as members. Supernode ids are dense in @c [0,
 * numberOfNodes()), so iteration never consults hasNode().
 *
 * Neighborhoods are aggregated on demand: every call to outNeighbors() walks the adjacency of the
 * member nodes and merges weights into one entry per adjacent supernode. The returned range is an
 * owning vector rather than a borrowed view -- there is no stored coarsened adjacency to point at.
 * Entries with non-positive aggregated weight are dropped.
 */
class CoarsenedGraphView {
public:
    /// Node ids are dense: the constructor compacts the partition, so every id in
    /// [0, numberOfNodes()) exists.
    static constexpr bool alwaysContiguousNodeIds = true;

    /**
     * Construct a coarsened graph view from the original graph and partition.
     * @param originalGraph The original CSR graph
     * @param partition Partition defining how nodes are grouped into supernodes
     */
    CoarsenedGraphView(const Graph &originalGraph, const Partition &partition);

    /**
     * Construct a coarsened graph view by coarsening an existing coarsened view.
     * The resulting view still references the original graph and composes the
     * original-node mapping directly.
     * @param baseView Existing coarsened view
     * @param partition Partition over the base view's supernodes
     */
    CoarsenedGraphView(const CoarsenedGraphView &baseView, const Partition &partition);

    /* GRAPHLIKE PRIMITIVES */

    /**
     * Get the number of nodes (supernodes) in the coarsened view
     */
    count numberOfNodes() const { return numSupernodes; }

    /**
     * Get the number of edges in the coarsened view
     *
     * Aggregates every supernode neighborhood once; O(sum of the members' degrees).
     */
    count numberOfEdges() const;

    /// Node ids are dense, so the upper node id bound is the supernode count.
    index upperNodeIdBound() const { return numSupernodes; }

    /// True when @a supernode is in [0, numberOfNodes()).
    bool hasNode(node supernode) const { return supernode < numSupernodes; }

    bool isEmpty() const { return numSupernodes == 0; }

    /**
     * Get the degree of a supernode (number of adjacent supernodes)
     */
    count degree(node supernode) const;

    /// The view is undirected regardless of the original graph.
    bool isDirected() const { return false; }

    /// The view is always weighted: aggregated edge weights are carried on the neighborhoods.
    bool isWeighted() const { return true; }

    /**
     * The number of supernodes carrying a positive-weight self-loop in this view.
     */
    count numberOfSelfLoops() const;

    /**
     * The aggregated out-neighbors of @a supernode: one entry per adjacent supernode, carrying the
     * summed edge weight. The range is an owning vector recomputed on each call; entries with
     * non-positive aggregated weight are dropped.
     */
    template <bool Weighted>
    std::vector<std::pair<node, edgeweight>> outNeighbors(node supernode) const {
        if (!hasNode(supernode))
            return {};
        return computeNeighbors(supernode);
    }

    /// The view is undirected: the in-neighbors are the out-neighbors.
    template <bool Weighted>
    std::vector<std::pair<node, edgeweight>> inNeighbors(node supernode) const {
        return outNeighbors<Weighted>(supernode);
    }

    auto neighborRange(node u) const { return outNeighbors<false>(u); }
    auto weightNeighborRange(node u) const { return outNeighbors<true>(u); }
    auto inNeighborRange(node u) const { return inNeighbors<false>(u); }
    auto weightInNeighborRange(node u) const { return inNeighbors<true>(u); }

    /* ITERATION, LOOKUPS, AND WEIGHT AGGREGATES (free operations) */

    /*
     * Thin forwarders to the free operations in GraphIterationOps, which are implemented once in
     * terms of the GraphLike primitives. A call goes straight to the compile-time-selected body,
     * so there is no runtime dispatch and no base class to inherit.
     */

    template <typename = void>
    bool hasContiguousNodeIds() const {
        return NetworKit::hasContiguousNodeIds(*this);
    }

    template <typename L>
    void forNodes(L handle) const {
        GraphIterationOps::forNodes(*this, handle);
    }

    template <typename L>
    void parallelForNodes(L handle) const {
        GraphIterationOps::parallelForNodes(*this, handle);
    }

    template <typename C, typename L>
    void forNodesWhile(C condition, L handle) const {
        GraphIterationOps::forNodesWhile(*this, condition, handle);
    }

    template <typename L>
    void forNodesInRandomOrder(L handle) const {
        GraphIterationOps::forNodesInRandomOrder(*this, handle);
    }

    template <typename L>
    void balancedParallelForNodes(L handle) const {
        GraphIterationOps::balancedParallelForNodes(*this, handle);
    }

    template <typename L>
    void forNodePairs(L handle) const {
        GraphIterationOps::forNodePairs(*this, handle);
    }

    template <typename L>
    void parallelForNodePairs(L handle) const {
        GraphIterationOps::parallelForNodePairs(*this, handle);
    }

    template <typename L>
    void forEdges(L handle) const {
        GraphIterationOps::forEdges(*this, handle);
    }

    template <typename L>
    void parallelForEdges(L handle) const {
        GraphIterationOps::parallelForEdges(*this, handle);
    }

    template <typename L>
    void forNeighborsOf(node u, L handle) const {
        GraphIterationOps::forNeighborsOf(*this, u, handle);
    }

    template <typename L>
    void forEdgesOf(node u, L handle) const {
        GraphIterationOps::forEdgesOf(*this, u, handle);
    }

    template <typename L>
    void forInNeighborsOf(node u, L handle) const {
        GraphIterationOps::forInNeighborsOf(*this, u, handle);
    }

    template <typename L>
    void forInEdgesOf(node u, L handle) const {
        GraphIterationOps::forInEdgesOf(*this, u, handle);
    }

    template <typename L>
    double parallelSumForNodes(L handle) const {
        return GraphIterationOps::parallelSumForNodes(*this, handle);
    }

    template <typename L>
    double parallelSumForEdges(L handle) const {
        return GraphIterationOps::parallelSumForEdges(*this, handle);
    }

    /**
     * Get the weighted degree of a supernode
     * @param countSelfLoopsTwice If true, count self-loops twice (for undirected graphs)
     */
    template <typename = void>
    edgeweight weightedDegree(node u, bool countSelfLoopsTwice = false) const {
        return GraphIterationOps::weightedDegree(*this, u, countSelfLoopsTwice);
    }

    template <typename = void>
    edgeweight weightedDegreeIn(node u, bool countSelfLoopsTwice = false) const {
        return GraphIterationOps::weightedDegreeIn(*this, u, countSelfLoopsTwice);
    }

    template <typename = void>
    edgeweight totalEdgeWeight() const {
        return GraphIterationOps::totalEdgeWeight(*this);
    }

    /// O(1): the undirected degree.
    count degreeOut(node u) const { return degree(u); }

    /// O(1): the undirected degree.
    count degreeIn(node u) const { return degree(u); }

    /// O(deg(u)): linear scan over the aggregated neighborhood.
    template <typename = void>
    bool hasEdge(node u, node v) const {
        if (!hasNode(u) || !hasNode(v))
            return false;
        return GraphIterationOps::hasEdge(*this, u, v);
    }

    /// O(deg(u)); nullWeight when the edge is absent.
    template <typename = void>
    edgeweight weight(node u, node v) const {
        if (!hasNode(u) || !hasNode(v))
            return nullWeight;
        return GraphIterationOps::weight(*this, u, v);
    }

    /* VIEW-SPECIFIC ACCESSORS */

    /**
     * Get the mapping from original nodes to supernodes
     */
    const std::vector<node> &getNodeMapping() const { return nodeMapping; }

    /**
     * Get original nodes that belong to a supernode
     */
    const std::vector<node> &getOriginalNodes(node supernode) const;

private:
    const Graph &originalGraph;
    std::vector<node> nodeMapping;                      // original_node -> supernode
    std::vector<std::vector<node>> supernodeToOriginal; // supernode -> [original_nodes]
    count numSupernodes;

    /**
     * Compute the aggregated neighbors of a supernode (on demand, no caching)
     */
    std::vector<std::pair<node, edgeweight>> computeNeighbors(node supernode) const;
};

static_assert(GraphLike<CoarsenedGraphView>,
              "the coarsened view must be usable wherever a graph type is");
static_assert(!IndexedGraph<CoarsenedGraphView>, "aggregated edges carry no edge ids");
static_assert(!MutableGraph<CoarsenedGraphView>, "a view cannot be mutated");
static_assert(CoarsenedGraphView::alwaysContiguousNodeIds);

} /* namespace NetworKit */

#endif // NETWORKIT_COARSENING_COARSENED_GRAPH_VIEW_HPP_
