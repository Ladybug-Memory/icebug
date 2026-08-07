/*
 * GraphR.hpp
 *
 *  Created on: Feb 8, 2026
 *  Read-only CSR-based graph implementation
 */

#ifndef NETWORKIT_GRAPH_GRAPH_R_HPP_
#define NETWORKIT_GRAPH_GRAPH_R_HPP_

#include <cassert>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

#include <arrow/api.h>

#include <networkit/Globals.hpp>
#include <networkit/graph/AttributedGraphBase.hpp>
#include <networkit/graph/EdgeIterators.hpp>
#include <networkit/graph/GraphConcepts.hpp>
#include <networkit/graph/GraphIterationOps.hpp>
#include <networkit/graph/GraphTypes.hpp>
#include <networkit/graph/NeighborIterators.hpp>
#include <networkit/graph/NodeIterators.hpp>

namespace NetworKit {

/**
 * @ingroup graph
 * GraphR - Read-only graph with CSR (Compressed Sparse Row) storage.
 *
 * This class provides a memory-efficient, immutable graph representation using
 * Arrow CSR arrays for zero-copy data sharing. It is optimized for read-only
 * operations and analysis algorithms.
 *
 * For graphs that require mutation (adding/removing nodes or edges), use GraphW instead.
 */
class GraphR final : public AttributedGraphBase<GraphR> {
    count n, m, storedNumberOfSelfLoops;
    node z;
    count t;
    bool weighted, directed;

    std::shared_ptr<arrow::UInt64Array> outEdgesCSRIndices, outEdgesCSRIndptr;
    std::shared_ptr<arrow::UInt64Array> inEdgesCSRIndices, inEdgesCSRIndptr;
    std::shared_ptr<arrow::DoubleArray> outEdgesCSRWeights, inEdgesCSRWeights;

    template <bool Weighted>
    static auto neighborsAt(const std::shared_ptr<arrow::UInt64Array> &indptr,
                            const std::shared_ptr<arrow::UInt64Array> &indices,
                            const std::shared_ptr<arrow::DoubleArray> &weights, node u) {
        const auto start = indptr->Value(u);
        const auto deg = indptr->Value(u + 1) - start;
        const node *first = reinterpret_cast<const node *>(indices->raw_values()) + start;

        if constexpr (Weighted) {
            assert(weights != nullptr);
            const edgeweight *firstWeight = weights->raw_values() + start;
            return std::ranges::subrange(NeighborWeightIter(first, firstWeight),
                                         NeighborWeightIter(first + deg, firstWeight + deg));
        } else {
            return std::ranges::subrange(NeighborIter(first), NeighborIter(first + deg));
        }
    }

public:
    /// CSR has no notion of a deleted node: every id in [0, n) exists for the graph's lifetime.
    static constexpr bool alwaysContiguousNodeIds = true;

    /// The CSR constructor requires sorted adjacency, so lookups may binary-search.
    static constexpr bool alwaysSortedNeighborhoods = true;

    using NeighborIter = NeighborIteratorBase<const node *>;
    using NeighborWeightIter = NeighborWeightIteratorBase<const node *, const edgeweight *>;

    using NodeIterator = NodeIteratorBase<GraphR>;
    using NodeRange = NodeRangeBase<GraphR>;
    using EdgeIterator = EdgeTypeIterator<GraphR, Edge>;
    using EdgeWeightIterator = EdgeTypeIterator<GraphR, WeightedEdge>;
    using EdgeRange = EdgeTypeRange<GraphR, Edge>;
    using EdgeWeightRange = EdgeTypeRange<GraphR, WeightedEdge>;

    /**
     * Constructor that creates a graph from Arrow CSR arrays for zero-copy memory efficiency.
     * @param n Number of nodes.
     * @param directed If set to @c true, the graph will be directed.
     * @param outIndices Arrow array containing neighbor node IDs for outgoing edges (CSR indices).
     * @param outIndptr Arrow array containing offsets into outIndices for each node (CSR indptr).
     * @param inIndices Arrow array containing neighbor node IDs for incoming edges (only for
     * directed graphs).
     * @param inIndptr Arrow array containing offsets into inIndices for each node (only for
     * directed graphs).
     * @param outWeights Arrow array containing edge weights for outgoing edges (optional).
     * @param inWeights Arrow array containing edge weights for incoming edges (optional, only for
     * directed graphs).
     */
    GraphR(count n, bool directed, std::shared_ptr<arrow::UInt64Array> outIndices,
           std::shared_ptr<arrow::UInt64Array> outIndptr,
           std::shared_ptr<arrow::UInt64Array> inIndices = nullptr,
           std::shared_ptr<arrow::UInt64Array> inIndptr = nullptr,
           std::shared_ptr<arrow::DoubleArray> outWeights = nullptr,
           std::shared_ptr<arrow::DoubleArray> inWeights = nullptr);

    GraphR(const GraphR &) = default;
    GraphR(GraphR &&) noexcept = default;
    GraphR &operator=(const GraphR &) = default;
    GraphR &operator=(GraphR &&) noexcept = default;
    ~GraphR() = default;

    /* ITERATION, LOOKUPS, AND WEIGHT AGGREGATES (free operations) */

    /*
     * Thin forwarders to the free operations in GraphIterationOps, which are implemented once in
     * terms of the GraphLike primitives. A call goes straight to the compile-time-selected body,
     * so there is no runtime dispatch and no base class to inherit.
     */

    /// True when node ids are dense, so iteration may skip the hasNode() check.
    /*
     * The defaulted template parameter defers this body to use-time: a plain member would be
     * instantiated while GraphR is still incomplete, before outNeighbors's deduced return type
     * exists, which fails the GraphLike constraint the operation is checked against.
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

    /* GLOBAL PROPERTIES */

    count numberOfNodes() const noexcept { return n; }
    count numberOfEdges() const noexcept { return m; }
    count numberOfSelfLoops() const noexcept { return storedNumberOfSelfLoops; }
    index upperNodeIdBound() const noexcept { return z; }
    bool isEmpty() const noexcept { return !n; }
    bool isWeighted() const noexcept { return weighted; }
    bool isDirected() const noexcept { return directed; }
    count time() const noexcept { return t; }

    /// CSR carries no edge ids; see edgeId().
    bool hasEdgeIds() const noexcept { return false; }
    index upperEdgeIdBound() const noexcept { return 0; }

    bool getKeepEdgesSorted() const noexcept { return alwaysSortedNeighborhoods; }
    bool getMaintainCompactEdges() const noexcept { return true; }
    bool hasSortedNeighborhoods() const noexcept { return alwaysSortedNeighborhoods; }

    bool checkConsistency() const;

    /* NODE AND EDGE PROPERTIES */

    bool hasNode(node v) const noexcept { return v < z; }
    bool hasEdge(node u, node v) const;
    count degree(node v) const;
    count degreeIn(node v) const;
    count degreeOut(node v) const { return degree(v); }
    bool isIsolated(node v) const;

    /**
     * Return edge weight of edge {@a u,@a v}. Returns 0 if edge does not
     * exist. If weights are provided during construction, returns the actual
     * edge weight; otherwise returns defaultEdgeWeight (1.0).
     */
    edgeweight weight(node u, node v) const;

    /// Always throws: the CSR constructor assigns no edge ids.
    edgeid edgeId(node u, node v) const;
    std::pair<node, node> edgeById(index id) const;

    /* INDEXED NEIGHBOR ACCESS */

    index indexOfNeighbor(node u, node v) const { return indexInOutEdgeArray(u, v); }
    index indexInInEdgeArray(node v, node u) const;
    index indexInOutEdgeArray(node u, node v) const;

    node getIthNeighbor(Unsafe, node u, index i) const;
    node getIthNeighbor(node u, index i) const;
    node getIthInNeighbor(node u, index i) const;
    edgeweight getIthNeighborWeight(node u, index i) const;
    edgeweight getIthNeighborWeight(Unsafe, node u, index i) const {
        return getIthNeighborWeight(u, i);
    }
    std::pair<node, edgeweight> getIthNeighborWithWeight(node u, index i) const;
    std::pair<node, edgeweight> getIthNeighborWithWeight(Unsafe, node u, index i) const {
        return getIthNeighborWithWeight(u, i);
    }
    /// Always throws: the CSR constructor assigns no edge ids.
    std::pair<node, edgeid> getIthNeighborWithId(node u, index i) const;

    std::vector<node> getNeighborsVector(node u, bool inEdges = false) const;
    std::pair<std::vector<node>, std::vector<edgeweight>>
    getNeighborsWithWeightsVector(node u, bool inEdges = false) const;

    /* RANGES */

    NodeRange nodeRange() const noexcept { return NodeRange(*this); }
    EdgeRange edgeRange() const noexcept { return EdgeRange(*this); }
    EdgeWeightRange edgeWeightRange() const noexcept { return EdgeWeightRange(*this); }

    /**
     * The out-neighbors of @a u, as a range over the CSR arrays. The range borrows this graph's
     * Arrow buffers and must not outlive it.
     *
     * @tparam Weighted select the (node, weight) payload. Requires isWeighted(); an unweighted
     * graph has no weight array, and callers that want a synthesized weight iterate the
     * unweighted range and read neighborWeight() off the node.
     */
    template <bool Weighted>
    auto outNeighbors(node u) const {
        return neighborsAt<Weighted>(outEdgesCSRIndptr, outEdgesCSRIndices, outEdgesCSRWeights, u);
    }

    /// The in-neighbors of @a u. On an undirected graph these are the out-neighbors.
    template <bool Weighted>
    auto inNeighbors(node u) const {
        if (!directed) {
            return outNeighbors<Weighted>(u);
        }
        return neighborsAt<Weighted>(inEdgesCSRIndptr, inEdgesCSRIndices, inEdgesCSRWeights, u);
    }

    auto neighborRange(node u) const { return outNeighbors<false>(u); }
    auto weightNeighborRange(node u) const { return outNeighbors<true>(u); }
    auto inNeighborRange(node u) const { return inNeighbors<false>(u); }
    auto weightInNeighborRange(node u) const { return inNeighbors<true>(u); }

private:
    /// The slot of edge (@a u, @a v) in @a indices, or none. Bisects; adjacency is sorted.
    index findInAdjacency(const std::shared_ptr<arrow::UInt64Array> &indptr,
                          const std::shared_ptr<arrow::UInt64Array> &indices, node u, node v) const;
    count degreeCSR(node u, bool incoming = false) const;
    std::pair<const node *, count> getCSROutNeighbors(node u) const;
    std::pair<const node *, count> getCSRInNeighbors(node u) const;
};

static_assert(GraphLike<GraphR>);
static_assert(std::derived_from<GraphR, AttributedGraphBase<GraphR>>);
static_assert(!IndexedGraph<GraphR>, "CSR carries no edge ids; see GraphR::edgeId.");
static_assert(!MutableGraph<GraphR>);
static_assert(GraphR::alwaysContiguousNodeIds);

} // namespace NetworKit

// ReferenceGraph's member templates (for* family etc.) are defined in ReferenceGraphImpl.hpp, which
// additionally requires GraphW and GraphR to be complete. Include it here so any TU that uses a
// concrete graph class -- without necessarily pulling in the umbrella Graph.hpp (e.g. test files
// including only GraphR.hpp) -- still gets those template definitions.
#include <networkit/graph/ReferenceGraphImpl.hpp>

#endif // NETWORKIT_GRAPH_GRAPH_R_HPP_
