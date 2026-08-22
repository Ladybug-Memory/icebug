/*
 * InducedSubgraphView.hpp
 *
 *  A zero-copy induced subgraph over any GraphLike base graph.
 */

#ifndef NETWORKIT_GRAPH_INDUCED_SUBGRAPH_VIEW_HPP_
#define NETWORKIT_GRAPH_INDUCED_SUBGRAPH_VIEW_HPP_

#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

#include <networkit/Globals.hpp>
#include <networkit/graph/GraphConcepts.hpp>
#include <networkit/graph/GraphIterationOps.hpp>
#include <networkit/graph/GraphW.hpp>

namespace NetworKit {

/**
 * @ingroup graph
 * InducedSubgraphView - a zero-copy induced subgraph of a base graph.
 *
 * The view spans a subset of the base graph's nodes and, implicitly, every base edge whose two
 * endpoints are in the subset. It works over any type satisfying the GraphLike concept --
 * GraphR, GraphW, CoarsenedGraphView, another InducedSubgraphView -- and keeps the base graph's
 * node ids, so results computed on the view map straight back onto the base graph.
 *
 * Membership is edited in batches (@c addNodes / @c removeNodes). The induced degrees, the edge
 * count and the self-loop count are maintained incrementally while the batch is applied, which
 * keeps every GraphLike primitive O(1). Neighborhoods are never materialized: each range filters
 * the base adjacency on the fly, so the view's memory footprint is one presence flag and one or
 * two degree counters per base node.
 *
 * The base graph must outlive the view and must not be modified while the view exists; ranges
 * borrowed from the view are additionally invalidated by any addNodes()/removeNodes() call, just
 * like ranges borrowed from a GraphW are invalidated by mutation.
 */
template <typename BaseGraph>
class InducedSubgraphView {
    static_assert(GraphLike<BaseGraph>, "the base graph of an induced view must be GraphLike");

public:
    /**
     * An empty view on @a base.
     * @param base The graph the view spans. Must outlive the view.
     */
    explicit InducedSubgraphView(const BaseGraph &base)
        : base_(&base), directed_(base.isDirected()) {
        const index z = base.upperNodeIdBound();
        exists_.assign(z, 0);
        outDegree_.assign(z, 0);
        if (directed_)
            inDegree_.assign(z, 0);
    }

    /**
     * A view on @a base spanning @a subset. Node ids must exist in the base graph; duplicates are
     * ignored.
     */
    template <typename NodeRange>
    InducedSubgraphView(const BaseGraph &base, const NodeRange &subset)
        : InducedSubgraphView(base) {
        addNodes(subset);
    }

    /// The initializer-list flavor of the subset constructor above.
    InducedSubgraphView(const BaseGraph &base, std::initializer_list<node> subset)
        : InducedSubgraphView(base) {
        addNodes(subset);
    }

    InducedSubgraphView(const InducedSubgraphView &) = default;
    InducedSubgraphView(InducedSubgraphView &&) noexcept = default;
    InducedSubgraphView &operator=(const InducedSubgraphView &) = default;
    InducedSubgraphView &operator=(InducedSubgraphView &&) noexcept = default;

    /* MEMBERSHIP EDITS */

    /// Adds the single node @a u to the subset. Throws if @a u is not in the base graph.
    void addNode(node u) { addNodes(std::initializer_list<node>{u}); }

    /// Adds @a nodes to the subset. Unknown ids throw; already present ids are ignored.
    void addNodes(std::initializer_list<node> nodes) { addNodesImpl(nodes); }

    /// Adds every node of @a nodes to the subset. Unknown ids throw; already present ids are
    /// ignored. The order of @a nodes does not matter.
    template <typename NodeRange>
    void addNodes(const NodeRange &nodes) {
        addNodesImpl(nodes);
    }

    /// Removes the single node @a u from the subset, if present.
    void removeNode(node u) { removeNodes(std::initializer_list<node>{u}); }

    /// Removes @a nodes from the subset. Absent ids are ignored.
    void removeNodes(std::initializer_list<node> nodes) { removeNodesImpl(nodes); }

    /// Removes every node of @a nodes from the subset. Absent ids are ignored.
    template <typename NodeRange>
    void removeNodes(const NodeRange &nodes) {
        removeNodesImpl(nodes);
    }

    /* GRAPHLIKE PRIMITIVES */

    count numberOfNodes() const noexcept { return n_; }

    /// The induced edge count; maintained incrementally, O(1).
    count numberOfEdges() const noexcept { return m_; }

    /// The base graph's upper node id bound: the view keeps the base ids.
    index upperNodeIdBound() const noexcept { return exists_.size(); }

    bool hasNode(node u) const noexcept { return u < exists_.size() && exists_[u]; }

    /// The induced out-degree of @a u; O(1). Requires hasNode(u).
    count degree(node u) const {
        assert(hasNode(u));
        return outDegree_[u];
    }

    bool isEmpty() const noexcept { return n_ == 0; }

    bool isDirected() const noexcept { return directed_; }

    /// Mirrors the base graph: the view carries whatever weights the base carries.
    bool isWeighted() const { return base_->isWeighted(); }

    /// The induced self-loop count; maintained incrementally, O(1).
    count numberOfSelfLoops() const noexcept { return selfLoops_; }

    /**
     * The out-neighbors of @a u inside the view: the base neighborhood filtered by membership.
     * The range borrows the base graph's storage and this view's flags, and is invalidated by any
     * membership edit. @tparam Weighted selects the (node, weight) payload; an unweighted base
     * has no weight array, so only request it when isWeighted() holds.
     */
    template <bool Weighted>
    auto outNeighbors(node u) const {
        assert(hasNode(u));
        return std::ranges::owning_view(base_->template outNeighbors<Weighted>(u))
               | std::views::filter([this](const auto &nb) { return hasNode(neighborTarget(nb)); });
    }

    /// The in-neighbors of @a u inside the view. On an undirected base these are the out-neighbors.
    template <bool Weighted>
    auto inNeighbors(node u) const {
        assert(hasNode(u));
        return std::ranges::owning_view(base_->template inNeighbors<Weighted>(u))
               | std::views::filter([this](const auto &nb) { return hasNode(neighborTarget(nb)); });
    }

    auto neighborRange(node u) const { return outNeighbors<false>(u); }
    auto weightNeighborRange(node u) const { return outNeighbors<true>(u); }
    auto inNeighborRange(node u) const { return inNeighbors<false>(u); }
    auto weightInNeighborRange(node u) const { return inNeighbors<true>(u); }

    /// Honest at runtime: filtering preserves the base neighborhoods' order.
    bool hasSortedNeighborhoods() const {
        if constexpr (requires { base_->hasSortedNeighborhoods(); })
            return base_->hasSortedNeighborhoods();
        else
            return false;
    }

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

    /// O(1). Requires hasNode(u).
    count degreeOut(node u) const { return degree(u); }

    /// O(1). Requires hasNode(u).
    count degreeIn(node u) const {
        assert(hasNode(u));
        return directed_ ? inDegree_[u] : outDegree_[u];
    }

    bool isIsolated(node u) const {
        if (!hasNode(u))
            throw std::runtime_error("Error, the node does not exist!");
        return degree(u) == 0 && (!directed_ || degreeIn(u) == 0);
    }

    /// O(deg(u)). True when both endpoints are in the view and the base carries the edge.
    template <typename = void>
    bool hasEdge(node u, node v) const {
        if (!hasNode(u) || !hasNode(v))
            return false;
        return GraphIterationOps::hasEdge(*this, u, v);
    }

    /// O(deg(u)); nullWeight when the edge is absent from the view.
    template <typename = void>
    edgeweight weight(node u, node v) const {
        if (!hasNode(u) || !hasNode(v))
            return nullWeight;
        return GraphIterationOps::weight(*this, u, v);
    }

    /* VIEW-SPECIFIC OPERATIONS */

    /// The graph the view spans.
    const BaseGraph &getBaseGraph() const noexcept { return *base_; }

    /// The members of the subgraph, in ascending id order. O(z).
    std::vector<node> getNodeSubset() const {
        std::vector<node> result;
        result.reserve(n_);
        for (node u = 0; u < exists_.size(); ++u)
            if (exists_[u])
                result.push_back(u);
        return result;
    }

    /**
     * Nodes outside the view that are out-neighbors of a node inside it. Ascending and unique.
     * This is the search frontier when the view grows one layer at a time.
     */
    std::vector<node> frontier() const {
        std::vector<node> result;
        for (node u : getNodeSubset())
            for (const auto nb : base_->template outNeighbors<false>(u)) {
                const node v = neighborTarget(nb);
                if (!hasNode(v))
                    result.push_back(v);
            }
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());
        return result;
    }

    /**
     * Constructs an explicit mutable subgraph equivalent to this view.
     * @param compact Whether the subgraph should get compact node ids instead of the base ids.
     */
    GraphW realize(bool compact = false) const {
        GraphW out(compact ? n_ : upperNodeIdBound(), isWeighted(), directed_);
        if (compact) {
            std::vector<node> newId(upperNodeIdBound(), none);
            node next = 0;
            forNodes([&](node u) { newId[u] = next++; });
            forEdges([&](node u, node v, edgeweight w) {
                out.addEdge(newId[u], newId[v], isWeighted() ? w : defaultEdgeWeight);
            });
        } else {
            for (node u = 0; u < upperNodeIdBound(); ++u)
                if (!exists_[u])
                    out.removeNode(u);
            forEdges([&](node u, node v, edgeweight w) {
                out.addEdge(u, v, isWeighted() ? w : defaultEdgeWeight);
            });
        }
        return out;
    }

private:
    const BaseGraph *base_;
    /// One presence flag per base node; indexed by the base ids the view keeps.
    std::vector<char> exists_;
    std::vector<count> outDegree_;
    /// Only populated on a directed base.
    std::vector<count> inDegree_;
    count n_ = 0, m_ = 0, selfLoops_ = 0;
    bool directed_;

    template <typename NodeRange>
    void addNodesImpl(const NodeRange &nodes) {
        for (const node v : nodes) {
            if (!base_->hasNode(v))
                throw std::runtime_error("InducedSubgraphView: node is not in the base graph");
            insertNode(v);
        }
    }

    template <typename NodeRange>
    void removeNodesImpl(const NodeRange &nodes) {
        for (const node v : nodes)
            eraseNode(v);
    }

    /// Brings the counters in line with @a v joining the subset. Nodes are applied one at a
    /// time, so edges to nodes added earlier in the same batch are counted exactly once.
    void insertNode(node v) {
        if (exists_[v])
            return;
        exists_[v] = 1;
        ++n_;

        count outDeg = 0;
        for (const auto nb : base_->template outNeighbors<false>(v)) {
            const node u = neighborTarget(nb);
            if (!exists_[u])
                continue;
            if (!directed_) {
                if (u == v)
                    ++selfLoops_;
                else
                    ++outDegree_[u];
            } else {
                // the edge (v, u) is an in-edge of u
                ++inDegree_[u];
                if (u == v)
                    ++selfLoops_;
            }
            ++outDeg;
        }
        outDegree_[v] = outDeg;
        m_ += outDeg;

        if (directed_) {
            count inDeg = 0;
            for (const auto nb : base_->template inNeighbors<false>(v)) {
                const node u = neighborTarget(nb);
                // a self-loop was already accounted for by the out-scan above
                if (u == v || !exists_[u])
                    continue;
                ++outDegree_[u];
                ++inDeg;
            }
            inDegree_[v] += inDeg;
            m_ += inDeg;
        }
    }

    /// The inverse of insertNode(). @a v stays marked present until its own adjacency has been
    /// scanned, so a self-loop is seen exactly once.
    void eraseNode(node v) {
        // ids outside the base's id space were never inserted; indexing exists_ with them
        // would read past its storage
        if (v >= exists_.size() || !exists_[v])
            return;

        count removedOut = 0;
        for (const auto nb : base_->template outNeighbors<false>(v)) {
            const node u = neighborTarget(nb);
            if (!exists_[u])
                continue;
            if (!directed_) {
                if (u == v)
                    --selfLoops_;
                else
                    --outDegree_[u];
            } else {
                --inDegree_[u];
                if (u == v)
                    --selfLoops_;
            }
            ++removedOut;
        }
        m_ -= removedOut;

        if (directed_) {
            count removedIn = 0;
            for (const auto nb : base_->template inNeighbors<false>(v)) {
                const node u = neighborTarget(nb);
                if (u == v || !exists_[u])
                    continue;
                --outDegree_[u];
                ++removedIn;
            }
            m_ -= removedIn;
            inDegree_[v] = 0;
        }

        outDegree_[v] = 0;
        exists_[v] = 0;
        --n_;
    }
};

/*
 * The concept claims are checked on the two concrete base graphs; a base that satisfies
 * GraphLike yields an InducedSubgraphView that does too, and any algorithm written as
 * template <GraphLike G> accepts the view.
 */
static_assert(GraphLike<InducedSubgraphView<GraphW>>);
static_assert(GraphLike<InducedSubgraphView<GraphR>>);
static_assert(!IndexedGraph<InducedSubgraphView<GraphW>>,
              "the view does not re-expose the base's edge ids");
static_assert(!MutableGraph<InducedSubgraphView<GraphW>>,
              "batch membership edits are not the MutableGraph protocol");

} // namespace NetworKit

#endif // NETWORKIT_GRAPH_INDUCED_SUBGRAPH_VIEW_HPP_
