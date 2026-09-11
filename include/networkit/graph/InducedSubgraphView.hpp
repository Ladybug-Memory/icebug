/*
 * InducedSubgraphView.hpp
 *
 *  A zero-copy induced subgraph over any GraphLike base graph.
 */

#ifndef NETWORKIT_GRAPH_INDUCED_SUBGRAPH_VIEW_HPP_
#define NETWORKIT_GRAPH_INDUCED_SUBGRAPH_VIEW_HPP_

#include <algorithm>
#include <bit>
#include <cassert>
#include <initializer_list>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <networkit/Globals.hpp>
#include <networkit/auxiliary/FlatMap.hpp>
#include <networkit/graph/GraphConcepts.hpp>
#include <networkit/graph/GraphIterationOps.hpp>
#include <networkit/graph/GraphW.hpp>
#include <networkit/graph/ReferenceGraph.hpp>

namespace NetworKit {

namespace InducedSubgraphViewDetail {
/**
 * Stand-in base for bases that are not handle arms (a view of a view, say): there is no embedded
 * handle to carry, so the base contributes nothing. Keeping it empty is what lets the real base,
 * SelfHandled, stay true to its documented requirement that Derived be a variant alternative.
 */
struct NoEmbeddedHandle {};
} // namespace InducedSubgraphViewDetail

/**
 * @ingroup graph
 * InducedSubgraphView - a zero-copy induced subgraph of a base graph.
 *
 * The view spans a subset of the base graph's nodes and, implicitly, every base edge whose two
 * endpoints are in the subset. It works over any type satisfying the GraphLike concept --
 * GraphR, GraphW, CoarsenedGraphView, another InducedSubgraphView.
 *
 * By default the view keeps the base graph's node ids, so results computed on the view map
 * straight back onto the base graph. With @c compact=true the view instead presents dense ids
 * @c [0, n): the compact id of a member is its rank in the ascending base-id order, so
 * @c getNodeSubset()[c] is the base id behind compact id @c c. Compact ids stay dense across
 * membership edits -- which renumbers the survivors -- so translate results back with
 * @c toBaseId() before touching the base graph. The dense ids let algorithms that allocate
 * per-node storage run on the view without paying for the base graph's id space, the same
 * reason @c GraphTools::subgraphFromNodes offers its @c compact flag, but without
 * materializing a copy.
 *
 * Id spaces at a glance (compact view): membership edits (@c addNodes / @c removeNodes),
 * @c getNodeSubset(), @c frontier() and @c toCompactId() speak base-graph ids; every
 * GraphLike primitive (@c hasNode, @c degree, neighborhoods, @c forNodes, ...) speaks compact
 * view ids. In the default non-compact view both spaces coincide.
 *
 * Membership is edited in batches (@c addNodes / @c removeNodes). The induced degrees, the edge
 * count and the self-loop count are maintained incrementally while the batch is applied, which
 * keeps every GraphLike primitive O(1). Neighborhoods are never materialized: each range either
 * filters the base adjacency on the fly or, when the base degree dwarfs the view, walks the view's
 * members and probes the base for adjacency instead (see NeighborRange), so the view's memory
 * footprint is one presence flag, one or two degree counters per base node -- n-sized flat_maps
 * on a compact view -- plus a member index that is rebuilt lazily after edits.
 *
 * The base graph must outlive the view and must not be modified while the view exists; ranges
 * borrowed from the view are additionally invalidated by any addNodes()/removeNodes() call, just
 * like ranges borrowed from a GraphW are invalidated by mutation. The invalidation is caught by
 * an assertion in debug builds; in release it is silent, so keep the edit/iterate phases apart.
 */
template <typename BaseGraph>
class InducedSubgraphView : public std::conditional_t<std::is_same_v<BaseGraph, GraphW>
                                                          || std::is_same_v<BaseGraph, GraphR>,
                                                      SelfHandled<InducedSubgraphView<BaseGraph>>,
                                                      InducedSubgraphViewDetail::NoEmbeddedHandle> {
    static_assert(GraphLike<BaseGraph>, "the base graph of an induced view must be GraphLike");

public:
    /**
     * An empty view on @a base.
     * @param base The graph the view spans. Must outlive the view.
     * @param compact When true, the view presents dense ids @c [0, n) instead of the base ids.
     */
    explicit InducedSubgraphView(const BaseGraph &base, bool compact = false)
        : base_(&base), directed_(base.isDirected()), compact_(compact) {
        const index z = base.upperNodeIdBound();
        exists_.assign(z, 0);
        // A compact view keeps its degree bookkeeping in the n-sized flat_maps below rather
        // than in z-sized arrays: the compact id of a member is its rank among the members,
        // so nothing needs indexing by the base id space.
        if (!compact_) {
            outDegree_.assign(z, 0);
            if (directed_)
                inDegree_.assign(z, 0);
        }
    }

    /**
     * A view on @a base spanning @a subset. Node ids must exist in the base graph; duplicates are
     * ignored. @a subset always holds base-graph ids, even for a compact view.
     */
    template <typename NodeRange>
    InducedSubgraphView(const BaseGraph &base, const NodeRange &subset, bool compact = false)
        : InducedSubgraphView(base, compact) {
        addNodes(subset);
    }

    /// The initializer-list flavor of the subset constructor above.
    InducedSubgraphView(const BaseGraph &base, std::initializer_list<node> subset,
                        bool compact = false)
        : InducedSubgraphView(base, compact) {
        addNodes(subset);
    }

    InducedSubgraphView(const InducedSubgraphView &) = default;
    InducedSubgraphView(InducedSubgraphView &&) noexcept = default;
    InducedSubgraphView &operator=(const InducedSubgraphView &) = default;
    InducedSubgraphView &operator=(InducedSubgraphView &&) noexcept = default;

    /* MEMBERSHIP EDITS */

    /// Adds the single base-graph node @a u to the subset. Throws if @a u is not in the base
    /// graph; the batch is atomic, so a failed call adds nothing. On a compact view the
    /// compact ids of pre-existing members may shift.
    void addNode(node u) { addNodes(std::initializer_list<node>{u}); }

    /// Adds base-graph @a nodes to the subset. Unknown ids throw and leave the view unchanged
    /// (the batch is atomic); already present ids are ignored. On a compact view the compact
    /// ids of pre-existing members may shift.
    void addNodes(std::initializer_list<node> nodes) { addNodesImpl(nodes); }

    /// Adds every base-graph node of @a nodes to the subset. Unknown ids throw and leave the
    /// view unchanged (the batch is atomic); already present ids are ignored. The order of
    /// @a nodes does not matter. On a compact view the compact ids of pre-existing members
    /// may shift.
    template <typename NodeRange>
    void addNodes(const NodeRange &nodes) {
        addNodesImpl(nodes);
    }

    /// Removes the single base-graph node @a u from the subset, if present. On a compact view
    /// the survivors are renumbered to stay dense.
    void removeNode(node u) { removeNodes(std::initializer_list<node>{u}); }

    /// Removes base-graph @a nodes from the subset. Absent ids are ignored. On a compact view
    /// the survivors are renumbered to stay dense.
    void removeNodes(std::initializer_list<node> nodes) { removeNodesImpl(nodes); }

    /// Removes every base-graph node of @a nodes from the subset. Absent ids are ignored. On a
    /// compact view the survivors are renumbered to stay dense.
    template <typename NodeRange>
    void removeNodes(const NodeRange &nodes) {
        removeNodesImpl(nodes);
    }

    /* GRAPHLIKE PRIMITIVES */

    count numberOfNodes() const noexcept { return n_; }

    /// The induced edge count; maintained incrementally, O(1).
    count numberOfEdges() const noexcept { return m_; }

    /// The view's upper node id bound: the base bound by default, the member count when compact.
    index upperNodeIdBound() const noexcept { return compact_ ? n_ : exists_.size(); }

    /// Whether @a u -- a base-graph id by default, a compact id when compact -- is in the view.
    bool hasNode(node u) const noexcept {
        if (compact_)
            return u < n_;
        return u < exists_.size() && exists_[u];
    }

    /// Whether the view presents dense ids @c [0, n) instead of the base ids.
    bool isCompact() const noexcept { return compact_; }

    /// The base-graph id behind the view id @a u. Precondition: hasNode(u), asserted like
    /// the other GraphLike primitives rather than checked with an exception.
    node toBaseId(node u) const {
        assert(hasNode(u));
        return toBaseUnsafe(u);
    }

    /// The view id of the base-graph node @a v, or @c none when @a v is outside the subset.
    node toCompactId(node v) const {
        if (v >= exists_.size() || !exists_[v])
            return none;
        return toCompactUnsafe(v);
    }

    /// The induced degree of @a u, given as a view id; O(1). Requires hasNode(u).
    count degree(node u) const { return degreeOut(u); }

    bool isEmpty() const noexcept { return n_ == 0; }

    bool isDirected() const noexcept { return directed_; }

    /// Mirrors the base graph: the view carries whatever weights the base carries.
    bool isWeighted() const { return base_->isWeighted(); }

    /// The induced self-loop count; maintained incrementally, O(1).
    count numberOfSelfLoops() const noexcept { return selfLoops_; }

    /**
     * The neighborhood of @a u restricted to the view, as a forward range. Elements are bare
     * nodes when @a Weighted is false and (node, weight) pairs otherwise, regardless of how the
     * base graph represents its neighborhoods.
     *
     * Two evaluation strategies produce the same members; the range picks one at construction:
     *
     * - scanning @a u's base neighborhood and keeping the subset members (order: base adjacency
     *   order), or
     * - walking the subset members ascending and probing the base graph for adjacency (order:
     *   ascending ids).
     *
     * @a u is a view id; on a compact view the yielded neighbors are compact ids as well. The
     * id translation is order-preserving, so ascending stays ascending on sorted bases.
     *
     * The second strategy engages only while the base neighborhoods stay sorted, so both orders
     * agree whenever it can run at all. It exists for small views over graphs with skewed
     * degrees: scanning a hub's whole adjacency costs O(deg(u)) no matter how few induced
     * neighbors turn up, while the probe-per-member walk costs one binary search each and wins
     * once deg(u) overshoots n * bit_width(deg(u)) by 4x.
     */
    template <bool Weighted, bool Incoming>
    class NeighborRange {
        template <bool W>
        using BaseOutRange = decltype(std::views::all(
            std::declval<const BaseGraph &>().template outNeighbors<W>(0)));
        template <bool W>
        using BaseInRange =
            decltype(std::views::all(std::declval<const BaseGraph &>().template inNeighbors<W>(0)));
        using BaseRange =
            std::conditional_t<Incoming, BaseInRange<Weighted>, BaseOutRange<Weighted>>;
        // The shared handle below hands out a const range, so iterators come off the const view.
        using BaseIt = std::ranges::iterator_t<const BaseRange>;
        using BaseSent = std::ranges::sentinel_t<const BaseRange>;

        static std::shared_ptr<const BaseRange> makeBaseRange(const InducedSubgraphView &view,
                                                              node u) {
            if constexpr (Incoming)
                return std::make_shared<BaseRange>(
                    std::views::all(view.base_->template inNeighbors<Weighted>(u)));
            else
                return std::make_shared<BaseRange>(
                    std::views::all(view.base_->template outNeighbors<Weighted>(u)));
        }

        static count baseDegree(const InducedSubgraphView &view, node u) {
            if constexpr (Incoming && requires { view.base_->degreeIn(u); })
                return view.base_->degreeIn(u);
            else if constexpr (!Incoming && requires { view.base_->degree(u); })
                return view.base_->degree(u);
            else if constexpr (Incoming)
                return GraphIterationOps::degreeIn(*view.base_, u);
            else
                return GraphIterationOps::degreeOut(*view.base_, u);
        }

        static bool chooseScan(const InducedSubgraphView &view, node u) {
            constexpr bool sortable = requires(const BaseGraph &g) { g.hasSortedNeighborhoods(); };
            if constexpr (!sortable)
                return false;
            else {
                const count deg = baseDegree(view, u);
                return deg > 0 && view.hasSortedNeighborhoods()
                       && deg > view.n_ * std::bit_width(deg) * 4;
            }
        }

    public:
        using Payload = std::conditional_t<Weighted, std::pair<node, edgeweight>, node>;

        NeighborRange(const InducedSubgraphView &view, node u)
            : view_(&view), u_(view.toBaseUnsafe(u)), epoch_(view.edits_),
              scanSubset_(chooseScan(view, u_)), base_(makeBaseRange(view, u_)),
              members_(&view.members()) {}

        /**
         * A self-contained cursor: it carries its own copy of the borrowed base range (kept
         * alive through the shared handle), so iterators survive the temporary range object they
         * were created from -- which the type-erased handles rely on.
         */
        class iterator {
        public:
            using value_type = Payload;
            using reference = Payload;
            using difference_type = std::ptrdiff_t;
            using iterator_concept = std::forward_iterator_tag;

            iterator() noexcept = default;

            iterator(const NeighborRange &range, bool atEnd)
                : view_(range.view_), u_(range.u_), epoch_(range.epoch_),
                  scanSubset_(range.scanSubset_), base_(range.base_), members_(range.members_),
                  atEnd_(atEnd) {
                if (atEnd_)
                    return;
                if (scanSubset_) {
                    pos_ = 0;
                    nextSubset();
                } else {
                    baseIt_ = base_->begin();
                    baseEnd_ = base_->end();
                    nextBase();
                }
            }

            const Payload &operator*() const noexcept { return current_; }

            iterator &operator++() {
                // Membership edits invalidate a borrowed range's backing storage; caught in
                // debug builds, silent in release, like ranges borrowed from a GraphW.
                assert(view_->edits_ == epoch_ && "membership edits invalidate this range");
                if (scanSubset_)
                    nextSubset();
                else
                    nextBase();
                return *this;
            }

            iterator operator++(int) {
                iterator copy(*this);
                ++(*this);
                return copy;
            }

            bool operator==(const iterator &other) const noexcept {
                if (atEnd_ || other.atEnd_)
                    return atEnd_ && other.atEnd_;
                return scanSubset_ ? pos_ == other.pos_ : baseIt_ == other.baseIt_;
            }

        private:
            void nextBase() {
                while (baseIt_ != baseEnd_) {
                    const auto nb = *baseIt_;
                    const node v = neighborTarget(nb);
                    ++baseIt_;
                    if (!view_->exists_[v])
                        continue;
                    // u_ and members_ are base ids; the payload translates back to view ids.
                    const node w = view_->toCompactUnsafe(v);
                    if constexpr (Weighted)
                        current_ = {w, neighborWeight(nb)};
                    else
                        current_ = w;
                    return;
                }
                atEnd_ = true;
            }

            void nextSubset() {
                const auto &members = *members_;
                while (pos_ < members.size()) {
                    const node v = members[pos_++];
                    const node from = Incoming ? v : u_;
                    const node to = Incoming ? u_ : v;
                    if (!GraphIterationOps::hasEdge(*view_->base_, from, to))
                        continue;
                    const node w = view_->toCompactUnsafe(v);
                    if constexpr (Weighted)
                        current_ = {w, GraphIterationOps::weight(*view_->base_, from, to)};
                    else
                        current_ = w;
                    return;
                }
                atEnd_ = true;
            }

            const InducedSubgraphView *view_ = nullptr;
            node u_ = none;
            count epoch_ = 0;
            bool scanSubset_ = false;
            std::shared_ptr<const BaseRange> base_{};
            const std::vector<node> *members_ = nullptr;
            bool atEnd_ = false;
            std::size_t pos_ = 0;
            BaseIt baseIt_{};
            BaseSent baseEnd_{};
            Payload current_{};
        };

        iterator begin() const { return iterator(*this, false); }
        iterator end() const { return iterator(*this, true); }

    private:
        const InducedSubgraphView *view_;
        node u_;
        count epoch_;
        bool scanSubset_;
        std::shared_ptr<const BaseRange> base_;
        const std::vector<node> *members_;
    };

public:
    /**
     * The out-neighbors of @a u inside the view, with @a u given as a view id. The range
     * borrows the base graph's storage and this view's flags, and is invalidated by any
     * membership edit. @tparam Weighted selects the (node, weight) payload; an unweighted base
     * has no weight array, so only request it when isWeighted() holds.
     */
    template <bool Weighted>
    NeighborRange<Weighted, false> outNeighbors(node u) const {
        assert(hasNode(u));
        return NeighborRange<Weighted, false>(*this, u);
    }

    /// The in-neighbors of @a u inside the view, with @a u given as a view id. On an undirected
    /// base these are the out-neighbors.
    template <bool Weighted>
    NeighborRange<Weighted, true> inNeighbors(node u) const {
        assert(hasNode(u));
        return NeighborRange<Weighted, true>(*this, u);
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

    /// Whether lookups may bisect: exactly when the borrowed neighborhoods are sorted.
    bool getKeepEdgesSorted() const { return hasSortedNeighborhoods(); }

    /// Tag for the type-erased handle: this arm's ranges travel through AnyNeighborCursor.
    constexpr bool isInducedSubgraph() const noexcept { return true; }

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

    /// O(1). Requires hasNode(u); @a u is a view id.
    count degreeOut(node u) const {
        assert(hasNode(u));
        return compact_ ? outDegreeC_.values()[u] : outDegree_[u];
    }

    /// O(1). Requires hasNode(u); @a u is a view id. On an undirected base the in-degree is
    /// the out-degree.
    count degreeIn(node u) const {
        assert(hasNode(u));
        return directed_ ? (compact_ ? inDegreeC_.values()[u] : inDegree_[u]) : degreeOut(u);
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

    /* EDGE IDS AND INDEXED NEIGHBOR ACCESS */

    /// A view stores no edges of its own, so it assigns no edge ids.
    bool hasEdgeIds() const noexcept { return false; }

    index upperEdgeIdBound() const noexcept { return none; }

    /// Nothing to maintain without edge ids; mirrors the concrete graphs' knob.
    bool getMaintainCompactEdges() const noexcept { return true; }

    /**
     * Recomputes the node, edge, and self-loop counts straight from the base graph and reports
     * whether they agree with the incrementally maintained ones. O(z + m).
     */
    bool checkConsistency() const {
        count n = 0;
        count ordered = 0; // sightings of an induced edge from one of its endpoints
        count loops = 0;
        for (const node u : members()) {
            ++n;
            for (const auto nb : base_->template outNeighbors<false>(u)) {
                const node v = neighborTarget(nb);
                if (!exists_[v])
                    continue;
                ++ordered;
                if (u == v)
                    ++loops;
            }
        }
        const count m = directed_ ? ordered : (ordered + loops) / 2;
        return n == n_ && m == m_ && loops == selfLoops_;
    }

    /**
     * The @a i-th member of @a u's induced neighborhood, or @c none when @a i is out of range.
     * Members appear in the adaptive range's order -- base adjacency order, which is ascending on
     * sorted bases. O(deg(u)).
     */
    node getIthNeighbor(node u, index i) const {
        if (!hasNode(u) || i >= degree(u))
            return none;
        return getIthNeighbor(Unsafe{}, u, i);
    }

    node getIthNeighbor(Unsafe, node u, index i) const {
        count j = 0;
        for (const auto nb : outNeighbors<false>(u))
            if (j++ == i)
                return neighborTarget(nb);
        return none;
    }

    /// As getIthNeighbor, over the induced in-neighborhood.
    node getIthInNeighbor(node u, index i) const {
        if (!hasNode(u) || i >= degreeIn(u))
            return none;
        count j = 0;
        for (const auto nb : inNeighbors<false>(u))
            if (j++ == i)
                return neighborTarget(nb);
        return none;
    }

    /// The weight of the @a i-th induced neighbor of @a u, or @c none when @a i is out of range.
    edgeweight getIthNeighborWeight(node u, index i) const {
        if (!hasNode(u) || i >= degree(u))
            return nullWeight;
        return getIthNeighborWeight(Unsafe{}, u, i);
    }

    edgeweight getIthNeighborWeight(Unsafe, node u, index i) const {
        count j = 0;
        for (const auto &nb : outNeighbors<true>(u))
            if (j++ == i)
                return neighborWeight(nb);
        return nullWeight;
    }

    /// The @a i-th induced neighbor of @a u with its weight, or (@c none, @c nullWeight).
    std::pair<node, edgeweight> getIthNeighborWithWeight(node u, index i) const {
        if (!hasNode(u) || i >= degree(u))
            return {none, nullWeight};
        return getIthNeighborWithWeight(Unsafe{}, u, i);
    }

    std::pair<node, edgeweight> getIthNeighborWithWeight(Unsafe, node u, index i) const {
        count j = 0;
        for (const auto &nb : outNeighbors<true>(u))
            if (j++ == i)
                return {neighborTarget(nb), neighborWeight(nb)};
        return {none, nullWeight};
    }

    /// The position of @a v in @a u's induced neighborhood, or @c none when absent. O(deg(u)).
    index indexOfNeighbor(node u, node v) const {
        if (!hasNode(u) || !hasNode(v))
            return none;
        index i = 0;
        for (const auto nb : outNeighbors<false>(u)) {
            if (neighborTarget(nb) == v)
                return i;
            ++i;
        }
        return none;
    }

    /* VIEW-SPECIFIC OPERATIONS */

    /// The graph the view spans.
    const BaseGraph &getBaseGraph() const noexcept { return *base_; }

    /// The members of the subgraph as base-graph ids, in ascending id order. Ascending is what
    /// the adaptive neighbor ranges rely on. On a non-compact view the index is rebuilt lazily
    /// after membership edits (O(z) once per batch, O(1) otherwise); on a compact view it is
    /// the compact degree flat_map's keys, maintained incrementally (O(1)). In both cases
    /// entry @c c is the base id behind compact id @c c.
    const std::vector<node> &getNodeSubset() const { return members(); }

    /**
     * Base-graph nodes outside the view that are out-neighbors of a node inside it. Ascending
     * and unique. This is the search frontier when the view grows one layer at a time.
     */
    std::vector<node> frontier() const {
        std::vector<node> result;
        for (node u : members())
            for (const auto nb : base_->template outNeighbors<false>(u)) {
                const node v = neighborTarget(nb);
                // base-id space: hasNode() speaks view ids on a compact view.
                if (!isMember(v))
                    result.push_back(v);
            }
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());
        return result;
    }

    /**
     * Constructs an explicit mutable subgraph equivalent to this view.
     * @param compact Whether the subgraph should get compact node ids instead of the base ids.
     * Independent of whether the view itself is compact.
     */
    GraphW realize(bool compact = false) const {
        if (compact) {
            GraphW out(n_, isWeighted(), directed_);
            std::vector<node> newId(upperNodeIdBound(), none);
            node next = 0;
            forNodes([&](node u) { newId[u] = next++; });
            forEdges([&](node u, node v, edgeweight w) {
                out.addEdge(newId[u], newId[v], isWeighted() ? w : defaultEdgeWeight);
            });
            return out;
        }
        // Non-compact output keeps the base ids; forEdges() speaks view ids, so translate.
        GraphW out(static_cast<count>(exists_.size()), isWeighted(), directed_);
        for (node u = 0; u < exists_.size(); ++u)
            if (!exists_[u])
                out.removeNode(u);
        forEdges([&](node u, node v, edgeweight w) {
            out.addEdge(toBaseUnsafe(u), toBaseUnsafe(v), isWeighted() ? w : defaultEdgeWeight);
        });
        return out;
    }

private:
    const BaseGraph *base_;
    /// One presence flag per base node; indexed by the base ids the view keeps.
    std::vector<char> exists_;
    /// Degree bookkeeping indexed by the base ids; used as-is on a non-compact view. On a
    /// compact view these stay empty and the flat_maps below carry the degrees instead, so
    /// nothing is allocated in the base id space.
    std::vector<count> outDegree_;
    /// Only populated on a directed base.
    std::vector<count> inDegree_;
    /// Compact-mode degree storage: keyed by base id (so the incremental updates during
    /// membership edits find their entries), with keys() sorted ascending. keys() is therefore
    /// the member set -- a compact id indexes keys()/values() directly -- and the flat_map is
    /// updated in place by every edit, so readers never observe a stale mapping.
    Aux::flat_map<node, count> outDegreeC_;
    /// Only populated on a directed base.
    Aux::flat_map<node, count> inDegreeC_;
    count n_ = 0, m_ = 0, selfLoops_ = 0;
    bool directed_;
    /// When true, the GraphLike primitives present dense ids @c [0, n) while the membership
    /// flags above stay in base-id space. Immutable after construction.
    bool compact_ = false;
    /// Bumped by every membership-edit batch; ranges borrowed from the view record it so
    /// invalidation by an edit is detectable (in debug builds).
    count edits_ = 0;

    /// Ascending member ids, derived from exists_ on demand on a non-compact view; see
    /// NeighborRange and getNodeSubset(). Mutable because a const range must be able to
    /// rebuild it. Unused on a compact view, where members() reads the flat_map's keys.
    mutable std::vector<node> sortedMembers_;
    mutable bool membersDirty_ = true;

    /// Base-id-space membership test: @a b is in the subset. Distinct from hasNode(), whose
    /// argument is a view id (the compact id when compact).
    bool isMember(node b) const noexcept { return b < exists_.size() && exists_[b]; }

    /// The members as ascending base ids. On a non-compact view the index is rebuilt lazily
    /// after membership edits; on a compact view this is the compact degree flat_map's keys,
    /// which track the membership incrementally. In both cases entry @c c is the base id
    /// behind compact id @c c.
    const std::vector<node> &members() const {
        if (!compact_) {
            if (membersDirty_) {
                sortedMembers_.clear();
                sortedMembers_.reserve(n_);
                for (node v = 0; v < exists_.size(); ++v)
                    if (exists_[v])
                        sortedMembers_.push_back(v);
                membersDirty_ = false;
            }
            return sortedMembers_;
        }
        return outDegreeC_.keys();
    }

    /// View id @a u to base id. Precondition: hasNode(u); callers establish that (outNeighbors
    /// asserts it, iterators only translate members). O(1) in both modes.
    node toBaseUnsafe(node u) const {
        if (!compact_)
            return u;
        assert(u < n_);
        return members()[u];
    }

    /// Base id @a b to view id. Precondition: @a b is a member. O(log n) on a compact view --
    /// the rank comes from a binary search over the ascending members.
    node toCompactUnsafe(node b) const {
        if (!compact_)
            return b;
        const auto &ks = members();
        const auto it = std::lower_bound(ks.begin(), ks.end(), b);
        assert(base_->hasNode(b) && it != ks.end() && *it == b);
        return static_cast<node>(it - ks.begin());
    }

    /// A member's stored out-degree, for the incremental updates during edits. On a compact
    /// view flat_map::operator[] finds the member's entry (inserting a zero for a node that
    /// is joining, which the caller then overwrites).
    count &outDegreeAt(node b) { return compact_ ? outDegreeC_[b] : outDegree_[b]; }

    /// As outDegreeAt, over the in-degrees. Only called on a directed base.
    count &inDegreeAt(node b) { return compact_ ? inDegreeC_[b] : inDegree_[b]; }

    template <typename NodeRange>
    void addNodesImpl(const NodeRange &nodes) {
        // Validate the whole batch before touching anything: the edit is atomic -- either all
        // nodes join or none do -- and the insert loop below is exception-free, which lets
        // code gen treat it as a noexcept hot path.
        for (const node v : nodes)
            if (!base_->hasNode(v))
                throw std::runtime_error("InducedSubgraphView: node is not in the base graph");
        ++edits_;
        for (const node v : nodes)
            insertNode(v);
    }

    template <typename NodeRange>
    void removeNodesImpl(const NodeRange &nodes) {
        ++edits_;
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
        membersDirty_ = true;

        count outDeg = 0;
        for (const auto nb : base_->template outNeighbors<false>(v)) {
            const node u = neighborTarget(nb);
            if (!isMember(u))
                continue;
            if (!directed_) {
                if (u == v)
                    ++selfLoops_;
                else
                    ++outDegreeAt(u);
            } else {
                // the edge (v, u) is an in-edge of u
                ++inDegreeAt(u);
                if (u == v)
                    ++selfLoops_;
            }
            ++outDeg;
        }
        outDegreeAt(v) = outDeg;
        m_ += outDeg;

        if (directed_) {
            count inDeg = 0;
            for (const auto nb : base_->template inNeighbors<false>(v)) {
                const node u = neighborTarget(nb);
                // a self-loop was already accounted for by the out-scan above
                if (u == v || !isMember(u))
                    continue;
                ++outDegreeAt(u);
                ++inDeg;
            }
            inDegreeAt(v) += inDeg;
            m_ += inDeg;
        }
    }

    /// The inverse of insertNode(). @a v stays marked present until its own adjacency has been
    /// scanned, so a self-loop is seen exactly once.
    void eraseNode(node v) {
        // ids outside the base's id space were never inserted; indexing exists_ with them
        // would read past its storage
        if (!isMember(v))
            return;

        count removedOut = 0;
        for (const auto nb : base_->template outNeighbors<false>(v)) {
            const node u = neighborTarget(nb);
            if (!isMember(u))
                continue;
            if (!directed_) {
                if (u == v)
                    --selfLoops_;
                else
                    --outDegreeAt(u);
            } else {
                --inDegreeAt(u);
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
                if (u == v || !isMember(u))
                    continue;
                --outDegreeAt(u);
                ++removedIn;
            }
            m_ -= removedIn;
        }

        if (compact_) {
            // the flat_maps' keys must equal the membership: drop @a v's entries
            outDegreeC_.erase(v);
            if (directed_)
                inDegreeC_.erase(v);
        } else {
            if (directed_)
                inDegree_[v] = 0;
            outDegree_[v] = 0;
        }
        exists_[v] = 0;
        --n_;
        membersDirty_ = true;
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
