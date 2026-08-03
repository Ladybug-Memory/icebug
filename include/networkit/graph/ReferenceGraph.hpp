/*
 * ReferenceGraph.hpp
 *
 *  Non-owning handle to a graph whose concrete type is not known at the call site.
 */

#ifndef NETWORKIT_GRAPH_REFERENCE_GRAPH_HPP_
#define NETWORKIT_GRAPH_REFERENCE_GRAPH_HPP_

#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <networkit/Globals.hpp>
#include <networkit/graph/Attributes.hpp>
#include <networkit/graph/EdgeIterators.hpp>
#include <networkit/graph/GraphConcepts.hpp>
#include <networkit/graph/GraphTypes.hpp>
#include <networkit/graph/NeighborIterators.hpp>
#include <networkit/graph/NodeIterators.hpp>

namespace NetworKit {

class GraphR;
class GraphW;

/**
 * A handle to some concrete graph, resolved statically rather than through a vtable.
 *
 * Non-owning: the referenced graph must outlive the handle. Copying copies a pointer, never a
 * graph. Only reads live here; mutating a graph, including attaching an attribute, goes through
 * the concrete type.
 */
class ReferenceGraph {
    std::variant<const GraphW *, const GraphR *> arm_;

public:
    using NodeIterator = NodeIteratorBase<ReferenceGraph>;
    using NodeRange = NodeRangeBase<ReferenceGraph>;
    using EdgeIterator = EdgeTypeIterator<ReferenceGraph, Edge>;
    using EdgeWeightIterator = EdgeTypeIterator<ReferenceGraph, WeightedEdge>;
    using EdgeRange = EdgeTypeRange<ReferenceGraph, Edge>;
    using EdgeWeightRange = EdgeTypeRange<ReferenceGraph, WeightedEdge>;

    using NodeIntAttribute = Attribute<PerNode, ReferenceGraph, int, false>;
    using NodeDoubleAttribute = Attribute<PerNode, ReferenceGraph, double, false>;
    using NodeStringAttribute = Attribute<PerNode, ReferenceGraph, std::string, false>;
    using EdgeIntAttribute = Attribute<PerEdge, ReferenceGraph, int, false>;
    using EdgeDoubleAttribute = Attribute<PerEdge, ReferenceGraph, double, false>;
    using EdgeStringAttribute = Attribute<PerEdge, ReferenceGraph, std::string, false>;

    /// An arm's iterator paired with that arm's end, so that termination is a unary visit.
    template <typename ArmIter>
    struct NeighborCursor {
        ArmIter cur, last;
        bool done() const { return cur == last; }
        decltype(auto) get() const { return *cur; }
        void next() { ++cur; }
        bool operator==(const NeighborCursor &) const = default;
    };

    template <typename Payload, typename... ArmIters>
    class ErasedNeighborIterator {
        std::variant<NeighborCursor<ArmIters>...> cursor_;

    public:
        using value_type = Payload;
        using reference = Payload;
        using pointer = void;
        using iterator_category = std::forward_iterator_tag;
        using difference_type = ptrdiff_t;

        ErasedNeighborIterator() = default;
        template <typename Cursor>
        explicit ErasedNeighborIterator(Cursor c) : cursor_(c) {}

        Payload operator*() const {
            return std::visit([](const auto &c) -> Payload { return c.get(); }, cursor_);
        }
        ErasedNeighborIterator &operator++() {
            std::visit([](auto &c) { c.next(); }, cursor_);
            return *this;
        }
        ErasedNeighborIterator operator++(int) {
            auto tmp = *this;
            ++*this;
            return tmp;
        }
        bool operator==(const ErasedNeighborIterator &o) const {
            return std::visit(
                [&o](const auto &a) {
                    const auto *b = std::get_if<std::decay_t<decltype(a)>>(&o.cursor_);
                    return b != nullptr && a.cur == b->cur;
                },
                cursor_);
        }
        bool operator!=(const ErasedNeighborIterator &o) const { return !(*this == o); }
    };

    using NeighborIterator =
        ErasedNeighborIterator<node, NeighborIteratorBase<std::vector<node>::const_iterator>,
                               NeighborIteratorBase<const node *>>;
    using NeighborWeightIterator = ErasedNeighborIterator<
        std::pair<node, edgeweight>,
        NeighborWeightIteratorBase<std::vector<node>::const_iterator,
                                   std::vector<edgeweight>::const_iterator>,
        NeighborWeightIteratorBase<const node *, const edgeweight *>>;

    /// The neighbors of a node, borrowed from the arm's own storage; nothing is copied.
    template <bool InEdges = false>
    class NeighborRange {
        NeighborIterator first, last;

    public:
        NeighborRange(const ReferenceGraph &G, node u);
        NeighborRange() = default;

        NeighborIterator begin() const { return first; }
        NeighborIterator end() const { return last; }
    };

    /// The (node, weight) counterpart of NeighborRange.
    template <bool InEdges = false>
    class NeighborWeightRange {
        NeighborWeightIterator first, last;

    public:
        NeighborWeightRange(const ReferenceGraph &G, node u);
        NeighborWeightRange() = default;

        NeighborWeightIterator begin() const { return first; }
        NeighborWeightIterator end() const { return last; }
    };

    using OutNeighborRange = NeighborRange<false>;
    using InNeighborRange = NeighborRange<true>;
    using OutNeighborWeightRange = NeighborWeightRange<false>;
    using InNeighborWeightRange = NeighborWeightRange<true>;

    /// References a shared empty read-only graph. Required so Cython can materialize temporaries.
    ReferenceGraph() noexcept;

    /*
     * Explicit so that converting a concrete graph by value goes through SelfHandled's
     * conversion operator. With both routes available g++ silently materializes a temporary
     * while clang++ rejects the call as ambiguous.
     */
    explicit ReferenceGraph(const GraphW &g) noexcept : arm_(&g) {}
    explicit ReferenceGraph(const GraphR &g) noexcept : arm_(&g) {}

    /// Calls @a f with a reference to the concrete graph, instantiating it once per arm.
    template <typename Fn>
    decltype(auto) visit(Fn &&f) const;

    /// Lets a handle held by value be used with the pointer syntax algorithms already use.
    const ReferenceGraph *operator->() const noexcept { return this; }
    const ReferenceGraph &operator*() const noexcept { return *this; }

    /* GLOBAL PROPERTIES */

    count numberOfNodes() const noexcept;
    count numberOfEdges() const noexcept;
    count numberOfSelfLoops() const noexcept;
    index upperNodeIdBound() const noexcept;
    index upperEdgeIdBound() const noexcept;
    bool isEmpty() const noexcept;
    bool isWeighted() const noexcept;
    bool isDirected() const noexcept;
    bool hasEdgeIds() const noexcept;
    bool getKeepEdgesSorted() const noexcept;
    bool getMaintainCompactEdges() const noexcept;
    edgeweight totalEdgeWeight() const noexcept;
    bool checkConsistency() const;

    const AttributeMap<PerNode, ReferenceGraph> &nodeAttributes() const noexcept;
    const AttributeMap<PerEdge, ReferenceGraph> &edgeAttributes() const noexcept;

    /* NODE AND EDGE PROPERTIES */

    bool hasNode(node v) const noexcept;
    bool hasEdge(node u, node v) const;
    bool isIsolated(node v) const;
    count degree(node v) const;
    count degreeIn(node v) const;
    count degreeOut(node v) const;
    edgeweight weightedDegree(node u, bool countSelfLoopsTwice = false) const;
    edgeweight weightedDegreeIn(node u, bool countSelfLoopsTwice = false) const;
    edgeweight weight(node u, node v) const;
    edgeid edgeId(node u, node v) const;
    std::pair<node, node> edgeById(index id) const;

    /* INDEXED NEIGHBOR ACCESS */

    index indexOfNeighbor(node u, node v) const;
    node getIthNeighbor(node u, index i) const;
    node getIthNeighbor(Unsafe, node u, index i) const;
    node getIthInNeighbor(node u, index i) const;
    edgeweight getIthNeighborWeight(node u, index i) const;
    edgeweight getIthNeighborWeight(Unsafe, node u, index i) const;
    std::pair<node, edgeweight> getIthNeighborWithWeight(node u, index i) const;
    std::pair<node, edgeweight> getIthNeighborWithWeight(Unsafe, node u, index i) const;
    std::pair<node, edgeid> getIthNeighborWithId(node u, index i) const;

    /* RANGES */

    NodeRange nodeRange() const noexcept;
    EdgeRange edgeRange() const noexcept;
    EdgeWeightRange edgeWeightRange() const noexcept;
    NeighborRange<false> neighborRange(node u) const;
    NeighborWeightRange<false> weightNeighborRange(node u) const;
    NeighborRange<true> inNeighborRange(node u) const;
    NeighborWeightRange<true> weightInNeighborRange(node u) const;

    /* ITERATION */

    template <typename L>
    void forNodes(L handle) const;

    template <typename L>
    void parallelForNodes(L handle) const;

    template <typename C, typename L>
    void forNodesWhile(C condition, L handle) const;

    template <typename L>
    void forNodesInRandomOrder(L handle) const;

    template <typename L>
    void balancedParallelForNodes(L handle) const;

    template <typename L>
    void forNodePairs(L handle) const;

    template <typename L>
    void parallelForNodePairs(L handle) const;

    template <typename L>
    void forEdges(L handle) const;

    template <typename L>
    void parallelForEdges(L handle) const;

    template <typename L>
    void forNeighborsOf(node u, L handle) const;

    template <typename L>
    void forEdgesOf(node u, L handle) const;

    template <typename L>
    void forInNeighborsOf(node u, L handle) const;

    template <typename L>
    void forInEdgesOf(node u, L handle) const;

    template <typename L>
    double parallelSumForNodes(L handle) const;

    template <typename L>
    double parallelSumForEdges(L handle) const;
};

/**
 * A handle to @a Derived, stored inside @a Derived and living exactly as long as it.
 *
 * Lets a caller take @c "const Graph &" of a concrete graph without materializing a temporary, so
 * algorithms may hold the address of one. The handle is identity-bound: copying or moving a graph
 * yields a handle on the new object, which is why the special members here ignore their argument
 * and why a derived class may leave its own as @c "= default".
 *
 * @tparam Derived must be an alternative of ReferenceGraph's variant.
 *
 * ReferenceGraph must never inherit this -- the base stores one by value, so the type would be
 * infinitely sized.
 */
template <typename Derived>
class SelfHandled {
    ReferenceGraph handle_;

    ReferenceGraph seat() const noexcept {
        return ReferenceGraph(*static_cast<const Derived *>(this));
    }

protected:
    SelfHandled() noexcept : handle_(seat()) {}
    SelfHandled(const SelfHandled &) noexcept : handle_(seat()) {}
    SelfHandled(SelfHandled &&) noexcept : handle_(seat()) {}
    SelfHandled &operator=(const SelfHandled &) noexcept { return *this; }
    SelfHandled &operator=(SelfHandled &&) noexcept { return *this; }
    ~SelfHandled() = default;

public:
    /// The handle denoting this graph. Valid for as long as the graph is.
    const ReferenceGraph &asGraph() const noexcept { return handle_; }

    operator const ReferenceGraph &() const noexcept { return handle_; }
};

} // namespace NetworKit

#endif // NETWORKIT_GRAPH_REFERENCE_GRAPH_HPP_
