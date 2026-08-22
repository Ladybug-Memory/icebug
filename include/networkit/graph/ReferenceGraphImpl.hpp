/*
 * ReferenceGraphImpl.hpp
 *
 *  Inline bodies for ReferenceGraph. Separate from ReferenceGraph.hpp because the forwarders
 *  need the concrete graphs complete, while those graphs need ReferenceGraph declared.
 */

#ifndef NETWORKIT_GRAPH_REFERENCE_GRAPH_IMPL_HPP_
#define NETWORKIT_GRAPH_REFERENCE_GRAPH_IMPL_HPP_

#include <ranges>
#include <utility>
#include <variant>

#include <networkit/graph/GraphIteration.hpp>
#include <networkit/graph/GraphR.hpp>
#include <networkit/graph/GraphW.hpp>
#include <networkit/graph/ReferenceGraph.hpp>

namespace NetworKit {

template <typename Fn>
decltype(auto) ReferenceGraph::visit(Fn &&f) const {
    return std::visit([&f](const auto *g) -> decltype(auto) { return f(*g); }, arm_);
}

namespace ReferenceGraphDetail {
/// Backs the default constructor, so that no arm is ever null and visit() need not check.
inline const GraphR &emptyGraph() {
    static const GraphR empty(0, false, nullptr, nullptr);
    return empty;
}
} // namespace ReferenceGraphDetail

inline ReferenceGraph::ReferenceGraph() noexcept : arm_(&ReferenceGraphDetail::emptyGraph()) {}

/// Cython has no way to spell a conversion; these give it one.
inline ReferenceGraph refGraphW(const GraphW &g) {
    return ReferenceGraph(g);
}
inline ReferenceGraph refGraphR(const GraphR &g) {
    return ReferenceGraph(g);
}

/* GLOBAL PROPERTIES */

inline count ReferenceGraph::numberOfNodes() const noexcept {
    return visit([](const auto &g) { return g.numberOfNodes(); });
}
inline count ReferenceGraph::numberOfEdges() const noexcept {
    return visit([](const auto &g) { return g.numberOfEdges(); });
}
inline count ReferenceGraph::numberOfSelfLoops() const noexcept {
    return visit([](const auto &g) { return g.numberOfSelfLoops(); });
}
inline index ReferenceGraph::upperNodeIdBound() const noexcept {
    return visit([](const auto &g) { return g.upperNodeIdBound(); });
}
inline index ReferenceGraph::upperEdgeIdBound() const noexcept {
    return visit([](const auto &g) { return g.upperEdgeIdBound(); });
}
inline bool ReferenceGraph::isEmpty() const noexcept {
    return visit([](const auto &g) { return g.isEmpty(); });
}
inline bool ReferenceGraph::isWeighted() const noexcept {
    return visit([](const auto &g) { return g.isWeighted(); });
}
inline bool ReferenceGraph::isDirected() const noexcept {
    return visit([](const auto &g) { return g.isDirected(); });
}
inline bool ReferenceGraph::hasEdgeIds() const noexcept {
    return visit([](const auto &g) { return g.hasEdgeIds(); });
}
inline bool ReferenceGraph::getKeepEdgesSorted() const noexcept {
    return visit([](const auto &g) { return g.getKeepEdgesSorted(); });
}
inline bool ReferenceGraph::getMaintainCompactEdges() const noexcept {
    return visit([](const auto &g) { return g.getMaintainCompactEdges(); });
}
inline edgeweight ReferenceGraph::totalEdgeWeight() const noexcept {
    return visit([](const auto &g) { return g.totalEdgeWeight(); });
}
inline bool ReferenceGraph::checkConsistency() const {
    return visit([](const auto &g) { return g.checkConsistency(); });
}
inline const AttributeMap<PerNode, ReferenceGraph> &
ReferenceGraph::nodeAttributes() const noexcept {
    return visit([](const auto &g) -> decltype(auto) { return g.nodeAttributes(); });
}
inline const AttributeMap<PerEdge, ReferenceGraph> &
ReferenceGraph::edgeAttributes() const noexcept {
    return visit([](const auto &g) -> decltype(auto) { return g.edgeAttributes(); });
}

/* NODE AND EDGE PROPERTIES */

inline bool ReferenceGraph::hasNode(node v) const noexcept {
    return visit([v](const auto &g) { return g.hasNode(v); });
}
inline bool ReferenceGraph::hasEdge(node u, node v) const {
    return visit([u, v](const auto &g) { return g.hasEdge(u, v); });
}
inline bool ReferenceGraph::isIsolated(node v) const {
    return visit([v](const auto &g) { return g.isIsolated(v); });
}
inline count ReferenceGraph::degree(node v) const {
    return visit([v](const auto &g) { return g.degree(v); });
}
inline count ReferenceGraph::degreeIn(node v) const {
    return visit([v](const auto &g) { return g.degreeIn(v); });
}
inline count ReferenceGraph::degreeOut(node v) const {
    return visit([v](const auto &g) { return g.degreeOut(v); });
}
inline edgeweight ReferenceGraph::weightedDegree(node u, bool countSelfLoopsTwice) const {
    return visit([&](const auto &g) { return g.weightedDegree(u, countSelfLoopsTwice); });
}
inline edgeweight ReferenceGraph::weightedDegreeIn(node u, bool countSelfLoopsTwice) const {
    return visit([&](const auto &g) { return g.weightedDegreeIn(u, countSelfLoopsTwice); });
}
inline edgeweight ReferenceGraph::weight(node u, node v) const {
    return visit([u, v](const auto &g) { return g.weight(u, v); });
}
inline edgeid ReferenceGraph::edgeId(node u, node v) const {
    return visit([u, v](const auto &g) { return g.edgeId(u, v); });
}
inline std::pair<node, node> ReferenceGraph::edgeById(index id) const {
    return visit([id](const auto &g) { return g.edgeById(id); });
}

/* INDEXED NEIGHBOR ACCESS */

inline index ReferenceGraph::indexOfNeighbor(node u, node v) const {
    return visit([u, v](const auto &g) { return g.indexOfNeighbor(u, v); });
}
inline node ReferenceGraph::getIthNeighbor(node u, index i) const {
    return visit([u, i](const auto &g) { return g.getIthNeighbor(u, i); });
}
inline node ReferenceGraph::getIthNeighbor(Unsafe, node u, index i) const {
    return visit([u, i](const auto &g) { return g.getIthNeighbor(Unsafe{}, u, i); });
}
inline node ReferenceGraph::getIthInNeighbor(node u, index i) const {
    return visit([u, i](const auto &g) { return g.getIthInNeighbor(u, i); });
}
inline edgeweight ReferenceGraph::getIthNeighborWeight(node u, index i) const {
    return visit([u, i](const auto &g) { return g.getIthNeighborWeight(u, i); });
}
inline edgeweight ReferenceGraph::getIthNeighborWeight(Unsafe, node u, index i) const {
    return visit([u, i](const auto &g) { return g.getIthNeighborWeight(Unsafe{}, u, i); });
}
inline std::pair<node, edgeweight> ReferenceGraph::getIthNeighborWithWeight(node u, index i) const {
    return visit([u, i](const auto &g) { return g.getIthNeighborWithWeight(u, i); });
}
inline std::pair<node, edgeweight> ReferenceGraph::getIthNeighborWithWeight(Unsafe, node u,
                                                                            index i) const {
    return visit([u, i](const auto &g) { return g.getIthNeighborWithWeight(Unsafe{}, u, i); });
}
inline std::pair<node, edgeid> ReferenceGraph::getIthNeighborWithId(node u, index i) const {
    return visit([u, i](const auto &g) { return g.getIthNeighborWithId(u, i); });
}
/* RANGES */

inline ReferenceGraph::NodeRange ReferenceGraph::nodeRange() const noexcept {
    return NodeRange(*this);
}
inline ReferenceGraph::EdgeRange ReferenceGraph::edgeRange() const noexcept {
    return EdgeRange(*this);
}
inline ReferenceGraph::EdgeWeightRange ReferenceGraph::edgeWeightRange() const noexcept {
    return EdgeWeightRange(*this);
}

/*
 * Seating a cursor on the arm's own iterators. This is why the Impl header exists: naming an arm's
 * range type requires that arm to be complete, while the arms need ReferenceGraph declared.
 */
namespace ReferenceGraphDetail {
template <bool InEdges, bool Weighted, typename G>
auto neighborsOf(const G &g, node u) {
    if constexpr (InEdges)
        return g.template inNeighbors<Weighted>(u);
    else
        return g.template outNeighbors<Weighted>(u);
}
} // namespace ReferenceGraphDetail

template <bool InEdges>
ReferenceGraph::NeighborRange<InEdges>::NeighborRange(const ReferenceGraph &G, node u) {
    G.visit([&](const auto &g) {
        auto r = ReferenceGraphDetail::neighborsOf<InEdges, false>(g, u);
        using Cur = NeighborCursor<std::ranges::iterator_t<decltype(r)>>;
        first = NeighborIterator(Cur{r.begin(), r.end()});
        last = NeighborIterator(Cur{r.end(), r.end()});
    });
}

template <bool InEdges>
ReferenceGraph::NeighborWeightRange<InEdges>::NeighborWeightRange(const ReferenceGraph &G, node u) {
    G.visit([&](const auto &g) {
        auto r = ReferenceGraphDetail::neighborsOf<InEdges, true>(g, u);
        using Cur = NeighborCursor<std::ranges::iterator_t<decltype(r)>>;
        first = NeighborWeightIterator(Cur{r.begin(), r.end()});
        last = NeighborWeightIterator(Cur{r.end(), r.end()});
    });
}

inline ReferenceGraph::NeighborRange<false> ReferenceGraph::neighborRange(node u) const {
    return NeighborRange<false>(*this, u);
}
inline ReferenceGraph::NeighborWeightRange<false>
ReferenceGraph::weightNeighborRange(node u) const {
    return NeighborWeightRange<false>(*this, u);
}
inline ReferenceGraph::NeighborRange<true> ReferenceGraph::inNeighborRange(node u) const {
    return NeighborRange<true>(*this, u);
}
inline ReferenceGraph::NeighborWeightRange<true>
ReferenceGraph::weightInNeighborRange(node u) const {
    return NeighborWeightRange<true>(*this, u);
}

/* ITERATION */

template <typename L>
void ReferenceGraph::forNodes(L handle) const {
    visit([&](const auto &g) { g.forNodes(handle); });
}
template <typename L>
void ReferenceGraph::parallelForNodes(L handle) const {
    visit([&](const auto &g) { g.parallelForNodes(handle); });
}
template <typename C, typename L>
void ReferenceGraph::forNodesWhile(C condition, L handle) const {
    visit([&](const auto &g) { g.forNodesWhile(condition, handle); });
}
template <typename L>
void ReferenceGraph::forNodesInRandomOrder(L handle) const {
    visit([&](const auto &g) { g.forNodesInRandomOrder(handle); });
}
template <typename L>
void ReferenceGraph::balancedParallelForNodes(L handle) const {
    visit([&](const auto &g) { g.balancedParallelForNodes(handle); });
}
template <typename L>
void ReferenceGraph::forNodePairs(L handle) const {
    visit([&](const auto &g) { g.forNodePairs(handle); });
}
template <typename L>
void ReferenceGraph::parallelForNodePairs(L handle) const {
    visit([&](const auto &g) { g.parallelForNodePairs(handle); });
}
template <typename L>
void ReferenceGraph::forEdges(L handle) const {
    visit([&](const auto &g) { g.forEdges(handle); });
}
template <typename L>
void ReferenceGraph::parallelForEdges(L handle) const {
    visit([&](const auto &g) { g.parallelForEdges(handle); });
}
template <typename L>
void ReferenceGraph::forNeighborsOf(node u, L handle) const {
    visit([&](const auto &g) { g.forNeighborsOf(u, handle); });
}
template <typename L>
void ReferenceGraph::forEdgesOf(node u, L handle) const {
    visit([&](const auto &g) { g.forEdgesOf(u, handle); });
}
template <typename L>
void ReferenceGraph::forInNeighborsOf(node u, L handle) const {
    visit([&](const auto &g) { g.forInNeighborsOf(u, handle); });
}
template <typename L>
void ReferenceGraph::forInEdgesOf(node u, L handle) const {
    visit([&](const auto &g) { g.forInEdgesOf(u, handle); });
}
template <typename L>
double ReferenceGraph::parallelSumForNodes(L handle) const {
    return visit([&](const auto &g) { return g.parallelSumForNodes(handle); });
}
template <typename L>
double ReferenceGraph::parallelSumForEdges(L handle) const {
    return visit([&](const auto &g) { return g.parallelSumForEdges(handle); });
}

static_assert(GraphLike<ReferenceGraph>,
              "the type-erased handle must be usable wherever a graph type is");
static_assert(!IndexedGraph<ReferenceGraph>,
              "edge ids are only carried by graphs that index their edges");
static_assert(!MutableGraph<ReferenceGraph>, "a handle cannot mutate the referenced graph");

} // namespace NetworKit

#endif // NETWORKIT_GRAPH_REFERENCE_GRAPH_IMPL_HPP_
