/*
 * GraphConcepts.hpp
 *
 *  Capability contracts a graph class signs up to.
 */

#ifndef NETWORKIT_GRAPH_GRAPH_CONCEPTS_HPP_
#define NETWORKIT_GRAPH_GRAPH_CONCEPTS_HPP_

#include <concepts>
#include <ranges>
#include <tuple>
#include <utility>

#include <networkit/Globals.hpp>
#include <networkit/graph/NeighborIterators.hpp>

namespace NetworKit {

/**
 * Uniform access to the value types the neighbor ranges yield: a bare node, a
 * (node, weight) pair, or a (node, weight, id) tuple.
 */
inline node neighborTarget(node nb) {
    return nb;
}
inline node neighborTarget(const std::pair<node, edgeweight> &nb) {
    return nb.first;
}
inline node neighborTarget(const std::tuple<node, edgeweight, edgeid> &nb) {
    return std::get<0>(nb);
}

/// An unweighted neighborhood synthesizes the default weight rather than storing one.
inline edgeweight neighborWeight(node) {
    return defaultEdgeWeight;
}
inline edgeweight neighborWeight(const std::pair<node, edgeweight> &nb) {
    return nb.second;
}
inline edgeweight neighborWeight(const std::tuple<node, edgeweight, edgeid> &nb) {
    return std::get<1>(nb);
}

inline edgeid neighborId(const std::tuple<node, edgeweight, edgeid> &nb) {
    return std::get<2>(nb);
}

/*
 * An indexed neighborhood on an unweighted graph carries (node, id) and no weight, mirroring
 * Graph::getOutEdgeWeight<false>, which returns defaultEdgeWeight without reading storage.
 */
inline node neighborTarget(const std::pair<node, edgeid> &nb) {
    return nb.first;
}
inline edgeweight neighborWeight(const std::pair<node, edgeid> &) {
    return defaultEdgeWeight;
}
inline edgeid neighborId(const std::pair<node, edgeid> &nb) {
    return nb.second;
}

template <typename N>
concept NeighborLike = requires(const N nb) {
    { neighborTarget(nb) } -> std::convertible_to<node>;
    { neighborWeight(nb) } -> std::convertible_to<edgeweight>;
};

template <typename Derived>
class GraphIterationMixin;

/**
 * The base capability: a node universe, out-neighbor iteration, and the two axes.
 *
 * Membership is by construction, not by tag: a class satisfies this only by deriving from
 * GraphIterationMixin<itself>, which is also what supplies the whole @c for* family. Structural
 * matching alone is not enough, and there is nothing to remember to declare.
 *
 * Everything listed here is a *primitive* -- it cannot be computed from the others without
 * changing an algorithm's complexity. Anything derivable lives in the mixin instead, so a new
 * graph type implements this list and gets the rest. @c numberOfSelfLoops is the documented
 * exception: derivable in principle, but call sites use it as an entry predicate where an
 * O(n * lookup) default would be wrong.
 *
 * What a class guarantees by satisfying this:
 * - @c degree(u) is O(1). Iteration does not rely on it, but callers do.
 * - @c outNeighbors<Weighted>(u) may be traversed more than once, and yields neighbors in a
 *   stable order for as long as the graph is unmodified.
 * - The returned range borrows the graph's storage and must not outlive the graph.
 *
 * @c outNeighbors is a template because weightedness is a runtime flag while the range's type
 * depends on it: an unweighted graph has no weight array to point at. Callers of the @c for*
 * family never see this; implementers of a graph class do.
 */
template <typename G>
concept GraphLike = std::derived_from<G, GraphIterationMixin<G>> && requires(const G g, node u) {
    { g.numberOfNodes() } -> std::convertible_to<count>;
    { g.numberOfEdges() } -> std::convertible_to<count>;
    { g.upperNodeIdBound() } -> std::convertible_to<index>;
    { g.hasNode(u) } -> std::convertible_to<bool>;
    { g.degree(u) } -> std::convertible_to<count>;
    { g.isDirected() } -> std::convertible_to<bool>;
    { g.isWeighted() } -> std::convertible_to<bool>;
    { g.numberOfSelfLoops() } -> std::convertible_to<count>;

    { g.template outNeighbors<false>(u) } -> std::ranges::forward_range;
    { g.template outNeighbors<true>(u) } -> std::ranges::forward_range;
    { g.template inNeighbors<false>(u) } -> std::ranges::forward_range;
    { g.template inNeighbors<true>(u) } -> std::ranges::forward_range;
    requires NeighborLike<std::ranges::range_value_t<decltype(g.template outNeighbors<false>(u))>>;
    requires NeighborLike<std::ranges::range_value_t<decltype(g.template outNeighbors<true>(u))>>;
};

/**
 * Adds edge ids carried on the neighbors, so asking an unindexed graph for an edge id is a
 * compile error rather than a runtime throw.
 */
template <typename G>
concept IndexedGraph = GraphLike<G> && requires(const G g, node u) {
    { g.hasEdgeIds() } -> std::convertible_to<bool>;
    { g.upperEdgeIdBound() } -> std::convertible_to<index>;
    { g.edgeId(u, u) } -> std::convertible_to<edgeid>;
    { g.template outNeighborsIndexed<false>(u) } -> std::ranges::forward_range;
    { g.template outNeighborsIndexed<true>(u) } -> std::ranges::forward_range;
    { g.template inNeighborsIndexed<false>(u) } -> std::ranges::forward_range;
    { g.template inNeighborsIndexed<true>(u) } -> std::ranges::forward_range;
    {
        neighborId(*std::ranges::begin(g.template outNeighborsIndexed<false>(u)))
    } -> std::convertible_to<edgeid>;
    {
        neighborId(*std::ranges::begin(g.template outNeighborsIndexed<true>(u)))
    } -> std::convertible_to<edgeid>;
};

template <typename G>
concept MutableGraph = GraphLike<G> && requires(G g, node u, node v, edgeweight ew) {
    { g.addNode() } -> std::convertible_to<node>;
    g.addEdge(u, v, ew);
    g.removeEdge(u, v);
    g.removeNode(u);
    g.setWeight(u, v, ew);
};

} // namespace NetworKit

#endif // NETWORKIT_GRAPH_GRAPH_CONCEPTS_HPP_
