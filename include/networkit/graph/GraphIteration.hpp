/*
 * GraphIteration.hpp
 *
 *  Callback-shape dispatch and the per-edge loops the for* family is built from.
 */

#ifndef NETWORKIT_GRAPH_GRAPH_ITERATION_HPP_
#define NETWORKIT_GRAPH_GRAPH_ITERATION_HPP_

#include <cstddef>
#include <type_traits>
#include <utility>

#include <omp.h>

#include <networkit/Globals.hpp>
#include <networkit/auxiliary/FunctionTraits.hpp>
#include <networkit/graph/GraphConcepts.hpp>

namespace NetworKit {

/*
 * Aux::FunctionTraits selects which of these is viable for a given callback, so a callback is
 * handed exactly the parameters it declares and the range is never asked for a payload nobody
 * reads. The decltype return type doubles as the arity check: a callback with the wrong shape
 * makes every overload but the diagnostic one non-viable.
 */

template <class F, void * = (void *)0>
typename Aux::FunctionTraits<F>::result_type edgeLambda(F &, ...) {
    static_assert(!std::is_same<F, F>::value,
                  "Your lambda does not support the required parameters or the "
                  "parameters have the wrong type.");
    return std::declval<typename Aux::FunctionTraits<F>::result_type>();
}

template <
    class F,
    typename std::enable_if<
        (Aux::FunctionTraits<F>::arity >= 3)
        && std::is_same<edgeweight, typename Aux::FunctionTraits<F>::template arg<2>::type>::value
        && std::is_same<edgeid, typename Aux::FunctionTraits<F>::template arg<3>::type>::value>::
        type * = (void *)0>
auto edgeLambda(F &f, node u, node v, edgeweight ew, edgeid id) -> decltype(f(u, v, ew, id)) {
    return f(u, v, ew, id);
}

template <class F,
          typename std::enable_if<
              (Aux::FunctionTraits<F>::arity >= 2)
              && std::is_same<edgeid, typename Aux::FunctionTraits<F>::template arg<2>::type>::value
              && std::is_same<node, typename Aux::FunctionTraits<F>::template arg<1>::type>::value
              /* prevent f(v, weight, eid) */
              >::type * = (void *)0>
auto edgeLambda(F &f, node u, node v, edgeweight, edgeid id) -> decltype(f(u, v, id)) {
    return f(u, v, id);
}

template <class F, typename std::enable_if<
                       (Aux::FunctionTraits<F>::arity >= 2)
                       && std::is_same<edgeweight, typename Aux::FunctionTraits<F>::template arg<
                                                       2>::type>::value>::type * = (void *)0>
auto edgeLambda(F &f, node u, node v, edgeweight ew, edgeid) -> decltype(f(u, v, ew)) {
    return f(u, v, ew);
}

template <class F, typename std::enable_if<
                       (Aux::FunctionTraits<F>::arity >= 1)
                       && std::is_same<node, typename Aux::FunctionTraits<F>::template arg<
                                                 1>::type>::value>::type * = (void *)0>
auto edgeLambda(F &f, node u, node v, edgeweight, edgeid) -> decltype(f(u, v)) {
    return f(u, v);
}

template <class F, typename std::enable_if<
                       (Aux::FunctionTraits<F>::arity >= 1)
                       && std::is_same<edgeweight, typename Aux::FunctionTraits<F>::template arg<
                                                       1>::type>::value>::type * = (void *)0>
auto edgeLambda(F &f, node, node v, edgeweight ew, edgeid) -> decltype(f(v, ew)) {
    return f(v, ew);
}

template <class F, void * = (void *)0>
auto edgeLambda(F &f, node, node v, edgeweight, edgeid) -> decltype(f(v)) {
    return f(v);
}

namespace GraphIterationDetail {

/// An undirected edge is visited from its smaller-id endpoint, so each is seen exactly once.
template <bool Directed>
inline bool useEdge(node u, node v) {
    if constexpr (Directed)
        return true;
    else
        return u <= v;
}

/**
 * Hands @a nb to @a handle with whatever payload the callback declares.
 *
 * @tparam HasId must come from the caller, which knows which range it is walking. Probing the
 * neighbor type with `requires { neighborId(nb); }` would answer yes for an unindexed weighted
 * neighbor, because std::pair<node, edgeweight> converts to std::pair<node, edgeid> and the
 * weight would be handed over as the id.
 */
template <bool HasId, typename L, typename N>
inline void applyEdge(L &handle, node u, const N &nb) {
    edgeLambda(handle, u, neighborTarget(nb), neighborWeight(nb), [&] {
        if constexpr (HasId)
            return neighborId(nb);
        else
            return none;
    }());
}

template <bool Weighted, bool Indexed, bool Directed, bool Dedup, GraphLike G, typename L>
inline void forOutEdgesOf(const G &g, node u, L &handle) {
    if constexpr (Indexed && IndexedGraph<G>) {
        for (const auto nb : g.template outNeighborsIndexed<Weighted>(u)) {
            if (Dedup && !useEdge<Directed>(u, neighborTarget(nb)))
                continue;
            applyEdge<true>(handle, u, nb);
        }
    } else {
        for (const auto nb : g.template outNeighbors<Weighted>(u)) {
            if (Dedup && !useEdge<Directed>(u, neighborTarget(nb)))
                continue;
            applyEdge<false>(handle, u, nb);
        }
    }
}

/**
 * The in-edge counterpart of forOutEdgesOf().
 *
 * Named rather than inlined into a lambda: inside a lambda the graph is a capture of concrete
 * type, so a discarded `if constexpr` branch still undergoes member lookup and an unindexed
 * graph fails to compile on the indexed branch.
 */
template <bool Weighted, bool Indexed, GraphLike G, typename L>
inline void forInEdgesOfNode(const G &g, node u, L &handle) {
    if constexpr (Indexed && IndexedGraph<G>) {
        for (const auto nb : g.template inNeighborsIndexed<Weighted>(u))
            applyEdge<true>(handle, u, nb);
    } else {
        for (const auto nb : g.template inNeighbors<Weighted>(u))
            applyEdge<false>(handle, u, nb);
    }
}

/**
 * The reduction form of the edge loop.
 *
 * This exists as a named function rather than a lambda passed to withEdgeFlags() because clang's
 * OpenMP rejects a reduction clause whose variable, or whose callback, is a local of an enclosing
 * function. Both @a sum and @a handle must therefore belong to the function holding the pragma.
 */
template <bool Weighted, bool Indexed, bool Directed, GraphLike G, typename L>
inline double sumForEdgesImpl(const G &g, L &handle) {
    double sum = 0.0;
    const bool dense = hasContiguousNodeIds(g);
    const auto z = static_cast<omp_index>(g.upperNodeIdBound());
#pragma omp parallel for reduction(+ : sum) schedule(guided)
    for (omp_index i = 0; i < z; ++i) {
        const node u = static_cast<node>(i);
        if (!dense)
            if (!g.hasNode(u))
                continue;
        auto accumulate = [&sum, &handle](node s, node t, edgeweight ew, edgeid id) {
            sum += edgeLambda(handle, s, t, ew, id);
        };
        forOutEdgesOf<Weighted, Indexed, Directed, true>(g, u, accumulate);
    }
    return sum;
}

/// Folds to a literal false for a graph that cannot carry edge ids.
template <GraphLike G>
inline bool isIndexed(const G &g) {
    if constexpr (IndexedGraph<G>)
        return g.hasEdgeIds();
    else
        return false;
}

/// Dispatches the three runtime flags once per call, never per edge.
template <GraphLike G, typename Body>
inline void withEdgeFlags(const G &g, Body body) {
    switch (g.isWeighted() * 4 + g.isDirected() * 2 + isIndexed(g)) {
    case 0:
        return body.template operator()<false, false, false>();
    case 1:
        return body.template operator()<false, true, false>();
    case 2:
        return body.template operator()<false, false, true>();
    case 3:
        return body.template operator()<false, true, true>();
    case 4:
        return body.template operator()<true, false, false>();
    case 5:
        return body.template operator()<true, true, false>();
    case 6:
        return body.template operator()<true, false, true>();
    default:
        return body.template operator()<true, true, true>();
    }
}

} // namespace GraphIterationDetail

} // namespace NetworKit

#endif // NETWORKIT_GRAPH_GRAPH_ITERATION_HPP_
