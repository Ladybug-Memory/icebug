/*
 * GraphIterationOps.hpp
 *
 *  The for* family, lookups, and weight aggregates as free customization points over GraphLike.
 *
 *  Every operation here is implemented once, in terms of the primitives a GraphLike type declares,
 *  and is constrained to GraphLike so a missing primitive surfaces as a concept error naming the
 *  missing member. A graph class need not inherit anything to get the whole family -- it simply
 *  provides the primitives listed in GraphConcepts.hpp and these free functions apply to it. The
 *  concrete read-only/mutable graphs re-expose them as thin forwarding members (see GraphW,
 *  GraphR) so the member-call interface @c g.forNodes(...) keeps working, but the logic itself
 *  lives here, shared by every graph type.
 *
 *  The free operations are deliberately NOT constrained by GraphLike: GraphW and GraphR call into
 *  this layer from their own class definitions, where the graph type is still incomplete and
 *  outNeighbors's deduced return type does not yet exist, so a GraphLike constraint would reject
 *  every such call. The concept instead gates the abstract layer -- algorithm templates declared as
 *  @c template <GraphLike G> and the archetype tests -- where the graph is always complete.
 */

#ifndef NETWORKIT_GRAPH_GRAPH_ITERATION_OPS_HPP_
#define NETWORKIT_GRAPH_GRAPH_ITERATION_OPS_HPP_

#include <algorithm>
#include <vector>

#include <omp.h>

#include <networkit/Globals.hpp>
#include <networkit/auxiliary/Random.hpp>
#include <networkit/graph/GraphConcepts.hpp>
#include <networkit/graph/GraphIteration.hpp>

namespace NetworKit {

namespace GraphIterationOps {

/* NODE ITERATORS */

/// Visits every existing node in ascending id order.
/*
 * The upper bound is re-read each step, not hoisted: a callback is allowed to add nodes while
 * iterating, and the nodes it adds are visited.
 */
template <typename G, typename L>
inline void forNodes(const G &g, L handle) {
    const bool dense = hasContiguousNodeIds(g);
    for (node u = 0; u < g.upperNodeIdBound(); ++u) {
        if (!dense && !g.hasNode(u))
            continue;
        handle(u);
    }
}

template <typename G, typename L>
inline void parallelForNodes(const G &g, L handle) {
    const bool dense = hasContiguousNodeIds(g);
    const auto z = static_cast<omp_index>(g.upperNodeIdBound());
#pragma omp parallel for
    for (omp_index u = 0; u < z; ++u) {
        if (!dense && !g.hasNode(static_cast<node>(u)))
            continue;
        handle(static_cast<node>(u));
    }
}

/// Visits nodes in ascending id order until @a condition returns false.
template <typename G, typename C, typename L>
inline void forNodesWhile(const G &g, C condition, L handle) {
    const bool dense = hasContiguousNodeIds(g);
    const index z = g.upperNodeIdBound();
    for (node u = 0; u < z; ++u) {
        if (!dense && !g.hasNode(u))
            continue;
        if (!condition())
            break;
        handle(u);
    }
}

/// Visits nodes in a random order.
template <typename G, typename L>
inline void forNodesInRandomOrder(const G &g, L handle) {
    std::vector<node> randVec;
    randVec.reserve(g.numberOfNodes());
    forNodes(g, [&](node u) { randVec.push_back(u); });
    std::shuffle(randVec.begin(), randVec.end(), Aux::Random::getURNG());
    for (node v : randVec) {
        handle(v);
    }
}

/// Uses schedule(guided) to remedy load imbalance from an unequal degree distribution.
template <typename G, typename L>
inline void balancedParallelForNodes(const G &g, L handle) {
    const bool dense = hasContiguousNodeIds(g);
    const auto z = static_cast<omp_index>(g.upperNodeIdBound());
#pragma omp parallel for schedule(guided)
    for (omp_index u = 0; u < z; ++u) {
        if (!dense && !g.hasNode(static_cast<node>(u)))
            continue;
        handle(static_cast<node>(u));
    }
}

/// Visits each unordered pair {u, v} of distinct existing nodes once, with u < v.
template <typename G, typename L>
inline void forNodePairs(const G &g, L handle) {
    const bool dense = hasContiguousNodeIds(g);
    const index z = g.upperNodeIdBound();
    for (node u = 0; u < z; ++u) {
        if (!dense && !g.hasNode(u))
            continue;
        for (node v = u + 1; v < z; ++v) {
            if (!dense && !g.hasNode(v))
                continue;
            handle(u, v);
        }
    }
}

template <typename G, typename L>
inline void parallelForNodePairs(const G &g, L handle) {
    const bool dense = hasContiguousNodeIds(g);
    const auto z = static_cast<omp_index>(g.upperNodeIdBound());
#pragma omp parallel for schedule(guided)
    for (omp_index u = 0; u < z; ++u) {
        if (!dense && !g.hasNode(static_cast<node>(u)))
            continue;
        for (node v = static_cast<node>(u) + 1; v < static_cast<node>(z); ++v) {
            if (!dense && !g.hasNode(v))
                continue;
            handle(static_cast<node>(u), v);
        }
    }
}

/* NEIGHBORHOOD ITERATORS (out-edges of a single node) */

/// Visits the out-edges of @a u; differs from forNeighborsOf() only in the callback payload.
template <typename G, typename L>
inline void forEdgesOf(const G &g, node u, L handle) {
    using namespace GraphIterationDetail;
    withEdgeFlags(g,
                  [&]<bool W, bool I, bool D>() { forOutEdgesOf<W, I, D, false>(g, u, handle); });
}

/// Visits the in-edges of @a u.
template <typename G, typename L>
inline void forInEdgesOf(const G &g, node u, L handle) {
    using namespace GraphIterationDetail;
    withEdgeFlags(g, [&]<bool W, bool I, bool D>() { forInEdgesOfNode<W, I>(g, u, handle); });
}

/// Visits the out-neighbors of @a u in the graph's stored order.
template <typename G, typename L>
inline void forNeighborsOf(const G &g, node u, L handle) {
    forEdgesOf(g, u, handle);
}

/// Visits the in-neighbors of @a u. On an undirected graph these are the out-neighbors.
template <typename G, typename L>
inline void forInNeighborsOf(const G &g, node u, L handle) {
    forInEdgesOf(g, u, handle);
}

/* EDGE ITERATORS (all edges) */

/**
 * Visits every edge once. A directed graph yields each edge as (source, target); an undirected
 * graph yields it once from the smaller-id endpoint, so (u, v) satisfies u <= v.
 *
 * The payload handed to @a handle is chosen from its arity: (u, v), (u, v, weight), or
 * (u, v, weight, edgeid). What the callback does not take is never read from storage.
 */
template <typename G, typename L>
inline void forEdges(const G &g, L handle) {
    using namespace GraphIterationDetail;
    withEdgeFlags(g, [&]<bool W, bool I, bool D>() {
        forNodes(g, [&](node u) { forOutEdgesOf<W, I, D, true>(g, u, handle); });
    });
}

/**
 * As forEdges(), with the outer node loop parallelised.
 *
 * The omp directive must sit in the function that owns @a handle: clang's OpenMP rejects
 * capturing a callback across an extra dispatch lambda layer, so this cannot be a thin wrapper.
 */
template <typename G, typename L>
inline void parallelForEdges(const G &g, L handle) {
    using namespace GraphIterationDetail;
    withEdgeFlags(g, [&]<bool W, bool I, bool D>() {
        const bool dense = hasContiguousNodeIds(g);
        const auto z = static_cast<omp_index>(g.upperNodeIdBound());
#pragma omp parallel for schedule(guided)
        for (omp_index i = 0; i < z; ++i) {
            const node u = static_cast<node>(i);
            if (!dense && !g.hasNode(u))
                continue;
            forOutEdgesOf<W, I, D, true>(g, u, handle);
        }
    });
}

/* REDUCTION ITERATORS */

template <typename G, typename L>
inline double parallelSumForNodes(const G &g, L handle) {
    double sum = 0.0;
    const bool dense = hasContiguousNodeIds(g);
    const auto z = static_cast<omp_index>(g.upperNodeIdBound());
#pragma omp parallel for reduction(+ : sum)
    for (omp_index u = 0; u < z; ++u) {
        if (!dense && !g.hasNode(static_cast<node>(u)))
            continue;
        sum += handle(static_cast<node>(u));
    }
    return sum;
}

/// The reduction lives in sumForEdgesImpl(), which owns the pragma clang's OpenMP requires it
/// to be declared alongside.
template <typename G, typename L>
inline double parallelSumForEdges(const G &g, L handle) {
    using namespace GraphIterationDetail;
    double sum = 0.0;
    withEdgeFlags(g, [&]<bool Weighted, bool Indexed, bool Directed>() {
        sum = sumForEdgesImpl<Weighted, Indexed, Directed>(g, handle);
    });
    return sum;
}

/* LOOKUPS */

template <typename G>
inline count degreeOut(const G &g, node u) {
    return g.degree(u);
}

/// O(1) when undirected, otherwise O(in-degree) -- a graph storing in-degrees should override.
template <typename G>
inline count degreeIn(const G &g, node u) {
    if (!g.isDirected())
        return g.degree(u);
    count d = 0;
    for ([[maybe_unused]] auto nb : g.template inNeighbors<false>(u))
        ++d;
    return d;
}

/// O(log d) on sorted neighborhoods, O(d) otherwise.
template <typename G>
inline bool hasEdge(const G &g, node u, node v) {
    /*
     * Bisection needs the unweighted payload to order against a bare node. A neighborhood may
     * instead carry (node, weight) pairs -- e.g. a view that aggregates weights on the fly --
     * which sorts by target just fine but does not compare with v, so it falls back to the scan.
     */
    constexpr bool bisectable = requires(const G &gg, node uu, node vv) {
        std::ranges::binary_search(gg.template outNeighbors<false>(uu), vv);
    };
    if constexpr (bisectable) {
        if (hasSortedNeighborhoods(g))
            return std::ranges::binary_search(g.template outNeighbors<false>(u), v);
    }
    for (auto nb : g.template outNeighbors<false>(u))
        if (neighborTarget(nb) == v)
            return true;
    return false;
}

/// O(d); nullWeight when the edge is absent. On an unweighted graph an existing edge weighs
/// defaultEdgeWeight, mirroring the concrete graphs' own weight().
template <typename G>
inline edgeweight weight(const G &g, node u, node v) {
    if (!g.isWeighted()) {
        for (auto nb : g.template outNeighbors<false>(u))
            if (neighborTarget(nb) == v)
                return defaultEdgeWeight;
        return nullWeight;
    }
    for (auto nb : g.template outNeighbors<true>(u))
        if (neighborTarget(nb) == v)
            return neighborWeight(nb);
    return nullWeight;
}

/* WEIGHT AGGREGATES */

namespace detail {
template <typename G>
inline edgeweight weightedDegreeImpl(const G &g, node u, bool inDegree, bool countSelfLoopsTwice) {
    if (g.isWeighted()) {
        edgeweight sum = 0.0;
        auto sumWeights = [&](node v, edgeweight w) {
            sum += (countSelfLoopsTwice && u == v) ? 2. * w : w;
        };
        if (inDegree)
            forInNeighborsOf(g, u, sumWeights);
        else
            forNeighborsOf(g, u, sumWeights);
        return sum;
    }

    count sum = inDegree ? degreeIn(g, u) : degreeOut(g, u);
    auto countSelfLoops = [&](node v) { sum += (u == v); };

    if (countSelfLoopsTwice && g.numberOfSelfLoops()) {
        if (inDegree)
            forInNeighborsOf(g, u, countSelfLoops);
        else
            forNeighborsOf(g, u, countSelfLoops);
    }

    return static_cast<edgeweight>(sum);
}
} // namespace detail

/**
 * The sum of the weights of @a u's incident edges, counting a self-loop once unless
 * @a countSelfLoopsTwice.
 *
 * On an unweighted graph this is the degree, so no neighbor is visited unless self-loops make
 * the answer differ from it.
 */
template <typename G>
inline edgeweight weightedDegree(const G &g, node u, bool countSelfLoopsTwice = false) {
    return detail::weightedDegreeImpl(g, u, false, countSelfLoopsTwice);
}

template <typename G>
inline edgeweight weightedDegreeIn(const G &g, node u, bool countSelfLoopsTwice = false) {
    return detail::weightedDegreeImpl(g, u, true, countSelfLoopsTwice);
}

/// The sum of the weights of all edges; the edge count when the graph is unweighted.
///
/// Computed with a deterministic sequential reduction. The OMP parallel sum is nondeterministic
/// for floating point (schedule-dependent accumulation order), which made repeated calls return
/// slightly different values -- e.g. two callers comparing the volume against totalEdgeWeight
/// could disagree by ~1 ulp.
template <typename G>
inline edgeweight totalEdgeWeight(const G &g) {
    if (g.isWeighted()) {
        edgeweight total = 0.0;
        forEdges(g, [&total](node, node, edgeweight ew) { total += ew; });
        return total;
    }
    return g.numberOfEdges() * defaultEdgeWeight;
}

} // namespace GraphIterationOps

} // namespace NetworKit

#endif // NETWORKIT_GRAPH_GRAPH_ITERATION_OPS_HPP_
