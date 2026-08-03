/*
 * GraphIterationMixin.hpp
 *
 *  The for* family, written once for every graph satisfying the concepts.
 */

#ifndef NETWORKIT_GRAPH_GRAPH_ITERATION_MIXIN_HPP_
#define NETWORKIT_GRAPH_GRAPH_ITERATION_MIXIN_HPP_

#include <algorithm>
#include <vector>

#include <omp.h>

#include <networkit/Globals.hpp>
#include <networkit/auxiliary/Random.hpp>
#include <networkit/graph/GraphConcepts.hpp>
#include <networkit/graph/GraphIteration.hpp>

namespace NetworKit {

/**
 * CRTP mixin supplying every graph traversal that can be expressed in terms of the primitives.
 *
 * A graph inheriting this provides only the primitives and gets the whole family. Members reach
 * the graph through self(), so a derived class overriding one of them has its override respected
 * by every other member that builds on it -- a graph whose node set is not a range of ids need
 * only override forNodes() for the edge loops to follow.
 *
 * Derived is unconstrained because the base is instantiated while Derived is still incomplete.
 * The GraphLike requirement is checked when a member is first used.
 */
template <typename Derived>
class GraphIterationMixin {
    const Derived &self() const { return static_cast<const Derived &>(*this); }

public:
    /* STRUCTURAL PROPERTIES */

    /*
     * Each answers at runtime what a class may instead pin for all its instances by declaring the
     * matching static constexpr member. A class declares only what holds for every instance, so a
     * tag cannot lie, and where one is present the query folds to a constant.
     */

    /// True when node ids are dense, so iteration may skip the hasNode() check.
    bool hasContiguousNodeIds() const {
        if constexpr (requires { Derived::alwaysContiguousNodeIds; })
            return Derived::alwaysContiguousNodeIds;
        else
            return self().numberOfNodes() == self().upperNodeIdBound();
    }

    /// True when neighborhoods are in ascending id order, so lookups may bisect.
    bool hasSortedNeighborhoods() const {
        if constexpr (requires { Derived::alwaysSortedNeighborhoods; })
            return Derived::alwaysSortedNeighborhoods;
        else
            return false;
    }

    /* NODE ITERATORS */

    /// Visits every existing node in ascending id order.
    /*
     * The upper bound is re-read each step, not hoisted: a callback is allowed to add nodes while
     * iterating, and the nodes it adds are visited.
     */
    template <typename L>
    void forNodes(L handle) const {
        const bool dense = self().hasContiguousNodeIds();
        for (node u = 0; u < self().upperNodeIdBound(); ++u) {
            if (!dense && !self().hasNode(u))
                continue;
            handle(u);
        }
    }

    template <typename L>
    void parallelForNodes(L handle) const {
        const bool dense = self().hasContiguousNodeIds();
        const auto z = static_cast<omp_index>(self().upperNodeIdBound());
#pragma omp parallel for
        for (omp_index u = 0; u < z; ++u) {
            if (!dense && !self().hasNode(static_cast<node>(u)))
                continue;
            handle(static_cast<node>(u));
        }
    }

    /// Visits nodes in ascending id order until @a condition returns false.
    template <typename C, typename L>
    void forNodesWhile(C condition, L handle) const {
        const bool dense = self().hasContiguousNodeIds();
        const index z = self().upperNodeIdBound();
        for (node u = 0; u < z; ++u) {
            if (!dense && !self().hasNode(u))
                continue;
            if (!condition())
                break;
            handle(u);
        }
    }

    template <typename L>
    void forNodesInRandomOrder(L handle) const {
        std::vector<node> randVec;
        randVec.reserve(self().numberOfNodes());
        self().forNodes([&](node u) { randVec.push_back(u); });
        std::shuffle(randVec.begin(), randVec.end(), Aux::Random::getURNG());
        for (node v : randVec) {
            handle(v);
        }
    }

    /// Uses schedule(guided) to remedy load imbalance from an unequal degree distribution.
    template <typename L>
    void balancedParallelForNodes(L handle) const {
        const bool dense = self().hasContiguousNodeIds();
        const auto z = static_cast<omp_index>(self().upperNodeIdBound());
#pragma omp parallel for schedule(guided)
        for (omp_index u = 0; u < z; ++u) {
            if (!dense && !self().hasNode(static_cast<node>(u)))
                continue;
            handle(static_cast<node>(u));
        }
    }

    /// Visits each unordered pair {u, v} of distinct existing nodes once, with u < v.
    template <typename L>
    void forNodePairs(L handle) const {
        const bool dense = self().hasContiguousNodeIds();
        const index z = self().upperNodeIdBound();
        for (node u = 0; u < z; ++u) {
            if (!dense && !self().hasNode(u))
                continue;
            for (node v = u + 1; v < z; ++v) {
                if (!dense && !self().hasNode(v))
                    continue;
                handle(u, v);
            }
        }
    }

    template <typename L>
    void parallelForNodePairs(L handle) const {
        const bool dense = self().hasContiguousNodeIds();
        const auto z = static_cast<omp_index>(self().upperNodeIdBound());
#pragma omp parallel for schedule(guided)
        for (omp_index u = 0; u < z; ++u) {
            if (!dense && !self().hasNode(static_cast<node>(u)))
                continue;
            for (node v = static_cast<node>(u) + 1; v < static_cast<node>(z); ++v) {
                if (!dense && !self().hasNode(v))
                    continue;
                handle(static_cast<node>(u), v);
            }
        }
    }

    /* EDGE ITERATORS */

    /**
     * Visits every edge once. A directed graph yields each edge as (source, target); an undirected
     * graph yields it once from the smaller-id endpoint, so (u, v) satisfies u <= v.
     *
     * The payload handed to @a handle is chosen from its arity: (u, v), (u, v, weight), or
     * (u, v, weight, edgeid). What the callback does not take is never read from storage.
     */
    template <typename L>
    void forEdges(L handle) const {
        using namespace GraphIterationDetail;
        withEdgeFlags(self(), [&]<bool W, bool I, bool D>() {
            self().forNodes([&](node u) { forOutEdgesOf<W, I, D, true>(self(), u, handle); });
        });
    }

    /**
     * As forEdges(), with the outer node loop parallelised.
     *
     * The omp directive must sit in the function that owns @a handle: clang's OpenMP rejects
     * capturing a callback across an extra dispatch lambda layer, so this cannot be a thin wrapper.
     */
    template <typename L>
    void parallelForEdges(L handle) const {
        using namespace GraphIterationDetail;
        withEdgeFlags(self(), [&]<bool W, bool I, bool D>() {
            const bool dense = self().hasContiguousNodeIds();
            const auto z = static_cast<omp_index>(self().upperNodeIdBound());
#pragma omp parallel for schedule(guided)
            for (omp_index i = 0; i < z; ++i) {
                const node u = static_cast<node>(i);
                if (!dense && !self().hasNode(u))
                    continue;
                forOutEdgesOf<W, I, D, true>(self(), u, handle);
            }
        });
    }

    /* NEIGHBORHOOD ITERATORS */

    /// Visits the out-neighbors of @a u in the graph's stored order.
    template <typename L>
    void forNeighborsOf(node u, L handle) const {
        self().forEdgesOf(u, handle);
    }

    /// Visits the out-edges of @a u; differs from forNeighborsOf() only in the callback payload.
    template <typename L>
    void forEdgesOf(node u, L handle) const {
        using namespace GraphIterationDetail;
        withEdgeFlags(self(), [&]<bool W, bool I, bool D>() {
            forOutEdgesOf<W, I, D, false>(self(), u, handle);
        });
    }

    /// Visits the in-neighbors of @a u. On an undirected graph these are the out-neighbors.
    template <typename L>
    void forInNeighborsOf(node u, L handle) const {
        self().forInEdgesOf(u, handle);
    }

    template <typename L>
    void forInEdgesOf(node u, L handle) const {
        using namespace GraphIterationDetail;
        withEdgeFlags(self(),
                      [&]<bool W, bool I, bool D>() { forInEdgesOfNode<W, I>(self(), u, handle); });
    }

    /* REDUCTION ITERATORS */

    template <typename L>
    double parallelSumForNodes(L handle) const {
        double sum = 0.0;
        const bool dense = self().hasContiguousNodeIds();
        const auto z = static_cast<omp_index>(self().upperNodeIdBound());
#pragma omp parallel for reduction(+ : sum)
        for (omp_index u = 0; u < z; ++u) {
            if (!dense && !self().hasNode(static_cast<node>(u)))
                continue;
            sum += handle(static_cast<node>(u));
        }
        return sum;
    }

    /// The reduction lives in sumForEdgesImpl(), which owns the pragma clang's OpenMP requires it
    /// to be declared alongside.
    template <typename L>
    double parallelSumForEdges(L handle) const {
        using namespace GraphIterationDetail;
        double sum = 0.0;
        withEdgeFlags(self(), [&]<bool Weighted, bool Indexed, bool Directed>() {
            sum = sumForEdgesImpl<Weighted, Indexed, Directed>(self(), handle);
        });
        return sum;
    }

    /* LOOKUPS */

    count degreeOut(node u) const { return self().degree(u); }

    /// O(1) when undirected, otherwise O(in-degree) -- a graph storing in-degrees should override.
    count degreeIn(node u) const {
        if (!self().isDirected())
            return self().degree(u);
        count d = 0;
        for ([[maybe_unused]] auto nb : self().template inNeighbors<false>(u))
            ++d;
        return d;
    }

    /// O(log d) on sorted neighborhoods, O(d) otherwise.
    bool hasEdge(node u, node v) const {
        if (self().hasSortedNeighborhoods())
            return std::ranges::binary_search(self().template outNeighbors<false>(u), v);
        for (auto nb : self().template outNeighbors<false>(u))
            if (neighborTarget(nb) == v)
                return true;
        return false;
    }

    /// O(d); nullWeight when the edge is absent.
    edgeweight weight(node u, node v) const {
        for (auto nb : self().template outNeighbors<true>(u))
            if (neighborTarget(nb) == v)
                return neighborWeight(nb);
        return nullWeight;
    }

    /* WEIGHT AGGREGATES */

    /**
     * The sum of the weights of @a u's incident edges, counting a self-loop once unless
     * @a countSelfLoopsTwice.
     *
     * On an unweighted graph this is the degree, so no neighbor is visited unless self-loops make
     * the answer differ from it.
     */
    edgeweight weightedDegree(node u, bool countSelfLoopsTwice = false) const {
        return weightedDegreeImpl(u, false, countSelfLoopsTwice);
    }

    edgeweight weightedDegreeIn(node u, bool countSelfLoopsTwice = false) const {
        return weightedDegreeImpl(u, true, countSelfLoopsTwice);
    }

    /// The sum of the weights of all edges; the edge count when the graph is unweighted.
    edgeweight totalEdgeWeight() const {
        if (self().isWeighted())
            return self().parallelSumForEdges([](node, node, edgeweight ew) { return ew; });
        return self().numberOfEdges() * defaultEdgeWeight;
    }

private:
    edgeweight weightedDegreeImpl(node u, bool inDegree, bool countSelfLoopsTwice) const {
        if (self().isWeighted()) {
            edgeweight sum = 0.0;
            auto sumWeights = [&](node v, edgeweight w) {
                sum += (countSelfLoopsTwice && u == v) ? 2. * w : w;
            };
            if (inDegree)
                self().forInNeighborsOf(u, sumWeights);
            else
                self().forNeighborsOf(u, sumWeights);
            return sum;
        }

        count sum = inDegree ? self().degreeIn(u) : self().degreeOut(u);
        auto countSelfLoops = [&](node v) { sum += (u == v); };

        if (countSelfLoopsTwice && self().numberOfSelfLoops()) {
            if (inDegree)
                self().forInNeighborsOf(u, countSelfLoops);
            else
                self().forNeighborsOf(u, countSelfLoops);
        }

        return static_cast<edgeweight>(sum);
    }
};

} // namespace NetworKit

#endif // NETWORKIT_GRAPH_GRAPH_ITERATION_MIXIN_HPP_
