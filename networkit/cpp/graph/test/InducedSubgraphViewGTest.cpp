/*
 * InducedSubgraphViewGTest.cpp
 *
 *  Tests for the zero-copy induced subgraph view over GraphLike bases.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <networkit/auxiliary/Random.hpp>
#include <networkit/auxiliary/Vector2Arrow.hpp>
#include <networkit/generators/ErdosRenyiGenerator.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/graph/GraphIterationOps.hpp>
#include <networkit/graph/GraphR.hpp>
#include <networkit/graph/GraphTools.hpp>
#include <networkit/graph/GraphW.hpp>
#include <networkit/graph/InducedSubgraphView.hpp>

namespace NetworKit {

namespace {

/// A fixed house graph plus a self-loop on node 0:
/// edges 3-1, 1-0, 0-2, 2-1, 1-4, 4-3, 3-2, 2-4, in that insertion order.
GraphW houseBase(bool weighted = true) {
    const count n = 5;
    GraphW G(n, weighted, false);
    std::vector<std::pair<node, node>> edges{{3, 1}, {1, 0}, {0, 2}, {2, 1},
                                             {1, 4}, {4, 3}, {3, 2}, {2, 4}};
    edgeweight w = 1.0;
    for (const auto &[u, v] : edges) {
        G.addEdge(u, v, w);
        if (weighted)
            w += 1.0;
    }
    G.addEdge(0, 0); // self-loop
    return G;
}

/// A fixed unweighted undirected GraphR: the house graph without the self-loop, CSR-encoded.
GraphR houseCSR() {
    const std::vector<std::vector<node>> adjacency{
        {1, 2}, {0, 2, 3, 4}, {0, 1, 3, 4}, {1, 2, 4}, {1, 2, 3}};
    std::vector<uint64_t> indptr{0};
    std::vector<uint64_t> indices;
    for (const auto &neighbors : adjacency) {
        indices.insert(indices.end(), neighbors.begin(), neighbors.end());
        indptr.push_back(indices.size());
    }

    // An undirected GraphR shares the out arrays with the in arrays.
    return GraphR(5, false, Aux::vectorToArrow<uint64_t, arrow::UInt64Array>(std::move(indices)),
                  Aux::vectorToArrow<uint64_t, arrow::UInt64Array>(std::move(indptr)));
}

/// The induced edge set of @a G on @a subset, computed independently of the view.
std::vector<std::tuple<node, node, edgeweight>> expectedEdges(const GraphW &G,
                                                              const std::set<node> &subset) {
    std::vector<std::tuple<node, node, edgeweight>> result;
    for (const node u : subset)
        G.forNeighborsOf(u, [&](node v, edgeweight w) {
            if (!subset.count(v))
                return;
            if (!G.isDirected() && u > v)
                return;
            result.emplace_back(u, v, w);
        });
    std::sort(result.begin(), result.end());
    return result;
}

template <typename View>
void expectSameEdges(const GraphW &G, const std::set<node> &subset, const View &view) {
    using Entry = std::tuple<node, node, edgeweight>;
    std::vector<Entry> viewEdges;
    view.forEdges([&](node u, node v, edgeweight w) { viewEdges.emplace_back(u, v, w); });
    std::sort(viewEdges.begin(), viewEdges.end());
    EXPECT_EQ(expectedEdges(G, subset), viewEdges);
}

} // namespace

class InducedSubgraphViewGTest : public testing::Test {
protected:
    GraphW base = houseBase();
};

TEST_F(InducedSubgraphViewGTest, testConstructionFromSubset) {
    const std::set<node> subset{0, 1, 2, 4};
    InducedSubgraphView<GraphW> view(base, subset);

    EXPECT_EQ(4u, view.numberOfNodes());
    EXPECT_EQ(base.upperNodeIdBound(), view.upperNodeIdBound());
    EXPECT_TRUE(view.isWeighted());
    EXPECT_FALSE(view.isDirected());

    // house edges among {0,1,2,4}: 1-0, 0-2, 2-1, 1-4, 2-4 -- plus the self-loop
    EXPECT_EQ(6u, view.numberOfEdges());
    EXPECT_EQ(1u, view.numberOfSelfLoops());

    for (const node u : subset)
        ASSERT_TRUE(view.hasNode(u));
    EXPECT_FALSE(view.hasNode(3));
    EXPECT_FALSE(view.hasNode(base.upperNodeIdBound()));
}

TEST_F(InducedSubgraphViewGTest, testDegreesMatchFilteredNeighborhoods) {
    const std::set<node> subset{1, 2, 3};
    InducedSubgraphView<GraphW> view(base, subset);

    // induced house edges among {1,2,3}: 3-1, 2-1, 3-2 -- a triangle, no self-loops
    EXPECT_EQ(3u, view.numberOfEdges());
    EXPECT_EQ(0u, view.numberOfSelfLoops());
    EXPECT_EQ(2u, view.degree(1));
    EXPECT_EQ(2u, view.degree(2));
    EXPECT_EQ(2u, view.degree(3));

    count seen = 0;
    view.forNodes([&](node u) {
        count neighbors = 0;
        view.forNeighborsOf(u, [&](node) { ++neighbors; });
        ASSERT_EQ(view.degree(u), neighbors);
        ++seen;
    });
    EXPECT_EQ(3u, seen);
}

TEST_F(InducedSubgraphViewGTest, testNeighborhoodsCarryWeights) {
    const std::set<node> subset{0, 1, 2};
    InducedSubgraphView<GraphW> view(base, subset);

    std::vector<std::pair<node, edgeweight>> out0;
    for (const auto nb : view.outNeighbors<true>(0))
        out0.push_back(nb);
    std::sort(out0.begin(), out0.end());

    // base weights around 0: self-loop 1.0, 1-0 has 2.0, 0-2 has 3.0
    ASSERT_EQ(3u, out0.size());
    EXPECT_EQ(0u, out0[0].first);
    EXPECT_DOUBLE_EQ(1.0, out0[0].second);
    EXPECT_EQ(1u, out0[1].first);
    EXPECT_DOUBLE_EQ(2.0, out0[1].second);
    EXPECT_EQ(2u, out0[2].first);
    EXPECT_DOUBLE_EQ(3.0, out0[2].second);

    // unweighted payload yields bare targets: 0's in-subject neighbors are its self-loop, 1, 2
    std::vector<node> plain;
    for (const auto v : view.outNeighbors<false>(0))
        plain.push_back(v);
    std::sort(plain.begin(), plain.end());
    EXPECT_EQ(std::vector<node>({0, 1, 2}), plain);
}

TEST_F(InducedSubgraphViewGTest, testBatchAddAndRemoveKeepsCountersConsistent) {
    InducedSubgraphView<GraphW> view(base);
    EXPECT_EQ(0u, view.numberOfNodes());
    EXPECT_EQ(0u, view.numberOfEdges());

    // adding {0..4} one by one must reproduce the whole base graph
    for (node u = 0; u < base.upperNodeIdBound(); ++u)
        view.addNode(u);

    ASSERT_EQ(base.numberOfNodes(), view.numberOfNodes());
    ASSERT_EQ(base.numberOfEdges(), view.numberOfEdges());
    ASSERT_EQ(base.numberOfSelfLoops(), view.numberOfSelfLoops());
    expectSameEdges(base, {0, 1, 2, 3, 4}, view);

    // batch removal, including an id that is already gone
    view.removeNodes({0, 2, 4});
    view.removeNode(4);
    const std::set<node> subset{1, 3};
    ASSERT_EQ(subset.size(), view.numberOfNodes());
    ASSERT_EQ(1u, view.numberOfEdges()); // 3-1
    ASSERT_EQ(0u, view.numberOfSelfLoops());
    expectSameEdges(base, subset, view);

    // growing back in one batch, order shuffled, duplicate included
    view.addNodes({4, 0, 2, 4});
    const std::set<node> regrown{0, 1, 2, 3, 4};
    ASSERT_EQ(regrown.size(), view.numberOfNodes());
    ASSERT_EQ(base.numberOfEdges(), view.numberOfEdges());
    ASSERT_EQ(base.numberOfSelfLoops(), view.numberOfSelfLoops());
    expectSameEdges(base, regrown, view);
}

TEST_F(InducedSubgraphViewGTest, testUnknownNodeThrows) {
    InducedSubgraphView<GraphW> view(base);
    EXPECT_THROW(view.addNode(base.upperNodeIdBound()), std::runtime_error);
    EXPECT_THROW(view.addNodes({0, 99}), std::runtime_error);
    // removal of unknown ids is a no-op, not an error
    EXPECT_NO_THROW(view.removeNode(99));
}

TEST_F(InducedSubgraphViewGTest, testInNeighborsOnUndirectedBase) {
    const std::set<node> subset{1, 3, 4};
    InducedSubgraphView<GraphW> view(base, subset);

    std::vector<node> inOf1;
    for (const auto nb : view.inNeighbors<false>(1))
        inOf1.push_back(neighborTarget(nb)); // 3 and 4 stay; 0 and 2 are filtered out
    EXPECT_EQ(std::vector<node>({3, 4}), inOf1);
}

TEST(InducedSubgraphViewDirectedGTest, testDirectedCountersAndRanges) {
    GraphW D(4, true, true);
    D.addEdge(0, 1, 1.0);
    D.addEdge(1, 2, 2.0);
    D.addEdge(2, 1, 3.0);
    D.addEdge(3, 3, 4.0); // directed self-loop
    D.addEdge(3, 0, 5.0);

    using Entry = std::tuple<node, node, edgeweight>;
    std::vector<Entry> expected;
    D.forEdges([&](node u, node v, edgeweight w) { expected.emplace_back(u, v, w); });

    const std::set<node> subset{0, 1, 2, 3};
    InducedSubgraphView<GraphW> view(D, subset);

    std::vector<Entry> got;
    view.forEdges([&](node u, node v, edgeweight w) { got.emplace_back(u, v, w); });
    std::sort(got.begin(), got.end());
    EXPECT_EQ(expected.size(), got.size());

    // in-arcs of 1: 0->1 and 2->1; out-arcs: 1->2
    EXPECT_EQ(2u, view.degreeIn(1));
    EXPECT_EQ(1u, view.degreeOut(1));
    EXPECT_EQ(1u, view.degreeIn(3));
    EXPECT_EQ(2u, view.degreeOut(3));
    EXPECT_EQ(1u, view.numberOfSelfLoops());

    // shrink to {1, 2}: only the 1 <-> 2 pair survives, self-loop and both cross arcs vanish
    view.removeNodes({0, 3});
    EXPECT_EQ(2u, view.numberOfNodes());
    EXPECT_EQ(2u, view.numberOfEdges());
    EXPECT_EQ(0u, view.numberOfSelfLoops());
    EXPECT_EQ(1u, view.degreeIn(1));
    EXPECT_EQ(1u, view.degreeOut(1));
    EXPECT_EQ(1u, view.degreeIn(2));
    EXPECT_EQ(1u, view.degreeOut(2));
    EXPECT_DOUBLE_EQ(2.0, view.weight(1, 2));
    EXPECT_DOUBLE_EQ(3.0, view.weight(2, 1));
}

TEST_F(InducedSubgraphViewGTest, testFrontier) {
    const std::set<node> subset{0, 1};
    InducedSubgraphView<GraphW> view(base, subset);
    // out-neighbors outside {0,1}: 0 reaches 2; 1 reaches 2, 3 and 4
    const auto f = view.frontier();
    const std::set<node> expected(f.begin(), f.end());
    EXPECT_EQ(std::set<node>({2, 3, 4}), expected);
    EXPECT_TRUE(std::is_sorted(f.begin(), f.end()));

    view.addNodes({2, 3, 4});
    EXPECT_TRUE(view.frontier().empty());
}

TEST_F(InducedSubgraphViewGTest, testRealizeMatchesSubgraphFromNodes) {
    const std::set<node> subset{0, 1, 3, 4};

    InducedSubgraphView<GraphW> view(base, subset);
    GraphW realizedKeep = view.realize(false);
    GraphW referenceKeep = GraphTools::subgraphFromNodes(base, subset.begin(), subset.end(), false);
    EXPECT_EQ(referenceKeep.numberOfNodes(), realizedKeep.numberOfNodes());
    EXPECT_EQ(referenceKeep.numberOfEdges(), realizedKeep.numberOfEdges());
    EXPECT_EQ(referenceKeep.numberOfSelfLoops(), realizedKeep.numberOfSelfLoops());
    EXPECT_EQ(referenceKeep.totalEdgeWeight(), realizedKeep.totalEdgeWeight());

    GraphW realizedCompact = view.realize(true);
    GraphW referenceCompact =
        GraphTools::subgraphFromNodes(base, subset.begin(), subset.end(), true);
    EXPECT_EQ(referenceCompact.numberOfNodes(), realizedCompact.numberOfNodes());
    EXPECT_EQ(referenceCompact.numberOfEdges(), realizedCompact.numberOfEdges());
    EXPECT_EQ(4u, realizedCompact.upperNodeIdBound());
    EXPECT_EQ(referenceCompact.totalEdgeWeight(), realizedCompact.totalEdgeWeight());

    // compact ids are dense
    count visited = 0;
    realizedCompact.forNodes([&](node) { ++visited; });
    EXPECT_EQ(realizedCompact.numberOfNodes(), visited);
    EXPECT_EQ(realizedCompact.upperNodeIdBound(), visited);
}

TEST_F(InducedSubgraphViewGTest, testFreeOperationsOverView) {
    const std::set<node> subset{1, 2, 3, 4};
    InducedSubgraphView<GraphW> view(base, subset);

    namespace Ops = GraphIterationOps;

    count nodes = 0;
    Ops::forNodes(view, [&](node) { ++nodes; });
    EXPECT_EQ(4u, nodes);

    count edges = 0;
    Ops::forEdges(view, [&](node u, node v, edgeweight w) {
        EXPECT_TRUE(view.hasEdge(u, v));
        EXPECT_GT(w, 0.0);
        ++edges;
    });
    EXPECT_EQ(view.numberOfEdges(), edges);

    // weighted degree sums over the filtered neighborhood
    view.forNodes([&](node u) {
        edgeweight sum = 0.0;
        view.forEdgesOf(u, [&](node, edgeweight w) { sum += w; });
        EXPECT_DOUBLE_EQ(sum, view.weightedDegree(u));
    });

    edgeweight total = 0.0;
    view.forEdges([&](node, node, edgeweight w) { total += w; });
    EXPECT_DOUBLE_EQ(total, view.totalEdgeWeight());

    EXPECT_FALSE(Ops::hasEdge(view, 1, 0)) << "0 left the subset";
    EXPECT_TRUE(Ops::hasEdge(view, 3, 2));
    EXPECT_GT(Ops::weight(view, 3, 2), 0.0);
    EXPECT_EQ(nullWeight, Ops::weight(view, 1, 0));

    EXPECT_NO_THROW(Ops::parallelForEdges(view, [](node, node) {}));
    EXPECT_NO_THROW(Ops::parallelSumForEdges(view, [](node, node, edgeweight) { return 1.0; }));
}

TEST_F(InducedSubgraphViewGTest, testViewOverViewComposes) {
    InducedSubgraphView<GraphW> inner(base, {0, 1, 2, 3, 4});
    inner.removeNodes({0});

    static_assert(
        GraphLike<InducedSubgraphView<InducedSubgraphView<GraphW>>>,
        "an induced view over an induced view must itself be usable wherever a graph type is");

    InducedSubgraphView<InducedSubgraphView<GraphW>> outer(inner, {1, 2, 4});

    EXPECT_EQ(3u, outer.numberOfNodes());
    expectSameEdges(base, {1, 2, 4}, outer);
}

TEST(InducedSubgraphViewOverGraphRGTest, testReadOnlyBase) {
    GraphR R = houseCSR();

    const std::set<node> subset{1, 2, 3};
    InducedSubgraphView<GraphR> view(R, subset);

    EXPECT_FALSE(view.isWeighted());
    EXPECT_EQ(3u, view.numberOfEdges()); // the induced triangle
    EXPECT_EQ(0u, view.numberOfSelfLoops());
    EXPECT_EQ(defaultEdgeWeight, view.weight(1, 2));
    EXPECT_EQ(nullWeight, view.weight(1, 4));

    count edges = 0;
    view.forEdges([&](node u, node v, edgeweight w) {
        EXPECT_DOUBLE_EQ(defaultEdgeWeight, w);
        ++edges;
    });
    EXPECT_EQ(3u, edges);

    // free ops see an unweighted graph and fall back to degree-based aggregates
    EXPECT_DOUBLE_EQ(2.0, view.weightedDegree(1));

    // the base's sorted neighborhoods survive filtering, so lookups may bisect
    EXPECT_TRUE(view.hasSortedNeighborhoods());
}

TEST_F(InducedSubgraphViewGTest, testRandomizedAgainstGroundTruth) {
    for (int trial = 0; trial < 20; ++trial) {
        ErdosRenyiGenerator gen(12, 0.35, trial % 2 == 1);
        GraphW G = gen.generate();

        InducedSubgraphView<GraphW> view(G);
        std::set<node> subset;
        for (node u = 0; u < G.upperNodeIdBound(); ++u)
            if (Aux::Random::probability() < 0.6) {
                view.addNode(u);
                subset.insert(u);
            }

        ASSERT_EQ(subset.size(), view.numberOfNodes());

        count mExpected = 0;
        const bool directed = G.isDirected();
        for (const node u : subset)
            G.forNeighborsOf(u, [&](node v) {
                if (!subset.count(v))
                    return;
                if (!directed && u > v)
                    return;
                ++mExpected;
            });
        ASSERT_EQ(mExpected, view.numberOfEdges())
            << "trial " << trial << ": edge bookkeeping diverged";

        view.forNodes([&](node u) {
            count d = 0;
            G.forNeighborsOf(u, [&](node v) { d += subset.count(v) ? 1 : 0; });
            ASSERT_EQ(d, view.degree(u)) << "trial " << trial << ", node " << u;
        });

        count loops = 0;
        for (const node u : subset)
            if (G.hasEdge(u, u))
                ++loops;
        ASSERT_EQ(loops, view.numberOfSelfLoops()) << "trial " << trial;

        expectSameEdges(G, subset, view);
    }
}

} // namespace NetworKit
