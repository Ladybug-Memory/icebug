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
#include <networkit/centrality/CoreDecomposition.hpp>
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
    for (const auto &nb : view.outNeighbors<true>(0))
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
    // a failed batch is atomic: nothing was inserted
    EXPECT_EQ(0u, view.numberOfNodes());
    EXPECT_FALSE(view.hasNode(0));
    // removal of unknown ids is a no-op, not an error -- including ids beyond the base's id
    // space, which must not touch the presence/degree storage (regression: OOB write)
    EXPECT_NO_THROW(view.removeNode(99));
    EXPECT_NO_THROW(view.removeNodes({99, base.upperNodeIdBound() + 100}));
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

        InducedSubgraphView<GraphW> view(G, /*compact=*/trial % 3 == 2);
        std::set<node> subset;
        for (node u = 0; u < G.upperNodeIdBound(); ++u)
            if (Aux::Random::probability() < 0.6) {
                view.addNode(u);
                subset.insert(u);
            }

        ASSERT_EQ(subset.size(), view.numberOfNodes());
        if (view.isCompact())
            ASSERT_TRUE(NetworKit::hasContiguousNodeIds(view));

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
            const node b = view.toBaseId(u);
            count d = 0;
            G.forNeighborsOf(b, [&](node v) { d += subset.count(v) ? 1 : 0; });
            ASSERT_EQ(d, view.degree(u)) << "trial " << trial << ", node " << u;
            ASSERT_TRUE(subset.count(view.toBaseId(u))) << "compact id out of the subset";
        });

        count loops = 0;
        for (const node u : subset)
            if (G.hasEdge(u, u))
                ++loops;
        ASSERT_EQ(loops, view.numberOfSelfLoops()) << "trial " << trial;

        // every view edge is a base edge with both endpoints in the subset
        view.forEdges([&](node u, node v, edgeweight w) {
            const node bu = view.toBaseId(u), bv = view.toBaseId(v);
            ASSERT_TRUE(subset.count(bu) && subset.count(bv));
            EXPECT_EQ(G.weight(bu, bv), w);
        });
    }
}

TEST_F(InducedSubgraphViewGTest, testAdaptiveHubNeighborhoods) {
    // A star: node 0 with 200 leaves. The view keeps the hub and three of its leaves, so
    // deg_base(0) = 200 overshoots n_view * bit_width(200) * 4 = 4 * 8 * 4 = 128: both the
    // out- and the in-neighborhood ranges must take the subset-scanning strategy, which walks
    // members ascending rather than scanning all 200 base neighbors.
    GraphW star(201, true, false);
    for (node leaf = 1; leaf <= 200; ++leaf)
        star.addEdge(0, leaf, static_cast<edgeweight>(leaf));

    const std::set<node> subset{0, 5, 100, 200};
    InducedSubgraphView<GraphW> view(star, subset);
    ASSERT_EQ(3u, view.numberOfEdges());

    std::vector<node> plain;
    for (const auto v : view.outNeighbors<false>(0))
        plain.push_back(v);
    EXPECT_EQ((std::vector<node>{5, 100, 200}), plain); // ascending: subset-scan order

    std::vector<std::pair<node, edgeweight>> weighted;
    for (const auto &nb : view.outNeighbors<true>(0))
        weighted.push_back(nb);
    EXPECT_EQ(3u, weighted.size());
    for (const auto &[v, w] : weighted) {
        EXPECT_EQ(w, static_cast<edgeweight>(v)); // weights carried through either strategy
    }

    // undirected base: in-neighbors agree with out-neighbors
    std::vector<node> incoming;
    view.forInNeighborsOf(0, [&](node v) { incoming.push_back(v); });
    EXPECT_EQ((std::vector<node>{5, 100, 200}), incoming);

    // a big view over the same hub stays on the base scan and must produce identical members
    std::set<node> wide{0};
    for (node leaf = 1; leaf <= 150; ++leaf)
        wide.insert(leaf);
    InducedSubgraphView<GraphW> wideView(star, wide);
    std::vector<node> fromWide;
    wideView.forNeighborsOf(0, [&](node v) { fromWide.push_back(v); });
    ASSERT_EQ(150u, fromWide.size()); // every kept leaf
    EXPECT_TRUE(std::is_sorted(fromWide.begin(), fromWide.end()));
}

/*
 * The type-erased handle accepts the two view instantiations as arms, so an unmodified algorithm
 * -- here CoreDecomposition, the case Ian asked to compile unchanged -- consumes a view directly.
 */
TEST_F(InducedSubgraphViewGTest, testHandleRunsExistingAlgorithms) {
    const std::set<node> subset{1, 2, 3, 4};
    InducedSubgraphView<GraphW> view(base, subset);
    GraphW reference = GraphTools::subgraphFromNodes(base, subset.begin(), subset.end(), false);

    // Implicit handle conversion at the call site; no copy of the view, no wrapper type.
    CoreDecomposition kcore(view);
    kcore.run();
    CoreDecomposition referenceKcore(reference);
    referenceKcore.run();

    for (const node u : subset)
        EXPECT_EQ(referenceKcore.score(u), kcore.score(u)) << "node " << u;
}

/*
 * The conversion hands out the handle embedded inside the view: one address per view, stable
 * across statements -- which is what makes "construct an algorithm, then run it" sound rather
 * than merely lucky -- and re-seated when the view is copied, so a copy never aliases the
 * original's handle.
 */
TEST_F(InducedSubgraphViewGTest, testHandleIsEmbeddedAndIdentityBound) {
    const std::set<node> subset{1, 2, 3, 4};
    InducedSubgraphView<GraphW> view(base, subset);

    const ReferenceGraph &borrowed = view;
    EXPECT_EQ(&borrowed, &view.asGraph());

    GraphW reference = GraphTools::subgraphFromNodes(base, subset.begin(), subset.end(), false);
    CoreDecomposition expected(reference);
    expected.run();
    CoreDecomposition kcore(borrowed); // stored across statements by the algorithm class
    kcore.run();
    for (const node u : subset)
        EXPECT_EQ(expected.score(u), kcore.score(u)) << "node " << u;

    InducedSubgraphView<GraphW> copy = view;
    const ReferenceGraph &copiedHandle = copy;
    EXPECT_NE(&copiedHandle, &borrowed); // identity-bound: the copy carries its own handle
    EXPECT_EQ(borrowed.numberOfEdges(), copiedHandle.numberOfEdges());
}

TEST_F(InducedSubgraphViewGTest, testCompactViewPresentsDenseIds) {
    const std::set<node> subset{1, 2, 3, 4};
    InducedSubgraphView<GraphW> view(base, subset, true);

    EXPECT_TRUE(view.isCompact());
    EXPECT_EQ(4u, view.numberOfNodes());
    EXPECT_EQ(4u, view.upperNodeIdBound());
    EXPECT_EQ(6u, view.numberOfEdges());
    EXPECT_EQ(0u, view.numberOfSelfLoops());
    EXPECT_TRUE(NetworKit::hasContiguousNodeIds(view));

    // view ids are dense; base ids above n_ - 1 are not members
    for (node c = 0; c < 4; ++c)
        EXPECT_TRUE(view.hasNode(c));
    EXPECT_FALSE(view.hasNode(4));

    // compact id is the rank in ascending base-id order
    EXPECT_EQ(std::vector<node>({1, 2, 3, 4}), view.getNodeSubset());
    for (node c = 0; c < 4; ++c)
        EXPECT_EQ(c + 1, view.toBaseId(c));
    EXPECT_EQ(0u, view.toCompactId(1));
    EXPECT_EQ(3u, view.toCompactId(4));
    EXPECT_EQ(none, view.toCompactId(0));
    EXPECT_EQ(none, view.toCompactId(base.upperNodeIdBound() + 10));

    // degrees in compact space: every member keeps 3 induced neighbors
    for (node c = 0; c < 4; ++c)
        EXPECT_EQ(3u, view.degree(c));

    // neighborhoods yield compact ids covering the same base edges
    std::vector<node> nbhd;
    for (const auto v : view.outNeighbors<false>(0))
        nbhd.push_back(view.toBaseId(v));
    std::sort(nbhd.begin(), nbhd.end());
    EXPECT_EQ(std::vector<node>({2, 3, 4}), nbhd);

    // weights ride along with the translated targets
    std::vector<std::pair<node, edgeweight>> weighted;
    for (const auto &nb : view.outNeighbors<true>(0))
        weighted.emplace_back(view.toBaseId(nb.first), nb.second);
    std::sort(weighted.begin(), weighted.end());
    ASSERT_EQ(3u, weighted.size());
    EXPECT_DOUBLE_EQ(4.0, weighted[0].second); // base edge 1-2
    EXPECT_DOUBLE_EQ(1.0, weighted[1].second); // base edge 3-1
    EXPECT_DOUBLE_EQ(5.0, weighted[2].second); // base edge 1-4

    // lookups and aggregates speak compact ids
    EXPECT_TRUE(view.hasEdge(0, 1));
    EXPECT_FALSE(view.hasEdge(0, 0));
    EXPECT_DOUBLE_EQ(4.0, view.weight(0, 1));
    EXPECT_EQ(nullWeight, view.weight(0, 0));
    // base edges incident to 1 carry weights 1.0 (3-1), 4.0 (2-1) and 5.0 (1-4)
    EXPECT_DOUBLE_EQ(10.0, view.weightedDegree(0));
    const index pos = view.indexOfNeighbor(0, 1);
    EXPECT_NE(none, pos);
    EXPECT_EQ(1u, view.getIthNeighbor(0, pos));
    EXPECT_EQ(none, view.indexOfNeighbor(0, 0));
}

TEST_F(InducedSubgraphViewGTest, testCompactViewMatchesCompactSubgraph) {
    const std::vector<node> subset{1, 2, 3, 4}; // ascending, so both mappings agree
    InducedSubgraphView<GraphW> view(base, subset, true);
    GraphW reference = GraphTools::subgraphFromNodes(base, subset.begin(), subset.end(), true);

    EXPECT_EQ(reference.numberOfNodes(), view.numberOfNodes());
    EXPECT_EQ(reference.numberOfEdges(), view.numberOfEdges());
    EXPECT_EQ(reference.upperNodeIdBound(), view.upperNodeIdBound());
    EXPECT_DOUBLE_EQ(reference.totalEdgeWeight(), view.totalEdgeWeight());

    for (node c = 0; c < view.upperNodeIdBound(); ++c) {
        EXPECT_EQ(reference.degree(c), view.degree(c));
        std::vector<node> viewNbhd;
        for (const auto v : view.outNeighbors<false>(c))
            viewNbhd.push_back(v);
        std::vector<node> refNbhd;
        reference.forNeighborsOf(c, [&](node v) { refNbhd.push_back(v); });
        std::sort(viewNbhd.begin(), viewNbhd.end());
        std::sort(refNbhd.begin(), refNbhd.end());
        EXPECT_EQ(refNbhd, viewNbhd);
    }
}

TEST_F(InducedSubgraphViewGTest, testCompactViewEditsRenumber) {
    InducedSubgraphView<GraphW> view(base, {1, 2, 3, 4}, true);
    ASSERT_EQ(4u, view.upperNodeIdBound());

    // removing base node 1 leaves the 2-3-4 triangle, renumbered to 0..2
    view.removeNode(1);
    EXPECT_EQ(3u, view.numberOfNodes());
    EXPECT_EQ(3u, view.upperNodeIdBound());
    EXPECT_EQ(std::vector<node>({2, 3, 4}), view.getNodeSubset());
    for (node c = 0; c < 3; ++c) {
        EXPECT_TRUE(view.hasNode(c));
        EXPECT_EQ(2u, view.degree(c));
        EXPECT_EQ(c + 2, view.toBaseId(c));
    }
    EXPECT_EQ(3u, view.numberOfEdges());

    // growing back prepends base node 0 as compact id 0 and shifts the rest
    view.addNode(0);
    EXPECT_EQ(4u, view.numberOfNodes());
    EXPECT_EQ(std::vector<node>({0, 2, 3, 4}), view.getNodeSubset());
    EXPECT_EQ(0u, view.toCompactId(0));
    EXPECT_EQ(1u, view.toCompactId(2));
    // base node 0 brings its self-loop plus the 0-2 edge into the view
    EXPECT_EQ(1u, view.numberOfSelfLoops());
    EXPECT_EQ(5u, view.numberOfEdges());

    // an empty compact view stays dense
    view.removeNodes({0, 2, 3, 4});
    EXPECT_EQ(0u, view.numberOfNodes());
    EXPECT_EQ(0u, view.upperNodeIdBound());
    EXPECT_TRUE(view.isEmpty());
}

TEST_F(InducedSubgraphViewGTest, testCompactRealizeFromCompactView) {
    const std::vector<node> subset{1, 2, 3, 4};
    InducedSubgraphView<GraphW> view(base, subset, true);

    // realize(false) restores the base ids however the view itself is laid out
    GraphW keep = view.realize(false);
    GraphW referenceKeep = GraphTools::subgraphFromNodes(base, subset.begin(), subset.end(), false);
    EXPECT_EQ(5u, keep.upperNodeIdBound());
    EXPECT_EQ(referenceKeep.numberOfEdges(), keep.numberOfEdges());
    EXPECT_DOUBLE_EQ(referenceKeep.totalEdgeWeight(), keep.totalEdgeWeight());
    EXPECT_TRUE(keep.hasEdge(1, 2));
    EXPECT_FALSE(keep.hasNode(0));

    // realize(true) on a compact view agrees with the compact materialization
    GraphW dense = view.realize(true);
    EXPECT_EQ(4u, dense.upperNodeIdBound());
    EXPECT_EQ(6u, dense.numberOfEdges());
    EXPECT_DOUBLE_EQ(31.0, dense.totalEdgeWeight());
}

TEST_F(InducedSubgraphViewGTest, testNonCompactIdHelpersAreIdentity) {
    InducedSubgraphView<GraphW> view(base, {1, 2});
    EXPECT_FALSE(view.isCompact());
    EXPECT_EQ(1u, view.toBaseId(1));
    EXPECT_EQ(2u, view.toCompactId(2));
    EXPECT_EQ(none, view.toCompactId(0));
    EXPECT_FALSE(view.hasNode(0));
}

TEST(InducedSubgraphViewCompactDirectedGTest, testDirectedCompactIds) {
    GraphW D(5, true, true);
    D.addEdge(0, 4, 1.0);
    D.addEdge(4, 1, 2.0);
    D.addEdge(1, 4, 3.0);
    D.addEdge(4, 4, 4.0); // directed self-loop

    // sorted subset [1, 4]: compact 0 is base 1, compact 1 is base 4
    InducedSubgraphView<GraphW> view(D, {1, 4}, true);
    EXPECT_EQ(2u, view.upperNodeIdBound());
    EXPECT_EQ(0u, view.toCompactId(1));
    EXPECT_EQ(1u, view.toCompactId(4));
    EXPECT_EQ(4u, view.toBaseId(1));
    EXPECT_EQ(3u, view.numberOfEdges());
    EXPECT_EQ(1u, view.numberOfSelfLoops());
    EXPECT_EQ(1u, view.degreeOut(0)); // base 1 -> base 4
    EXPECT_EQ(1u, view.degreeIn(0));  // base 4 -> base 1
    EXPECT_EQ(2u, view.degreeOut(1)); // base 4 -> base 1 and the loop
    EXPECT_EQ(2u, view.degreeIn(1));  // base 1 -> base 4 and the loop
    EXPECT_DOUBLE_EQ(2.0, view.weight(1, 0));
    EXPECT_DOUBLE_EQ(3.0, view.weight(0, 1));

    std::vector<node> inOf0;
    for (const auto v : view.inNeighbors<false>(0))
        inOf0.push_back(v);
    std::sort(inOf0.begin(), inOf0.end());
    EXPECT_EQ(std::vector<node>({1}), inOf0);
}

TEST(InducedSubgraphViewCompactOverGraphRGTest, testReadOnlyCompactBase) {
    GraphR R = houseCSR();
    InducedSubgraphView<GraphR> view(R, {1, 2, 3}, true);

    EXPECT_TRUE(view.isCompact());
    EXPECT_EQ(3u, view.upperNodeIdBound());
    EXPECT_EQ(3u, view.numberOfEdges());
    for (node c = 0; c < 3; ++c)
        EXPECT_EQ(2u, view.degree(c));
    EXPECT_EQ(2u, view.toBaseId(1));
    EXPECT_EQ(none, view.toCompactId(0));

    // the frontier still speaks base ids
    InducedSubgraphView<GraphR> grown(R, {1}, true);
    EXPECT_EQ(std::vector<node>({0, 2, 3, 4}), grown.frontier());
}

TEST_F(InducedSubgraphViewGTest, testCompactViewRunsExistingAlgorithms) {
    const std::set<node> subset{1, 2, 3, 4};
    InducedSubgraphView<GraphW> view(base, subset, true);
    GraphW reference = GraphTools::subgraphFromNodes(base, subset.begin(), subset.end(), false);

    CoreDecomposition kcore(view);
    kcore.run();
    CoreDecomposition referenceKcore(reference);
    referenceKcore.run();

    for (const node b : subset)
        EXPECT_EQ(referenceKcore.score(b), kcore.score(view.toCompactId(b))) << "node " << b;
}

TEST_F(InducedSubgraphViewGTest, testHandleSurfaceOverViewArms) {
    const std::set<node> subset{1, 2, 3, 4};
    InducedSubgraphView<GraphW> wView(base, subset);
    const ReferenceGraph &h = wView; // implicit, same conversion the algorithms take

    EXPECT_EQ(4u, h.numberOfNodes());
    EXPECT_EQ(6u, h.numberOfEdges()); // every base edge with both endpoints kept
    EXPECT_EQ(0u, h.numberOfSelfLoops());
    EXPECT_EQ(5u, h.upperNodeIdBound());
    EXPECT_FALSE(h.hasNode(0));
    EXPECT_TRUE(h.hasEdge(1, 2));
    EXPECT_FALSE(h.hasEdge(0, 1));
    EXPECT_EQ(3u, h.degree(1));
    EXPECT_TRUE(h.isWeighted());
    EXPECT_FALSE(h.hasEdgeIds());
    EXPECT_TRUE(h.checkConsistency());
    EXPECT_DOUBLE_EQ(1.0 + 4.0 + 5.0 + 6.0 + 7.0 + 8.0, h.totalEdgeWeight());

    // indexed access follows the base adjacency order of each row, filtered to the members
    EXPECT_EQ(3u, h.getIthNeighbor(1, 0));
    EXPECT_EQ(none, h.getIthNeighbor(1, 3));
    EXPECT_DOUBLE_EQ(4.0, h.getIthNeighborWeight(1, 1));
    EXPECT_EQ(0u, h.indexOfNeighbor(2, 1));

    // erased neighbor ranges agree with the view's own
    count seen = 0;
    h.forNeighborsOf(4, [&](node v) {
        EXPECT_TRUE(wView.hasNode(v));
        ++seen;
    });
    EXPECT_EQ(3u, seen);

    count edgeCount = 0;
    edgeweight weightSum = 0.0;
    h.forEdges([&](node, node, edgeweight w) {
        ++edgeCount;
        weightSum += w;
    });
    EXPECT_EQ(6u, edgeCount);
    EXPECT_DOUBLE_EQ(h.totalEdgeWeight(), weightSum);

    // capabilities the views deliberately do not carry
    EXPECT_THROW(h.edgeId(1, 2), std::runtime_error);
    EXPECT_THROW((void)h.nodeAttributes(), std::runtime_error);

    // the same surface over a CSR base
    const GraphR R = houseCSR();
    InducedSubgraphView<GraphR> rView(R, {2});
    const ReferenceGraph &rh = rView;
    EXPECT_EQ(1u, rh.numberOfNodes());
    EXPECT_EQ(0u, rh.numberOfEdges());
    EXPECT_EQ(0u, rh.degree(2));
    EXPECT_TRUE(rh.checkConsistency());
    EXPECT_THROW(rh.edgeById(0), std::runtime_error);
}

} // namespace NetworKit
