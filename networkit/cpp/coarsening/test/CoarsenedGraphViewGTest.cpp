/*
 * CoarsenedGraphViewGTest.cpp
 *
 *  Tests for the zero-copy coarsened graph view as a GraphLike type.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <tuple>
#include <vector>

#include <networkit/coarsening/CoarsenedGraphView.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/graph/GraphIterationOps.hpp>
#include <networkit/graph/GraphW.hpp>
#include <networkit/graph/InducedSubgraphView.hpp>
#include <networkit/structures/Partition.hpp>

namespace NetworKit {

namespace {

/// The weighted house graph with a self-loop on node 0; edge i carries weight i + 1 in the
/// order 3-1, 1-0, 0-2, 2-1, 1-4, 4-3, 3-2, 2-4, and the self-loop carries 1.0.
GraphW houseGraph() {
    GraphW G(5, true, false);
    const std::vector<std::pair<node, node>> edges{{3, 1}, {1, 0}, {0, 2}, {2, 1},
                                                   {1, 4}, {4, 3}, {3, 2}, {2, 4}};
    edgeweight w = 1.0;
    for (const auto &[u, v] : edges) {
        G.addEdge(u, v, w);
        w += 1.0;
    }
    G.addEdge(0, 0);
    return G;
}

/// Assigns @a groups[s] to supernode s.
Partition partitionOf(count z, const std::vector<std::vector<node>> &groups) {
    Partition p(z);
    for (index s = 0; s < groups.size(); ++s)
        for (const node u : groups[s])
            p[u] = s;
    return p;
}

std::vector<std::tuple<node, node, edgeweight>> sortedEdges(const CoarsenedGraphView &view) {
    std::vector<std::tuple<node, node, edgeweight>> edges;
    view.forEdges([&](node u, node v, edgeweight w) { edges.emplace_back(u, v, w); });
    std::sort(edges.begin(), edges.end());
    return edges;
}

} // namespace

class CoarsenedGraphViewGTest : public testing::Test {
protected:
    GraphW base = houseGraph();
};

TEST_F(CoarsenedGraphViewGTest, testSingletonPartitionMirrorsOriginalGraph) {
    Partition singleton(base.upperNodeIdBound());
    singleton.allToSingletons();

    CoarsenedGraphView view(base, singleton);

    EXPECT_EQ(base.numberOfNodes(), view.numberOfNodes());
    EXPECT_EQ(base.numberOfEdges(), view.numberOfEdges());
    EXPECT_EQ(base.numberOfSelfLoops(), view.numberOfSelfLoops());
    EXPECT_TRUE(view.isWeighted());
    EXPECT_FALSE(view.isDirected());

    base.forNodes([&](node u) {
        ASSERT_EQ(base.degree(u), view.degree(u));
        ASSERT_EQ(base.weightedDegree(u), view.weightedDegree(u));
        std::vector<std::pair<node, edgeweight>> expected;
        base.forEdgesOf(u, [&](node v, edgeweight w) { expected.push_back({v, w}); });
        std::sort(expected.begin(), expected.end());
        std::vector<std::pair<node, edgeweight>> got(view.outNeighbors<true>(u));
        std::sort(got.begin(), got.end());
        ASSERT_EQ(expected, got) << "supernode " << u;
    });
}

TEST_F(CoarsenedGraphViewGTest, testAggregatedWeightsDegreesAndLoops) {
    // {0,1} -> A, {2,3} -> B, {4} -> C
    CoarsenedGraphView view(base, partitionOf(5, {{0, 1}, {2, 3}, {4}}));

    ASSERT_EQ(3u, view.numberOfNodes());
    ASSERT_EQ(3u, view.upperNodeIdBound());

    // A={0,1}, B={2,3}, C={4}: A-B collects (3,1)+(0,2)+(2,1) = 1+3+4 = 8; the internal blocks
    // A-A (edge 1-0 plus the self-loop) and B-B (edge 3-2) are each counted once.
    EXPECT_EQ(5u, view.numberOfEdges()); // A-B, A-C, B-C, and the two internal blocks
    EXPECT_EQ(2u, view.numberOfSelfLoops());

    EXPECT_TRUE(view.hasEdge(0, 1)); // A-B
    EXPECT_TRUE(view.hasEdge(0, 2)); // A-C
    EXPECT_TRUE(view.hasEdge(1, 2)); // B-C
    EXPECT_TRUE(view.hasEdge(0, 0)); // A-A
    EXPECT_TRUE(view.hasEdge(1, 1)); // B-B

    EXPECT_DOUBLE_EQ(8.0, view.weight(0, 1));
    EXPECT_DOUBLE_EQ(5.0, view.weight(0, 2));  // (1,4)
    EXPECT_DOUBLE_EQ(14.0, view.weight(1, 2)); // (4,3) + (2,4)
    EXPECT_DOUBLE_EQ(3.0, view.weight(0, 0));  // edge 1-0 (2.0) + self-loop (1.0)
    EXPECT_DOUBLE_EQ(7.0, view.weight(1, 1));  // edge 3-2

    EXPECT_EQ(3u, view.degree(0));
    EXPECT_EQ(3u, view.degree(1));
    EXPECT_EQ(2u, view.degree(2));

    EXPECT_DOUBLE_EQ(16.0, view.weightedDegree(0));
    EXPECT_DOUBLE_EQ(19.0, view.weightedDegree(0, true)); // the 3.0 loop block counted twice
    EXPECT_DOUBLE_EQ(37.0, view.totalEdgeWeight());

    // every reported edge exists in both directions on the undirected view
    view.forEdges([&](node u, node v, edgeweight w) {
        ASSERT_TRUE(view.hasEdge(u, v));
        ASSERT_DOUBLE_EQ(w, view.weight(u, v));
        ASSERT_DOUBLE_EQ(w, view.weight(v, u));
    });
}

TEST_F(CoarsenedGraphViewGTest, testLayeredCoarseningEqualsComposedPartition) {
    const auto first = partitionOf(5, {{0, 1}, {2, 3}, {4}});
    CoarsenedGraphView layer1(base, first);

    // {A,B} -> X, {C} -> Y
    const auto second = partitionOf(layer1.numberOfNodes(), {{0, 1}, {2}});
    CoarsenedGraphView layer2(layer1, second);

    // coarsening the original graph directly with the composed partition {0,1,2,3}, {4}
    CoarsenedGraphView direct(base, partitionOf(5, {{0, 1, 2, 3}, {4}}));

    EXPECT_EQ(direct.numberOfNodes(), layer2.numberOfNodes());
    EXPECT_EQ(direct.numberOfEdges(), layer2.numberOfEdges());
    EXPECT_EQ(direct.numberOfSelfLoops(), layer2.numberOfSelfLoops());
    EXPECT_EQ(sortedEdges(direct), sortedEdges(layer2));

    // X-Y collects A-C (5.0) and B-C (14.0); X-X collects A-B (8.0) and both internal blocks
    EXPECT_DOUBLE_EQ(19.0, layer2.weight(0, 1));
    EXPECT_DOUBLE_EQ(18.0, layer2.weight(0, 0));
    EXPECT_EQ(1u, layer2.numberOfSelfLoops());
}

TEST_F(CoarsenedGraphViewGTest, testFreeOperationsOverView) {
    CoarsenedGraphView view(base, partitionOf(5, {{0, 1}, {2, 3}, {4}}));

    namespace Ops = GraphIterationOps;

    static_assert(GraphLike<CoarsenedGraphView>);

    count nodes = 0;
    Ops::forNodes(view, [&](node) { ++nodes; });
    EXPECT_EQ(3u, nodes);

    count edges = 0;
    Ops::forEdges(view, [&](node u, node v, edgeweight w) {
        EXPECT_TRUE(view.hasEdge(u, v));
        EXPECT_GT(w, 0.0);
        ++edges;
    });
    EXPECT_EQ(5u, edges);

    edgeweight sum = 0.0;
    Ops::forEdges(view, [&](node, node, edgeweight w) { sum += w; });
    EXPECT_DOUBLE_EQ(37.0, sum);
    EXPECT_DOUBLE_EQ(37.0, Ops::totalEdgeWeight(view));

    // node ids are dense, so the free layer skips hasNode checks
    EXPECT_TRUE(hasContiguousNodeIds(view));

    EXPECT_NO_THROW(Ops::parallelForNodes(view, [](node) {}));
    EXPECT_NO_THROW(Ops::parallelForEdges(view, [](node, node) {}));
    EXPECT_NO_THROW(Ops::forNodesInRandomOrder(view, [](node) {}));
    EXPECT_DOUBLE_EQ(3.0, Ops::parallelSumForNodes(view, [](node) { return 1.0; }));
    EXPECT_DOUBLE_EQ(5.0,
                     Ops::parallelSumForEdges(view, [](node, node, edgeweight) { return 1.0; }));

    // in-neighbors are the out-neighbors on the undirected view
    std::vector<node> inOf0;
    Ops::forInNeighborsOf(view, 0, [&](node v) { inOf0.push_back(v); });
    EXPECT_EQ(3u, inOf0.size());
}

TEST_F(CoarsenedGraphViewGTest, testInducedViewOverCoarsenedView) {
    CoarsenedGraphView coarsened(base, partitionOf(5, {{0, 1}, {2, 3}, {4}}));

    // keep the supernodes A (0) and C (2): the A-A loop block and A-C survive
    InducedSubgraphView<CoarsenedGraphView> induced(coarsened, {0, 2});

    static_assert(GraphLike<InducedSubgraphView<CoarsenedGraphView>>);

    EXPECT_EQ(2u, induced.numberOfNodes());
    EXPECT_EQ(2u, induced.numberOfEdges()); // A-A and A-C
    EXPECT_EQ(1u, induced.numberOfSelfLoops());
    EXPECT_DOUBLE_EQ(3.0, induced.weight(0, 0));
    EXPECT_DOUBLE_EQ(5.0, induced.weight(0, 2));
    EXPECT_EQ(2u, induced.degree(0));

    count edges = 0;
    induced.forEdges([&](node, node, edgeweight) { ++edges; });
    EXPECT_EQ(2u, edges);
}

TEST_F(CoarsenedGraphViewGTest, testPartitionWithGapsIsCompacted) {
    Partition p(5);
    p[0] = 7;
    p[1] = 7;
    p[2] = 3;
    p[3] = 3;
    p[4] = 3; // ids 3 and 7 with a gap; supernode 2 unused

    // compact() renumbers by sorted id order: {2,3,4} -> 0, {0,1} -> 1
    CoarsenedGraphView view(base, p);
    EXPECT_EQ(2u, view.numberOfNodes());
    EXPECT_TRUE(view.hasNode(0));
    EXPECT_TRUE(view.hasNode(1));
    EXPECT_FALSE(view.hasNode(2));

    EXPECT_EQ(3u, view.numberOfEdges()); // 0-1 crossing, plus the loop blocks on both sides
    EXPECT_EQ(2u, view.numberOfSelfLoops());

    EXPECT_DOUBLE_EQ(13.0, view.weight(0, 1)); // edges 3-1 + 0-2 + 2-1 + 1-4
    EXPECT_DOUBLE_EQ(21.0, view.weight(0, 0)); // edges inside {2,3,4}
    EXPECT_DOUBLE_EQ(3.0, view.weight(1, 1));  // edge 1-0 plus the self-loop inside {0,1}
    EXPECT_EQ(2u, view.degree(0));
    EXPECT_EQ(2u, view.degree(1));
}

TEST_F(CoarsenedGraphViewGTest, testNodeMappingAccessors) {
    const auto p = partitionOf(5, {{0, 1}, {2, 3}, {4}});
    CoarsenedGraphView view(base, p);

    const auto &mapping = view.getNodeMapping();
    EXPECT_EQ(0u, mapping[0]);
    EXPECT_EQ(0u, mapping[1]);
    EXPECT_EQ(1u, mapping[2]);
    EXPECT_EQ(1u, mapping[3]);
    EXPECT_EQ(2u, mapping[4]);

    const auto &members = view.getOriginalNodes(0);
    EXPECT_EQ(std::vector<node>({0, 1}), members);

    EXPECT_EQ(std::vector<node>(), view.getOriginalNodes(42));
}

} // namespace NetworKit
