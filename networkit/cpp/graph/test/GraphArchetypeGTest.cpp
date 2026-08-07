/*
 * GraphArchetypeGTest.cpp
 *
 *  A graph type that implements the GraphLike primitives and nothing else.
 */

#include <ranges>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <networkit/Globals.hpp>
#include <networkit/graph/GraphConcepts.hpp>
#include <networkit/graph/GraphIterationMixin.hpp>

/*
 * Deliberately not namespace NetworKit. A mixin member that reaches a helper only through
 * argument-dependent lookup compiles for GraphW and GraphR -- which are NetworKit types, so ADL
 * finds NetworKit -- and fails here. That asymmetry is the point of the foreign namespace.
 */
namespace ArchetypeTest {

using NetworKit::count;
using NetworKit::edgeweight;
using NetworKit::index;
using NetworKit::node;

/**
 * The four neighbor ranges and the storage behind them, shared by the archetype and by the
 * negative controls so that they differ only in the property under test.
 */
class Neighborhoods {
public:
    template <bool Weighted>
    auto outNeighbors(node u) const {
        if constexpr (Weighted)
            return std::views::all(outWeighted_[u]);
        else
            return std::views::all(out_[u]);
    }

    template <bool Weighted>
    auto inNeighbors(node u) const {
        return outNeighbors<Weighted>(u);
    }

protected:
    /// A path 0-1-2 plus the edge 1-3, stored undirected.
    Neighborhoods()
        : out_{{1}, {0, 2, 3}, {1}, {1}},
          outWeighted_{{{1, 1.0}}, {{0, 1.0}, {2, 1.0}, {3, 1.0}}, {{1, 1.0}}, {{1, 1.0}}} {}

    std::vector<std::vector<node>> out_;
    std::vector<std::vector<std::pair<node, edgeweight>>> outWeighted_;
};

/// Implements every GraphLike primitive and inherits the rest.
class MinimalGraph : public Neighborhoods, public NetworKit::GraphIterationMixin<MinimalGraph> {
public:
    count numberOfNodes() const { return 4; }
    count numberOfEdges() const { return 3; }
    index upperNodeIdBound() const { return 4; }
    bool hasNode(node u) const { return u < 4; }
    count degree(node u) const { return out_[u].size(); }
    bool isDirected() const { return false; }
    bool isWeighted() const { return false; }
    count numberOfSelfLoops() const { return 0; }
};

/// Structurally a graph, but does not derive from the mixin.
class WithoutMixin : public Neighborhoods {
public:
    count numberOfNodes() const { return 4; }
    count numberOfEdges() const { return 3; }
    index upperNodeIdBound() const { return 4; }
    bool hasNode(node u) const { return u < 4; }
    count degree(node u) const { return out_[u].size(); }
    bool isDirected() const { return false; }
    bool isWeighted() const { return false; }
    count numberOfSelfLoops() const { return 0; }
};

/// Derives from the mixin, but omits degree().
class WithoutDegree : public Neighborhoods, public NetworKit::GraphIterationMixin<WithoutDegree> {
public:
    count numberOfNodes() const { return 4; }
    count numberOfEdges() const { return 3; }
    index upperNodeIdBound() const { return 4; }
    bool hasNode(node u) const { return u < 4; }
    bool isDirected() const { return false; }
    bool isWeighted() const { return false; }
    count numberOfSelfLoops() const { return 0; }
};

static_assert(NetworKit::GraphLike<MinimalGraph>,
              "the primitive list and the mixin base are together sufficient");
static_assert(!NetworKit::GraphLike<WithoutMixin>,
              "structural matching alone must not satisfy the concept");
static_assert(!NetworKit::GraphLike<WithoutDegree>, "a missing primitive must be rejected");

/*
 * Satisfying the concept only proves the primitives are callable. C++20 never checks a constrained
 * template's body against its own constraints, so each derived member is additionally instantiated
 * below -- that, not the static_asserts, is what catches a body reaching past the primitive list.
 */
TEST(GraphArchetypeGTest, testEveryDerivedMemberInstantiates) {
    const MinimalGraph G;
    count nodes = 0, edges = 0;

    G.forNodes([&](node) { ++nodes; });
    G.parallelForNodes([&](node) {});
    G.forNodesWhile([] { return true; }, [](node) {});
    G.forNodesInRandomOrder([](node) {});
    G.balancedParallelForNodes([](node) {});
    G.forNodePairs([](node, node) {});
    G.parallelForNodePairs([](node, node) {});

    G.forEdges([&](node, node) { ++edges; });
    G.forEdges([](node, node, edgeweight) {});
    G.parallelForEdges([](node, node) {});

    G.forNeighborsOf(1, [](node) {});
    G.forEdgesOf(1, [](node, node) {});
    G.forEdgesOf(1, [](node, node, edgeweight) {});
    G.forInNeighborsOf(1, [](node) {});
    G.forInEdgesOf(1, [](node, node) {});

    EXPECT_EQ(4u, nodes);
    EXPECT_EQ(3u, edges);
    EXPECT_DOUBLE_EQ(4.0, G.parallelSumForNodes([](node) { return 1.0; }));
    EXPECT_DOUBLE_EQ(3.0, G.parallelSumForEdges([](node, node, edgeweight) { return 1.0; }));

    EXPECT_EQ(3u, G.degreeOut(1));
    EXPECT_EQ(3u, G.degreeIn(1));
    EXPECT_TRUE(G.hasEdge(1, 3));
    EXPECT_FALSE(G.hasEdge(0, 2));
    EXPECT_DOUBLE_EQ(NetworKit::defaultEdgeWeight, G.weight(1, 3));
    EXPECT_DOUBLE_EQ(1.0, G.weightedDegree(0));
    EXPECT_DOUBLE_EQ(1.0, G.weightedDegreeIn(0));
    EXPECT_DOUBLE_EQ(3.0, G.totalEdgeWeight());
}

} // namespace ArchetypeTest
