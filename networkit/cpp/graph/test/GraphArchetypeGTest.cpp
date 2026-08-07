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
#include <networkit/graph/GraphIterationOps.hpp>

/*
 * Deliberately not namespace NetworKit. The free operations are found by qualified
 * call (GraphIterationOps::forNodes(...)) rather than argument-dependent lookup, so a foreign
 * namespace cannot hide an unintended ADL dependence.
 */
namespace ArchetypeTest {

using NetworKit::count;
using NetworKit::edgeweight;
using NetworKit::index;
using NetworKit::node;
namespace Ops = NetworKit::GraphIterationOps;

/**
 * The four neighbor ranges and the storage behind them, shared by the archetype and by the
 * negated controls so that they differ only in the property under test.
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

/// Implements every GraphLike primitive and nothing else; structurally satisfies GraphLike.
class MinimalGraph : public Neighborhoods {
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

/// Omits degree(); structurally a graph apart from the missing primitive.
class WithoutDegree : public Neighborhoods {
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
              "the primitive list alone must be sufficient: membership is structural");
static_assert(!NetworKit::GraphLike<WithoutDegree>, "a missing primitive must be rejected");

/*
 * Satisfying the concept only proves the primitives are callable. C++20 never checks a constrained
 * template's body against its own constraints, so each free operation is additionally instantiated
 * below -- that, not the static_asserts, is what catches a body reaching past the primitive list.
 */
TEST(GraphArchetypeGTest, testEveryDerivedOperationInstantiates) {
    const MinimalGraph G;
    count nodes = 0, edges = 0;

    Ops::forNodes(G, [&](node) { ++nodes; });
    Ops::parallelForNodes(G, [](node) {});
    Ops::forNodesWhile(G, [] { return true; }, [](node) {});
    Ops::forNodesInRandomOrder(G, [](node) {});
    Ops::balancedParallelForNodes(G, [](node) {});
    Ops::forNodePairs(G, [](node, node) {});
    Ops::parallelForNodePairs(G, [](node, node) {});

    Ops::forEdges(G, [&](node, node) { ++edges; });
    Ops::forEdges(G, [](node, node, edgeweight) {});
    Ops::parallelForEdges(G, [](node, node) {});

    Ops::forNeighborsOf(G, 1, [](node) {});
    Ops::forEdgesOf(G, 1, [](node, node) {});
    Ops::forEdgesOf(G, 1, [](node, node, edgeweight) {});
    Ops::forInNeighborsOf(G, 1, [](node) {});
    Ops::forInEdgesOf(G, 1, [](node, node) {});

    EXPECT_EQ(4u, nodes);
    EXPECT_EQ(3u, edges);
    EXPECT_DOUBLE_EQ(4.0, Ops::parallelSumForNodes(G, [](node) { return 1.0; }));
    EXPECT_DOUBLE_EQ(3.0, Ops::parallelSumForEdges(G, [](node, node, edgeweight) { return 1.0; }));

    EXPECT_EQ(3u, Ops::degreeOut(G, 1));
    EXPECT_EQ(3u, Ops::degreeIn(G, 1));
    EXPECT_TRUE(Ops::hasEdge(G, 1, 3));
    EXPECT_FALSE(Ops::hasEdge(G, 0, 2));
    EXPECT_DOUBLE_EQ(NetworKit::defaultEdgeWeight, Ops::weight(G, 1, 3));
    EXPECT_DOUBLE_EQ(1.0, Ops::weightedDegree(G, 0));
    EXPECT_DOUBLE_EQ(1.0, Ops::weightedDegreeIn(G, 0));
    EXPECT_DOUBLE_EQ(3.0, Ops::totalEdgeWeight(G));
}

} // namespace ArchetypeTest
