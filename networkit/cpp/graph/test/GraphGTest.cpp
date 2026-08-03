/*
 * GraphGTest.cpp
 *
 *  Created on: 01.06.2014
 *      Author: Klara Reichard (klara.reichard@gmail.com), Marvin Ritter
 * (marvin.ritter@gmail.com)
 */

#include <algorithm>
#include <atomic>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include <networkit/auxiliary/Log.hpp>
#include <networkit/auxiliary/NumericTools.hpp>
#include <networkit/auxiliary/Parallel.hpp>
#include <networkit/auxiliary/Vector2Arrow.hpp>
#include <networkit/generators/ErdosRenyiGenerator.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/graph/GraphBuilder.hpp>
#include <networkit/graph/GraphIteration.hpp>
#include <networkit/graph/GraphR.hpp>
#include <networkit/graph/GraphTools.hpp>
#include <networkit/graph/GraphW.hpp>
#include <networkit/io/METISGraphReader.hpp>

namespace NetworKit {

/**
 * A graph independent of any implementation: node count, the two axes, and the edges in
 * insertion order. Every graph a test needs is described by one of these and handed to
 * buildGraph<G>(), which is the only code a new graph type has to supply.
 */
struct GraphSpec {
    count n;
    bool weighted;
    bool directed;
    std::vector<std::tuple<node, node, edgeweight>> edges;
};

/**
 * Builds @a spec as a @a G. Every graph type supplies one specialization, and one entry in
 * GraphCases below.
 *
 * A type that cannot represent a given spec says so through canRepresent<G>() rather than
 * building something approximate.
 */
template <typename G>
G buildGraph(const GraphSpec &spec);

template <typename G>
bool canRepresent(const GraphSpec &spec);

template <>
inline GraphW buildGraph<GraphW>(const GraphSpec &spec) {
    GraphW G(spec.n, spec.weighted, spec.directed);
    for (const auto &[u, v, ew] : spec.edges) {
        G.addEdge(u, v, ew);
    }
    return G;
}

template <>
inline bool canRepresent<GraphW>(const GraphSpec &) {
    return true;
}

/// Aux::vectorToArrow yields nullptr for an empty vector, but an edgeless graph still needs a
/// present array: GraphR reads its weightedness off whether the weight pointer is null.
template <typename T, typename Arr>
std::shared_ptr<Arr> toArrow(std::vector<T> v) {
    if (v.empty()) {
        return std::make_shared<Arr>(
            0, std::shared_ptr<arrow::Buffer>(std::move(*arrow::AllocateBuffer(0))));
    }
    return Aux::vectorToArrow<T, Arr>(std::move(v));
}

template <>
inline GraphR buildGraph<GraphR>(const GraphSpec &spec) {
    // Mirrors GraphW::addEdge: an undirected edge is stored at both endpoints, an undirected
    // self-loop only once.
    auto adjacency = [&spec](bool inEdges) {
        std::vector<std::vector<std::pair<node, edgeweight>>> adj(spec.n);
        for (const auto &[u, v, ew] : spec.edges) {
            if (spec.directed) {
                adj[inEdges ? v : u].emplace_back(inEdges ? u : v, ew);
            } else {
                adj[u].emplace_back(v, ew);
                if (u != v) {
                    adj[v].emplace_back(u, ew);
                }
            }
        }
        return adj;
    };

    auto toCSR = [&spec](const std::vector<std::vector<std::pair<node, edgeweight>>> &adj) {
        std::vector<uint64_t> indptr{0}, indices;
        std::vector<double> weights;
        for (auto neighbors : adj) {
            std::sort(neighbors.begin(), neighbors.end()); // GraphR assumes sorted adjacency
            for (const auto &[v, ew] : neighbors) {
                indices.push_back(v);
                weights.push_back(ew);
            }
            indptr.push_back(indices.size());
        }
        return std::make_tuple(
            toArrow<uint64_t, arrow::UInt64Array>(std::move(indices)),
            toArrow<uint64_t, arrow::UInt64Array>(std::move(indptr)),
            spec.weighted ? toArrow<double, arrow::DoubleArray>(std::move(weights)) : nullptr);
    };

    const auto [outIndices, outIndptr, outWeights] = toCSR(adjacency(false));
    if (!spec.directed) {
        return GraphR(spec.n, false, outIndices, outIndptr, outIndices, outIndptr, outWeights,
                      outWeights);
    }

    const auto [inIndices, inIndptr, inWeights] = toCSR(adjacency(true));
    return GraphR(spec.n, true, outIndices, outIndptr, inIndices, inIndptr, outWeights, inWeights);
}

template <>
inline bool canRepresent<GraphR>(const GraphSpec &spec) {
    if (spec.directed) {
        return true;
    }
    // An undirected self-loop occupies one CSR slot, but the edge count is derived as
    // slots / 2, so the graph would report a wrong m.
    return std::none_of(spec.edges.begin(), spec.edges.end(),
                        [](const auto &e) { return std::get<0>(e) == std::get<1>(e); });
}

/*
 *    0
 *   . \
 *  /   \
 * /     .
 * 1 <-- 2
 * ^ \  .|
 * |  \/ |
 * | / \ |
 * |/   ..
 * 3 <-- 4
 *
 * move you pen from node to node:
 * 3 -> 1 -> 0 -> 2 -> 1 -> 4 -> 3 -> 2 -> 4
 */
inline std::vector<std::pair<node, node>> houseEdges() {
    return {{3, 1}, {1, 0}, {0, 2}, {2, 1}, {1, 4}, {4, 3}, {3, 2}, {2, 4}};
}

inline GraphSpec houseSpec(bool weighted, bool directed) {
    GraphSpec spec{5, weighted, directed, {}};
    edgeweight ew = 1.0;
    for (const auto &[u, v] : houseEdges()) {
        spec.edges.emplace_back(u, v, ew);
        if (weighted) {
            ew += 1.0;
        }
    }
    return spec;
}

inline std::vector<std::vector<edgeweight>> adjacencyMatrix(const GraphSpec &spec) {
    std::vector<std::vector<edgeweight>> A{spec.n, std::vector<edgeweight>(spec.n, 0.0)};
    for (const auto &[u, v, ew] : spec.edges) {
        A[u][v] = ew;
        if (!spec.directed) {
            A[v][u] = ew;
        }
    }
    return A;
}

/**
 * One graph type at one point of the weighted x directed grid. gtest cannot combine a typed
 * suite with a value-parameterized one, so the axes live in the type list.
 */
template <typename G, bool Weighted, bool Directed>
struct Case {
    using Graph_t = G;
    static constexpr bool weighted = Weighted;
    static constexpr bool directed = Directed;
};

using GraphCases =
    testing::Types<Case<GraphW, false, false>, Case<GraphW, true, false>, Case<GraphW, false, true>,
                   Case<GraphW, true, true>, Case<GraphR, false, false>, Case<GraphR, true, false>,
                   Case<GraphR, false, true>, Case<GraphR, true, true>>;

/*
 * Every type in GraphCases must be GraphLike; the narrower capabilities are what decide which
 * further suites it belongs in. A new graph type states its capabilities here and the compiler
 * checks the claim.
 */
static_assert(GraphLike<GraphW> && IndexedGraph<GraphW> && MutableGraph<GraphW>
              && std::derived_from<GraphW, AttributedGraphBase<GraphW>>);
static_assert(GraphLike<GraphR> && std::derived_from<GraphR, AttributedGraphBase<GraphR>>);
static_assert(!IndexedGraph<GraphR> && !MutableGraph<GraphR>);

/// What every graph type must satisfy. Narrower capabilities get their own suite below.
template <typename TCase>
class GraphGTest : public testing::Test {
protected:
    using Graph_t = typename TCase::Graph_t;

    static constexpr bool isWeighted() { return TCase::weighted; }
    static constexpr bool isDirected() { return TCase::directed; }

    static constexpr bool isUnweightedUndirected() { return !isWeighted() && !isDirected(); }
    static constexpr bool isWeightedUndirected() { return isWeighted() && !isDirected(); }
    static constexpr bool isUnweightedDirected() { return !isWeighted() && isDirected(); }
    static constexpr bool isWeightedDirected() { return isWeighted() && isDirected(); }

    static Graph_t build(const GraphSpec &spec) { return buildGraph<Graph_t>(spec); }
    static bool canBuild(const GraphSpec &spec) { return canRepresent<Graph_t>(spec); }

    GraphSpec spec = houseSpec(isWeighted(), isDirected());
    Graph_t Ghouse = build(spec);
    std::vector<std::pair<node, node>> houseEdgesOut = houseEdges();
    std::vector<std::vector<edgeweight>> Ahouse = adjacencyMatrix(spec);
    count n_house = 5;
    count m_house = 8;
};

TYPED_TEST_SUITE(GraphGTest, GraphCases, /*Comma needed for variadic macro.*/);

/** NODE PROPERTIES **/

TYPED_TEST(GraphGTest, testDegree) {
    if (this->isDirected()) {
        ASSERT_EQ(1u, this->Ghouse.degree(0));
        ASSERT_EQ(2u, this->Ghouse.degree(1));
        ASSERT_EQ(2u, this->Ghouse.degree(2));
        ASSERT_EQ(2u, this->Ghouse.degree(3));
        ASSERT_EQ(1u, this->Ghouse.degree(4));
    } else {
        ASSERT_EQ(2u, this->Ghouse.degree(0));
        ASSERT_EQ(4u, this->Ghouse.degree(1));
        ASSERT_EQ(4u, this->Ghouse.degree(2));
        ASSERT_EQ(3u, this->Ghouse.degree(3));
        ASSERT_EQ(3u, this->Ghouse.degree(4));
    }
}

TYPED_TEST(GraphGTest, testDegreeIn) {
    if (this->isDirected()) {
        ASSERT_EQ(1u, this->Ghouse.degreeIn(0));
        ASSERT_EQ(2u, this->Ghouse.degreeIn(1));
        ASSERT_EQ(2u, this->Ghouse.degreeIn(2));
        ASSERT_EQ(1u, this->Ghouse.degreeIn(3));
        ASSERT_EQ(2u, this->Ghouse.degreeIn(4));
    } else {
        ASSERT_EQ(2u, this->Ghouse.degreeIn(0));
        ASSERT_EQ(4u, this->Ghouse.degreeIn(1));
        ASSERT_EQ(4u, this->Ghouse.degreeIn(2));
        ASSERT_EQ(3u, this->Ghouse.degreeIn(3));
        ASSERT_EQ(3u, this->Ghouse.degreeIn(4));
    }
}

TYPED_TEST(GraphGTest, testDegreeOut) {
    if (this->isDirected()) {
        ASSERT_EQ(1u, this->Ghouse.degreeOut(0));
        ASSERT_EQ(2u, this->Ghouse.degreeOut(1));
        ASSERT_EQ(2u, this->Ghouse.degreeOut(2));
        ASSERT_EQ(2u, this->Ghouse.degreeOut(3));
        ASSERT_EQ(1u, this->Ghouse.degreeOut(4));
    } else {
        ASSERT_EQ(2u, this->Ghouse.degreeOut(0));
        ASSERT_EQ(4u, this->Ghouse.degreeOut(1));
        ASSERT_EQ(4u, this->Ghouse.degreeOut(2));
        ASSERT_EQ(3u, this->Ghouse.degreeOut(3));
        ASSERT_EQ(3u, this->Ghouse.degreeOut(4));
    }
}

/** EDGE PROPERTIES **/

TYPED_TEST(GraphGTest, testHasEdge) {
    auto containsEdge = [&](std::pair<node, node> e) {
        auto it = std::find(this->houseEdgesOut.begin(), this->houseEdgesOut.end(), e);
        return it != this->houseEdgesOut.end();
    };

    for (node u = 0; u < this->Ghouse.upperNodeIdBound(); u++) {
        for (node v = 0; v < this->Ghouse.upperNodeIdBound(); v++) {
            auto edge = std::make_pair(u, v);
            auto edgeReverse = std::make_pair(v, u);
            bool hasEdge = containsEdge(edge);
            bool hasEdgeReverse = containsEdge(edgeReverse);
            if (this->Ghouse.isDirected()) {
                ASSERT_EQ(hasEdge, this->Ghouse.hasEdge(u, v));
            } else {
                ASSERT_EQ(hasEdge || hasEdgeReverse, this->Ghouse.hasEdge(u, v));
            }
        }
    }
}

TYPED_TEST(GraphGTest, testWeight) {
    this->Ghouse.forNodes([&](node u) {
        this->Ghouse.forNodes(
            [&](node v) { ASSERT_EQ(this->Ahouse[u][v], this->Ghouse.weight(u, v)); });
    });
}

/** GLOBAL PROPERTIES **/

TYPED_TEST(GraphGTest, testIsWeighted) {
    ASSERT_EQ(this->isWeighted(), this->Ghouse.isWeighted());
}

TYPED_TEST(GraphGTest, testIsDirected) {
    ASSERT_EQ(this->isDirected(), this->Ghouse.isDirected());
}

TYPED_TEST(GraphGTest, testNumberOfNodes) {
    ASSERT_EQ(this->n_house, this->Ghouse.numberOfNodes());
}

TYPED_TEST(GraphGTest, testNumberOfEdges) {
    ASSERT_EQ(this->m_house, this->Ghouse.numberOfEdges());
}

TYPED_TEST(GraphGTest, testUpperNodeIdBound) {
    ASSERT_EQ(5u, this->Ghouse.upperNodeIdBound());
}

TYPED_TEST(GraphGTest, testTotalEdgeWeight) {
    if (this->Ghouse.isWeighted()) {
        ASSERT_EQ(36.0, this->Ghouse.totalEdgeWeight());
    } else {
        ASSERT_EQ(8 * defaultEdgeWeight, this->Ghouse.totalEdgeWeight());
    }
}

/** Collections **/

TYPED_TEST(GraphGTest, testNeighborsIterators) {
    auto range = this->Ghouse.neighborRange(1);
    auto iter = range.begin();
    this->Ghouse.forNeighborsOf(1, [&](node v) {
        ASSERT_TRUE(*iter == v);
        ++iter;
    });
    ASSERT_TRUE(iter == range.end());

    if (this->Ghouse.isWeighted()) {
        auto rangeW = this->Ghouse.weightNeighborRange(1);
        auto iterW = rangeW.begin();
        this->Ghouse.forNeighborsOf(1, [&](node v, edgeweight w) {
            ASSERT_TRUE((*iterW).first == v);
            ASSERT_TRUE((*iterW).second == w);
            ++iterW;
        });
        ASSERT_TRUE(iterW == rangeW.end());
    }

    if (this->Ghouse.isDirected()) {
        auto inRange = this->Ghouse.inNeighborRange(1);
        auto inIter = inRange.begin();
        this->Ghouse.forInNeighborsOf(1, [&](node v) {
            ASSERT_TRUE(*inIter == v);
            ++inIter;
        });
        ASSERT_TRUE(inIter == inRange.end());

        if (this->Ghouse.isWeighted()) {
            auto inRangeW = this->Ghouse.weightInNeighborRange(1);
            auto iterW = inRangeW.begin();
            this->Ghouse.forInNeighborsOf(1, [&](node v, edgeweight w) {
                ASSERT_TRUE((*iterW).first == v);
                ASSERT_TRUE((*iterW).second == w);
                ++iterW;
            });
            ASSERT_TRUE(iterW == inRangeW.end());
        }
    }
}

/** NODE ITERATORS **/

TYPED_TEST(GraphGTest, testParallelForNodes) {
    std::vector<node> visited(this->Ghouse.upperNodeIdBound());
    this->Ghouse.parallelForNodes([&](node u) { visited[u] = u; });

    Aux::Parallel::sort(visited.begin(), visited.end());

    ASSERT_EQ(5u, visited.size());
    for (index i = 0; i < this->Ghouse.upperNodeIdBound(); i++) {
        ASSERT_EQ(i, visited[i]);
    }
}

/** NEIGHBORHOOD ITERATORS **/

TYPED_TEST(GraphGTest, testForNeighborsOf) {
    std::vector<node> visited;
    this->Ghouse.forNeighborsOf(3, [&](node u) { visited.push_back(u); });

    Aux::Parallel::sort(visited.begin(), visited.end());

    if (this->isDirected()) {
        ASSERT_EQ(2u, visited.size());
        ASSERT_EQ(1u, visited[0]);
        ASSERT_EQ(2u, visited[1]);
    } else {
        ASSERT_EQ(3u, visited.size());
        ASSERT_EQ(1u, visited[0]);
        ASSERT_EQ(2u, visited[1]);
        ASSERT_EQ(4u, visited[2]);
    }
}

TYPED_TEST(GraphGTest, testForWeightedNeighborsOf) {
    std::vector<std::pair<node, edgeweight>> visited;
    this->Ghouse.forNeighborsOf(
        3, [&](node u, edgeweight ew) { visited.push_back(std::make_pair(u, ew)); });

    // should sort after the first element
    Aux::Parallel::sort(visited.begin(), visited.end());

    if (this->isUnweightedUndirected()) {
        ASSERT_EQ(3u, visited.size());
        ASSERT_EQ(1u, visited[0].first);
        ASSERT_EQ(2u, visited[1].first);
        ASSERT_EQ(4u, visited[2].first);
        ASSERT_EQ(defaultEdgeWeight, visited[0].second);
        ASSERT_EQ(defaultEdgeWeight, visited[1].second);
        ASSERT_EQ(defaultEdgeWeight, visited[2].second);
    }

    if (this->isWeightedUndirected()) {
        ASSERT_EQ(3u, visited.size());
        ASSERT_EQ(1u, visited[0].first);
        ASSERT_EQ(2u, visited[1].first);
        ASSERT_EQ(4u, visited[2].first);
        ASSERT_EQ(1.0, visited[0].second);
        ASSERT_EQ(7.0, visited[1].second);
        ASSERT_EQ(6.0, visited[2].second);
    }

    if (this->isUnweightedDirected()) {
        ASSERT_EQ(2u, visited.size());
        ASSERT_EQ(1u, visited[0].first);
        ASSERT_EQ(2u, visited[1].first);
        ASSERT_EQ(defaultEdgeWeight, visited[0].second);
        ASSERT_EQ(defaultEdgeWeight, visited[1].second);
    }

    if (this->isWeightedDirected()) {
        ASSERT_EQ(2u, visited.size());
        ASSERT_EQ(1u, visited[0].first);
        ASSERT_EQ(2u, visited[1].first);
        ASSERT_EQ(1.0, visited[0].second);
        ASSERT_EQ(7.0, visited[1].second);
    }
}

TYPED_TEST(GraphGTest, testForEdgesOf) {
    count m = 0;
    std::vector<int> visited(this->m_house, 0);

    this->Ghouse.forNodes([&](node u) {
        this->Ghouse.forEdgesOf(u, [&](node v, node w) {
            // edges should be v to w, so if we iterate over edges from u, u should be
            // equal v
            EXPECT_EQ(u, v);

            auto e = std::make_pair(v, w);
            auto it = std::find(this->houseEdgesOut.begin(), this->houseEdgesOut.end(), e);
            if (!this->isDirected() && it == this->houseEdgesOut.end()) {
                auto e2 = std::make_pair(w, v);
                it = std::find(this->houseEdgesOut.begin(), this->houseEdgesOut.end(), e2);
            }

            EXPECT_TRUE(it != this->houseEdgesOut.end());

            // find index in edge array
            int i = std::distance(this->houseEdgesOut.begin(), it);
            if (this->isDirected()) {
                // make sure edge was not visited before (would be visited twice)
                EXPECT_EQ(0, visited[i]);
            }

            // mark edge as visited
            visited[i]++;
            m++;
        });
    });

    if (this->isDirected()) {
        // we iterated over all outgoing edges once
        EXPECT_EQ(this->m_house, m);
        for (auto c : visited) {
            EXPECT_EQ(1, c);
        }
    } else {
        // we iterated over all edges in both directions
        EXPECT_EQ(2 * this->m_house, m);
        for (auto c : visited) {
            EXPECT_EQ(2, c);
        }
    }
}

TYPED_TEST(GraphGTest, testForWeightedEdgesOf) {
    count m = 0;
    std::vector<int> visited(this->m_house, 0);
    double sumOfWeights = 0;

    this->Ghouse.forNodes([&](node u) {
        this->Ghouse.forEdgesOf(u, [&](node v, node w, edgeweight ew) {
            // edges should be v to w, so if we iterate over edges from u, u should be
            // equal v
            EXPECT_EQ(u, v);
            sumOfWeights += ew;
            auto e = std::make_pair(v, w);
            auto it = std::find(this->houseEdgesOut.begin(), this->houseEdgesOut.end(), e);
            if (!this->isDirected() && it == this->houseEdgesOut.end()) {
                auto e2 = std::make_pair(w, v);
                it = std::find(this->houseEdgesOut.begin(), this->houseEdgesOut.end(), e2);
            }

            EXPECT_TRUE(it != this->houseEdgesOut.end());

            // find index in edge array
            int i = std::distance(this->houseEdgesOut.begin(), it);
            if (this->isDirected()) {
                // make sure edge was not visited before (would be visited twice)
                EXPECT_EQ(0, visited[i]);
            }

            // mark edge as visited
            visited[i]++;
            m++;
        });
    });

    if (this->isUnweightedUndirected()) {
        EXPECT_EQ(sumOfWeights, m);
        EXPECT_EQ(2 * this->m_house, m);
        for (auto c : visited) {
            EXPECT_EQ(2, c);
        }
    }

    if (this->isWeightedUndirected()) {
        // we iterated over all edges in both directions
        EXPECT_EQ(2 * this->m_house, m);
        EXPECT_EQ(sumOfWeights, 72);
        for (auto c : visited) {
            EXPECT_EQ(2, c);
        }
    }

    if (this->isUnweightedDirected()) {
        // we iterated over all outgoing edges once
        EXPECT_EQ(this->m_house, m);
        EXPECT_EQ(sumOfWeights, m);
        for (auto c : visited) {
            EXPECT_EQ(1, c);
        }
    }

    if (this->isWeightedDirected()) {
        EXPECT_EQ(sumOfWeights, 36);
        EXPECT_EQ(this->m_house, m);
        for (auto c : visited) {
            EXPECT_EQ(1, c);
        }
    }
}

TYPED_TEST(GraphGTest, testForInNeighborsOf) {
    std::vector<node> visited;
    this->Ghouse.forInNeighborsOf(2, [&](node v) { visited.push_back(v); });
    Aux::Parallel::sort(visited.begin(), visited.end());

    if (this->isDirected()) {
        EXPECT_EQ(2u, visited.size());
        EXPECT_EQ(0u, visited[0]);
        EXPECT_EQ(3u, visited[1]);
    } else {
        EXPECT_EQ(4u, visited.size());
        EXPECT_EQ(0u, visited[0]);
        EXPECT_EQ(1u, visited[1]);
        EXPECT_EQ(3u, visited[2]);
        EXPECT_EQ(4u, visited[3]);
    }
}

TYPED_TEST(GraphGTest, testForWeightedInNeighborsOf) {
    std::vector<std::pair<node, edgeweight>> visited;
    this->Ghouse.forInNeighborsOf(3, [&](node v, edgeweight ew) { visited.push_back({v, ew}); });
    Aux::Parallel::sort(visited.begin(), visited.end());

    if (this->isUnweightedUndirected()) {
        ASSERT_EQ(3u, visited.size());
        ASSERT_EQ(1u, visited[0].first);
        ASSERT_EQ(2u, visited[1].first);
        ASSERT_EQ(4u, visited[2].first);
        ASSERT_EQ(defaultEdgeWeight, visited[0].second);
        ASSERT_EQ(defaultEdgeWeight, visited[1].second);
        ASSERT_EQ(defaultEdgeWeight, visited[2].second);
    }

    if (this->isWeightedUndirected()) {
        ASSERT_EQ(3u, visited.size());
        ASSERT_EQ(1u, visited[0].first);
        ASSERT_EQ(2u, visited[1].first);
        ASSERT_EQ(4u, visited[2].first);
        ASSERT_EQ(1.0, visited[0].second);
        ASSERT_EQ(7.0, visited[1].second);
        ASSERT_EQ(6.0, visited[2].second);
    }

    if (this->isUnweightedDirected()) {
        ASSERT_EQ(1u, visited.size());
        ASSERT_EQ(4u, visited[0].first);
        ASSERT_EQ(defaultEdgeWeight, visited[0].second);
    }

    if (this->isWeightedDirected()) {
        ASSERT_EQ(1u, visited.size());
        ASSERT_EQ(4u, visited[0].first);
        ASSERT_EQ(6.0, visited[0].second);
    }
}

TYPED_TEST(GraphGTest, testForInEdgesOf) {
    std::vector<bool> visited(this->n_house, false);
    this->Ghouse.forInEdgesOf(3, [&](node u, node v) {
        ASSERT_EQ(3u, u);
        if (this->isDirected()) {
            ASSERT_TRUE(this->Ahouse[v][u] > 0.0);
            ASSERT_TRUE(this->Ghouse.hasEdge(v, u));
        }
        ASSERT_FALSE(visited[v]);
        visited[v] = true;
    });

    if (this->isDirected()) {
        EXPECT_FALSE(visited[0]);
        EXPECT_FALSE(visited[1]);
        EXPECT_FALSE(visited[2]);
        EXPECT_FALSE(visited[3]);
        EXPECT_TRUE(visited[4]);
    } else {
        EXPECT_FALSE(visited[0]);
        EXPECT_TRUE(visited[1]);
        EXPECT_TRUE(visited[2]);
        EXPECT_FALSE(visited[3]);
        EXPECT_TRUE(visited[4]);
    }
}

/** EDGE ITERATORS **/

TYPED_TEST(GraphGTest, testForEdges) {
    const GraphSpec spec{4,
                         this->isWeighted(),
                         this->isDirected(),
                         {{0, 1, defaultEdgeWeight},   // 0 * 1 = 0
                          {1, 2, defaultEdgeWeight},   // 1 * 2 = 2
                          {3, 2, defaultEdgeWeight},   // 3 * 2 = 1 (mod 5)
                          {2, 2, defaultEdgeWeight},   // 2 * 2 = 4
                          {3, 1, defaultEdgeWeight}}}; // 3 * 1 = 3
    if (!this->canBuild(spec))
        GTEST_SKIP() << "graph type cannot represent this graph";
    auto G = this->build(spec);

    std::vector<bool> edgesSeen(5, false);

    G.forEdges([&](node u, node v) {
        ASSERT_TRUE(G.hasEdge(u, v));
        index id = (u * v) % 5;
        edgesSeen[id] = true;
    });

    for (auto b : edgesSeen) {
        ASSERT_TRUE(b);
    }
}

TYPED_TEST(GraphGTest, testForWeightedEdges) {
    double epsilon = 1e-6;

    const GraphSpec spec{4,
                         this->isWeighted(),
                         this->isDirected(),
                         {{0, 1, 0.1},   // 0 * 1 = 0
                          {3, 2, 0.2},   // 3 * 2 = 1 (mod 5)
                          {1, 2, 0.3},   // 1 * 2 = 2
                          {3, 1, 0.4},   // 3 * 1 = 3
                          {2, 2, 0.5}}}; // 2 * 2 = 4
    if (!this->canBuild(spec))
        GTEST_SKIP() << "graph type cannot represent this graph";
    auto G = this->build(spec);

    std::vector<bool> edgesSeen(5, false);

    edgeweight weightSum = 0;
    G.forEdges([&](node u, node v, edgeweight ew) {
        ASSERT_TRUE(G.hasEdge(u, v));
        ASSERT_EQ(G.weight(u, v), ew);

        index id = (u * v) % 5;
        edgesSeen[id] = true;
        if (G.isWeighted()) {
            ASSERT_NEAR((id + 1) * 0.1, ew, epsilon);
        } else {
            ASSERT_EQ(defaultEdgeWeight, ew);
        }
        weightSum += ew;
    });

    for (auto b : edgesSeen) {
        ASSERT_TRUE(b);
    }
    if (G.isWeighted()) {
        ASSERT_NEAR(1.5, weightSum, epsilon);
    } else {
        ASSERT_NEAR(5 * defaultEdgeWeight, weightSum, epsilon);
    }
}

/// The complete graph on four nodes: six edges whichever way they are oriented.
inline GraphSpec completeSpec(bool weighted, bool directed) {
    return {4,
            weighted,
            directed,
            {{0, 1, defaultEdgeWeight},
             {0, 2, defaultEdgeWeight},
             {0, 3, defaultEdgeWeight},
             {1, 2, defaultEdgeWeight},
             {1, 3, defaultEdgeWeight},
             {2, 3, defaultEdgeWeight}}};
}

TYPED_TEST(GraphGTest, testParallelForWeightedEdges) {
    auto G = this->build(completeSpec(this->isWeighted(), this->isDirected()));

    edgeweight weightSum = 0.0;
    G.parallelForEdges([&](node, node, edgeweight ew) {
#pragma omp atomic
        weightSum += ew;
    });

    ASSERT_EQ(6.0, weightSum) << "sum of edge weights should be 6 in every case";
}

TYPED_TEST(GraphGTest, testParallelForEdges) {
    auto G = this->build(completeSpec(this->isWeighted(), this->isDirected()));

    edgeweight weightSum = 0.0;
    G.parallelForEdges([&](node, node) {
#pragma omp atomic
        weightSum += 1;
    });

    ASSERT_EQ(6.0, weightSum) << "sum of edge weights should be 6 in every case";
}

/**
 * The balancedParallelForNodes + forInEdgesOf pattern PageRank uses, on a graph big enough for
 * the schedule to actually split. Every node is on the ring, so every node has in-edges.
 */
TYPED_TEST(GraphGTest, testPageRankStyleIteration) {
    constexpr count n = 5000;
    constexpr int extraEdges = 5;

    Aux::Random::setSeed(42, false);
    GraphSpec spec{n, this->isWeighted(), this->isDirected(), {}};
    for (node u = 0; u < n; ++u) {
        spec.edges.emplace_back(u, (u + 1) % n, defaultEdgeWeight);
        for (int k = 0; k < extraEdges; ++k) {
            const node v = Aux::Random::index(n);
            if (v != u) {
                spec.edges.emplace_back(u, v, defaultEdgeWeight);
            }
        }
    }
    auto G = this->build(spec);

    for (int iteration = 0; iteration < 50; ++iteration) {
        std::atomic<bool> success{true};
        G.balancedParallelForNodes([&](const node u) {
            count seen = 0;
            G.forInEdgesOf(u, [&](const node, const node, const edgeweight) { seen++; });
            if (seen == 0)
                success = false;
        });

        EXPECT_TRUE(success) << "Iteration " << iteration << " should process all nodes";
    }
}

/** REDUCTION ITERATORS **/

TYPED_TEST(GraphGTest, testParallelSumForNodes) {
    count n = 10;
    auto G = this->build({n, this->isWeighted(), this->isDirected(), {}});
    double sum = G.parallelSumForNodes([](node v) { return 2 * v + 0.5; });

    double expected_sum = n * (n - 1) + n * 0.5;
    ASSERT_EQ(expected_sum, sum);
}

TYPED_TEST(GraphGTest, testParallelSumForWeightedEdges) {
    double sum =
        this->Ghouse.parallelSumForEdges([](node, node, edgeweight ew) { return 1.5 * ew; });

    double expected_sum = 1.5 * this->Ghouse.totalEdgeWeight();
    ASSERT_EQ(expected_sum, sum);
}

TYPED_TEST(GraphGTest, testParallelForEdgesVisitsEveryEdge) {
    const auto &G = this->Ghouse;
    std::atomic<count> visited{0};
    G.parallelForEdges([&](node, node) { visited.fetch_add(1); });
    EXPECT_EQ(this->m_house, visited.load());
}

/*
 * Iterators taken from two separate neighborRange() calls must be comparable. They were not while
 * the erased range materialized a fresh buffer per call, which crashed
 * SampledGraphStructuralRandMeasure; the range now borrows the arm's storage, so the two address
 * the same memory.
 */
TYPED_TEST(GraphGTest, testNeighborRangeTemporariesAreInterchangeable) {
    const Graph G(this->Ghouse);
    const node u = 1;

    std::vector<node> fromTemporaries(G.neighborRange(u).begin(), G.neighborRange(u).end());

    std::vector<node> fromOneRange;
    for (node v : G.neighborRange(u))
        fromOneRange.push_back(v);

    EXPECT_EQ(fromOneRange, fromTemporaries);
    EXPECT_EQ(G.degree(u), fromTemporaries.size());
}

} /* namespace NetworKit */
