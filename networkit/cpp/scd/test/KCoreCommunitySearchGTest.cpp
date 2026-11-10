/*
 * KCoreCommunitySearchGTest.cpp
 *
 *  Created on: 06.08.2026
 *      Author: Ian Chen (ianchen3@illinois.edu)
 */

#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <gtest/gtest.h>
#include <networkit/auxiliary/Random.hpp>
#include <networkit/centrality/CoreDecomposition.hpp>
#include <networkit/components/ConnectedComponents.hpp>
#include <networkit/generators/HyperbolicGenerator.hpp>
#include <networkit/graph/GraphW.hpp>
#include <networkit/scd/ShellStruct.hpp>
#include <networkit/scd/SteinerKCore.hpp>

namespace NetworKit {

namespace {

GraphW getCliqueChain(const std::vector<count> &cliqueSizes) {
    count totalNodes = 0;
    for (count size : cliqueSizes)
        totalNodes += size;

    GraphW g(totalNodes, false, false);
    count start = 0;
    for (count size : cliqueSizes) {
        for (count u = start; u < start + size; ++u)
            for (count v = u + 1; v < start + size; ++v)
                g.addEdge(u, v);
        start += size;
    }

    start = 0;
    for (count i = 0; i < cliqueSizes.size() - 1; ++i) {
        start += cliqueSizes[i];
        g.addEdge(start - 1, start);
    }

    return g;
}

std::set<node> gtSearch(const std::vector<count> &cliques, const std::set<node> &query) {
    std::vector<count> boundaries;
    count currentSum = 0;
    for (count size : cliques) {
        currentSum += size;
        boundaries.push_back(currentSum);
    }

    count cMin = cliques.size();
    count cMax = 0;
    for (node q : query) {
        count cIdx = 0;
        for (count i = 0; i < boundaries.size(); ++i) {
            if (q < boundaries[i]) {
                cIdx = i;
                break;
            }
        }
        cMin = std::min(cMin, cIdx);
        cMax = std::max(cMax, cIdx);
    }

    count bottleneckSize = cliques[cMin];
    for (count i = cMin; i <= cMax; ++i)
        bottleneckSize = std::min(bottleneckSize, cliques[i]);

    count left = cMin;
    while (left > 0 && cliques[left - 1] >= bottleneckSize)
        left--;

    count right = cMax;
    while (right < cliques.size() - 1 && cliques[right + 1] >= bottleneckSize)
        right++;

    count vBeg = (left > 0) ? boundaries[left - 1] : 0;
    count vEnd = boundaries[right];

    std::set<node> expectedCommunity;
    for (count v = vBeg; v < vEnd; ++v) {
        expectedCommunity.insert(static_cast<node>(v));
    }

    return expectedCommunity;
}

std::vector<count> coreDecomp(const std::vector<count> &cliques) {
    std::vector<count> ret;
    for (count i : cliques)
        for (count j = 0; j < i; ++j)
            ret.push_back(i - 1);
    return ret;
}

void verifySingletons(SelectiveCommunityDetector &detector, const std::vector<count> &cliques) {
    count n = std::accumulate(cliques.begin(), cliques.end(), count(0));
    for (node v = 0; v < n; ++v) {
        std::set<node> query = {v};
        std::set<node> et = detector.expandOneCommunity(query);
        std::set<node> gt = gtSearch(cliques, query);

        EXPECT_EQ(et, gt) << "Failed on singleton query node: " << v;
    }
}

void verifyPairs(SelectiveCommunityDetector &detector, const std::vector<count> &cliques) {
    count n = std::accumulate(cliques.begin(), cliques.end(), count(0));
    for (node u = 0; u < n; ++u) {
        for (node v = u + 1; v < n; ++v) {
            std::set<node> query = {u, v};
            std::set<node> et = detector.expandOneCommunity(query);
            std::set<node> gt = gtSearch(cliques, query);

            EXPECT_EQ(et, gt) << "Failed on pair nodes: {" << u << ", " << v << "}";
        }
    }
}

std::vector<std::set<node>> asSets(const SteinerKCore::Communities &c) {
    std::vector<std::set<node>> sets;
    for (std::span<const node> members : c.communities)
        sets.emplace_back(members.begin(), members.end());
    return sets;
}

void verifyBatch(SteinerKCore &steiner, const std::vector<count> &cliques) {
    count n = std::accumulate(cliques.begin(), cliques.end(), count(0));
    std::vector<std::set<node>> queries = {
        {0}, {n - 1}, {0, 1}, {0, n - 1}, {n / 2, n / 2 + 1},
    };

    std::vector<std::set<node>> et = asSets(steiner.communities(queries));

    ASSERT_EQ(et.size(), queries.size());
    for (index i = 0; i < queries.size(); ++i) {
        std::set<node> gt = gtSearch(cliques, queries[i]);
        EXPECT_EQ(et[i], gt) << "Failed on batched query index " << i;
    }
}

template <typename Detector>
std::unique_ptr<Detector> makeDetector(const GraphW &g);

template <>
std::unique_ptr<ShellStruct> makeDetector(const GraphW &g) {
    auto shell = std::make_unique<ShellStruct>(g);
    shell->build();
    return shell;
}

template <>
std::unique_ptr<SteinerKCore> makeDetector(const GraphW &g) {
    return std::make_unique<SteinerKCore>(g);
}

} // namespace

template <typename Detector>
class KCoreCommunitySearchGTest : public ::testing::Test {};

struct DetectorName {
    template <typename Detector>
    static std::string GetName(int) {
        return std::is_same_v<Detector, ShellStruct> ? "ShellStruct" : "SteinerKCore";
    }
};

using Detectors = ::testing::Types<ShellStruct, SteinerKCore>;
TYPED_TEST_SUITE(KCoreCommunitySearchGTest, Detectors, DetectorName);

TYPED_TEST(KCoreCommunitySearchGTest, testSeparatedCoresSingleton) {
    std::vector<count> cliques{4, 3, 4};
    GraphW g = getCliqueChain(cliques);
    verifySingletons(*makeDetector<TypeParam>(g), cliques);
}

TYPED_TEST(KCoreCommunitySearchGTest, testSeparatedCoresPairs) {
    std::vector<count> cliques{4, 3, 4};
    GraphW g = getCliqueChain(cliques);
    verifyPairs(*makeDetector<TypeParam>(g), cliques);
}

TYPED_TEST(KCoreCommunitySearchGTest, testBridgedCoreSingleton) {
    std::vector<count> cliques{3, 4, 3};
    GraphW g = getCliqueChain(cliques);
    verifySingletons(*makeDetector<TypeParam>(g), cliques);
}

TYPED_TEST(KCoreCommunitySearchGTest, testBridgedCorePairs) {
    std::vector<count> cliques{3, 4, 3};
    GraphW g = getCliqueChain(cliques);
    verifyPairs(*makeDetector<TypeParam>(g), cliques);
}

TYPED_TEST(KCoreCommunitySearchGTest, testMediumSingleton) {
    std::vector<count> cliques{3, 5, 4, 3, 6, 5, 4, 6, 3};
    GraphW g = getCliqueChain(cliques);
    verifySingletons(*makeDetector<TypeParam>(g), cliques);
}

TYPED_TEST(KCoreCommunitySearchGTest, testMediumPairs) {
    std::vector<count> cliques{3, 5, 4, 3, 6, 5, 4, 6, 3};
    GraphW g = getCliqueChain(cliques);
    verifyPairs(*makeDetector<TypeParam>(g), cliques);
}

TYPED_TEST(KCoreCommunitySearchGTest, testLargeSingleton) {
    std::vector<count> cliques{3, 4, 5, 6, 5, 4, 3, 5, 7, 9, 11, 9, 7, 5, 3, 4, 5, 6, 4, 3};
    GraphW g = getCliqueChain(cliques);
    verifySingletons(*makeDetector<TypeParam>(g), cliques);
}

TYPED_TEST(KCoreCommunitySearchGTest, testLargePairs) {
    std::vector<count> cliques{3, 4, 5, 6, 5, 4, 3, 5, 7, 9, 11, 9, 7, 5, 3, 4, 5, 6, 4, 3};
    GraphW g = getCliqueChain(cliques);
    verifyPairs(*makeDetector<TypeParam>(g), cliques);
}

// Enough vertices for the parallel path of the shell grouping.
TEST(ShellStructGTest, testManyCliquesSingleton) {
    std::vector<count> cliques(2500);
    for (count i = 0; i < cliques.size(); ++i)
        cliques[i] = 3 + i % 4;
    GraphW g = getCliqueChain(cliques);
    ShellStruct shell(g);
    shell.build();
    verifySingletons(shell, cliques);
}

TEST(ShellStructGTest, testLoadSave) {
    std::vector<count> cliques{3, 4, 5, 6, 5, 4, 3, 5, 7, 9, 11, 9, 7, 5, 3, 4, 5, 6, 4, 3};
    GraphW g = getCliqueChain(cliques);
    ShellStruct shell(g);
    shell.build();
    shell.save("output/shellstruct_comp.feather", "output/shellstruct_tree.feather");

    ShellStruct index(g);
    index.load("output/shellstruct_comp.feather", "output/shellstruct_tree.feather");
    verifyPairs(index, cliques);
}

TEST(ShellStructGTest, testLoadSaveValidate) {
    std::vector<count> cliques{3, 4, 5, 6, 5, 4, 3, 5, 7, 9, 11, 9, 7, 5, 3, 4, 5, 6, 4, 3};
    GraphW g = getCliqueChain(cliques);
    ShellStruct shell(g);
    shell.build();
    shell.save("output/shellstruct_comp.feather", "output/shellstruct_tree.feather");

    ShellStruct index(g, true);
    index.load("output/shellstruct_comp.feather", "output/shellstruct_tree.feather");
    verifyPairs(index, cliques);
}

TEST(ShellStructGTest, testLoadOtherGraphThrows) {
    GraphW g = getCliqueChain({3, 4, 3});
    ShellStruct shell(g);
    shell.build();
    shell.save("output/shellstruct_comp.feather", "output/shellstruct_tree.feather");

    GraphW other = getCliqueChain({3, 4, 4});
    ShellStruct index(other);
    EXPECT_THROW(index.load("output/shellstruct_comp.feather", "output/shellstruct_tree.feather"),
                 std::runtime_error);
}

TEST(ShellStructGTest, testEmptyGraph) {
    GraphW g(0, false, false);
    ShellStruct shell(g);
    shell.build();
    EXPECT_TRUE(shell.community({}).empty());
    shell.save("output/shellstruct_comp.feather", "output/shellstruct_tree.feather");

    ShellStruct index(g, true);
    index.load("output/shellstruct_comp.feather", "output/shellstruct_tree.feather");
    EXPECT_TRUE(index.expandOneCommunity(std::set<node>{}).empty());
}

TEST(ShellStructGTest, testInvalidSeeds) {
    GraphW g = getCliqueChain({3, 4, 3});
    ShellStruct shell(g);
    EXPECT_THROW(shell.community({0}), std::runtime_error);
    shell.build();
    EXPECT_THROW(shell.community({g.upperNodeIdBound()}), std::out_of_range);
    EXPECT_THROW(shell.score({}), std::invalid_argument);
    EXPECT_EQ(shell.score({0, 9}), count{2});
    EXPECT_EQ(shell.score({3, 4}), count{3});
}

TEST(ShellStructGTest, testBuildFromCoreDecomposition) {
    std::vector<count> cliques{3, 5, 4, 3, 6, 5, 4, 6, 3};
    GraphW g = getCliqueChain(cliques);
    CoreDecomposition coreDecomposition(g);
    coreDecomposition.run();
    const std::vector<double> &scores = coreDecomposition.scores();
    ShellStruct shell(g);
    shell.build(std::vector<count>(scores.begin(), scores.end()));
    verifyPairs(shell, cliques);
}

TEST(ShellStructGTest, testBuildWrongSizeThrows) {
    GraphW g = getCliqueChain({3, 4, 3});
    ShellStruct shell(g);
    EXPECT_THROW(shell.build(std::vector<count>(g.upperNodeIdBound() - 1)), std::invalid_argument);
}

// Same vertex count and tree shape, so only validate detects the mismatch.
TEST(ShellStructGTest, testLoadMismatchedValidateThrows) {
    GraphW g = getCliqueChain({3, 4, 3});
    ShellStruct shell(g);
    shell.build();
    shell.save("output/shellstruct_comp.feather", "output/shellstruct_tree.feather");

    GraphW other = getCliqueChain({4, 3, 3});
    ShellStruct otherShell(other);
    otherShell.build();
    otherShell.save("output/shellstruct_other_comp.feather",
                    "output/shellstruct_other_tree.feather");

    ShellStruct trusting(g);
    EXPECT_NO_THROW(
        trusting.load("output/shellstruct_comp.feather", "output/shellstruct_other_tree.feather"));
    ShellStruct validating(g, true);
    EXPECT_THROW(
        validating.load("output/shellstruct_comp.feather", "output/shellstruct_other_tree.feather"),
        std::runtime_error);
}

TEST(SteinerKCoreGTest, testSeparatedCoresBatch) {
    std::vector<count> cliques{4, 3, 4};
    GraphW g = getCliqueChain(cliques);
    std::vector<count> coreness = coreDecomp(cliques);
    SteinerKCore steiner(g, coreness);
    verifyBatch(steiner, cliques);
}

TEST(SteinerKCoreGTest, testBridgedCoreBatch) {
    std::vector<count> cliques{3, 4, 3};
    GraphW g = getCliqueChain(cliques);
    std::vector<count> coreness = coreDecomp(cliques);
    SteinerKCore steiner(g, coreness);
    verifyBatch(steiner, cliques);
}

TEST(SteinerKCoreGTest, testLargeBatch) {
    std::vector<count> cliques{3, 4, 5, 6, 5, 4, 3, 5, 7, 9, 11, 9, 7, 5, 3, 4, 5, 6, 4, 3};
    GraphW g = getCliqueChain(cliques);
    std::vector<count> coreness = coreDecomp(cliques);
    SteinerKCore steiner(g, coreness);
    verifyBatch(steiner, cliques);
}

// Enough vertices and queries for the parallel paths, with many queries sharing a component.
TEST(SteinerKCoreGTest, testManyCliquesBatch) {
    std::vector<count> cliques(2500);
    for (count i = 0; i < cliques.size(); ++i)
        cliques[i] = 3 + i % 4;
    GraphW g = getCliqueChain(cliques);
    std::vector<count> coreness = coreDecomp(cliques);
    SteinerKCore steiner(g, coreness);

    const count n = g.numberOfNodes();
    std::vector<std::set<node>> queries;
    for (node u = 0; u < n; u += 9) {
        queries.push_back({u});
        queries.push_back({u, (u * 7919 + 1) % n});
    }
    std::vector<std::set<node>> et = asSets(steiner.communities(queries));
    ASSERT_EQ(et.size(), queries.size());
    for (index i = 0; i < queries.size(); ++i)
        EXPECT_EQ(et[i], gtSearch(cliques, queries[i])) << "Failed on batched query index " << i;
}

TEST(SteinerKCoreGTest, testEmptyQueries) {
    std::vector<count> cliques{3, 4, 3};
    GraphW g = getCliqueChain(cliques);
    std::vector<count> coreness = coreDecomp(cliques);
    SteinerKCore steiner(g, coreness);
    EXPECT_TRUE(steiner.communities({}).communities.empty());
    EXPECT_TRUE(steiner.expandOneCommunity(std::set<node>{}).empty());
}

TEST(SteinerKCoreGTest, testInvalidSeeds) {
    GraphW g = getCliqueChain({3, 4, 3});
    SteinerKCore steiner(g);
    EXPECT_THROW(steiner.expandOneCommunity(std::set<node>{g.upperNodeIdBound()}),
                 std::out_of_range);
}

TEST(SteinerKCoreGTest, testDisconnectedSeeds) {
    GraphW g(8, false, false);
    for (node start : {0, 4})
        for (node u = start; u < start + 4; ++u)
            for (node v = u + 1; v < start + 4; ++v)
                g.addEdge(u, v);
    SteinerKCore steiner(g);
    const SteinerKCore::Communities c = steiner.communities({{0, 1}, {0, 4}, {5}});
    std::vector<std::set<node>> et = asSets(c);
    EXPECT_EQ(et[0], (std::set<node>{0, 1, 2, 3}));
    EXPECT_TRUE(et[1].empty());
    EXPECT_EQ(et[2], (std::set<node>{4, 5, 6, 7}));
    EXPECT_EQ(c.coreness, (std::vector<count>{3, none, 3}));
}

// ShellStruct answers the same queries; isolated seeds are skipped since ShellStruct places them
// in its coreness-0 root, which holds every vertex.
TEST(SteinerKCoreGTest, testMatchesShellStruct) {
    Aux::Random::setSeed(42, false);
    GraphW g = HyperbolicGenerator(2000, 16, 3).generate();
    ShellStruct shell(g);
    shell.build();
    ConnectedComponents cc(g);
    cc.run();
    SteinerKCore steiner(g);

    std::vector<std::set<node>> queries;
    for (index i = 0; i < 400; ++i) {
        std::set<node> q;
        while (q.size() < 1 + i % 4)
            q.insert(Aux::Random::integer(g.upperNodeIdBound() - 1));
        queries.push_back(q);
    }
    const SteinerKCore::Communities c = steiner.communities(queries);
    std::vector<std::set<node>> et = asSets(c);
    ASSERT_EQ(et.size(), queries.size());
    for (index i = 0; i < queries.size(); ++i) {
        const std::set<node> &q = queries[i];
        bool isolated = false, connected = true;
        for (node u : q) {
            isolated |= g.degree(u) == 0;
            connected &= cc.componentOfNode(u) == cc.componentOfNode(*q.begin());
        }
        if (isolated)
            continue;
        if (connected) {
            EXPECT_EQ(et[i], shell.expandOneCommunity(q)) << "Failed on query index " << i;
            EXPECT_EQ(c.coreness[i], shell.score(q)) << "Failed on query index " << i;
        } else {
            EXPECT_TRUE(et[i].empty()) << "Failed on disconnected query index " << i;
            EXPECT_EQ(c.coreness[i], none) << "Failed on disconnected query index " << i;
        }
    }
}

} /* namespace NetworKit */
