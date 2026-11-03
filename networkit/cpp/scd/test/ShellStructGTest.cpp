/*
 * ShellStructGTest.cpp
 *
 *  Created on: 06.08.2026
 *      Author: Ian Chen (ianchen3@illinois.edu)
 */

#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>
#include <networkit/centrality/CoreDecomposition.hpp>
#include <networkit/graph/GraphW.hpp>
#include <networkit/scd/ShellStruct.hpp>

namespace NetworKit {

class ShellStructGTest : public ::testing::Test {
protected:
    GraphW getCliqueChain(const std::vector<count> &cliqueSizes) {
        std::cerr << "[shellstruct-debug] getCliqueChain: numCliques=" << cliqueSizes.size()
                  << std::endl;
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

    void verifySingletons(ShellStruct &shell, const std::vector<count> &cliques) {
        count n = std::accumulate(cliques.begin(), cliques.end(), count(0));
        for (node v = 0; v < n; ++v) {
            std::set<node> query = {v};
            std::set<node> et = shell.expandOneCommunity(query);
            std::set<node> gt = gtSearch(cliques, query);

            EXPECT_EQ(et, gt) << "Failed on singleton query node: " << v;
        }
    }

    void verifyPairs(ShellStruct &shell, const std::vector<count> &cliques) {
        count n = std::accumulate(cliques.begin(), cliques.end(), count(0));
        for (node u = 0; u < n; ++u) {
            for (node v = u + 1; v < n; ++v) {
                std::set<node> query = {u, v};
                std::set<node> et = shell.expandOneCommunity(query);
                std::set<node> gt = gtSearch(cliques, query);

                EXPECT_EQ(et, gt) << "Failed on pair nodes: {" << u << ", " << v << "}";
            }
        }
    }
};

TEST_F(ShellStructGTest, testSeparatedCoresSingleton) {
    std::vector<count> cliques{4, 3, 4};
    GraphW g = getCliqueChain(cliques);
    ShellStruct shell(g);
    shell.build();
    verifySingletons(shell, cliques);
}

TEST_F(ShellStructGTest, testSeparatedCoresPairs) {
    std::cerr << "[shellstruct-debug] TEST testSeparatedCoresPairs start" << std::endl;
    std::vector<count> cliques{4, 3, 4};
    GraphW g = getCliqueChain(cliques);
    ShellStruct shell(g);
    shell.build();
    verifyPairs(shell, cliques);
    std::cerr << "[shellstruct-debug] TEST testSeparatedCoresPairs end" << std::endl;
}

TEST_F(ShellStructGTest, testBridgedCoreSingleton) {
    std::cerr << "[shellstruct-debug] TEST testBridgedCoreSingleton start" << std::endl;
    std::vector<count> cliques{3, 4, 3};
    GraphW g = getCliqueChain(cliques);
    std::cerr << "[shellstruct-debug] TEST testBridgedCoreSingleton graph built" << std::endl;
    ShellStruct shell(g);
    std::cerr << "[shellstruct-debug] TEST testBridgedCoreSingleton shell ctor done" << std::endl;
    shell.build();
    std::cerr << "[shellstruct-debug] TEST testBridgedCoreSingleton build done" << std::endl;
    verifySingletons(shell, cliques);
    std::cerr << "[shellstruct-debug] TEST testBridgedCoreSingleton end" << std::endl;
}

TEST_F(ShellStructGTest, testBridgedCorePairs) {
    std::vector<count> cliques{3, 4, 3};
    GraphW g = getCliqueChain(cliques);
    ShellStruct shell(g);
    shell.build();
    verifyPairs(shell, cliques);
}

TEST_F(ShellStructGTest, testMediumSingleton) {
    std::vector<count> cliques{3, 5, 4, 3, 6, 5, 4, 6, 3};
    GraphW g = getCliqueChain(cliques);
    ShellStruct shell(g);
    shell.build();
    verifySingletons(shell, cliques);
}

TEST_F(ShellStructGTest, testMediumPairs) {
    std::vector<count> cliques{3, 5, 4, 3, 6, 5, 4, 6, 3};
    GraphW g = getCliqueChain(cliques);
    ShellStruct shell(g);
    shell.build();
    verifyPairs(shell, cliques);
}

TEST_F(ShellStructGTest, testLargeSingleton) {
    std::vector<count> cliques{3, 4, 5, 6, 5, 4, 3, 5, 7, 9, 11, 9, 7, 5, 3, 4, 5, 6, 4, 3};
    GraphW g = getCliqueChain(cliques);
    ShellStruct shell(g);
    shell.build();
    verifySingletons(shell, cliques);
}

TEST_F(ShellStructGTest, testLargePairs) {
    std::vector<count> cliques{3, 4, 5, 6, 5, 4, 3, 5, 7, 9, 11, 9, 7, 5, 3, 4, 5, 6, 4, 3};
    GraphW g = getCliqueChain(cliques);
    ShellStruct shell(g);
    shell.build();
    verifyPairs(shell, cliques);
}

TEST_F(ShellStructGTest, testLoadSave) {
    std::vector<count> cliques{3, 4, 5, 6, 5, 4, 3, 5, 7, 9, 11, 9, 7, 5, 3, 4, 5, 6, 4, 3};
    GraphW g = getCliqueChain(cliques);
    ShellStruct shell(g);
    shell.build();
    shell.save("output/shellstruct_comp.feather", "output/shellstruct_tree.feather");

    ShellStruct index(g);
    index.load("output/shellstruct_comp.feather", "output/shellstruct_tree.feather");
    verifyPairs(index, cliques);
}

TEST_F(ShellStructGTest, testLoadSaveValidate) {
    std::vector<count> cliques{3, 4, 5, 6, 5, 4, 3, 5, 7, 9, 11, 9, 7, 5, 3, 4, 5, 6, 4, 3};
    GraphW g = getCliqueChain(cliques);
    ShellStruct shell(g);
    shell.build();
    shell.save("output/shellstruct_comp.feather", "output/shellstruct_tree.feather");

    ShellStruct index(g, true);
    index.load("output/shellstruct_comp.feather", "output/shellstruct_tree.feather");
    verifyPairs(index, cliques);
}

TEST_F(ShellStructGTest, testLoadOtherGraphThrows) {
    GraphW g = getCliqueChain({3, 4, 3});
    ShellStruct shell(g);
    shell.build();
    shell.save("output/shellstruct_comp.feather", "output/shellstruct_tree.feather");

    GraphW other = getCliqueChain({3, 4, 4});
    ShellStruct index(other);
    EXPECT_THROW(index.load("output/shellstruct_comp.feather", "output/shellstruct_tree.feather"),
                 std::runtime_error);
}

TEST_F(ShellStructGTest, testEmptyGraph) {
    GraphW g(0, false, false);
    ShellStruct shell(g);
    shell.build();
    EXPECT_TRUE(shell.community({}).empty());
    shell.save("output/shellstruct_comp.feather", "output/shellstruct_tree.feather");

    ShellStruct index(g, true);
    index.load("output/shellstruct_comp.feather", "output/shellstruct_tree.feather");
    EXPECT_TRUE(index.expandOneCommunity(std::set<node>{}).empty());
}

TEST_F(ShellStructGTest, testInvalidSeeds) {
    GraphW g = getCliqueChain({3, 4, 3});
    ShellStruct shell(g);
    EXPECT_THROW(shell.community({0}), std::runtime_error);
    shell.build();
    EXPECT_THROW(shell.community({g.upperNodeIdBound()}), std::out_of_range);
    EXPECT_THROW(shell.score({}), std::invalid_argument);
    EXPECT_EQ(shell.score({0, 9}), count{2});
    EXPECT_EQ(shell.score({3, 4}), count{3});
}

// Enough vertices for the parallel path of the shell grouping.
TEST_F(ShellStructGTest, testManyCliquesSingleton) {
    std::vector<count> cliques(2500);
    for (count i = 0; i < cliques.size(); ++i)
        cliques[i] = 3 + i % 4;
    GraphW g = getCliqueChain(cliques);
    ShellStruct shell(g);
    shell.build();
    verifySingletons(shell, cliques);
}

TEST_F(ShellStructGTest, testBuildFromCoreDecomposition) {
    std::vector<count> cliques{3, 5, 4, 3, 6, 5, 4, 6, 3};
    GraphW g = getCliqueChain(cliques);
    CoreDecomposition coreDecomposition(g);
    coreDecomposition.run();
    const std::vector<double> &scores = coreDecomposition.scores();
    ShellStruct shell(g);
    shell.build(std::vector<count>(scores.begin(), scores.end()));
    verifyPairs(shell, cliques);
}

TEST_F(ShellStructGTest, testBuildWrongSizeThrows) {
    GraphW g = getCliqueChain({3, 4, 3});
    ShellStruct shell(g);
    EXPECT_THROW(shell.build(std::vector<count>(g.upperNodeIdBound() - 1)), std::invalid_argument);
}

// Same vertex count and tree shape, so only validate detects the mismatch.
TEST_F(ShellStructGTest, testLoadMismatchedValidateThrows) {
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

} /* namespace NetworKit */
