/*
 * ParallelEdgeIterationGTest.cpp
 *
 * Test for parallel edge iteration on CSR graphs
 */

#include <atomic>
#include <arrow/api.h>
#include <arrow/array.h>
#include <arrow/buffer.h>
#include <gtest/gtest.h>
#include <networkit/graph/Graph.hpp>

namespace NetworKit {

class ParallelEdgeIterationGTest : public testing::Test {};

TEST_F(ParallelEdgeIterationGTest, testParallelForEdgesCSR) {
    // Create a simple undirected CSR graph: 0-1, 1-2, 2-0
    const count n = 3;

    // For undirected graph, we need both directions in CSR format
    std::vector<uint64_t> indices = {1, 2, 0, 2, 1, 0}; // edges: 0->1,2  1->0,2  2->1,0
    std::vector<uint64_t> indptr = {0, 2, 4, 6};

    // Create Arrow arrays using Arrow's API properly
    arrow::UInt64Builder indicesBuilder;
    ASSERT_TRUE(indicesBuilder.AppendValues(indices).ok());
    std::shared_ptr<arrow::Array> indicesArray;
    ASSERT_TRUE(indicesBuilder.Finish(&indicesArray).ok());
    auto indicesArrow = std::static_pointer_cast<arrow::UInt64Array>(indicesArray);

    arrow::UInt64Builder indptrBuilder;
    ASSERT_TRUE(indptrBuilder.AppendValues(indptr).ok());
    std::shared_ptr<arrow::Array> indptrArray;
    ASSERT_TRUE(indptrBuilder.Finish(&indptrArray).ok());
    auto indptrArrow = std::static_pointer_cast<arrow::UInt64Array>(indptrArray);

    Graph G(n, false, indicesArrow, indptrArrow, indicesArrow, indptrArrow);

    EXPECT_EQ(G.numberOfNodes(), 3);
    EXPECT_EQ(G.numberOfEdges(), 6); // Undirected stores both directions

    // Test parallelForEdges - count edges
    std::atomic<count> edgeCount{0};
    G.parallelForEdges([&](node u, node v) { edgeCount++; });

    EXPECT_EQ(edgeCount.load(), 3) << "parallelForEdges should iterate over 3 edges (undirected)";

    // Test parallelSumForEdges - sum node IDs
    double sumNodeIds =
        G.parallelSumForEdges([&](node u, node v) { return static_cast<double>(u + v); });

    // Expected: (0+1) + (1+2) + (0+2) = 6
    EXPECT_EQ(sumNodeIds, 6.0) << "Sum of node IDs should be 6";
}

TEST_F(ParallelEdgeIterationGTest, testParallelForEdgesDirectedCSR) {
    // Create a simple directed CSR graph: 0->1, 1->2, 2->0
    const count n = 3;

    std::vector<uint64_t> out_indices = {1, 2, 0}; // edges: 0->1, 1->2, 2->0
    std::vector<uint64_t> out_indptr = {0, 1, 2, 3};
    std::vector<uint64_t> in_indices = {2, 0, 1}; // incoming: 0<-2, 1<-0, 2<-1
    std::vector<uint64_t> in_indptr = {0, 1, 2, 3};

    arrow::UInt64Builder outIndicesBuilder;
    ASSERT_TRUE(outIndicesBuilder.AppendValues(out_indices).ok());
    std::shared_ptr<arrow::Array> outIndicesArray;
    ASSERT_TRUE(outIndicesBuilder.Finish(&outIndicesArray).ok());
    auto outIndicesArrow = std::static_pointer_cast<arrow::UInt64Array>(outIndicesArray);

    arrow::UInt64Builder outIndptrBuilder;
    ASSERT_TRUE(outIndptrBuilder.AppendValues(out_indptr).ok());
    std::shared_ptr<arrow::Array> outIndptrArray;
    ASSERT_TRUE(outIndptrBuilder.Finish(&outIndptrArray).ok());
    auto outIndptrArrow = std::static_pointer_cast<arrow::UInt64Array>(outIndptrArray);

    arrow::UInt64Builder inIndicesBuilder;
    ASSERT_TRUE(inIndicesBuilder.AppendValues(in_indices).ok());
    std::shared_ptr<arrow::Array> inIndicesArray;
    ASSERT_TRUE(inIndicesBuilder.Finish(&inIndicesArray).ok());
    auto inIndicesArrow = std::static_pointer_cast<arrow::UInt64Array>(inIndicesArray);

    arrow::UInt64Builder inIndptrBuilder;
    ASSERT_TRUE(inIndptrBuilder.AppendValues(in_indptr).ok());
    std::shared_ptr<arrow::Array> inIndptrArray;
    ASSERT_TRUE(inIndptrBuilder.Finish(&inIndptrArray).ok());
    auto inIndptrArrow = std::static_pointer_cast<arrow::UInt64Array>(inIndptrArray);

    Graph G(n, true, outIndicesArrow, outIndptrArrow, inIndicesArrow, inIndptrArrow);

    EXPECT_EQ(G.numberOfNodes(), 3);
    EXPECT_EQ(G.numberOfEdges(), 3);
    EXPECT_TRUE(G.isDirected());

    // Test parallelForEdges
    std::atomic<count> edgeCount{0};
    G.parallelForEdges([&](node u, node v) { edgeCount++; });

    EXPECT_EQ(edgeCount.load(), 3) << "parallelForEdges should iterate over 3 directed edges";

    // Test parallelSumForEdges
    double sumNodeIds =
        G.parallelSumForEdges([&](node u, node v) { return static_cast<double>(u + v); });

    // Expected: (0+1) + (1+2) + (2+0) = 6
    EXPECT_EQ(sumNodeIds, 6.0) << "Sum of node IDs should be 6";
}

TEST_F(ParallelEdgeIterationGTest, testParallelForEdgesLargerGraph) {
    // Create a larger graph to test parallelism with multiple runs
    const count n = 1000;
    std::vector<uint64_t> indices;
    std::vector<uint64_t> indptr = {0};

    // Create a ring graph: 0-1-2-...-999-0
    for (count u = 0; u < n; u++) {
        count next = (u + 1) % n;
        count prev = (u + n - 1) % n;

        // For undirected, add both neighbors (maintaining sorted order)
        if (prev < u || (prev == n - 1 && u == 0)) {
            indices.push_back(prev);
        }
        if (next > u || (next == 0 && u == n - 1)) {
            indices.push_back(next);
        }

        indptr.push_back(indices.size());
    }

    arrow::UInt64Builder indicesBuilder;
    ASSERT_TRUE(indicesBuilder.AppendValues(indices).ok());
    std::shared_ptr<arrow::Array> indicesArray;
    ASSERT_TRUE(indicesBuilder.Finish(&indicesArray).ok());
    auto indicesArrow = std::static_pointer_cast<arrow::UInt64Array>(indicesArray);

    arrow::UInt64Builder indptrBuilder;
    ASSERT_TRUE(indptrBuilder.AppendValues(indptr).ok());
    std::shared_ptr<arrow::Array> indptrArray;
    ASSERT_TRUE(indptrBuilder.Finish(&indptrArray).ok());
    auto indptrArrow = std::static_pointer_cast<arrow::UInt64Array>(indptrArray);

    Graph G(n, false, indicesArrow, indptrArrow, indicesArrow, indptrArrow);

    EXPECT_EQ(G.numberOfNodes(), n);
    EXPECT_EQ(G.numberOfEdges(), n * 2); // Undirected stores both directions

    // Run multiple times to catch race conditions
    for (int run = 0; run < 20; run++) {
        // Test parallelForEdges with atomic counter
        std::atomic<count> edgeCount{0};
        G.parallelForEdges([&](node u, node v) { edgeCount++; });

        EXPECT_EQ(edgeCount.load(), n)
            << "parallelForEdges should iterate over all edges (run " << run << ")";

        // Test parallelSumForEdges
        double totalDegree = G.parallelSumForEdges([&](node u, node v) { return 1.0; });

        EXPECT_EQ(totalDegree, static_cast<double>(n))
            << "Sum should count all edges (run " << run << ")";
    }
}

TEST_F(ParallelEdgeIterationGTest, testParallelForInEdgesOf) {
    // Test forInEdgesOf which is used by PageRank within balancedParallelForNodes
    const count n = 100;
    std::vector<uint64_t> indices;
    std::vector<uint64_t> indptr = {0};

    // Create a ring graph
    for (count u = 0; u < n; u++) {
        count next = (u + 1) % n;
        count prev = (u + n - 1) % n;

        if (prev < u || (prev == n - 1 && u == 0)) {
            indices.push_back(prev);
        }
        if (next > u || (next == 0 && u == n - 1)) {
            indices.push_back(next);
        }

        indptr.push_back(indices.size());
    }

    arrow::UInt64Builder indicesBuilder;
    ASSERT_TRUE(indicesBuilder.AppendValues(indices).ok());
    std::shared_ptr<arrow::Array> indicesArray;
    ASSERT_TRUE(indicesBuilder.Finish(&indicesArray).ok());
    auto indicesArrow = std::static_pointer_cast<arrow::UInt64Array>(indicesArray);

    arrow::UInt64Builder indptrBuilder;
    ASSERT_TRUE(indptrBuilder.AppendValues(indptr).ok());
    std::shared_ptr<arrow::Array> indptrArray;
    ASSERT_TRUE(indptrBuilder.Finish(&indptrArray).ok());
    auto indptrArrow = std::static_pointer_cast<arrow::UInt64Array>(indptrArray);

    Graph G(n, false, indicesArrow, indptrArrow, indicesArrow, indptrArrow);

    // Simulate PageRank's usage: balancedParallelForNodes calling forInEdgesOf
    for (int run = 0; run < 10; run++) {
        std::vector<double> pr(n, 0.0);
        std::vector<double> scoreData(n, 1.0 / n);

        G.balancedParallelForNodes([&](const node u) {
            pr[u] = 0.0;
            G.forInEdgesOf(u, [&](const node u, const node v, const edgeweight w) {
                pr[u] += scoreData[v] / 2.0; // divide by degree
            });
        });

        // Check that all nodes were processed
        double sum = 0.0;
        for (count u = 0; u < n; u++) {
            sum += pr[u];
        }
        EXPECT_GT(sum, 0.0) << "PageRank-like iteration should compute non-zero values (run " << run
                            << ")";
    }
}

} // namespace NetworKit
