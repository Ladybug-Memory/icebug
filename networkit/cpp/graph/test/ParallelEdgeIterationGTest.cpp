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
#include <networkit/graph/GraphR.hpp>

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

    GraphR G(n, false, indicesArrow, indptrArrow, indicesArrow, indptrArrow);

    EXPECT_EQ(G.numberOfNodes(), 3);
    EXPECT_EQ(G.numberOfEdges(), 6); // Undirected stores both directions

    // Test parallelForEdges - count edges
    std::atomic<count> edgeCount{0};
    G.parallelForEdges([&]([[maybe_unused]] node u, [[maybe_unused]] node v) { edgeCount++; });

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

    GraphR G(n, true, outIndicesArrow, outIndptrArrow, inIndicesArrow, inIndptrArrow);

    EXPECT_EQ(G.numberOfNodes(), 3);
    EXPECT_EQ(G.numberOfEdges(), 3);
    EXPECT_TRUE(G.isDirected());

    // Test parallelForEdges
    std::atomic<count> edgeCount{0};
    G.parallelForEdges([&]([[maybe_unused]] node u, [[maybe_unused]] node v) { edgeCount++; });

    EXPECT_EQ(edgeCount.load(), 3) << "parallelForEdges should iterate over 3 directed edges";

    // Test parallelSumForEdges
    double sumNodeIds =
        G.parallelSumForEdges([&](node u, node v) { return static_cast<double>(u + v); });

    // Expected: (0+1) + (1+2) + (2+0) = 6
    EXPECT_EQ(sumNodeIds, 6.0) << "Sum of node IDs should be 6";
}

TEST_F(ParallelEdgeIterationGTest, DISABLED_testParallelForEdgesLargeGraph) {
    // Create a larger graph to test parallelism with ~1M edges
    const count n = 10000;
    std::vector<uint64_t> indices;
    std::vector<uint64_t> indptr(n + 1);

    // Create a dense-ish random graph
    // Each node connects to ~100 random other nodes
    const int avg_degree = 100;
    indices.reserve(n * avg_degree);

    srand(42); // Deterministic seed
    for (count u = 0; u < n; u++) {
        indptr[u] = indices.size();
        for (int k = 0; k < avg_degree; k++) {
            count v = (u + 1 + (rand() % (n - 1))) % n;
            if (v >= u)
                v = (v + 1) % n;
            indices.push_back(v);
        }
    }
    indptr[n] = indices.size();

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

    GraphR G(n, false, indicesArrow, indptrArrow, indicesArrow, indptrArrow);

    EXPECT_EQ(G.numberOfNodes(), n);

    // Run multiple times to catch race conditions
    for (int run = 0; run < 10; run++) {
        std::atomic<count> edgeCount{0};
        G.parallelForEdges([&]([[maybe_unused]] node u, [[maybe_unused]] node v) { edgeCount++; });

        EXPECT_EQ(edgeCount.load(), 500142) << "Run " << run << " should count correct edges";

        // Test parallelSumForEdges
        double total = G.parallelSumForEdges(
            [&]([[maybe_unused]] node u, [[maybe_unused]] node v) { return 1.0; });

        EXPECT_EQ(total, 500142.0) << "Run " << run << " should sum correctly";
    }
}

TEST_F(ParallelEdgeIterationGTest, testPageRankStyleIteration) {
    // Test balancedParallelForNodes + forInEdgesOf pattern used by PageRank
    // Just verify it doesn't hang and produces consistent results
    const count n = 5000;
    std::vector<uint64_t> indices;
    std::vector<uint64_t> indptr(n + 1);

    // Create a ring plus some random edges
    const int extra_edges = 5;
    indices.reserve(n * (2 + extra_edges));

    srand(42); // Deterministic seed
    for (count u = 0; u < n; u++) {
        indptr[u] = indices.size();
        // Ring edges
        indices.push_back((u + 1) % n);
        indices.push_back((u + n - 1) % n);
        // Random edges
        for (int k = 0; k < extra_edges; k++) {
            count v = rand() % n;
            if (v != u)
                indices.push_back(v);
        }
    }
    indptr[n] = indices.size();

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

    GraphR G(n, false, indicesArrow, indptrArrow, indicesArrow, indptrArrow);

    // Run the pattern many times to ensure no hangs
    for (int iteration = 0; iteration < 50; iteration++) {
        std::atomic<bool> success{true};
        G.balancedParallelForNodes([&](const node u) {
            int count = 0;
            G.forInEdgesOf(u, [&](const node, const node, const edgeweight) { count++; });
            if (count == 0)
                success = false;
        });

        EXPECT_TRUE(success) << "Iteration " << iteration << " should process all nodes";
    }
}

} // namespace NetworKit
