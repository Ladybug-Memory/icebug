/*
 * HierarchicalBucketPQBenchmark.cpp
 *
 *  Created on: 26.10.2026
 *      Author: Ian Chen (ianchen3@illinois.edu)
 */

#include <gtest/gtest.h>

#include <networkit/auxiliary/BucketPQ.hpp>
#include <networkit/auxiliary/HierarchicalBucketPQ.hpp>
#include <networkit/auxiliary/Timer.hpp>

namespace NetworKit {

namespace {

/**
 * Synthetic round trace: repeatedly extract the minimum batch, then every extracted value lowers
 * the keys of pseudo-random values in [0, targetRange) by one. Updates run in parallel on
 * HierarchicalBucketPQ and serially on BucketPQ.
 * @return number of extractMin calls
 */
template <typename Queue>
count runTrace(Queue &queue, count decrementsPerValue, count targetRange);

} // namespace

class HierarchicalBucketPQBenchmark : public testing::Test {};

/**
 * Time of the round trace on Aux::BucketPQ, the serial baseline.
 */
TEST_F(HierarchicalBucketPQBenchmark, benchmarkTraceBucketPQ) {
    GTEST_SKIP();
}

/**
 * Time of the round trace on HierarchicalBucketPQ with one thread.
 */
TEST_F(HierarchicalBucketPQBenchmark, benchmarkTraceSerial) {
    GTEST_SKIP();
}

/**
 * Time of the round trace on HierarchicalBucketPQ at the environment's thread count.
 */
TEST_F(HierarchicalBucketPQBenchmark, benchmarkTraceParallel) {
    GTEST_SKIP();
}

/**
 * Time of the round trace on HierarchicalBucketPQ at the environment's thread count with the
 * decrements concentrated on few values.
 */
TEST_F(HierarchicalBucketPQBenchmark, benchmarkTraceContended) {
    GTEST_SKIP();
}

} // namespace NetworKit
