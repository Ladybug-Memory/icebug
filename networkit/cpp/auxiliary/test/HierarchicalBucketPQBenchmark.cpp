/*
 * HierarchicalBucketPQBenchmark.cpp
 *
 *  Created on: 26.10.2026
 *      Author: Ian Chen (ianchen3@illinois.edu)
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <omp.h>
#include <random>
#include <span>
#include <string>
#include <vector>

#include <networkit/auxiliary/BucketPQ.hpp>
#include <networkit/auxiliary/HierarchicalBucketPQ.hpp>
#include <networkit/auxiliary/Timer.hpp>

namespace NetworKit {

namespace {

using PQ = Aux::HierarchicalBucketPQ;

constexpr count numValues = count{1} << 22;
constexpr count maxInitialKey = 1000;
constexpr count decrementsPerValue = 8;

std::vector<int64_t> initialKeys() {
    std::mt19937_64 rng(1);
    std::vector<int64_t> keys(numValues);
    for (auto &key : keys)
        key = static_cast<int64_t>(rng() % (maxInitialKey + 1));
    return keys;
}

PQ::Value target(PQ::Value value, count j, count targetRange) {
    return static_cast<PQ::Value>((value * 0x9E3779B97F4A7C15ull + j * 0xC2B2AE3D27D4EB4Full)
                                  % targetRange);
}

void report(const std::string &name, double buildMs, double traceMs, count rounds) {
    std::cout << name << ": build " << buildMs << " ms, trace " << traceMs << " ms, " << rounds
              << " extractMin calls\n";
}

/**
 * Synthetic round trace: repeatedly extract the minimum batch, then every extracted value lowers
 * the keys of pseudo-random values in [0, targetRange) by one. Updates run in parallel.
 * @return number of extractMin calls
 */
count runTrace(PQ &queue, count targetRange) {
    count rounds = 0;
    while (!queue.empty()) {
        const std::span<const PQ::Value> batch = queue.extractMin().second;
        ++rounds;
#pragma omp parallel for
        for (omp_index i = 0; i < static_cast<omp_index>(batch.size()); ++i)
            for (count j = 0; j < decrementsPerValue; ++j)
                queue.decreaseKeyRelative(1, target(batch[i], j, targetRange),
                                          omp_get_thread_num());
    }
    return rounds;
}

/**
 * The same trace on Aux::BucketPQ, serially, with keys kept at or above the last extracted key.
 * @return number of extracted batches
 */
count runTrace(Aux::BucketPQ &queue, count targetRange) {
    count rounds = 0;
    std::vector<index> batch;
    while (!queue.empty()) {
        const int64_t key = queue.getMin().first;
        batch.clear();
        while (!queue.empty() && queue.getMin().first == key)
            batch.push_back(queue.extractMin().second);
        ++rounds;
        for (const index value : batch)
            for (count j = 0; j < decrementsPerValue; ++j) {
                const index u = target(static_cast<PQ::Value>(value), j, targetRange);
                if (queue.contains(u))
                    queue.changeKey(std::max(queue.getKey(u) - 1, key), u);
            }
    }
    return rounds;
}

void benchmarkHierarchical(const std::string &name, count targetRange) {
    const auto keys = initialKeys();
    Aux::Timer timer;
    timer.start();
    PQ queue(numValues, omp_get_max_threads());
#pragma omp parallel for
    for (omp_index value = 0; value < static_cast<omp_index>(numValues); ++value)
        queue.insert(static_cast<PQ::Key>(keys[value]), static_cast<PQ::Value>(value),
                     omp_get_thread_num());
    timer.stop();
    const double buildMs = timer.elapsedMilliseconds();

    timer.start();
    const count rounds = runTrace(queue, targetRange);
    timer.stop();
    report(name + " (" + std::to_string(omp_get_max_threads()) + " threads)", buildMs,
           timer.elapsedMilliseconds(), rounds);
}

} // namespace

class HierarchicalBucketPQBenchmark : public testing::Test {};

/**
 * Time of the round trace on Aux::BucketPQ, the serial baseline.
 */
TEST_F(HierarchicalBucketPQBenchmark, benchmarkTraceBucketPQ) {
    Aux::Timer timer;
    timer.start();
    Aux::BucketPQ queue(initialKeys(), 0, static_cast<int64_t>(maxInitialKey));
    timer.stop();
    const double buildMs = timer.elapsedMilliseconds();

    timer.start();
    const count rounds = runTrace(queue, numValues);
    timer.stop();
    report("BucketPQ", buildMs, timer.elapsedMilliseconds(), rounds);
}

/**
 * Time of the round trace on HierarchicalBucketPQ with one thread.
 */
TEST_F(HierarchicalBucketPQBenchmark, benchmarkTraceSerial) {
    const int threads = omp_get_max_threads();
    omp_set_num_threads(1);
    benchmarkHierarchical("HierarchicalBucketPQ", numValues);
    omp_set_num_threads(threads);
}

/**
 * Time of the round trace on HierarchicalBucketPQ at the environment's thread count.
 */
TEST_F(HierarchicalBucketPQBenchmark, benchmarkTraceParallel) {
    benchmarkHierarchical("HierarchicalBucketPQ", numValues);
}

/**
 * Time of the round trace on HierarchicalBucketPQ at the environment's thread count with the
 * decrements concentrated on few values.
 */
TEST_F(HierarchicalBucketPQBenchmark, benchmarkTraceContended) {
    benchmarkHierarchical("HierarchicalBucketPQ contended", 1024);
}

} // namespace NetworKit
