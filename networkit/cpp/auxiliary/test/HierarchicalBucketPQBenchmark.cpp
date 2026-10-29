/*
 * HierarchicalBucketPQBenchmark.cpp
 *
 *  Created on: 26.10.2026
 *      Author: Ian Chen (ianchen3@illinois.edu)
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <span>
#include <string>
#include <vector>

#include <networkit/auxiliary/BucketPQ.hpp>
#include <networkit/auxiliary/HierarchicalBucketPQ.hpp>
#include <networkit/auxiliary/Random.hpp>
#include <networkit/auxiliary/Timer.hpp>
#include <networkit/generators/PowerlawDegreeSequence.hpp>

namespace NetworKit {

namespace {

// Two workloads: uniform or power-law Zipf keys/targets (see below).
// 1. Every value starts with its initial key.
// 2. A round extracts every value with the minimum key k.
// 3. Each extracted value v makes decrementsPerValue updates. Each update decreases the key of a
//    target (random with larger initial key) by one, clamped at k. No-op if target not in queue.
// 4. Rounds repeat until the queue is empty.

using HBS = Aux::HierarchicalBucketPQ;

constexpr count numValues = count{1} << 22;
constexpr count maxUniformKey = 1000;
constexpr count maxZipfKey = count{1} << 20;
constexpr double zipfExponent = 2.0;
constexpr count decrementsPerValue = 8;
constexpr uint64_t traceSeed = 1;

std::vector<int64_t> uniformKeys(count maxKey, uint64_t seed) {
    Aux::Random::setSeed(seed, false);
    std::vector<int64_t> keys(numValues);
    for (auto &key : keys)
        key = static_cast<int64_t>(Aux::Random::integer(maxKey));
    return keys;
}

std::vector<int64_t> zipfKeys(count maxKey, double exponent, uint64_t seed) {
    Aux::Random::setSeed(seed, false);
    PowerlawDegreeSequence sequence(1, maxKey, -exponent);
    sequence.run();
    const std::vector<count> degrees = sequence.getDegreeSequence(numValues);
    return {degrees.begin(), degrees.end()};
}

// Values in increasing order of initial key.
std::vector<HBS::Value> sortByKey(const std::vector<int64_t> &keys) {
    std::vector<HBS::Value> order(numValues);
    std::iota(order.begin(), order.end(), 0);
    std::ranges::stable_sort(order, {}, [&](HBS::Value value) { return keys[value]; });
    return order;
}

/**
 * Draw the targets of step 3 of the trace uniformly among the values with a larger initial key.
 * @return targets, where the j-th update of value v goes to targets[v * decrementsPerValue + j]
 */
std::vector<HBS::Value> uniformTargets(const std::vector<int64_t> &keys, uint64_t seed) {
    Aux::Random::setSeed(seed, false);
    const std::vector<HBS::Value> order = sortByKey(keys);
    std::vector<int64_t> sortedKeys(numValues);
    for (index i = 0; i < numValues; ++i)
        sortedKeys[i] = keys[order[i]];

    std::vector<HBS::Value> targets(numValues * decrementsPerValue);
    for (index value = 0; value < numValues; ++value) {
        const index above = std::ranges::upper_bound(sortedKeys, keys[value]) - sortedKeys.begin();
        for (count j = 0; j < decrementsPerValue; ++j)
            // With no larger key, the target is the value itself, which is extracted: a no-op.
            targets[value * decrementsPerValue + j] =
                above < numValues ? order[Aux::Random::integer(above, numValues - 1)]
                                  : static_cast<HBS::Value>(value);
    }
    return targets;
}

/**
 * Draw the targets of step 3 of the trace among the values with a larger initial key, with
 * probability proportional to initial key.
 * @return targets, where the j-th update of value v goes to targets[v * decrementsPerValue + j]
 */
std::vector<HBS::Value> weightedTargets(const std::vector<int64_t> &keys, uint64_t seed) {
    Aux::Random::setSeed(seed, false);
    const std::vector<HBS::Value> order = sortByKey(keys);
    std::vector<int64_t> sortedKeys(numValues);
    for (index i = 0; i < numValues; ++i)
        sortedKeys[i] = keys[order[i]];
    // prefix[i] is the total initial key of order[0..i).
    std::vector<uint64_t> prefix(numValues + 1, 0);
    for (index i = 0; i < numValues; ++i)
        prefix[i + 1] = prefix[i] + static_cast<uint64_t>(sortedKeys[i]);

    std::vector<HBS::Value> targets(numValues * decrementsPerValue);
    for (index value = 0; value < numValues; ++value) {
        const index above = std::ranges::upper_bound(sortedKeys, keys[value]) - sortedKeys.begin();
        for (count j = 0; j < decrementsPerValue; ++j) {
            // With no larger key, the target is the value itself, which is extracted: a no-op.
            HBS::Value target = static_cast<HBS::Value>(value);
            if (above < numValues) {
                const uint64_t point = Aux::Random::integer(prefix[above], prefix[numValues] - 1);
                target = order[std::ranges::upper_bound(prefix, point) - prefix.begin() - 1];
            }
            targets[value * decrementsPerValue + j] = target;
        }
    }
    return targets;
}

void report(const std::string &name, double buildMs, double traceMs, count rounds) {
    std::cout << name << ": build " << buildMs << " ms, trace " << traceMs << " ms, " << rounds
              << " rounds\n";
}

/**
 * Run the trace on @p queue with updates in parallel.
 * @param targets the j-th update of value v goes to targets[v * decrementsPerValue + j]
 * @return number of rounds
 */
count runTraceHBS(HBS &queue, const std::vector<HBS::Value> &targets) {
    count rounds = 0;
    while (!queue.empty()) {
        const std::span<const HBS::Value> batch = queue.extractMin().second;
        ++rounds;
#pragma omp parallel for
        for (omp_index i = 0; i < static_cast<omp_index>(batch.size()); ++i)
            for (count j = 0; j < decrementsPerValue; ++j)
                queue.decreaseKeyRelative(1, targets[batch[i] * decrementsPerValue + j],
                                          omp_get_thread_num());
    }
    return rounds;
}

/**
 * Run the trace on @p queue serially.
 * @param targets the j-th update of value v goes to targets[v * decrementsPerValue + j]
 * @return number of rounds
 */
count runTraceBucketPQ(Aux::BucketPQ &queue, const std::vector<HBS::Value> &targets) {
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
                const index u = targets[value * decrementsPerValue + j];
                if (queue.contains(u))
                    queue.changeKey(std::max(queue.getKey(u) - 1, key), u);
            }
    }
    return rounds;
}

void benchmarkBucketPQ(const std::string &name, const std::vector<int64_t> &keys,
                       const std::vector<HBS::Value> &targets, count maxKey) {
    Aux::Timer timer;
    timer.start();
    Aux::BucketPQ queue(keys, 0, static_cast<int64_t>(maxKey));
    timer.stop();
    const double buildMs = timer.elapsedMilliseconds();

    timer.start();
    const count rounds = runTraceBucketPQ(queue, targets);
    timer.stop();
    report(name, buildMs, timer.elapsedMilliseconds(), rounds);
}

void benchmarkHierarchical(const std::string &name, const std::vector<int64_t> &keys,
                           const std::vector<HBS::Value> &targets, count log2SingleBuckets = 3,
                           count numIntermediateBuckets = 6) {
    Aux::Timer timer;
    timer.start();
    HBS queue(numValues, omp_get_max_threads(), log2SingleBuckets, numIntermediateBuckets);
#pragma omp parallel for
    for (omp_index value = 0; value < static_cast<omp_index>(numValues); ++value)
        queue.insert(static_cast<HBS::Key>(keys[value]), static_cast<HBS::Value>(value),
                     omp_get_thread_num());
    timer.stop();
    const double buildMs = timer.elapsedMilliseconds();

    timer.start();
    const count rounds = runTraceHBS(queue, targets);
    timer.stop();
    report(name + " (" + std::to_string(omp_get_max_threads()) + " threads)", buildMs,
           timer.elapsedMilliseconds(), rounds);
}

} // namespace

class HierarchicalBucketPQBenchmark : public testing::Test {};

/**
 * Time of the trace on Aux::BucketPQ, the serial baseline.
 */
TEST_F(HierarchicalBucketPQBenchmark, benchmarkTraceBucketPQ) {
    const auto keys = uniformKeys(maxUniformKey, traceSeed);
    benchmarkBucketPQ("BucketPQ", keys, uniformTargets(keys, traceSeed), maxUniformKey);
}

/**
 * Time of the trace on HierarchicalBucketPQ with one thread.
 */
TEST_F(HierarchicalBucketPQBenchmark, benchmarkTraceSerial) {
    const int threads = omp_get_max_threads();
    omp_set_num_threads(1);
    const auto keys = uniformKeys(maxUniformKey, traceSeed);
    benchmarkHierarchical("HierarchicalBucketPQ", keys, uniformTargets(keys, traceSeed));
    omp_set_num_threads(threads);
}

/**
 * Time of the trace on HierarchicalBucketPQ with one thread and a single bucket for every
 * key, as in Aux::BucketPQ.
 */
TEST_F(HierarchicalBucketPQBenchmark, benchmarkTraceSerialFlat) {
    const int threads = omp_get_max_threads();
    omp_set_num_threads(1);
    const auto keys = uniformKeys(maxUniformKey, traceSeed);
    benchmarkHierarchical("HierarchicalBucketPQ flat", keys, uniformTargets(keys, traceSeed),
                          std::bit_width(maxUniformKey), 1);
    omp_set_num_threads(threads);
}

/**
 * Time of the trace with Zipf-distributed initial keys on Aux::BucketPQ.
 */
TEST_F(HierarchicalBucketPQBenchmark, benchmarkTraceZipfBucketPQ) {
    const auto keys = zipfKeys(maxZipfKey, zipfExponent, traceSeed);
    benchmarkBucketPQ("BucketPQ zipf", keys, weightedTargets(keys, traceSeed), maxZipfKey);
}

/**
 * Time of the trace with Zipf-distributed initial keys on HierarchicalBucketPQ with one
 * thread.
 */
TEST_F(HierarchicalBucketPQBenchmark, benchmarkTraceZipfSerial) {
    const int threads = omp_get_max_threads();
    omp_set_num_threads(1);
    const auto keys = zipfKeys(maxZipfKey, zipfExponent, traceSeed);
    benchmarkHierarchical("HierarchicalBucketPQ zipf", keys, weightedTargets(keys, traceSeed));
    omp_set_num_threads(threads);
}

/**
 * Time of the trace with Zipf-distributed initial keys on HierarchicalBucketPQ with one
 * thread and a single bucket for every key.
 */
TEST_F(HierarchicalBucketPQBenchmark, benchmarkTraceZipfSerialFlat) {
    const int threads = omp_get_max_threads();
    omp_set_num_threads(1);
    const auto keys = zipfKeys(maxZipfKey, zipfExponent, traceSeed);
    benchmarkHierarchical("HierarchicalBucketPQ zipf flat", keys, weightedTargets(keys, traceSeed),
                          std::bit_width(maxZipfKey), 1);
    omp_set_num_threads(threads);
}

/**
 * Time of the trace on HierarchicalBucketPQ at the environment's thread count.
 */
TEST_F(HierarchicalBucketPQBenchmark, benchmarkTraceParallel) {
    const auto keys = uniformKeys(maxUniformKey, traceSeed);
    benchmarkHierarchical("HierarchicalBucketPQ", keys, uniformTargets(keys, traceSeed));
}

/**
 * Time of the trace with Zipf-distributed initial keys on HierarchicalBucketPQ at the
 * environment's thread count, where values with large keys receive many concurrent updates.
 */
TEST_F(HierarchicalBucketPQBenchmark, benchmarkTraceZipfParallel) {
    const auto keys = zipfKeys(maxZipfKey, zipfExponent, traceSeed);
    benchmarkHierarchical("HierarchicalBucketPQ zipf", keys, weightedTargets(keys, traceSeed));
}

} // namespace NetworKit
