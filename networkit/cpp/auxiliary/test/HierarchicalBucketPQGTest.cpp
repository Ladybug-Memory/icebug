/*
 * HierarchicalBucketPQGTest.cpp
 *
 *  Created on: 26.10.2026
 *      Author: Ian Chen (ianchen3@illinois.edu)
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <omp.h>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include <networkit/auxiliary/BucketPQ.hpp>
#include <networkit/auxiliary/HierarchicalBucketPQ.hpp>

namespace NetworKit {

namespace {

using PQ = Aux::HierarchicalBucketPQ;
using Key = PQ::Key;
using Value = PQ::Value;

/**
 * Serial reference implementation of HierarchicalBucketPQ on Aux::BucketPQ, for the update kinds
 * whose outcome does not depend on order.
 */
class BucketPQOracle {
public:
    BucketPQOracle(count capacity, Key maxKey)
        : pq(capacity, 0, static_cast<int64_t>(maxKey)), keys(capacity, PQ::infiniteKey),
          extracted(capacity) {}

    bool insert(Key key, Value value) {
        if (extracted[value])
            return false;
        key = std::max(key, lastExtractedKey);
        if (pq.contains(value)) {
            lowerTo(key, value);
            return false;
        }
        pq.insert(static_cast<int64_t>(key), value);
        keys[value] = key;
        return true;
    }

    bool decreaseKey(Key newKey, Value value) {
        return pq.contains(value) && lowerTo(std::max(newKey, lastExtractedKey), value);
    }

    bool decreaseKeyRelative(Key delta, Value value) {
        return pq.contains(value)
               && lowerTo(keys[value] - std::min(delta, keys[value] - lastExtractedKey), value);
    }

    std::pair<Key, std::vector<Value>> extractMin() {
        lastExtractedKey = static_cast<Key>(pq.getMin().first);
        std::vector<Value> values;
        while (!pq.empty() && static_cast<Key>(pq.getMin().first) == lastExtractedKey) {
            const Value value = static_cast<Value>(pq.extractMin().second);
            extracted[value] = true;
            values.push_back(value);
        }
        return {lastExtractedKey, values};
    }

    Key getKey(Value value) const { return keys[value]; }

    bool empty() const { return pq.empty(); }

private:
    bool lowerTo(Key key, Value value) {
        if (key >= keys[value])
            return false;
        pq.changeKey(static_cast<int64_t>(key), value);
        keys[value] = key;
        return true;
    }

    Aux::BucketPQ pq;
    std::vector<Key> keys;
    std::vector<bool> extracted;
    Key lastExtractedKey = 0;
};

std::vector<Value> sorted(std::span<const Value> values) {
    std::vector<Value> result(values.begin(), values.end());
    std::ranges::sort(result);
    return result;
}

void expectSameExtraction(PQ &pq, BucketPQOracle &oracle) {
    const auto [key, values] = pq.extractMin();
    const auto [expectedKey, expectedValues] = oracle.extractMin();
    EXPECT_EQ(key, expectedKey);
    EXPECT_EQ(sorted(values), sorted(expectedValues));
}

} // namespace

class HierarchicalBucketPQGTest : public testing::Test {};

/**
 * Values sharing a key come out together in one batch, separate from values with other keys, and
 * batches come out in increasing key order until the queue is empty.
 */
TEST_F(HierarchicalBucketPQGTest, testExtractMin) {
    constexpr count n = 2000;
    std::mt19937_64 rng(1);
    std::uniform_int_distribution<Key> keyOf(0, 3000);
    std::vector<Key> keys(n);
    PQ pq(n);
    for (Value value = 0; value < n; ++value) {
        keys[value] = keyOf(rng);
        pq.insert(keys[value], value);
    }

    std::vector<Key> distinctKeys = keys;
    std::ranges::sort(distinctKeys);
    distinctKeys.erase(std::unique(distinctKeys.begin(), distinctKeys.end()), distinctKeys.end());
    for (const Key key : distinctKeys) {
        std::vector<Value> expected;
        for (Value value = 0; value < n; ++value)
            if (keys[value] == key)
                expected.push_back(value);
        const auto [extractedKey, values] = pq.extractMin();
        EXPECT_EQ(extractedKey, key);
        EXPECT_EQ(sorted(values), expected);
    }
    EXPECT_TRUE(pq.empty());
}

/**
 * After an extractMin, every update that would take a key below the extracted key leaves it at the
 * extracted key instead.
 */
TEST_F(HierarchicalBucketPQGTest, testFloor) {
    PQ pq(5);
    pq.insert(5, 0);
    pq.insert(10, 1);
    pq.insert(20, 2);
    pq.insert(100, 3);
    EXPECT_EQ(pq.extractMin().first, 5u);

    EXPECT_TRUE(pq.insert(2, 4));
    EXPECT_TRUE(pq.decreaseKey(1, 1));
    EXPECT_TRUE(pq.decreaseKeyRelative(100, 2));
    for (const Value value : {1, 2, 4})
        EXPECT_EQ(pq.getKey(value), 5u);

    const auto [key, values] = pq.extractMin();
    EXPECT_EQ(key, 5u);
    EXPECT_EQ(sorted(values), (std::vector<Value>{1, 2, 4}));
}

/**
 * Only the first insert of a value reports success; later inserts lower its key but never raise
 * it, and an insert after extraction changes nothing.
 */
TEST_F(HierarchicalBucketPQGTest, testInsert) {
    PQ pq(1);
    EXPECT_TRUE(pq.insert(10, 0));
    EXPECT_FALSE(pq.insert(20, 0));
    EXPECT_EQ(pq.getKey(0), 10u);
    EXPECT_FALSE(pq.insert(5, 0));
    EXPECT_EQ(pq.getKey(0), 5u);

    pq.extractMin();
    EXPECT_FALSE(pq.insert(1, 0));
    EXPECT_FALSE(pq.contains(0));
    EXPECT_EQ(pq.getKey(0), 5u);
    EXPECT_TRUE(pq.empty());
}

/**
 * Decreases on a value never inserted or already extracted report no change and leave its key as
 * it was, as does a decrease to a larger key.
 */
TEST_F(HierarchicalBucketPQGTest, testDecreaseNotPresent) {
    PQ pq(2);
    EXPECT_FALSE(pq.decreaseKey(3, 0));
    EXPECT_FALSE(pq.decreaseKeyRelative(3, 0));
    EXPECT_EQ(pq.getKey(0), PQ::infiniteKey);

    pq.insert(10, 0);
    EXPECT_FALSE(pq.decreaseKey(20, 0));
    EXPECT_FALSE(pq.decreaseKeyRelative(0, 0));
    EXPECT_EQ(pq.getKey(0), 10u);

    pq.extractMin();
    EXPECT_FALSE(pq.decreaseKey(1, 0));
    EXPECT_FALSE(pq.decreaseKeyRelative(1, 0));
    EXPECT_EQ(pq.getKey(0), 10u);
}

/**
 * tryExtract succeeds only for a value at the key of the last extractMin and only once, and that
 * value is not returned by any later extractMin.
 */
TEST_F(HierarchicalBucketPQGTest, testTryExtract) {
    PQ pq(4);
    pq.insert(5, 0);
    pq.insert(7, 1);
    pq.insert(9, 2);
    pq.insert(300, 3);
    EXPECT_EQ(pq.extractMin().first, 5u);

    EXPECT_FALSE(pq.tryExtract(0));
    EXPECT_FALSE(pq.tryExtract(1));
    pq.decreaseKey(5, 1);
    pq.decreaseKey(5, 3);
    EXPECT_TRUE(pq.tryExtract(1));
    EXPECT_FALSE(pq.tryExtract(1));
    EXPECT_TRUE(pq.tryExtract(3));
    EXPECT_FALSE(pq.contains(1));
    EXPECT_EQ(pq.getKey(1), 5u);

    const auto [key, values] = pq.extractMin();
    EXPECT_EQ(key, 9u);
    EXPECT_EQ(sorted(values), (std::vector<Value>{2}));
    EXPECT_TRUE(pq.empty());
}

/**
 * getMinKey returns the key of the extractMin that follows it, however often it is called, skips
 * extracted values, and leaves keys free to drop below it down to the last extracted key.
 */
TEST_F(HierarchicalBucketPQGTest, testGetMinKey) {
    PQ pq(5);
    pq.insert(20, 0);
    pq.decreaseKey(5, 0);
    pq.insert(300, 1);
    pq.insert(300, 2);
    pq.insert(700, 3);
    EXPECT_EQ(pq.extractMin().first, 5u);

    EXPECT_EQ(pq.getMinKey(), 300u);
    EXPECT_EQ(pq.getMinKey(), 300u);
    EXPECT_EQ(pq.size(), 3u);

    pq.decreaseKey(10, 3);
    pq.insert(10, 4);
    EXPECT_EQ(pq.getMinKey(), 10u);
    const auto [key, values] = pq.extractMin();
    EXPECT_EQ(key, 10u);
    EXPECT_EQ(sorted(values), (std::vector<Value>{3, 4}));
    EXPECT_EQ(pq.getMinKey(), 300u);
}

/**
 * A value lowered to the key just extracted comes out in a later batch with that same key rather
 * than being skipped.
 */
TEST_F(HierarchicalBucketPQGTest, testReentry) {
    PQ pq(3);
    pq.insert(3, 0);
    pq.insert(10, 1);
    pq.insert(1000, 2);
    EXPECT_EQ(pq.extractMin().first, 3u);

    pq.decreaseKey(3, 1);
    pq.decreaseKeyRelative(997, 2);
    const auto [key, values] = pq.extractMin();
    EXPECT_EQ(key, 3u);
    EXPECT_EQ(sorted(values), (std::vector<Value>{1, 2}));
    EXPECT_TRUE(pq.empty());
}

/**
 * size and empty match the number of present values after each insert, tryExtract and extractMin.
 */
TEST_F(HierarchicalBucketPQGTest, testSize) {
    PQ pq(4);
    EXPECT_TRUE(pq.empty());
    pq.insert(1, 0);
    pq.insert(1, 1);
    pq.insert(1, 1);
    pq.insert(2, 2);
    pq.insert(2, 3);
    EXPECT_EQ(pq.size(), 4u);
    pq.extractMin();
    EXPECT_EQ(pq.size(), 2u);
    pq.decreaseKey(1, 2);
    pq.tryExtract(2);
    EXPECT_EQ(pq.size(), 1u);
    EXPECT_FALSE(pq.empty());
    pq.extractMin();
    EXPECT_EQ(pq.size(), 0u);
    EXPECT_TRUE(pq.empty());
}

/**
 * Keys on both sides of each internal bucket boundary, including keys lowered across one, come out
 * in increasing key order with each key in its own batch.
 */
TEST_F(HierarchicalBucketPQGTest, testBucketBoundaries) {
    std::vector<Key> keys;
    for (Key boundary = 8; boundary <= (Key{1} << 16); boundary *= 2)
        for (const Key key : {boundary - 1, boundary, boundary + 1})
            keys.push_back(key);
    const Key maxKey = Key{1} << 17;
    keys.push_back(maxKey);

    PQ pq(keys.size());
    BucketPQOracle oracle(keys.size(), maxKey);
    for (Value value = 0; value < keys.size(); ++value) {
        pq.insert(keys[value], value);
        oracle.insert(keys[value], value);
    }

    std::mt19937_64 rng(2);
    while (!oracle.empty()) {
        expectSameExtraction(pq, oracle);
        if (oracle.empty())
            break;
        const Value value = static_cast<Value>(rng() % keys.size());
        const Key delta = std::uniform_int_distribution<Key>(1, oracle.getKey(value) / 2 + 1)(rng);
        EXPECT_EQ(pq.decreaseKeyRelative(delta, value), oracle.decreaseKeyRelative(delta, value));
    }
    EXPECT_TRUE(pq.empty());
}

/**
 * Concurrent phases of a single update kind leave the same keys, report the same changes per
 * value, and lead to the same extractions as BucketPQOracle applying those updates one at a time.
 */
TEST_F(HierarchicalBucketPQGTest, testConcurrentPhasesMatchBucketPQ) {
    constexpr count n = 20000;
    constexpr count opsPerPhase = 3 * n;
    constexpr Key maxKey = 5000;
    std::mt19937_64 rng(3);
    PQ pq(n, omp_get_max_threads());
    BucketPQOracle oracle(n, maxKey);

    std::vector<Value> values(opsPerPhase);
    std::vector<Key> arguments(opsPerPhase);
    std::vector<char> results(opsPerPhase);
    std::vector<count> changes(n), expectedChanges(n);
    // Which of several calls on one value reports the change depends on their order, so results
    // are compared per value: the number of reported changes for inserts, whether any for
    // decreases.
    auto runPhase = [&](auto update, auto oracleUpdate, Key maxArgument) {
        for (count i = 0; i < opsPerPhase; ++i) {
            values[i] = static_cast<Value>(rng() % n);
            arguments[i] = rng() % (maxArgument + 1);
        }
#pragma omp parallel for
        for (omp_index i = 0; i < static_cast<omp_index>(opsPerPhase); ++i)
            results[i] = update(arguments[i], values[i], static_cast<count>(omp_get_thread_num()));

        std::ranges::fill(changes, 0);
        std::ranges::fill(expectedChanges, 0);
        for (count i = 0; i < opsPerPhase; ++i) {
            changes[values[i]] += results[i];
            expectedChanges[values[i]] += oracleUpdate(arguments[i], values[i]);
        }
        for (Value value = 0; value < n; ++value) {
            ASSERT_EQ(pq.getKey(value), oracle.getKey(value));
            ASSERT_EQ(changes[value] > 0, expectedChanges[value] > 0);
        }
    };

    runPhase([&](Key key, Value value, count id) { return pq.insert(key, value, id); },
             [&](Key key, Value value) { return oracle.insert(key, value); }, maxKey);
    for (Value value = 0; value < n; ++value)
        ASSERT_EQ(changes[value], pq.contains(value) ? 1u : 0u);

    while (!oracle.empty()) {
        expectSameExtraction(pq, oracle);
        runPhase([&](Key key, Value value, count id) { return pq.decreaseKey(key, value, id); },
                 [&](Key key, Value value) { return oracle.decreaseKey(key, value); }, maxKey);
        if (oracle.empty())
            break;
        expectSameExtraction(pq, oracle);
        runPhase([&](Key delta, Value value,
                     count id) { return pq.decreaseKeyRelative(delta, value, id); },
                 [&](Key delta, Value value) { return oracle.decreaseKeyRelative(delta, value); },
                 64);
    }
    EXPECT_TRUE(pq.empty());
}

/**
 * Batches of thousands of values, inserted under every id and redistributed from a range bucket,
 * match BucketPQOracle and leave no extracted value present.
 */
TEST_F(HierarchicalBucketPQGTest, testLargeBatches) {
    constexpr count n = 40000;
    auto keyOf = [](Value value) -> Key { return value % 4 == 0 ? 5 : 300 + value % 3; };
    PQ pq(n, omp_get_max_threads());
    BucketPQOracle oracle(n, 302);
#pragma omp parallel for
    for (omp_index value = 0; value < static_cast<omp_index>(n); ++value)
        pq.insert(keyOf(value), static_cast<Value>(value), omp_get_thread_num());
    for (Value value = 0; value < n; ++value)
        oracle.insert(keyOf(value), value);

    while (!oracle.empty())
        expectSameExtraction(pq, oracle);
    EXPECT_TRUE(pq.empty());
    for (Value value = 0; value < n; ++value)
        ASSERT_FALSE(pq.contains(value));
}

/**
 * With tryExtract racing decreases, every extractMin batch agrees with contains and getKey,
 * extracted keys never decrease, tryExtract succeeds only at the key of the last extractMin, and
 * every value is extracted exactly once.
 */
TEST_F(HierarchicalBucketPQGTest, testConcurrentTryExtract) {
    constexpr count n = 50000;
    constexpr count degree = 8;
    std::mt19937_64 rng(4);
    PQ pq(n, omp_get_max_threads());
    for (Value value = 0; value < n; ++value)
        pq.insert(rng() % 2000, value);

    auto neighbor = [](Value value, count i) {
        return static_cast<Value>((value * 2654435761u + i * 40503u) % n);
    };
    std::vector<std::atomic<count>> timesExtracted(n);
    Key previousKey = 0;
    while (!pq.empty()) {
        const auto extracted = pq.extractMin();
        const Key key = extracted.first;
        const std::span<const Value> batch = extracted.second;
        EXPECT_GE(key, previousKey);
        previousKey = key;
        for (const Value value : batch) {
            EXPECT_FALSE(pq.contains(value));
            EXPECT_EQ(pq.getKey(value), key);
            ++timesExtracted[value];
        }

        std::atomic<count> wrongKey{0};
#pragma omp parallel for
        for (omp_index i = 0; i < static_cast<omp_index>(batch.size()); ++i) {
            const count id = omp_get_thread_num();
            for (count j = 0; j < degree; ++j) {
                const Value u = neighbor(batch[i], j);
                pq.decreaseKeyRelative(1, u, id);
                if (pq.tryExtract(u)) {
                    timesExtracted[u].fetch_add(1, std::memory_order_relaxed);
                    if (pq.getKey(u) != key)
                        wrongKey.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
        EXPECT_EQ(wrongKey.load(), 0u);
    }
    for (Value value = 0; value < n; ++value)
        ASSERT_EQ(timesExtracted[value].load(), 1u) << "value " << value;
}

} // namespace NetworKit
