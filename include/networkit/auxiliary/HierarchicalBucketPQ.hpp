/*
 * HierarchicalBucketPQ.hpp
 *
 *  Created on: 26.10.2026
 *      Author: Ian Chen (ianchen3@illinois.edu)
 */

#ifndef NETWORKIT_AUXILIARY_HIERARCHICAL_BUCKET_PQ_HPP_
#define NETWORKIT_AUXILIARY_HIERARCHICAL_BUCKET_PQ_HPP_

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cassert>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include <networkit/Globals.hpp>

namespace Aux {

using count = NetworKit::count;

/**
 * Monotone priority queue over values in [0, capacity) with decreasing non-negative integer keys.
 * Value(s) with the minimum key can be extracted.
 * Keys never go below the key of the last extractMin.
 *
 * Phase-concurrency. Methods are
 * - updates: insert, decreaseKey, decreaseKeyRelative, tryExtract,
 * - accessors: getKey, contains, empty, size,
 * - queries: extractMin, getMinKey.
 * Updates and accessors may run concurrently. Queries must be serial.
 *
 * Implementation: buckets follow the hierarchical bucketing structure of
 *
 * Youzhe Liu, Xiaojun Dong, Yan Gu, and Yihan Sun.
 * Parallel k-Core Decomposition: Theory and Practice.
 * Proc. ACM Manag. Data 3, 3 (SIGMOD), 2025.
 * https://doi.org/10.1145/3725332
 */
class HierarchicalBucketPQ final {
public:
    using Key = count;
    using Value = uint32_t;

    static constexpr Key infiniteKey = std::numeric_limits<Key>::max();
    static constexpr Key maxKey = std::numeric_limits<int64_t>::max() - 1;

    /**
     * Create an empty queue for values in [0, @p capacity). Requires @p capacity <=
     * std::numeric_limits<Value>::max().
     * @param capacity number of admissible values
     * @param numIds number of admissible ids; queries use at most @p numIds threads
     */
    explicit HierarchicalBucketPQ(count capacity, count numIds = 1);

    /**
     * Insert @p value with @p key, or lower its key if it is already present. No effect if
     * @p value was extracted. Requires @p key <= maxKey and @p id < numIds.
     * @param key key to insert with
     * @param value value to insert
     * @param id caller's id; concurrent calls must pass different ids
     * @return whether this call inserted @p value
     */
    bool insert(Key key, Value value, count id = 0);

    /**
     * Lower the key of @p value to @p newKey. No effect if @p value is not present. Requires
     * @p id < numIds.
     * @param newKey new key
     * @param value value to update
     * @param id caller's id; concurrent calls must pass different ids
     * @return whether this call lowered the key
     */
    bool decreaseKey(Key newKey, Value value, count id = 0);

    /**
     * Lower the key of @p value by @p delta. No effect if @p value is not present. Requires
     * @p id < numIds.
     * @param delta amount to subtract
     * @param value value to update
     * @param id caller's id; concurrent calls must pass different ids
     * @return whether this call lowered the key
     */
    bool decreaseKeyRelative(Key delta, Value value, count id = 0);

    /**
     * Extract @p value if it is present and its key equals the key of the last extractMin.
     * @param value value to extract
     * @return whether this call extracted @p value
     */
    bool tryExtract(Value value);

    /**
     * @param value value to look up
     * @return the key of @p value, or infiniteKey if it was never inserted
     */
    Key getKey(Value value) const;

    /**
     * @param value value to look up
     * @return whether @p value is present
     */
    bool contains(Value value) const;

    /**
     * Extract every present value with the minimum key. Requires !empty().
     * @return the minimum key and the extracted values; the span is valid until the next
     * extractMin
     */
    std::pair<Key, std::span<const Value>> extractMin();

    /**
     * Return the key that extractMin would return now, without extracting. Requires !empty().
     * Repeated calls repeat the same scan, which may visit values with keys far above the minimum.
     * @return the minimum key
     */
    Key getMinKey() const;

    /**
     * @return whether no value is present
     */
    bool empty() const { return size() == 0; }

    /**
     * @return number of present values
     */
    count size() const { return numQueued.load(std::memory_order_relaxed); }

private:
    // One word per value so that every update checks and changes the state and key atomically:
    // present holds the key, extracted holds -(key + 1), never inserted holds absentEntry.
    using Entry = int64_t;
    static constexpr Entry absentEntry = std::numeric_limits<Entry>::max();

    static constexpr count log2SingleBuckets = 3;
    static constexpr count numSingleBuckets = count{1} << log2SingleBuckets;
    static constexpr count numBuckets = numSingleBuckets + 6;

    // Aligned so that updates with different ids do not share cache lines.
    struct alignas(64) BucketSet {
        std::array<std::vector<Value>, numBuckets> buckets;
    };

    static bool isPresent(Entry entry) { return 0 <= entry && entry != absentEntry; }
    static Entry extractedEntry(Key key) { return -static_cast<Entry>(key) - 1; }

    // Bucket of key relative to baseKey; keys beyond the range buckets go to the last bucket.
    count bucketOf(Key key) const;
    // Lower the key of a present value to newKey(old entry), pushing the value into the bucket of
    // its new key if that bucket differs. Returns whether the key was lowered.
    template <typename NewKey>
    bool lower(Value value, count id, NewKey newKey);

    // Lowest bucket that is non-empty in the bucket set of any id.
    count firstNonEmptyBucket() const;
    // Move the present values in bucket from the bucket sets of all ids into out.
    void extractBucket(count bucket, std::vector<Value> &out);
    // Raise baseKey to the minimum key of values, rounded down to a multiple of
    // numSingleBuckets, and push values into their buckets relative to it.
    void redistribute(const std::vector<Value> &values);

    std::vector<std::atomic<Entry>> entries;
    std::vector<BucketSet> bucketSets; // indexed by id
    std::vector<Value> frontier;
    Key currentKey{0};
    Key baseKey{0}; // multiple of numSingleBuckets
    // Own cache line, so that changing it does not slow down reads of currentKey and baseKey.
    alignas(64) std::atomic<count> numQueued{0};
};

inline bool HierarchicalBucketPQ::insert(Key key, Value value, count id) {
    assert(key <= maxKey && id < bucketSets.size());
    const Entry newKey = static_cast<Entry>(std::max(key, currentKey));
    Entry expected = absentEntry;
    if (entries[value].compare_exchange_strong(expected, newKey, std::memory_order_relaxed)) {
        numQueued.fetch_add(1, std::memory_order_relaxed);
        bucketSets[id].buckets[bucketOf(newKey)].push_back(value);
        return true;
    }
    lower(value, id, [&](Entry) { return newKey; });
    return false;
}

inline bool HierarchicalBucketPQ::decreaseKey(Key newKey, Value value, count id) {
    assert(id < bucketSets.size());
    const Entry target = static_cast<Entry>(std::clamp(newKey, currentKey, maxKey));
    return lower(value, id, [&](Entry) { return target; });
}

inline bool HierarchicalBucketPQ::decreaseKeyRelative(Key delta, Value value, count id) {
    assert(id < bucketSets.size());
    return lower(value, id, [&](Entry old) {
        return old - static_cast<Entry>(std::min(delta, static_cast<Key>(old) - currentKey));
    });
}

inline bool HierarchicalBucketPQ::tryExtract(Value value) {
    Entry expected = static_cast<Entry>(currentKey);
    if (!entries[value].compare_exchange_strong(expected, extractedEntry(currentKey),
                                                std::memory_order_relaxed))
        return false;
    numQueued.fetch_sub(1, std::memory_order_relaxed);
    return true;
}

inline HierarchicalBucketPQ::Key HierarchicalBucketPQ::getKey(Value value) const {
    const Entry entry = entries[value].load(std::memory_order_relaxed);
    if (entry == absentEntry)
        return infiniteKey;
    return static_cast<Key>(entry >= 0 ? entry : -(entry + 1));
}

inline bool HierarchicalBucketPQ::contains(Value value) const {
    return isPresent(entries[value].load(std::memory_order_relaxed));
}

inline count HierarchicalBucketPQ::bucketOf(Key key) const {
    if (key < baseKey + numSingleBuckets)
        return key - baseKey;
    const count differingBit = std::bit_width(key ^ baseKey) - 1;
    return std::min(numBuckets - 1, numSingleBuckets + differingBit - log2SingleBuckets);
}

template <typename NewKey>
bool HierarchicalBucketPQ::lower(Value value, count id, NewKey newKey) {
    Entry old = entries[value].load(std::memory_order_relaxed);
    Entry key;
    do {
        if (!isPresent(old) || (key = newKey(old)) >= old)
            return false;
    } while (!entries[value].compare_exchange_weak(old, key, std::memory_order_relaxed));

    const count bucket = bucketOf(key);
    if (bucket != bucketOf(old))
        bucketSets[id].buckets[bucket].push_back(value);
    return true;
}

} // namespace Aux

#endif // NETWORKIT_AUXILIARY_HIERARCHICAL_BUCKET_PQ_HPP_
