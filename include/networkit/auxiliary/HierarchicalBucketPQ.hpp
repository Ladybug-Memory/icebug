/*
 * HierarchicalBucketPQ.hpp
 *
 *  Created on: 26.10.2026
 *      Author: Ian Chen (ianchen3@illinois.edu)
 */

#ifndef NETWORKIT_AUXILIARY_HIERARCHICAL_BUCKET_PQ_HPP_
#define NETWORKIT_AUXILIARY_HIERARCHICAL_BUCKET_PQ_HPP_

#include <atomic>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include <networkit/Globals.hpp>

namespace Aux {

/**
 * Monotone priority queue over values in [0, capacity) with decreasing non-negative integer keys.
 * Value(s) with the minimum key can be extracted.
 * Keys never go below the key of the last extractMin.
 *
 * Phase-concurrency. Methods are
 * - updates: insert, decreaseKey, decreaseKeyRelative, tryExtract,
 * - accessors: getKey, contains, empty, size,
 * - queries: extractMin, getMin.
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
    using Key = NetworKit::count;
    using Value = uint32_t;

    static constexpr Key infiniteKey = std::numeric_limits<Key>::max();
    static constexpr Key maxKey = std::numeric_limits<int64_t>::max() - 1;

    /**
     * Create an empty queue for values in [0, @p capacity). Requires @p capacity <=
     * std::numeric_limits<Value>::max().
     * @param capacity number of admissible values
     * @param numIds number of admissible ids
     */
    explicit HierarchicalBucketPQ(NetworKit::count capacity, NetworKit::count numIds = 1);

    /**
     * Insert @p value with @p key, or lower its key if it is already present. No effect if
     * @p value was extracted. Requires @p key <= maxKey and @p id < numIds.
     * @param key key to insert with
     * @param value value to insert
     * @param id caller's id; concurrent calls must pass different ids
     * @return whether this call inserted @p value
     */
    bool insert(Key key, Value value, NetworKit::count id = 0);

    /**
     * Lower the key of @p value to @p newKey. No effect if @p value is not present. Requires
     * @p id < numIds.
     * @param newKey new key
     * @param value value to update
     * @param id caller's id; concurrent calls must pass different ids
     * @return whether this call lowered the key
     */
    bool decreaseKey(Key newKey, Value value, NetworKit::count id = 0);

    /**
     * Lower the key of @p value by @p delta. No effect if @p value is not present. Requires
     * @p id < numIds.
     * @param delta amount to subtract
     * @param value value to update
     * @param id caller's id; concurrent calls must pass different ids
     * @return whether this call lowered the key
     */
    bool decreaseKeyRelative(Key delta, Value value, NetworKit::count id = 0);

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
     * Return what extractMin would return now, without extracting. Requires !empty().
     * @return the minimum key and the values holding it; the span is valid until the next
     * extractMin or update
     */
    std::pair<Key, std::span<const Value>> getMin();

    /**
     * @return whether no value is present
     */
    bool empty() const;

    /**
     * @return number of present values
     */
    NetworKit::count size() const;

private:
    // One word per value so that every update checks and changes the state and key atomically:
    // present holds the key, extracted holds -(key + 1), never inserted holds absentEntry.
    using Entry = int64_t;
    static constexpr Entry absentEntry = std::numeric_limits<Entry>::max();

    std::vector<std::atomic<Entry>> entries;
    std::atomic<NetworKit::count> numQueued{0};
    Key currentKey{0};
};

} // namespace Aux

#endif // NETWORKIT_AUXILIARY_HIERARCHICAL_BUCKET_PQ_HPP_
