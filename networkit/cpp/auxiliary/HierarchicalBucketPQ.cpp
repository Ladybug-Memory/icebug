/*
 * HierarchicalBucketPQ.cpp
 *
 *  Created on: 26.10.2026
 *      Author: Ian Chen (ianchen3@illinois.edu)
 */

#include <numeric>
#include <omp.h>

#include <networkit/auxiliary/HierarchicalBucketPQ.hpp>

using namespace NetworKit;

namespace Aux {

namespace {
constexpr count parallelThreshold = 4096;
} // namespace

HierarchicalBucketPQ::HierarchicalBucketPQ(count capacity, count numIds)
    : entries(capacity), bucketSets(numIds) {
    assert(capacity <= std::numeric_limits<Value>::max());
#pragma omp parallel for
    for (omp_index value = 0; value < static_cast<omp_index>(capacity); ++value)
        entries[value].store(absentEntry, std::memory_order_relaxed);
}

std::pair<HierarchicalBucketPQ::Key, std::span<const HierarchicalBucketPQ::Value>>
HierarchicalBucketPQ::extractMin() {
    assert(!empty());
    count bucket;
    do {
        bucket = firstNonEmptyBucket();
        extractBucket(bucket, frontier);
        if (bucket >= numSingleBuckets && !frontier.empty())
            redistribute(frontier);
    } while (bucket >= numSingleBuckets || frontier.empty());

    currentKey = baseKey + bucket;
#pragma omp parallel for if (frontier.size() >= parallelThreshold)
    for (omp_index i = 0; i < static_cast<omp_index>(frontier.size()); ++i)
        entries[frontier[i]].store(extractedEntry(currentKey), std::memory_order_relaxed);
    numQueued.fetch_sub(frontier.size(), std::memory_order_relaxed);
    return {currentKey, frontier};
}

// Unlike extractMin, this cannot redistribute: raising baseKey above currentKey would leave no
// bucket for keys that later updates may still lower into [currentKey, baseKey).
HierarchicalBucketPQ::Key HierarchicalBucketPQ::getMinKey() const {
    assert(!empty());
    for (count bucket = 0; bucket < numBuckets; ++bucket) {
        Key result = infiniteKey;
        for (const BucketSet &set : bucketSets) {
            for (const Value value : set.buckets[bucket]) {
                if (!contains(value))
                    continue;
                if (bucket < numSingleBuckets)
                    return baseKey + bucket;
                result = std::min(result, getKey(value));
            }
        }
        if (result != infiniteKey)
            return result;
    }
    assert(false);
    return infiniteKey;
}

count HierarchicalBucketPQ::firstNonEmptyBucket() const {
    for (count bucket = 0; bucket < numBuckets; ++bucket)
        for (const BucketSet &set : bucketSets)
            if (!set.buckets[bucket].empty())
                return bucket;
    assert(false);
    return numBuckets;
}

void HierarchicalBucketPQ::extractBucket(count bucket, std::vector<Value> &out) {
    const count numSets = bucketSets.size();
    count total = 0;
    for (const BucketSet &set : bucketSets)
        total += set.buckets[bucket].size();

    std::vector<count> offsets(numSets + 1, 0);
#pragma omp parallel for if (total >= parallelThreshold)
    for (omp_index id = 0; id < static_cast<omp_index>(numSets); ++id) {
        std::vector<Value> &source = bucketSets[id].buckets[bucket];
        std::erase_if(source, [&](Value value) {
            const Entry entry = entries[value].load(std::memory_order_relaxed);
            // A stale copy of a present value sits above its current bucket, which is gathered
            // first, so the value is no longer present when the stale copy is gathered.
            assert(!isPresent(entry) || bucketOf(entry) == bucket);
            return !isPresent(entry);
        });
        offsets[id + 1] = source.size();
    }
    std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());

    out.resize(offsets[numSets]);
#pragma omp parallel for if (total >= parallelThreshold)
    for (omp_index id = 0; id < static_cast<omp_index>(numSets); ++id) {
        std::vector<Value> &source = bucketSets[id].buckets[bucket];
        std::copy(source.begin(), source.end(), out.begin() + offsets[id]);
        source.clear();
    }
}

void HierarchicalBucketPQ::redistribute(const std::vector<Value> &values) {
    Key minKey = infiniteKey;
#pragma omp parallel for reduction(min : minKey) if (values.size() >= parallelThreshold)
    for (omp_index i = 0; i < static_cast<omp_index>(values.size()); ++i)
        minKey = std::min(minKey, getKey(values[i]));
    baseKey = minKey & ~(numSingleBuckets - 1);

    // Thread t pushes into the bucket set of id t, so at most numIds threads take part.
    const int numThreads =
        static_cast<int>(std::min<count>(bucketSets.size(), omp_get_max_threads()));
#pragma omp parallel num_threads(numThreads) if (values.size() >= parallelThreshold)
    {
        BucketSet &set = bucketSets[omp_get_thread_num()];
#pragma omp for
        for (omp_index i = 0; i < static_cast<omp_index>(values.size()); ++i)
            set.buckets[bucketOf(getKey(values[i]))].push_back(values[i]);
    }
}

} // namespace Aux
