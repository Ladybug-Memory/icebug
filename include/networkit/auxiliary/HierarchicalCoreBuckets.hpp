#ifndef NETWORKIT_AUXILIARY_HIERARCHICAL_CORE_BUCKETS_HPP_
#define NETWORKIT_AUXILIARY_HIERARCHICAL_CORE_BUCKETS_HPP_

#include <array>
#include <vector>
#include <networkit/Globals.hpp>

namespace Aux {
namespace CoreBuckets {
/**
 * @ingroup auxiliary
 * Bucket routing for online core peeling: eight exact levels and logarithmic intervals.
 * Callers own synchronization, stale-entry filtering and intermediate redistribution.
 * Base levels are multiples of eight; routed degrees must belong to the same aligned
 * 512-level window as the base. The peeling engine advances that window separately.
 */
constexpr NetworKit::count kLog2SingleBuckets = 3;
constexpr NetworKit::count kNumSingleBuckets = NetworKit::count{1} << kLog2SingleBuckets;
constexpr NetworKit::count kNumIntermediateBuckets = 6;
constexpr NetworKit::count kNumBuckets = kNumSingleBuckets + kNumIntermediateBuckets;
constexpr NetworKit::count kBucketMask = kNumSingleBuckets - 1;
constexpr NetworKit::count kBucketStride = kNumSingleBuckets << kNumIntermediateBuckets;

using Storage = std::array<std::vector<NetworKit::node>, kNumBuckets>;
inline NetworKit::count mostSignificantBit(NetworKit::count value) {
    NetworKit::count result = 0;
    while (value >>= 1) {
        ++result;
    }
    return result;
}

template <typename Buckets>
/** Insert a candidate at its current degree; candidates outside the window are ignored. */
void add(Buckets &target, NetworKit::node u, NetworKit::count degree, NetworKit::count baseLevel) {
    if (degree < baseLevel || degree >= baseLevel + kBucketStride) {
        return;
    }

    if (degree < baseLevel + kNumSingleBuckets) {
        target[degree & kBucketMask].push_back(u);
        return;
    }

    const NetworKit::count differingBit = mostSignificantBit(degree ^ baseLevel);
    const NetworKit::count bucket = kNumSingleBuckets + differingBit - kLog2SingleBuckets;
    target[bucket].push_back(u);
}

template <typename Buckets>
/** Record a decrement only when it crosses a bucket boundary (or reaches an exact bucket). */
void move(Buckets &target, NetworKit::node u, NetworKit::count degree, NetworKit::count baseLevel) {
    if (degree < baseLevel || degree >= baseLevel + kBucketStride) {
        return;
    }

    if (degree < baseLevel + kNumSingleBuckets) {
        target[degree & kBucketMask].push_back(u);
        return;
    }

    const NetworKit::count differingBit = mostSignificantBit(degree ^ baseLevel);
    const NetworKit::count previousDifferingBit = mostSignificantBit((degree + 1) ^ baseLevel);
    if (differingBit != previousDifferingBit) {
        const NetworKit::count bucket = kNumSingleBuckets + differingBit - kLog2SingleBuckets;
        target[bucket].push_back(u);
    }
}

} // namespace CoreBuckets
} // namespace Aux
#endif // NETWORKIT_AUXILIARY_HIERARCHICAL_CORE_BUCKETS_HPP_
