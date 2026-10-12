#ifndef NETWORKIT_CENTRALITY_PARALLEL_CORE_INTERNAL_HPP_
#define NETWORKIT_CENTRALITY_PARALLEL_CORE_INTERNAL_HPP_

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include <omp.h>

#include "CoreSampler.hpp"
#include <networkit/auxiliary/HierarchicalCoreBuckets.hpp>
#include <networkit/centrality/CoreDecomposition.hpp>
#include <networkit/centrality/ParallelCoreDecomposition.hpp>

namespace NetworKit {
namespace CoreDecompositionDetail {

constexpr count kSampleThreshold = 2000;
constexpr double kInitialReduceRatio = 0.1;
constexpr count kLog2ErrorFactor = 32;
constexpr count kBaselineExpectedHits =
    static_cast<count>(kLog2ErrorFactor / (kInitialReduceRatio * kInitialReduceRatio));
constexpr double kBiasFactor = 0.5;
constexpr double kErrorRateTolerance = 1e-10;

constexpr count kLocalQueueSize = 128;
constexpr count kNeighborParallelismThreshold = 128;
constexpr count kParallelFrontierThreshold = 256;
constexpr count kHbsSwitchCore = 16;

using BucketArray = Aux::CoreBuckets::Storage;
using namespace Aux::CoreBuckets;

struct ThreadScratch final {
    BucketArray buckets;
    std::vector<node> recounts;
};

struct EdgeBlock final {
    node source;
    index begin;
    index end;
};

inline uint32_t hash32(uint32_t value) {
    value = (value + 0x7ed55d16U) + (value << 12);
    value = (value ^ 0xc761c23cU) ^ (value >> 19);
    value = (value + 0x165667b1U) + (value << 5);
    value = (value + 0xd3a2646cU) ^ (value << 9);
    value = (value + 0xfd7046c5U) + (value << 3);
    value = (value ^ 0xb55a4f09U) ^ (value >> 16);
    return value;
}

inline count log2Up(count value) {
    if (value <= 1) {
        return 0;
    }

    count result = 0;
    --value;
    while (value > 0) {
        ++result;
        value >>= 1;
    }
    return result;
}

class ParallelKCoreSolver final {
public:
    ParallelKCoreSolver(const Graph &G, std::vector<double> &scores, bool samplingEnabled,
                        bool (*acceptRecount)(node, count) = nullptr)
        : G(G), scores(scores), samplingEnabled(samplingEnabled), acceptRecount(acceptRecount),
          z(G.upperNodeIdBound()), degrees(new std::atomic<count>[z]),
          active(new std::atomic<char>[z]), threadScratch(omp_get_max_threads()),
          filterCounts(omp_get_max_threads()), filterOffsets(omp_get_max_threads()),
          seenAtEpoch(z, 0) {}

    bool run() {
        const count n = G.numberOfNodes();
        if (n == 0) {
            return true;
        }

        initialize();

        std::vector<node> remaining(z);
#pragma omp parallel for schedule(static)
        for (omp_index u = 0; u < static_cast<omp_index>(z); ++u) {
            remaining[u] = static_cast<node>(u);
        }

        const count averageDegree = G.numberOfEdges() * 2 / n;
        count firstHbsLevel = 0;
        if (averageDegree < kHbsSwitchCore) {
            for (count level = 0; level < kHbsSwitchCore && !remaining.empty(); ++level) {
                buckets[0].clear();
                if (!buildSparseBucket(remaining, level)) {
                    return false;
                }
                if (!processLevel(level, level, false)) {
                    return false;
                }
                firstHbsLevel = level + 1;
            }
            filterRemaining(remaining);
        }

        for (count baseLevel = 0; !remaining.empty(); baseLevel += kBucketStride) {
            if (baseLevel > maximumInitialDegree) {
                return false;
            }

            clearBuckets();
            const count levelBegin = std::max(baseLevel, firstHbsLevel);
            if (!buildBuckets(remaining, levelBegin, baseLevel + kBucketStride, baseLevel, true)) {
                return false;
            }

            for (count level = levelBegin; level < baseLevel + kBucketStride && !remaining.empty();
                 ++level) {
                if (level != baseLevel && (level & kBucketMask) == 0) {
                    redistributeIntermediateBucket(level);
                }

                const count bucketBase = level & ~kBucketMask;
                if (!processLevel(level, bucketBase, true)) {
                    return false;
                }
            }

            filterRemaining(remaining);
        }

        return true;
    }

    index maxCoreNumber() const { return maxCore; }

private:
    const Graph &G;
    std::vector<double> &scores;
    bool samplingEnabled;
    // Per-run internal validation hook; never shared globally or exposed through the public API.
    bool (*acceptRecount)(node, count);
    count z;
    std::unique_ptr<std::atomic<count>[]> degrees;
    std::unique_ptr<Sampler[]> samplers;
    std::unique_ptr<std::atomic<bool>[]> recountQueued;
    // Atomics protect accesses within a peeling round. Relaxed ordering is sufficient:
    // OpenMP region/barrier boundaries publish sampler configuration and scratch buffers
    // between initialization, peeling and recount phases. No flag publishes other data.
    std::unique_ptr<std::atomic<char>[]> active;
    std::unique_ptr<std::atomic<char>[]> sampleMode;
    BucketArray buckets;
    std::vector<ThreadScratch> threadScratch;
    std::vector<count> filterCounts;
    std::vector<count> filterOffsets;
    std::vector<node> filterBuffer;
    std::vector<count> seenAtEpoch;
    count scratchInUse{1};
    bool samplingPossible{false};
    count currentEpoch{0};
    index maxCore{0};
    count maximumInitialDegree{0};

    void initialize() {
        count maxDegree = 0;
// MSVC classic OpenMP (NETWORKIT_OMP2) lacks reduction(max); fall back to serial.
#ifndef NETWORKIT_OMP2
#pragma omp parallel for reduction(max : maxDegree) schedule(static)
#endif
        for (omp_index u = 0; u < static_cast<omp_index>(z); ++u) {
            const count degree = G.degree(static_cast<node>(u));
            degrees[u].store(degree, std::memory_order_relaxed);
            active[u].store(1, std::memory_order_relaxed);
            maxDegree = std::max(maxDegree, degree);
        }
        maximumInitialDegree = maxDegree;
        samplingPossible = samplingEnabled && maxDegree * kInitialReduceRatio >= kSampleThreshold;

        if (samplingPossible) {
            // Most sparse inputs never sample. Avoid O(n) sampler construction and storage
            // until the degree scan establishes that sampling can actually be used.
            samplers.reset(new Sampler[z]);
            recountQueued.reset(new std::atomic<bool>[z]);
            sampleMode.reset(new std::atomic<char>[z]);
#pragma omp parallel for schedule(static)
            for (omp_index u = 0; u < static_cast<omp_index>(z); ++u) {
                recountQueued[u].store(false, std::memory_order_relaxed);
                setSampler(static_cast<node>(u), 0);
            }
        }
    }

    void setSampler(node u, count level) {
        if (!samplingPossible) {
            return;
        }

        const count degree = degrees[u].load(std::memory_order_relaxed);
        const double reducedDegree = degree * kInitialReduceRatio;
        const double remainingAfterReduction =
            (degree - std::min(degree, level)) * (1.0 - kInitialReduceRatio);

        if (reducedDegree >= kSampleThreshold && level < reducedDegree * kBiasFactor
            && kBaselineExpectedHits < remainingAfterReduction) {
            sampleMode[u].store(1, std::memory_order_relaxed);
            const count distance = degree - level;
            const count logarithm = log2Up(distance);
            const count expectedHits = kLog2ErrorFactor * logarithm * logarithm / 2;
            const double rate = expectedHits / ((1.0 - kInitialReduceRatio) * degree);
            samplers[u].reset(expectedHits, rate);
        } else {
            sampleMode[u].store(0, std::memory_order_relaxed);
        }
    }

    bool sampleIsSafe(node u, count horizon) const {
        const count degree = degrees[u].load(std::memory_order_relaxed);
        if (degree * kInitialReduceRatio * kBiasFactor < horizon || degree <= horizon) {
            return false;
        }

        const double rate = samplers[u].sampleRate();
        const double distance = static_cast<double>(degree - horizon);
        const double expected = distance * rate;
        if (expected <= 0.0) {
            return false;
        }

        const double hits = static_cast<double>(std::max<count>(1, samplers[u].numberOfHits()));
        const double exponent = -expected + 2.0 * hits - hits * hits / expected;
        return std::exp(exponent) < kErrorRateTolerance;
    }

    void resetScratch(count numberOfThreads) {
        for (count thread = 0; thread < numberOfThreads; ++thread) {
            auto &scratch = threadScratch[thread];
            scratch.recounts.clear();
            for (auto &bucket : scratch.buckets) {
                bucket.clear();
            }
        }
    }

    void mergeScratchBuckets(count numberOfThreads) {
        for (count thread = 0; thread < numberOfThreads; ++thread) {
            auto &scratch = threadScratch[thread];
            for (count bucket = 0; bucket < kNumBuckets; ++bucket) {
                auto &source = scratch.buckets[bucket];
                buckets[bucket].insert(buckets[bucket].end(), source.begin(), source.end());
            }
        }
    }

    std::vector<node> collectScratchRecounts(count numberOfThreads) {
        count size = 0;
        for (count thread = 0; thread < numberOfThreads; ++thread) {
            size += threadScratch[thread].recounts.size();
        }

        std::vector<node> recounts;
        recounts.reserve(size);
        for (count thread = 0; thread < numberOfThreads; ++thread) {
            const auto &source = threadScratch[thread].recounts;
            recounts.insert(recounts.end(), source.begin(), source.end());
        }
        return recounts;
    }

    void clearBuckets() {
        for (auto &bucket : buckets) {
            bucket.clear();
        }
    }

    bool buildBuckets(const std::vector<node> &remaining, count level, count horizon,
                      count baseLevel, bool useHbs) {
        resetScratch(threadScratch.size());
#pragma omp parallel for schedule(static)
        for (omp_index i = 0; i < static_cast<omp_index>(remaining.size()); ++i) {
            const node u = remaining[i];
            const count degree = degrees[u].load(std::memory_order_relaxed);
            auto &scratch = threadScratch[omp_get_thread_num()];
            Aux::CoreBuckets::add(scratch.buckets, u, degree, baseLevel);
            if (samplingPossible && sampleMode[u].load(std::memory_order_relaxed)
                && !sampleIsSafe(u, horizon)) {
                bool expected = false;
                if (recountQueued[u].compare_exchange_strong(expected, true,
                                                             std::memory_order_relaxed)) {
                    scratch.recounts.push_back(u);
                }
            }
        }
        mergeScratchBuckets(threadScratch.size());
        return recountVertices(collectScratchRecounts(threadScratch.size()), level, baseLevel,
                               useHbs);
    }

    bool buildSparseBucket(std::vector<node> &remaining, count level) {
        resetScratch(threadScratch.size());
        // Compact the previous level and build the next frontier in the same scan.
        filterRemaining(remaining, true, level);
        mergeScratchBuckets(threadScratch.size());
        return recountVertices(collectScratchRecounts(threadScratch.size()), level, level, false);
    }

    std::vector<node> uniqueActive(std::vector<node> candidates) {
        ++currentEpoch;
        if (currentEpoch == 0) {
            std::fill(seenAtEpoch.begin(), seenAtEpoch.end(), 0);
            ++currentEpoch;
        }

        size_t output = 0;
        for (node u : candidates) {
            if (active[u].load(std::memory_order_relaxed) && seenAtEpoch[u] != currentEpoch) {
                seenAtEpoch[u] = currentEpoch;
                candidates[output++] = u;
            }
        }
        candidates.resize(output);
        return candidates;
    }

    std::vector<node> takeExactBucket(count level, bool useHbs) {
        const count bucket = useHbs ? (level & kBucketMask) : 0;
        std::vector<node> candidates;
        candidates.swap(buckets[bucket]);
        candidates = uniqueActive(std::move(candidates));

        candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                        [&](node u) {
                                            return degrees[u].load(std::memory_order_relaxed)
                                                   != level;
                                        }),
                         candidates.end());
        return candidates;
    }

    void redistributeIntermediateBucket(count level) {
        count intermediate = kNumIntermediateBuckets;
        for (count i = kNumIntermediateBuckets; i-- > 0;) {
            const count mask = (kNumSingleBuckets << i) - 1;
            if ((level & mask) == 0) {
                intermediate = i;
                break;
            }
        }
        if (intermediate == kNumIntermediateBuckets) {
            return;
        }

        const count bucket = kNumSingleBuckets + intermediate;
        std::vector<node> candidates;
        candidates.swap(buckets[bucket]);
        candidates = uniqueActive(std::move(candidates));

        resetScratch(threadScratch.size());
#pragma omp parallel for schedule(static)
        for (omp_index i = 0; i < static_cast<omp_index>(candidates.size()); ++i) {
            const node u = candidates[i];
            const count degree = degrees[u].load(std::memory_order_relaxed);
            Aux::CoreBuckets::add(threadScratch[omp_get_thread_num()].buckets, u, degree, level);
        }
        mergeScratchBuckets(threadScratch.size());
    }

    void decrementNeighbor(node source, node target, count level, count bucketBase, bool useHbs,
                           bool allowLocalQueue, std::array<node, kLocalQueueSize> &localQueue,
                           count &rear, BucketArray &localBuckets,
                           std::vector<node> &localRecounts) {
        count oldDegree = degrees[target].load(std::memory_order_relaxed);
        if (oldDegree <= level) {
            return;
        }

        if (samplingPossible && sampleMode[target].load(std::memory_order_relaxed)) {
            // Unsigned multiplication wraps and the 32-bit key intentionally collides.
            // Exact recounts and recovery protect correctness from sampling collisions.
            const uint32_t key = static_cast<uint32_t>(source * z + target);
            if (samplers[target].sample(hash32(key))) {
                bool expected = false;
                if (recountQueued[target].compare_exchange_strong(expected, true,
                                                                  std::memory_order_relaxed)) {
                    localRecounts.push_back(target);
                }
            }
            return;
        }

        while (oldDegree > level) {
            if (degrees[target].compare_exchange_weak(oldDegree, oldDegree - 1,
                                                      std::memory_order_relaxed)) {
                const count newDegree = oldDegree - 1;
                if (newDegree == level && allowLocalQueue && rear < kLocalQueueSize) {
                    localQueue[rear++] = target;
                } else if (useHbs) {
                    Aux::CoreBuckets::move(localBuckets, target, newDegree, bucketBase);
                } else if (newDegree == level) {
                    localBuckets[0].push_back(target);
                }
                return;
            }
        }
    }

    count processRootSequential(node root, count level, count bucketBase, bool useHbs,
                                bool deferHighDegree, BucketArray &localBuckets,
                                std::vector<node> &localRecounts) {
        std::array<node, kLocalQueueSize> localQueue;
        count front = 0;
        count rear = 1;
        count processed = 0;
        localQueue[0] = root;

        while (front < rear) {
            const node u = localQueue[front++];
            // In the serial small-frontier path, continue a chain without another round.
            // Parallel roots retain a bounded work budget so one worker cannot monopolize
            // a long chain. High-degree vertices are still deferred to edge blocks.
            if (deferHighDegree && front == rear) {
                front = 0;
                rear = 0;
            }
            if (!active[u].load(std::memory_order_relaxed)) {
                continue;
            }

            // Use original degree: even a low residual degree still scans the full CSR row.
            if (deferHighDegree && G.degree(u) >= kNeighborParallelismThreshold) {
                const count bucket = useHbs ? (level & kBucketMask) : 0;
                localBuckets[bucket].push_back(u);
                continue;
            }

            active[u].store(0, std::memory_order_relaxed);
            scores[u] = level;
            ++processed;
            G.forNeighborsOf(u, [&](node v) {
                decrementNeighbor(u, v, level, bucketBase, useHbs, true, localQueue, rear,
                                  localBuckets, localRecounts);
            });
        }
        return processed;
    }

    count processLargeFrontier(const std::vector<node> &frontier, count level, count bucketBase,
                               bool useHbs) {
        scratchInUse = threadScratch.size();
        resetScratch(scratchInUse);
        count processed = 0;
#pragma omp parallel for schedule(dynamic, 64) reduction(+ : processed)
        for (omp_index i = 0; i < static_cast<omp_index>(frontier.size()); ++i) {
            auto &scratch = threadScratch[omp_get_thread_num()];
            processed += processRootSequential(frontier[i], level, bucketBase, useHbs, false,
                                               scratch.buckets, scratch.recounts);
        }
        mergeScratchBuckets(scratchInUse);
        return processed;
    }

    count processSmallFrontier(const std::vector<node> &frontier, count level, count bucketBase,
                               bool useHbs) {
        std::vector<node> highDegreeRoots;
        for (node root : frontier) {
            if (G.degree(root) >= kNeighborParallelismThreshold) {
                highDegreeRoots.push_back(root);
            }
        }

        scratchInUse = highDegreeRoots.empty() ? 1 : threadScratch.size();
        const bool directBuckets = highDegreeRoots.empty();
        if (directBuckets) {
            threadScratch[0].recounts.clear();
        } else {
            resetScratch(scratchInUse);
        }
        count processed = 0;
        for (node root : frontier) {
            if (G.degree(root) < kNeighborParallelismThreshold) {
                processed += processRootSequential(
                    root, level, bucketBase, useHbs, true,
                    directBuckets ? buckets : threadScratch[0].buckets, threadScratch[0].recounts);
            }
        }

        std::vector<EdgeBlock> edgeBlocks;
        for (node root : highDegreeRoots) {
            if (!active[root].load(std::memory_order_relaxed)) {
                continue;
            }

            active[root].store(0, std::memory_order_relaxed);
            scores[root] = level;
            ++processed;
            const count degree = G.degree(root);
            for (index begin = 0; begin < degree; begin += kNeighborParallelismThreshold) {
                edgeBlocks.push_back(
                    {root, begin, std::min<index>(degree, begin + kNeighborParallelismThreshold)});
            }
        }

        if (!edgeBlocks.empty()) {
#pragma omp parallel
            {
                auto &scratch = threadScratch[omp_get_thread_num()];
                std::array<node, kLocalQueueSize> unusedQueue;
                count unusedRear = 0;
#pragma omp for schedule(dynamic, 8)
                for (omp_index blockIndex = 0;
                     blockIndex < static_cast<omp_index>(edgeBlocks.size()); ++blockIndex) {
                    const auto &block = edgeBlocks[blockIndex];
                    for (index neighborIndex = block.begin; neighborIndex < block.end;
                         ++neighborIndex) {
                        const node v = G.getIthNeighbor(Unsafe{}, block.source, neighborIndex);
                        decrementNeighbor(block.source, v, level, bucketBase, useHbs, false,
                                          unusedQueue, unusedRear, scratch.buckets,
                                          scratch.recounts);
                    }
                }
            }
        }

        if (!directBuckets)
            mergeScratchBuckets(scratchInUse);
        return processed;
    }

    bool processLevel(count level, count bucketBase, bool useHbs) {
        std::vector<node> frontier = takeExactBucket(level, useHbs);
        while (!frontier.empty()) {
            const count processed = frontier.size() > kParallelFrontierThreshold
                                        ? processLargeFrontier(frontier, level, bucketBase, useHbs)
                                        : processSmallFrontier(frontier, level, bucketBase, useHbs);
            if (processed > 0) {
                maxCore = std::max<index>(maxCore, level);
            }

            const std::vector<node> recounts = collectScratchRecounts(scratchInUse);
            if (!recounts.empty() && !recountVertices(recounts, level, bucketBase, useHbs)) {
                return false;
            }
            frontier = takeExactBucket(level, useHbs);
        }
        return true;
    }

    bool recountVertices(const std::vector<node> &vertices, count level, count bucketBase,
                         bool useHbs) {
        if (vertices.empty()) {
            return true;
        }

        std::atomic<bool> samplingError{false};
        resetScratch(threadScratch.size());

#pragma omp parallel for schedule(dynamic, 1) if (vertices.size() > 1)
        for (omp_index i = 0; i < static_cast<omp_index>(vertices.size()); ++i) {
            const node u = vertices[i];
            recountQueued[u].store(false, std::memory_order_relaxed);
            if (!active[u].load(std::memory_order_relaxed)) {
                continue;
            }

            if (samplingEnabled && acceptRecount && !acceptRecount(u, level)) {
                samplingError.store(true, std::memory_order_relaxed);
                continue;
            }

            count exactDegree = 0;
            G.forNeighborsOf(
                u, [&](node v) { exactDegree += active[v].load(std::memory_order_relaxed) != 0; });

            if (exactDegree < level) {
                count previousRoundDegree = 0;
                G.forNeighborsOf(u, [&](node v) {
                    previousRoundDegree += active[v].load(std::memory_order_relaxed)
                                           || degrees[v].load(std::memory_order_relaxed) == level;
                });

                if (previousRoundDegree < level) {
                    samplingError.store(true, std::memory_order_relaxed);
                    continue;
                }
                exactDegree = level;
            }

            degrees[u].store(exactDegree, std::memory_order_relaxed);
            auto &scratch = threadScratch[omp_get_thread_num()];
            if (exactDegree == level) {
                const count bucket = useHbs ? (level & kBucketMask) : 0;
                scratch.buckets[bucket].push_back(u);
            } else if (useHbs) {
                Aux::CoreBuckets::add(scratch.buckets, u, exactDegree, bucketBase);
            }
            setSampler(u, level);
        }

        mergeScratchBuckets(threadScratch.size());
        return !samplingError.load(std::memory_order_relaxed);
    }

    void filterRemaining(std::vector<node> &remaining, bool buildSparse = false, count level = 0) {
        if (remaining.empty())
            return;
        const count inputSize = remaining.size();
        filterBuffer.resize(inputSize);
        count outputSize = 0;

#pragma omp parallel shared(outputSize)
        {
            const int thread = omp_get_thread_num();
            const int threadCount = omp_get_num_threads();
            const count begin = inputSize * thread / threadCount;
            const count end = inputSize * (thread + 1) / threadCount;
            count localCount = 0;
            for (count i = begin; i < end; ++i) {
                const node u = remaining[i];
                if (!active[u].load(std::memory_order_relaxed))
                    continue;
                ++localCount;
                if (buildSparse) {
                    auto &scratch = threadScratch[thread];
                    if (degrees[u].load(std::memory_order_relaxed) == level) {
                        scratch.buckets[0].push_back(u);
                    }
                    if (samplingPossible && sampleMode[u].load(std::memory_order_relaxed)
                        && !sampleIsSafe(u, level + kBucketStride)) {
                        bool expected = false;
                        if (recountQueued[u].compare_exchange_strong(expected, true,
                                                                     std::memory_order_relaxed)) {
                            scratch.recounts.push_back(u);
                        }
                    }
                }
            }
            filterCounts[thread] = localCount;

#pragma omp barrier
#pragma omp single
            {
                outputSize = 0;
                for (int owner = 0; owner < threadCount; ++owner) {
                    filterOffsets[owner] = outputSize;
                    outputSize += filterCounts[owner];
                }
            }
#pragma omp barrier

            count output = filterOffsets[thread];
            for (count i = begin; i < end; ++i) {
                const node u = remaining[i];
                if (active[u].load(std::memory_order_relaxed)) {
                    filterBuffer[output++] = u;
                }
            }
        }

        filterBuffer.resize(outputSize);
        remaining.swap(filterBuffer);
    }
};

struct CoreResult {
    index maxCore;
    count restarts;
};

// Reuse the established sparse peeling implementation without copying its score vector.
class SparseCoreSolver final : public CoreDecomposition {
public:
    explicit SparseCoreSolver(const Graph &graph) : CoreDecomposition(graph, false, false) {}

    void runInto(std::vector<double> &scores) {
        scoreData.swap(scores);
        run();
        scoreData.swap(scores);
    }
};

inline bool preferSparsePeeling(const Graph &graph) {
    const count n = graph.numberOfNodes();
    if (n == 0 || graph.numberOfEdges() / n >= kHbsSwitchCore / 2)
        return false;
    count maximumDegree = 0;
// MSVC classic OpenMP (NETWORKIT_OMP2) lacks reduction(max); fall back to serial.
#ifndef NETWORKIT_OMP2
#pragma omp parallel for reduction(max : maximumDegree) schedule(static) if (n > 256)
#endif
    for (omp_index u = 0; u < static_cast<omp_index>(n); ++u) {
        maximumDegree = std::max(maximumDegree, graph.degree(static_cast<node>(u)));
    }
    // Preserve sampling/edge-block peeling for stars and dense components in sparse graphs.
    return maximumDegree < kNeighborParallelismThreshold;
}

// Shared by the public algorithm and white-box recovery tests.
inline CoreResult runWithRecovery(const Graph &G, std::vector<double> &scores, bool sampling,
                                  bool (*acceptRecount)(node, count) = nullptr) {
    if (!acceptRecount && preferSparsePeeling(G)) {
        SparseCoreSolver sparse(G);
        sparse.runInto(scores);
        return {sparse.maxCoreNumber(), 0};
    }
    scores.assign(G.upperNodeIdBound(), 0.0);
    ParallelKCoreSolver solver(G, scores, sampling, acceptRecount);
    if (solver.run()) {
        return {solver.maxCoreNumber(), 0};
    }
    scores.assign(G.upperNodeIdBound(), 0.0);
    ParallelKCoreSolver exactSolver(G, scores, false);
    if (!exactSolver.run()) {
        throw std::runtime_error("ParallelCoreDecomposition exact recovery run failed");
    }
    return {exactSolver.maxCoreNumber(), 1};
}

} // namespace CoreDecompositionDetail
} // namespace NetworKit
#endif
