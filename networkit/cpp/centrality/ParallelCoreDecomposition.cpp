/*
 * ParallelCoreDecomposition.cpp
 *
 * Exact shared-memory k-core decomposition based on Liu et al., SIGMOD 2025.
 */

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

#include <networkit/centrality/ParallelCoreDecomposition.hpp>

namespace NetworKit {
namespace {

constexpr count log2SingleBuckets = 3;
constexpr count numSingleBuckets = count{1} << log2SingleBuckets;
constexpr count numIntermediateBuckets = 6;
constexpr count numBuckets = numSingleBuckets + numIntermediateBuckets;
constexpr count bucketMask = numSingleBuckets - 1;
constexpr count bucketStride = numSingleBuckets << numIntermediateBuckets;

constexpr count sampleThreshold = 2000;
constexpr double initialReduceRatio = 0.1;
constexpr count log2ErrorFactor = 32;
constexpr count baselineExpectedHits =
    static_cast<count>(log2ErrorFactor / (initialReduceRatio * initialReduceRatio));
constexpr double biasFactor = 0.5;
constexpr double errorRateTolerance = 1e-10;

constexpr count localQueueSize = 128;
constexpr count neighborParallelismThreshold = 128;
constexpr count parallelFrontierThreshold = 256;
constexpr count hbsSwitchCore = 16;

using BucketArray = std::array<std::vector<node>, numBuckets>;

struct ThreadScratch final {
    BucketArray buckets;
    std::vector<node> recounts;
};

struct EdgeBlock final {
    node source;
    index begin;
    index end;
};

uint32_t hash32(uint32_t value) {
    value = (value + 0x7ed55d16U) + (value << 12);
    value = (value ^ 0xc761c23cU) ^ (value >> 19);
    value = (value + 0x165667b1U) + (value << 5);
    value = (value + 0xd3a2646cU) ^ (value << 9);
    value = (value + 0xfd7046c5U) + (value << 3);
    value = (value ^ 0xb55a4f09U) ^ (value >> 16);
    return value;
}

count log2Up(count value) {
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

count mostSignificantBit(count value) {
    count result = 0;
    while (value >>= 1) {
        ++result;
    }
    return result;
}

class Sampler final {
public:
    void reset(count newExpectedHits, double newRate) {
        expectedHits = newExpectedHits;
        rate = std::min(1.0, newRate);
        threshold =
            static_cast<uint32_t>(rate * static_cast<double>(std::numeric_limits<uint32_t>::max()));
        hits.store(0, std::memory_order_relaxed);
    }

    bool sample(uint32_t randomNumber) {
        count current = hits.load(std::memory_order_relaxed);
        if (current >= expectedHits || randomNumber >= threshold) {
            return false;
        }

        current = hits.fetch_add(1, std::memory_order_relaxed);
        return current + 1 == expectedHits;
    }

    count numberOfHits() const { return hits.load(std::memory_order_relaxed); }

    count numberOfExpectedHits() const { return expectedHits; }

    double sampleRate() const { return rate; }

private:
    std::atomic<count> hits{0};
    count expectedHits{0};
    uint32_t threshold{0};
    double rate{0.0};
};

class ParallelKCoreSolver final {
public:
    ParallelKCoreSolver(const Graph &G, std::vector<double> &scores, bool samplingEnabled)
        : G(G), scores(scores), samplingEnabled(samplingEnabled), z(G.upperNodeIdBound()),
          degrees(new std::atomic<count>[z]), samplers(new Sampler[z]),
          recountQueued(new std::atomic<bool>[z]), active(z, 1), sampleMode(z, 0),
          threadScratch(omp_get_max_threads()), filterCounts(omp_get_max_threads()),
          filterOffsets(omp_get_max_threads()), seenAtEpoch(z, 0) {
        for (node u = 0; u < z; ++u) {
            recountQueued[u].store(false, std::memory_order_relaxed);
        }
    }

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
        if (averageDegree < hbsSwitchCore) {
            for (count level = 0; level < hbsSwitchCore && !remaining.empty(); ++level) {
                buckets[0].clear();
                if (!buildSparseBucket(remaining, level)) {
                    return false;
                }
                if (!processLevel(level, level, false)) {
                    return false;
                }
                filterRemaining(remaining);
                firstHbsLevel = level + 1;
            }
        }

        for (count baseLevel = 0; !remaining.empty(); baseLevel += bucketStride) {
            if (baseLevel > maximumInitialDegree) {
                return false;
            }

            clearBuckets();
            const count levelBegin = std::max(baseLevel, firstHbsLevel);
            if (!buildBuckets(remaining, levelBegin, baseLevel + bucketStride, baseLevel, true)) {
                return false;
            }

            for (count level = levelBegin; level < baseLevel + bucketStride && !remaining.empty();
                 ++level) {
                if (level != baseLevel && (level & bucketMask) == 0) {
                    redistributeIntermediateBucket(level);
                }

                const count bucketBase = level & ~bucketMask;
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
    count z;
    std::unique_ptr<std::atomic<count>[]> degrees;
    std::unique_ptr<Sampler[]> samplers;
    std::unique_ptr<std::atomic<bool>[]> recountQueued;
    std::vector<char> active;
    std::vector<char> sampleMode;
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
#pragma omp parallel for reduction(max : maxDegree) schedule(static)
        for (omp_index u = 0; u < static_cast<omp_index>(z); ++u) {
            const count degree = G.degree(static_cast<node>(u));
            degrees[u].store(degree, std::memory_order_relaxed);
            maxDegree = std::max(maxDegree, degree);
        }
        maximumInitialDegree = maxDegree;
        samplingPossible = samplingEnabled && maxDegree * initialReduceRatio >= sampleThreshold;

        if (samplingPossible) {
#pragma omp parallel for schedule(static)
            for (omp_index u = 0; u < static_cast<omp_index>(z); ++u) {
                setSampler(static_cast<node>(u), 0);
            }
        }
    }

    void setSampler(node u, count level) {
        if (!samplingPossible) {
            sampleMode[u] = 0;
            return;
        }

        const count degree = degrees[u].load(std::memory_order_relaxed);
        const double reducedDegree = degree * initialReduceRatio;
        const double remainingAfterReduction =
            (degree - std::min(degree, level)) * (1.0 - initialReduceRatio);

        if (reducedDegree >= sampleThreshold && level < reducedDegree * biasFactor
            && baselineExpectedHits < remainingAfterReduction) {
            sampleMode[u] = 1;
            const count distance = degree - level;
            const count logarithm = log2Up(distance);
            const count expectedHits = log2ErrorFactor * logarithm * logarithm / 2;
            const double rate = expectedHits / ((1.0 - initialReduceRatio) * degree);
            samplers[u].reset(expectedHits, rate);
        } else {
            sampleMode[u] = 0;
        }
    }

    bool sampleIsSafe(node u, count horizon) const {
        const count degree = degrees[u].load(std::memory_order_relaxed);
        if (degree * initialReduceRatio * biasFactor < horizon || degree <= horizon) {
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
        return std::exp(exponent) < errorRateTolerance;
    }

    template <typename Buckets>
    void addToBucket(Buckets &target, node u, count degree, count baseLevel) {
        if (degree < baseLevel || degree >= baseLevel + bucketStride) {
            return;
        }

        if (degree < baseLevel + numSingleBuckets) {
            target[degree & bucketMask].push_back(u);
            return;
        }

        const count differingBit = mostSignificantBit(degree ^ baseLevel);
        const count bucket = numSingleBuckets + differingBit - log2SingleBuckets;
        target[bucket].push_back(u);
    }

    template <typename Buckets>
    void moveBucket(Buckets &target, node u, count degree, count baseLevel) {
        if (degree < baseLevel || degree >= baseLevel + bucketStride) {
            return;
        }

        if (degree < baseLevel + numSingleBuckets) {
            target[degree & bucketMask].push_back(u);
            return;
        }

        const count differingBit = mostSignificantBit(degree ^ baseLevel);
        const count previousDifferingBit = mostSignificantBit((degree + 1) ^ baseLevel);
        if (differingBit != previousDifferingBit) {
            const count bucket = numSingleBuckets + differingBit - log2SingleBuckets;
            target[bucket].push_back(u);
        }
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
            for (count bucket = 0; bucket < numBuckets; ++bucket) {
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
            addToBucket(scratch.buckets, u, degree, baseLevel);
            if (samplingPossible && sampleMode[u] && !sampleIsSafe(u, horizon)) {
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

    bool buildSparseBucket(const std::vector<node> &remaining, count level) {
        resetScratch(threadScratch.size());
#pragma omp parallel for schedule(static)
        for (omp_index i = 0; i < static_cast<omp_index>(remaining.size()); ++i) {
            const node u = remaining[i];
            const count degree = degrees[u].load(std::memory_order_relaxed);
            auto &scratch = threadScratch[omp_get_thread_num()];
            if (degree == level) {
                scratch.buckets[0].push_back(u);
            }
            if (samplingPossible && sampleMode[u] && !sampleIsSafe(u, level + bucketStride)) {
                bool expected = false;
                if (recountQueued[u].compare_exchange_strong(expected, true,
                                                             std::memory_order_relaxed)) {
                    scratch.recounts.push_back(u);
                }
            }
        }
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
            if (active[u] && seenAtEpoch[u] != currentEpoch) {
                seenAtEpoch[u] = currentEpoch;
                candidates[output++] = u;
            }
        }
        candidates.resize(output);
        return candidates;
    }

    std::vector<node> takeExactBucket(count level, bool useHbs) {
        const count bucket = useHbs ? (level & bucketMask) : 0;
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
        count intermediate = numIntermediateBuckets;
        for (count i = numIntermediateBuckets; i-- > 0;) {
            const count mask = (numSingleBuckets << i) - 1;
            if ((level & mask) == 0) {
                intermediate = i;
                break;
            }
        }
        if (intermediate == numIntermediateBuckets) {
            return;
        }

        const count bucket = numSingleBuckets + intermediate;
        std::vector<node> candidates;
        candidates.swap(buckets[bucket]);
        candidates = uniqueActive(std::move(candidates));

        resetScratch(threadScratch.size());
#pragma omp parallel for schedule(static)
        for (omp_index i = 0; i < static_cast<omp_index>(candidates.size()); ++i) {
            const node u = candidates[i];
            const count degree = degrees[u].load(std::memory_order_relaxed);
            addToBucket(threadScratch[omp_get_thread_num()].buckets, u, degree, level);
        }
        mergeScratchBuckets(threadScratch.size());
    }

    void decrementNeighbor(node source, node target, count level, count bucketBase, bool useHbs,
                           bool allowLocalQueue, std::array<node, localQueueSize> &localQueue,
                           count &rear, BucketArray &localBuckets,
                           std::vector<node> &localRecounts) {
        count oldDegree = degrees[target].load(std::memory_order_relaxed);
        if (oldDegree <= level) {
            return;
        }

        if (samplingEnabled && sampleMode[target]) {
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
                if (newDegree == level && allowLocalQueue && rear < localQueueSize) {
                    localQueue[rear++] = target;
                } else if (useHbs) {
                    moveBucket(localBuckets, target, newDegree, bucketBase);
                } else if (newDegree == level) {
                    localBuckets[0].push_back(target);
                }
                return;
            }
        }
    }

    count processRootSequential(node root, count level, count bucketBase, bool useHbs,
                                bool deferHighDegree, ThreadScratch &scratch) {
        std::array<node, localQueueSize> localQueue;
        count front = 0;
        count rear = 1;
        count processed = 0;
        localQueue[0] = root;

        while (front < rear) {
            const node u = localQueue[front++];
            if (!active[u]) {
                continue;
            }

            if (deferHighDegree && G.degree(u) >= neighborParallelismThreshold) {
                const count bucket = useHbs ? (level & bucketMask) : 0;
                scratch.buckets[bucket].push_back(u);
                continue;
            }

            active[u] = 0;
            scores[u] = level;
            ++processed;
            G.forNeighborsOf(u, [&](node v) {
                decrementNeighbor(u, v, level, bucketBase, useHbs, true, localQueue, rear,
                                  scratch.buckets, scratch.recounts);
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
            processed +=
                processRootSequential(frontier[i], level, bucketBase, useHbs, false, scratch);
        }
        mergeScratchBuckets(scratchInUse);
        return processed;
    }

    count processSmallFrontier(const std::vector<node> &frontier, count level, count bucketBase,
                               bool useHbs) {
        std::vector<node> highDegreeRoots;
        for (node root : frontier) {
            if (G.degree(root) >= neighborParallelismThreshold) {
                highDegreeRoots.push_back(root);
            }
        }

        scratchInUse = highDegreeRoots.empty() ? 1 : threadScratch.size();
        resetScratch(scratchInUse);
        count processed = 0;
        for (node root : frontier) {
            if (G.degree(root) < neighborParallelismThreshold) {
                processed +=
                    processRootSequential(root, level, bucketBase, useHbs, true, threadScratch[0]);
            }
        }

        std::vector<EdgeBlock> edgeBlocks;
        for (node root : highDegreeRoots) {
            if (!active[root]) {
                continue;
            }

            active[root] = 0;
            scores[root] = level;
            ++processed;
            const count degree = G.degree(root);
            for (index begin = 0; begin < degree; begin += neighborParallelismThreshold) {
                edgeBlocks.push_back(
                    {root, begin, std::min<index>(degree, begin + neighborParallelismThreshold)});
            }
        }

        if (!edgeBlocks.empty()) {
#pragma omp parallel
            {
                auto &scratch = threadScratch[omp_get_thread_num()];
                std::array<node, localQueueSize> unusedQueue;
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

        mergeScratchBuckets(scratchInUse);
        return processed;
    }

    bool processLevel(count level, count bucketBase, bool useHbs) {
        std::vector<node> frontier = takeExactBucket(level, useHbs);
        while (!frontier.empty()) {
            const count processed = frontier.size() > parallelFrontierThreshold
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

#pragma omp parallel for schedule(dynamic, 1)
        for (omp_index i = 0; i < static_cast<omp_index>(vertices.size()); ++i) {
            const node u = vertices[i];
            recountQueued[u].store(false, std::memory_order_relaxed);
            if (!active[u]) {
                continue;
            }

            count exactDegree = 0;
            G.forNeighborsOf(u, [&](node v) { exactDegree += active[v] != 0; });

            if (exactDegree < level) {
                count previousRoundDegree = 0;
                G.forNeighborsOf(u, [&](node v) {
                    previousRoundDegree +=
                        active[v] || degrees[v].load(std::memory_order_relaxed) == level;
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
                const count bucket = useHbs ? (level & bucketMask) : 0;
                scratch.buckets[bucket].push_back(u);
            } else if (useHbs) {
                addToBucket(scratch.buckets, u, exactDegree, bucketBase);
            }
            setSampler(u, level);
        }

        mergeScratchBuckets(threadScratch.size());
        return !samplingError.load(std::memory_order_relaxed);
    }

    void filterRemaining(std::vector<node> &remaining) {
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
                localCount += active[remaining[i]] != 0;
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
                if (active[u]) {
                    filterBuffer[output++] = u;
                }
            }
        }

        filterBuffer.resize(outputSize);
        remaining.swap(filterBuffer);
    }
};

} // namespace

ParallelCoreDecomposition::ParallelCoreDecomposition(const Graph &G, bool normalized,
                                                     bool useSampling)
    : Centrality(G, normalized), useSampling(useSampling) {
    if (G.isDirected()) {
        throw std::runtime_error("ParallelCoreDecomposition requires an undirected graph");
    }
    if (G.numberOfSelfLoops()) {
        throw std::runtime_error("ParallelCoreDecomposition does not support self-loops; call "
                                 "Graph.removeSelfLoops() first");
    }
    if (G.numberOfNodes() != G.upperNodeIdBound()) {
        throw std::runtime_error("ParallelCoreDecomposition requires continuous node ids");
    }
}

void ParallelCoreDecomposition::run() {
    scoreData.assign(G.upperNodeIdBound(), 0.0);
    restarts = 0;

    ParallelKCoreSolver solver(G, scoreData, useSampling);
    if (!solver.run()) {
        ++restarts;
        scoreData.assign(G.upperNodeIdBound(), 0.0);
        ParallelKCoreSolver exactSolver(G, scoreData, false);
        if (!exactSolver.run()) {
            throw std::runtime_error("ParallelCoreDecomposition exact recovery run failed");
        }
        maxCore = exactSolver.maxCoreNumber();
    } else {
        maxCore = solver.maxCoreNumber();
    }

    if (normalized) {
        count maximumDegree = 0;
        G.forNodes([&](node u) { maximumDegree = std::max(maximumDegree, G.degree(u)); });
        if (maximumDegree > 0) {
            G.parallelForNodes([&](node u) { scoreData[u] /= maximumDegree; });
        }
    }

    hasRun = true;
}

Cover ParallelCoreDecomposition::getCover() const {
    assureFinished();
    Cover result(G.upperNodeIdBound());
    result.setUpperBound(G.upperNodeIdBound());
    G.forNodes([&](node u) {
        for (index core = 0; core <= scoreData[u]; ++core) {
            result.addToSubset(core, u);
        }
    });
    return result;
}

Partition ParallelCoreDecomposition::getPartition() const {
    assureFinished();
    Partition result(G.upperNodeIdBound());
    result.allToSingletons();
    G.forNodes([&](node u) { result.moveToSubset(static_cast<index>(scoreData[u]), u); });
    return result;
}

index ParallelCoreDecomposition::maxCoreNumber() const {
    assureFinished();
    return maxCore;
}

count ParallelCoreDecomposition::numberOfRestarts() const {
    assureFinished();
    return restarts;
}

double ParallelCoreDecomposition::maximum() {
    return G.numberOfNodes() > 0 ? static_cast<double>(G.numberOfNodes() - 1) : 0.0;
}

} // namespace NetworKit
