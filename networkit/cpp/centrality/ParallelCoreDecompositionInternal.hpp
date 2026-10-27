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
#include <networkit/auxiliary/HierarchicalBucketPQ.hpp>
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
constexpr count kSampleWindow = 512;

struct ThreadScratch final {
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
          z(G.upperNodeIdBound()), queue(z, omp_get_max_threads()),
          active(new std::atomic<char>[z]), threadScratch(omp_get_max_threads()) {}

    bool run() {
        const count n = G.numberOfNodes();
        if (n == 0) {
            return true;
        }

        initialize();

        while (!queue.empty()) {
            // Keys cannot drop below an extracted level, so sampled vertices are rechecked
            // before the queue reaches each window, including windows without vertices.
            if (samplingPossible && queue.getMinKey() >= nextSampleWindow) {
                if (!recountUnsafeSamples(nextSampleWindow)) {
                    return false;
                }
                nextSampleWindow += kSampleWindow;
                continue;
            }

            const auto extracted = queue.extractMin();
            if (!processLevel(extracted.first, std::vector<node>(extracted.second.begin(),
                                                                 extracted.second.end()))) {
                return false;
            }
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
    Aux::HierarchicalBucketPQ queue;
    std::unique_ptr<Sampler[]> samplers;
    std::unique_ptr<std::atomic<bool>[]> recountQueued;
    // Atomics protect accesses within a peeling round. Relaxed ordering is sufficient:
    // OpenMP region/barrier boundaries publish sampler configuration and scratch buffers
    // between initialization, peeling and recount phases. No flag publishes other data.
    // A vertex stays active until it is processed, which is after the queue extracts it.
    std::unique_ptr<std::atomic<char>[]> active;
    std::unique_ptr<std::atomic<char>[]> sampleMode;
    std::vector<ThreadScratch> threadScratch;
    // High-degree vertices extracted by the serial small-frontier path but not yet processed.
    std::vector<node> deferred;
    count scratchInUse{1};
    bool samplingPossible{false};
    count nextSampleWindow{0};
    index maxCore{0};

    void initialize() {
        count maxDegree = 0;
// MSVC classic OpenMP (NETWORKIT_OMP2) lacks reduction(max); fall back to serial.
#ifndef NETWORKIT_OMP2
#pragma omp parallel for reduction(max : maxDegree) schedule(static)
#endif
        for (omp_index u = 0; u < static_cast<omp_index>(z); ++u) {
            const count degree = G.degree(static_cast<node>(u));
            queue.insert(degree, static_cast<Aux::HierarchicalBucketPQ::Value>(u),
                         omp_get_thread_num());
            active[u].store(1, std::memory_order_relaxed);
            maxDegree = std::max(maxDegree, degree);
        }
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

        const count degree = queue.getKey(u);
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
        const count degree = queue.getKey(u);
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
            threadScratch[thread].recounts.clear();
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

    bool recountUnsafeSamples(count level) {
        resetScratch(threadScratch.size());
#pragma omp parallel for schedule(static)
        for (omp_index u = 0; u < static_cast<omp_index>(z); ++u) {
            if (active[u].load(std::memory_order_relaxed)
                && sampleMode[u].load(std::memory_order_relaxed)
                && !sampleIsSafe(static_cast<node>(u), level + kSampleWindow)) {
                bool expected = false;
                if (recountQueued[u].compare_exchange_strong(expected, true,
                                                             std::memory_order_relaxed)) {
                    threadScratch[omp_get_thread_num()].recounts.push_back(static_cast<node>(u));
                }
            }
        }
        return recountVertices(collectScratchRecounts(threadScratch.size()), level);
    }

    void decrementNeighbor(node source, node target, count level, count thread,
                           bool allowLocalQueue, std::array<node, kLocalQueueSize> &localQueue,
                           count &rear, std::vector<node> &localRecounts) {
        if (queue.getKey(target) <= level) {
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

        if (queue.decreaseKeyRelative(1, target, thread) && allowLocalQueue
            && rear < kLocalQueueSize && queue.getKey(target) == level
            && queue.tryExtract(target, thread)) {
            localQueue[rear++] = target;
        }
    }

    count processRootSequential(node root, count level, count thread, bool deferHighDegree,
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
                deferred.push_back(u);
                continue;
            }

            active[u].store(0, std::memory_order_relaxed);
            scores[u] = level;
            ++processed;
            G.forNeighborsOf(u, [&](node v) {
                decrementNeighbor(u, v, level, thread, true, localQueue, rear, localRecounts);
            });
        }
        return processed;
    }

    count processLargeFrontier(const std::vector<node> &frontier, count level) {
        scratchInUse = threadScratch.size();
        resetScratch(scratchInUse);
        count processed = 0;
#pragma omp parallel for schedule(dynamic, 64) reduction(+ : processed)
        for (omp_index i = 0; i < static_cast<omp_index>(frontier.size()); ++i) {
            const count thread = omp_get_thread_num();
            processed += processRootSequential(frontier[i], level, thread, false,
                                               threadScratch[thread].recounts);
        }
        return processed;
    }

    count processSmallFrontier(const std::vector<node> &frontier, count level) {
        std::vector<node> highDegreeRoots;
        for (node root : frontier) {
            if (G.degree(root) >= kNeighborParallelismThreshold) {
                highDegreeRoots.push_back(root);
            }
        }

        scratchInUse = highDegreeRoots.empty() ? 1 : threadScratch.size();
        resetScratch(scratchInUse);
        count processed = 0;
        for (node root : frontier) {
            if (G.degree(root) < kNeighborParallelismThreshold) {
                processed += processRootSequential(root, level, 0, true, threadScratch[0].recounts);
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
                const count thread = omp_get_thread_num();
                auto &scratch = threadScratch[thread];
                std::array<node, kLocalQueueSize> unusedQueue;
                count unusedRear = 0;
#pragma omp for schedule(dynamic, 8)
                for (omp_index blockIndex = 0;
                     blockIndex < static_cast<omp_index>(edgeBlocks.size()); ++blockIndex) {
                    const auto &block = edgeBlocks[blockIndex];
                    for (index neighborIndex = block.begin; neighborIndex < block.end;
                         ++neighborIndex) {
                        const node v = G.getIthNeighbor(Unsafe{}, block.source, neighborIndex);
                        decrementNeighbor(block.source, v, level, thread, false, unusedQueue,
                                          unusedRear, scratch.recounts);
                    }
                }
            }
        }

        return processed;
    }

    bool processLevel(count level, std::vector<node> frontier) {
        while (!frontier.empty()) {
            const count processed = frontier.size() > kParallelFrontierThreshold
                                        ? processLargeFrontier(frontier, level)
                                        : processSmallFrontier(frontier, level);
            if (processed > 0) {
                maxCore = std::max<index>(maxCore, level);
            }

            const std::vector<node> recounts = collectScratchRecounts(scratchInUse);
            if (!recounts.empty() && !recountVertices(recounts, level)) {
                return false;
            }
            frontier.swap(deferred);
            deferred.clear();
        }
        return true;
    }

    bool recountVertices(const std::vector<node> &vertices, count level) {
        if (vertices.empty()) {
            return true;
        }

        std::atomic<bool> samplingError{false};

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
                    previousRoundDegree +=
                        active[v].load(std::memory_order_relaxed) || queue.getKey(v) == level;
                });

                if (previousRoundDegree < level) {
                    samplingError.store(true, std::memory_order_relaxed);
                    continue;
                }
                exactDegree = level;
            }

            queue.decreaseKey(exactDegree, u, omp_get_thread_num());
            setSampler(u, level);
        }

        return !samplingError.load(std::memory_order_relaxed);
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
