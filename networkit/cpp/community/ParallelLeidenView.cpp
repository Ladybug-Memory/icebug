/*
 * ParallelLeidenView.cpp
 *
 *  Memory-efficient implementation of ParallelLeiden using CoarsenedGraphView
 */

#include <cstdint>
#include <cstdlib>
#include <networkit/community/ParallelLeidenView.hpp>
#ifndef _WIN32
#include <dlfcn.h>
#endif
#include <stdexcept>
#include <unordered_map>

namespace NetworKit {

count ParallelLeidenView::nodeSize(const Graph &graph, node u) {
    tlx::unused(graph);
    tlx::unused(u);
    return 1;
}

count ParallelLeidenView::nodeSize(const CoarsenedGraphView &graph, node u) {
    return graph.getOriginalNodes(u).size();
}

ParallelLeidenView::ParallelLeidenView(const Graph &graph, int iterations, bool randomize,
                                       double gamma)
    : CommunityDetectionAlgorithm(graph), gamma(gamma), numberOfIterations(iterations),
      random(randomize) {
    this->result = Partition(graph.numberOfNodes());
    this->result.allToSingletons();

    if (const char *epsilonEnv = std::getenv("NETWORKIT_LEIDEN_MOVE_EPS")) {
        try {
            const double parsed = std::stod(epsilonEnv);
            if (parsed >= 0.0) {
                moveGainMarginEpsilon = parsed;
            }
        } catch (...) {
            // Keep default if parsing fails.
        }
    }
    if (const char *maxInnerEnv = std::getenv("NETWORKIT_LEIDEN_MAX_INNER")) {
        try {
            const int parsed = std::stoi(maxInnerEnv);
            if (parsed > 0) {
                maxInnerIterations = parsed;
            }
        } catch (...) {
            // Keep default if parsing fails.
        }
    }
    if (const char *maxRequeuesPerNodeEnv = std::getenv("ICEBUG_LEIDEN_MAX_REQUEUES_PER_NODE")) {
        try {
            const int parsed = std::stoi(maxRequeuesPerNodeEnv);
            if (parsed > 0) {
                maxRequeuesPerNode = parsed;
            }
        } catch (...) {
            // Keep default if parsing fails.
        }
    }
    if (const char *vectorOversizeEnv = std::getenv("NETWORKIT_LEIDEN_VECTOR_OVERSIZE")) {
        try {
            const int parsed = std::stoi(vectorOversizeEnv);
            if (parsed >= 1) {
                VECTOR_OVERSIZE = parsed;
            }
        } catch (...) {
            // Keep default if parsing fails.
        }
    }
    if (const char *minReductionEnv = std::getenv("NETWORKIT_LEIDEN_MIN_COMM_REDUCTION")) {
        try {
            const double parsed = std::stod(minReductionEnv);
            if (parsed >= 0.0) {
                minCommunityReduction = parsed;
            }
        } catch (...) {
            // Keep default if parsing fails.
        }
    }
    if (const char *scoringLibEnv = std::getenv("NETWORKIT_LEIDEN_MOVE_SCORING_LIB")) {
        loadMoveScoringExtension(scoringLibEnv);
    }
}

ParallelLeidenView::~ParallelLeidenView() {
    unloadMoveScoringExtension();
    currentCoarsenedView.reset();
    communityVolumes.clear();
    communitySizes.clear();
}

void ParallelLeidenView::loadMoveScoringExtension(const std::string &sharedLibraryPath) {
#ifdef _WIN32
    throw std::runtime_error(
        "ParallelLeidenView shared-library scoring extensions are not supported on Windows");
#else
    void *handle = dlopen(sharedLibraryPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        throw std::runtime_error("Failed to load ParallelLeidenView scoring extension '"
                                 + sharedLibraryPath + "': " + dlerror());
    }

    dlerror();
    auto *communityScore = reinterpret_cast<ParallelLeidenCommunityScoreFunction>(
        dlsym(handle, "networkitParallelLeidenCommunityScore"));
    const char *communityScoreError = dlerror();
    if (communityScoreError != nullptr || communityScore == nullptr) {
        dlclose(handle);
        throw std::runtime_error(
            "ParallelLeidenView scoring extension '" + sharedLibraryPath
            + "' does not export required symbol networkitParallelLeidenCommunityScore");
    }

    dlerror();
    auto *thresholdScore = reinterpret_cast<ParallelLeidenCommunityScoreFunction>(
        dlsym(handle, "networkitParallelLeidenCurrentCommunityThreshold"));
    const char *thresholdScoreError = dlerror();
    if (thresholdScoreError != nullptr) {
        thresholdScore = &modularityThresholdScore;
    }

    unloadMoveScoringExtension();
    scoringExtensionHandle_ = handle;
    communityScoreFunction_ = communityScore;
    currentCommunityThresholdFunction_ =
        thresholdScore != nullptr ? thresholdScore : &modularityThresholdScore;
    dlerror();
    auto *refineRSet = reinterpret_cast<ParallelLeidenRefineSetConditionFunction>(
        dlsym(handle, "networkitParallelLeidenRefineRSetCondition"));
    const char *refineRSetError = dlerror();
    refineRSetConditionFunction_ = refineRSetError == nullptr && refineRSet != nullptr
                                       ? refineRSet
                                       : &modularityRefineRSetCondition;

    dlerror();
    auto *refineTSet = reinterpret_cast<ParallelLeidenRefineSetConditionFunction>(
        dlsym(handle, "networkitParallelLeidenRefineTSetCondition"));
    const char *refineTSetError = dlerror();
    refineTSetConditionFunction_ = refineTSetError == nullptr && refineTSet != nullptr
                                       ? refineTSet
                                       : &modularityRefineTSetCondition;
    scoringExtensionPath_ = sharedLibraryPath;
#endif
}

void ParallelLeidenView::unloadMoveScoringExtension() {
    communityScoreFunction_ = &modularityCommunityScore;
    currentCommunityThresholdFunction_ = &modularityThresholdScore;
    refineRSetConditionFunction_ = &modularityRefineRSetCondition;
    refineTSetConditionFunction_ = &modularityRefineTSetCondition;
    scoringExtensionPath_.clear();
#ifndef _WIN32
    if (scoringExtensionHandle_ != nullptr) {
        dlclose(scoringExtensionHandle_);
        scoringExtensionHandle_ = nullptr;
    }
#else
    scoringExtensionHandle_ = nullptr;
#endif
}

void ParallelLeidenView::run() {
    if (VECTOR_OVERSIZE < 1) {
        throw std::invalid_argument("VECTOR_OVERSIZE cant be smaller than 1");
    }
    auto totalTime = Aux::Timer();
    totalTime.start();

    do { // Leiden iteration
        INFO(numberOfIterations, " Leiden iteration(s) left");
        numberOfIterations--;
        changed = false;
        INFO("Using move gain epsilon=", std::setprecision(6), moveGainMarginEpsilon,
             " | max inner iterations=", maxInnerIterations,
             " | max requeues per node=", maxRequeuesPerNode,
             " | vector oversize=", VECTOR_OVERSIZE,
             " | min community reduction=", minCommunityReduction);

        // Start with the original graph
        const Graph *currentGraph = G;
        currentCoarsenedView.reset();

        Partition refined;

        // Calculate volumes for the current graph (either original or coarsened view)
        if (currentCoarsenedView) {
            calculateVolumes(*currentCoarsenedView);
        } else {
            calculateVolumes(*currentGraph);
        }

        int innerIterations = 0;
        count previousCommunities = result.numberOfSubsets();
        INFO("Starting inner loop with ", result.numberOfSubsets(), " communities");
        do {
            innerIterations++;
            if (innerIterations > maxInnerIterations) {
                INFO("Reached max inner iterations (", maxInnerIterations, ")");
                break;
            }
            handler.assureRunning();

            // Parallel move phase
            MoveStats moveStats;
            if (currentCoarsenedView) {
                moveStats = parallelMove(*currentCoarsenedView);
            } else {
                moveStats = parallelMove(*currentGraph);
            }

            const double avgGainMargin =
                moveStats.moved ? (moveStats.gainMarginSum / moveStats.moved) : 0.0;
            const double minGainMargin = moveStats.moved ? moveStats.gainMarginMin : 0.0;
            const double maxGainMargin = moveStats.moved ? moveStats.gainMarginMax : 0.0;
            const count candidateMoves = moveStats.moved + moveStats.marginalMovesRejected;
            const double rejectedPct = candidateMoves ? (100.0 * moveStats.marginalMovesRejected
                                                         / static_cast<double>(candidateMoves))
                                                      : 0.0;
            INFO("Inner iter ", innerIterations, ": moved ", moveStats.moved, " nodes, ",
                 result.numberOfSubsets(), " communities",
                 " | singleton moves=", moveStats.movedToSingleton,
                 " | marginal rejected=", moveStats.marginalMovesRejected, " (",
                 std::setprecision(4), rejectedPct, "%)",
                 " | avg gain margin=", std::setprecision(6), avgGainMargin,
                 " | min/max gain margin=", minGainMargin, "/", maxGainMargin);

            // If each community consists of exactly one node we're done
            count numNodes = currentCoarsenedView ? currentCoarsenedView->numberOfNodes()
                                                  : currentGraph->numberOfNodes();
            if (numNodes == result.numberOfSubsets()) {
                break;
            }

            handler.assureRunning();

            // Parallel refine phase
            bool refineMadeChanges = false;
            if (currentCoarsenedView) {
                refined = parallelRefine(*currentCoarsenedView, refineMadeChanges);
            } else {
                refined = parallelRefine(*currentGraph, refineMadeChanges);
            }

            const bool zeroMoveAndRefinementMadeNoChanges =
                moveStats.moved == 0 && !refineMadeChanges;

            if (zeroMoveAndRefinementMadeNoChanges) {
                INFO("Stopping inner loop at inner iteration ", innerIterations,
                     ": local moving made zero moves and refinement made no changes.");
                break;
            }

            handler.assureRunning();

            // Create coarsened view instead of new graph.
            if (currentCoarsenedView) {
                Partition compactRefined = refined;
                compactRefined.compact();
                auto newCoarsenedView =
                    std::make_shared<CoarsenedGraphView>(*currentCoarsenedView, compactRefined);

                // Create new partition for the new coarsened view
                Partition p(newCoarsenedView->numberOfNodes());
                p.setUpperBound(result.upperBound());

                currentCoarsenedView->parallelForNodes(
                    [&compactRefined, &p, this](node u) { p[compactRefined[u]] = result[u]; });

                result = std::move(p);
                currentCoarsenedView = newCoarsenedView;

            } else {
                // First coarsening from original graph
                ParallelPartitionCoarseningView ppcView(*currentGraph, refined);
                ppcView.run();
                auto newCoarsenedView = ppcView.getCoarsenedGraphView();
                const auto &map = ppcView.getFineToCoarseNodeMapping();

                // Maintain Partition, add every coarse Node to the community its fine Nodes were in
                Partition p(newCoarsenedView->numberOfNodes());
                p.setUpperBound(result.upperBound());
                currentGraph->parallelForNodes([&map, &p, this](node u) { p[map[u]] = result[u]; });

                result = std::move(p);

                currentCoarsenedView = newCoarsenedView;
                currentGraph = nullptr;
            }

            calculateVolumes(*currentCoarsenedView);

            const count currentCommunities = result.numberOfSubsets();
            if (minCommunityReduction > 0.0 && previousCommunities > 0) {
                const double reduction =
                    static_cast<double>(previousCommunities - currentCommunities)
                    / static_cast<double>(previousCommunities);
                if (reduction < minCommunityReduction) {
                    INFO("Stopping inner loop: community reduction ", reduction,
                         " below threshold ", minCommunityReduction);
                    break;
                }
            }
            previousCommunities = currentCommunities;

        } while (true);

        flattenPartition();
        INFO("Leiden iteration done, took ", totalTime.elapsedTag(), " so far");

    } while (changed && numberOfIterations > 0);

    hasRun = true;
}

template <typename GraphType>
void ParallelLeidenView::calculateVolumes(const GraphType &graph) {
    auto timer = Aux::Timer();
    timer.start();

    // thread safe reduction. Avoid atomic calculation of total graph volume for unweighted graphs.
    communityVolumes.clear();
    communitySizes.clear();
    communityVolumes.resize(result.upperBound() + VECTOR_OVERSIZE);
    communitySizes.resize(result.upperBound() + VECTOR_OVERSIZE);
    inverseGraphVolume = 0.0; // Reset to 0 before accumulation

    if (graph.isWeighted()) {
        std::vector<double> threadVolumes(omp_get_max_threads());
        graph.parallelForNodes([&](node a) {
            edgeweight ew = graph.weightedDegree(a, true);
            count size = nodeSize(graph, a);
#pragma omp atomic
            communityVolumes[result[a]] += ew;
#pragma omp atomic
            communitySizes[result[a]] += size;
            threadVolumes[omp_get_thread_num()] += ew;
        });
        for (const auto vol : threadVolumes) {
            inverseGraphVolume += vol;
        }
        inverseGraphVolume = 1 / inverseGraphVolume;
    } else {
        inverseGraphVolume = 1.0 / (2 * graph.numberOfEdges());
        graph.parallelForNodes([&](node a) {
            count size = nodeSize(graph, a);
#pragma omp atomic
            communityVolumes[result[a]] += graph.weightedDegree(a, true);
#pragma omp atomic
            communitySizes[result[a]] += size;
        });
    }
    TRACE("Calculating Volumes took " + timer.elapsedTag());
}

void ParallelLeidenView::flattenPartition() {
    auto timer = Aux::Timer();
    timer.start();
    if (!currentCoarsenedView) {
        return;
    }
    const auto &nodeMapping = currentCoarsenedView->getNodeMapping();
    Partition flattenedPartition(G->numberOfNodes());
    flattenedPartition.setUpperBound(result.upperBound());
    G->parallelForNodes([&](node a) { flattenedPartition[a] = result[nodeMapping[a]]; });
    flattenedPartition.compact(true);
    result = flattenedPartition;
    currentCoarsenedView.reset();
    TRACE("Flattening partition took " + timer.elapsedTag());
}

template <typename GraphType>
ParallelLeidenView::MoveStats ParallelLeidenView::parallelMove(const GraphType &graph) {
    enum : uint8_t { Idle, Queued, Processing, Reprocess };

    DEBUG("Local Moving : ", graph.numberOfNodes(), " Nodes ");
    std::vector<count> moved(omp_get_max_threads(), 0);
    std::vector<count> movedToSingleton(omp_get_max_threads(), 0);
    std::vector<count> marginalMovesRejected(omp_get_max_threads(), 0);
    std::vector<double> gainMarginSum(omp_get_max_threads(), 0.0);
    std::vector<double> gainMarginMin(omp_get_max_threads(), std::numeric_limits<double>::max());
    std::vector<double> gainMarginMax(omp_get_max_threads(), std::numeric_limits<double>::lowest());
    std::vector<count> totalNodesPerThread(omp_get_max_threads(), 0);
    std::vector<count> requeuesBlockedByNodeLimit(omp_get_max_threads(), 0);
    std::atomic_int singleton(0);

    // Track whether nodes are idle, queued, currently processed, or need another pass after the
    // current processing attempt finishes.
    std::vector<std::atomic<uint8_t>> nodeState(graph.upperNodeIdBound());
    for (auto &val : nodeState) {
        val.store(Idle);
    }

    std::vector<std::atomic<std::uint32_t>> requeuesPerNode(graph.upperNodeIdBound());
    for (auto &count : requeuesPerNode) {
        count.store(0);
    }

    std::queue<std::vector<node>> queue;
    std::mutex qlock;                      // queue lock
    std::condition_variable workAvailable; // waiting/notifying for new Nodes

    std::atomic_bool resize(false);
    std::atomic_int waitingForResize(0);
    std::atomic_int waitingForNodes(0);

    std::vector<int> order;
    int tshare;
    int tcount;
    uint64_t vectorSize = communityVolumes.capacity();
    std::atomic_int upperBound(result.upperBound());

#pragma omp parallel
    {
#pragma omp single
        {
            tcount = omp_get_num_threads();
            order.resize(tcount);
            for (int i = 0; i < tcount; i++) {
                order[i] = i;
            }
            if (random)
                std::shuffle(order.begin(), order.end(), Aux::Random::getURNG());
            tshare = 1 + graph.upperNodeIdBound() / tcount;
        }

        const int tid = omp_get_thread_num();
        auto &mt = Aux::Random::getURNG();
        std::vector<node> currentNodes;
        currentNodes.reserve(tshare);
        std::vector<node> newNodes;
        newNodes.reserve(WORKING_SIZE);
        // Sparse cutWeight[Community] storage avoids one dense community vector per thread.
        std::unordered_map<index, double> cutWeights;
        std::vector<index> pointers;

        auto admitNodeRequeue = [&](node candidate) {
            std::uint32_t used = requeuesPerNode[candidate].load(std::memory_order_relaxed);

            while (true) {
                if (used >= maxRequeuesPerNode) {
                    requeuesBlockedByNodeLimit[tid]++;
                    return false;
                }

                if (requeuesPerNode[candidate].compare_exchange_weak(
                        used, used + 1, std::memory_order_relaxed, std::memory_order_relaxed)) {
                    return true;
                }

                // On CAS failure, `used` has been refreshed with the current value.
            }
        };

        auto pushNewNode = [&](node queuedNode) {
            newNodes.push_back(queuedNode);
            // push new nodes to the queue in WORKING_SIZE steps
            if (newNodes.size() == WORKING_SIZE) {
                qlock.lock();
                queue.emplace(std::move(newNodes));
                qlock.unlock();
                // Notify threads that new is available
                workAvailable.notify_all();
                newNodes.clear();
                newNodes.reserve(WORKING_SIZE);
            }
        };

        auto scheduleNode = [&](node queuedNode) {
            std::uint8_t state = nodeState[queuedNode].load();

            while (true) {
                if (state == Idle) {
                    std::uint8_t expected = Idle;

                    if (nodeState[queuedNode].compare_exchange_strong(expected, Queued)) {
                        if (admitNodeRequeue(queuedNode)) {
                            pushNewNode(queuedNode);
                        } else {
                            std::uint8_t rollbackExpected = Queued;
                            const bool resetToIdle = nodeState[queuedNode].compare_exchange_strong(
                                rollbackExpected, Idle);
                            assert(resetToIdle);
                            tlx::unused(resetToIdle);
                        }
                        return;
                    }
                    state = expected;
                } else if (state == Processing) {
                    std::uint8_t expected = Processing;
                    if (nodeState[queuedNode].compare_exchange_strong(expected, Reprocess)) {
                        return;
                    }
                    state = expected;
                } else {
                    return;
                }
            }
        };

        auto finishProcessingNode = [&](node processedNode) {
            uint8_t expected = Processing;
            if (nodeState[processedNode].compare_exchange_strong(expected, Idle)) {
                return;
            }
            assert(expected == Reprocess);

            if (!admitNodeRequeue(processedNode)) {
                expected = Reprocess;
                const bool resetToIdle =
                    nodeState[processedNode].compare_exchange_strong(expected, Idle);
                assert(resetToIdle);
                tlx::unused(resetToIdle);
                return;
            }

            expected = Reprocess;
            const bool markedQueued =
                nodeState[processedNode].compare_exchange_strong(expected, Queued);
            assert(markedQueued);
            tlx::unused(markedQueued);
            pushNewNode(processedNode);
        };

        int start = tshare * order[omp_get_thread_num()];
        int end = (1 + order[omp_get_thread_num()]) * tshare;

        for (int i = start; i < end; i++) {
            if (graph.hasNode(i)) {
                currentNodes.push_back(i);
                nodeState[i].store(Queued);
            }
        }
        if (random)
            std::shuffle(currentNodes.begin(), currentNodes.end(), mt);
#pragma omp barrier
        do {
            handler.assureRunning();
            for (node u : currentNodes) {
                // If a vector resize is needed, yield until done
                if (resize) {
                    waitingForResize++;
                    while (resize) {
                        std::this_thread::yield();
                    }
                    waitingForResize--;
                }

                std::uint8_t expectedState = Queued;
                const bool startedProcessing =
                    nodeState[u].compare_exchange_strong(expectedState, Processing);
                assert(startedProcessing);
                tlx::unused(startedProcessing);

                index currentCommunity = result[u];
                double maxDelta = std::numeric_limits<double>::lowest();
                index bestCommunity = none;
                double degree = 0;
                count nodeMass = nodeSize(graph, u);
                for (auto z : pointers) {
                    // Reset the clearlist : Set all cutweights to 0 and clear the pointer vector
                    cutWeights.erase(z);
                }
                pointers.clear();

                graph.forNeighborsOf(u, [&](node neighbor, edgeweight ew) {
                    index neighborCommunity = result[neighbor];
                    if (cutWeights.find(neighborCommunity) == cutWeights.end()) {
                        pointers.push_back(neighborCommunity);
                    }
                    if (u == neighbor) {
                        degree += ew;
                    } else {
                        cutWeights[neighborCommunity] += ew;
                    }
                    degree += ew; // keep track of the nodes degree. Loops count twice
                });

                if (pointers.empty()) {
                    finishProcessingNode(u);
                    continue;
                }

                double singletonScore = scoreCommunity(0.0, degree, 0.0, nodeMass, 0);

                // Determine move score for all neighbor communities
                for (auto community : pointers) {
                    // "Moving" a node to its current community is pointless
                    if (community != currentCommunity) {
                        double delta;
                        const auto cutWeightIt = cutWeights.find(community);
                        const double cutWeight =
                            cutWeightIt == cutWeights.end() ? 0.0 : cutWeightIt->second;
                        delta = scoreCommunity(cutWeight, degree, communityVolumes[community],
                                               nodeMass, communitySizes[community]);
                        if (delta > maxDelta) {
                            maxDelta = delta;
                            bestCommunity = community;
                        }
                    }
                }
                const auto currentCutWeightIt = cutWeights.find(currentCommunity);
                const double currentCutWeight =
                    currentCutWeightIt == cutWeights.end() ? 0.0 : currentCutWeightIt->second;
                double modThreshold = scoreCurrentCommunityThreshold(
                    currentCutWeight, degree, communityVolumes[currentCommunity], nodeMass,
                    communitySizes[currentCommunity]);

                bool singletonMove = singletonScore > maxDelta;
                double selectedScore = singletonMove ? singletonScore : maxDelta;
                if (selectedScore > modThreshold) {
                    bool acceptedMove = false;
                    double gainMargin = 0.0;
                    gainMargin = selectedScore - modThreshold;
                    acceptedMove = true;
                    if (acceptedMove && gainMargin <= moveGainMarginEpsilon) {
                        marginalMovesRejected[omp_get_thread_num()]++;
                        acceptedMove = false;
                    }

                    if (acceptedMove) {
                        const int tid = omp_get_thread_num();
                        moved[tid]++;

                        if (singletonMove) { // move node to empty community
                            singleton++;
                            movedToSingleton[tid]++;
                            bestCommunity = upperBound++;
                            if (bestCommunity >= communityVolumes.size()) {
                                // Wait until all other threads yielded, then increase vector size
                                bool expected = false;
                                if (resize.compare_exchange_strong(expected, true)) {
                                    vectorSize += VECTOR_OVERSIZE;
                                    while (waitingForResize < tcount - 1) {
                                        std::this_thread::yield();
                                    }
                                    // all other threads are yielding, so resize is fine
                                    communityVolumes.resize(vectorSize);
                                    communitySizes.resize(vectorSize);
                                    expected = true;
                                    resize.compare_exchange_strong(expected, false);
                                } else {
                                    waitingForResize++;
                                    while (resize) {
                                        std::this_thread::yield();
                                    }
                                    waitingForResize--;
                                }
                            }
                        }
                        gainMarginSum[tid] += gainMargin;
                        gainMarginMin[tid] = std::min(gainMarginMin[tid], gainMargin);
                        gainMarginMax[tid] = std::max(gainMarginMax[tid], gainMargin);
                        result[u] = bestCommunity;
#pragma omp atomic
                        communityVolumes[bestCommunity] += degree;
#pragma omp atomic
                        communityVolumes[currentCommunity] -= degree;
#pragma omp atomic
                        communitySizes[bestCommunity] += nodeMass;
#pragma omp atomic
                        communitySizes[currentCommunity] -= nodeMass;
                        changed = true;

                        graph.forNeighborsOf(u, [&](node neighbor, edgeweight) {
                            if (neighbor != u) {
                                scheduleNode(neighbor);
                            }
                        });
                    } // if (acceptedMove)
                }
                finishProcessingNode(u);
            }

            // queue check/wait
            totalNodesPerThread[omp_get_thread_num()] += currentNodes.size();
            if (!newNodes.empty()) {
                std::swap(currentNodes, newNodes);
                newNodes.clear();
                continue;
            }

            std::unique_lock<std::mutex> uniqueLock(qlock);
            if (!queue.empty()) {
                std::swap(currentNodes, queue.front());
                queue.pop();
            } else { // queue empty && newNodes empty
                waitingForNodes++;
                if (waitingForNodes < tcount) { // Not all nodes are done yet, wait for new work
                    waitingForResize++;
                    while (queue.empty() && waitingForNodes < tcount) {
                        workAvailable.wait(uniqueLock);
                    }
                    if (waitingForNodes < tcount) {
                        // Notified and not all done means there's new work
                        std::swap(currentNodes, queue.front());
                        queue.pop();
                        waitingForNodes--;
                        waitingForResize--;
                        continue;
                    }
                }
                uniqueLock.unlock(); // Notified and all done, stop.
                workAvailable.notify_all();
                break;
            }
        } while (true);
        TRACE("Thread ", omp_get_thread_num(), " worked ",
              totalNodesPerThread[omp_get_thread_num()], "Nodes and moved ",
              moved[omp_get_thread_num()]);
    }
    result.setUpperBound(upperBound);
    assert(queue.empty());
    assert(waitingForNodes == tcount);

    const count totalMoved = std::accumulate(moved.begin(), moved.end(), (count)0);

    const count totalWorked = std::accumulate(totalNodesPerThread.begin(),
                                              totalNodesPerThread.end(), static_cast<count>(0));

    const count totalRequeuesBlockedByNodeLimit =
        std::accumulate(requeuesBlockedByNodeLimit.begin(), requeuesBlockedByNodeLimit.end(),
                        static_cast<count>(0));

    if (Aux::Log::isLogLevelEnabled(Aux::Log::LogLevel::DEBUG)) {
        tlx::unused(totalWorked);
        DEBUG("Total worked: ", totalWorked, " Total moved: ", totalMoved,
              " Moved to singleton community: ", singleton,
              " Requeues blocked by node limit: ", totalRequeuesBlockedByNodeLimit);
    }
    MoveStats stats;
    stats.moved = totalMoved;
    stats.movedToSingleton =
        std::accumulate(movedToSingleton.begin(), movedToSingleton.end(), static_cast<count>(0));
    stats.marginalMovesRejected = std::accumulate(
        marginalMovesRejected.begin(), marginalMovesRejected.end(), static_cast<count>(0));
    stats.gainMarginSum = std::accumulate(gainMarginSum.begin(), gainMarginSum.end(), 0.0);
    if (totalMoved > 0) {
        for (int i = 0; i < omp_get_max_threads(); ++i) {
            if (moved[i] == 0) {
                continue;
            }
            stats.gainMarginMin = std::min(stats.gainMarginMin, gainMarginMin[i]);
            stats.gainMarginMax = std::max(stats.gainMarginMax, gainMarginMax[i]);
        }
    } else {
        stats.gainMarginMin = 0.0;
        stats.gainMarginMax = 0.0;
    }
    return stats;
}

template <typename GraphType>
Partition ParallelLeidenView::parallelRefine(const GraphType &graph) {
    bool ignoredRefineMadeChanges = false;
    return parallelRefine(graph, ignoredRefineMadeChanges);
}

template <typename GraphType>
Partition ParallelLeidenView::parallelRefine(const GraphType &graph, bool &refineMadeChanges) {
    std::atomic<bool> anyRefinementChanges{false};
    refineMadeChanges = false;

    Partition refined(graph.numberOfNodes());
    refined.allToSingletons();
    DEBUG("Starting refinement with ", result.numberOfSubsets(), " partitions");
    std::vector<uint_fast8_t> singleton(refined.upperBound(), true);
    std::vector<double> cutCtoSminusC(refined.upperBound());
    std::vector<double> refinedVolumes(refined.upperBound()); // Community Volumes P_refined
    std::vector<count> refinedSizes(refined.upperBound());
    std::vector<std::mutex> locks(refined.upperBound());
    std::vector<node> nodes(graph.upperNodeIdBound(), none);

#pragma omp parallel
    {
        std::vector<index> neighComms;
        // Keeps track of relevant Neighbor communities. Needed to reset the clearlist fast
        std::unordered_map<index, double> cutWeights; // cut from Node to Communities
        auto &mt = Aux::Random::getURNG();
#pragma omp for
        for (omp_index u = 0; u < static_cast<omp_index>(graph.upperNodeIdBound()); u++) {
            if (graph.hasNode(u)) {
                nodes[u] = u;
                refinedSizes[u] = nodeSize(graph, u);
                graph.forNeighborsOf(u, [&](node neighbor, edgeweight ew) {
                    if (u != neighbor) {
                        if (result[neighbor] == result[u]) {
                            // Cut to communities in the refined partition that
                            // are in the same community in the original partition
                            cutCtoSminusC[u] += ew;
                        }
                    } else {
                        refinedVolumes[u] += ew;
                    }
                    refinedVolumes[u] += ew;
                });
            }
        }
        if (random) {
            int share = graph.upperNodeIdBound() / omp_get_num_threads();
            int start = omp_get_thread_num() * share;
            int end = (omp_get_thread_num() + 1) * share - 1;
            if (omp_get_thread_num() == omp_get_num_threads() - 1)
                end = nodes.size() - 1;
            if (start != end && end > start)
                std::shuffle(nodes.begin() + start, nodes.begin() + end, mt);
#pragma omp barrier
        }
        handler.assureRunning();
#pragma omp for schedule(dynamic, WORKING_SIZE)
        for (omp_index i = 0; i < static_cast<omp_index>(nodes.size()); i++) {
            node u = nodes[i];
            if (u == none || !singleton[u]) { // only consider singletons
                continue;
            }
            index S = result[u];                // Node's community ID in the original partition (S)
            for (auto neighComm : neighComms) { // Reset the clearlist : Set all cutweights to 0
                if (neighComm != none)
                    cutWeights.erase(neighComm);
            }

            neighComms.clear();

            std::vector<node> criticalNodes;
            // Nodes whose community ID equals their Node ID. These may be singletons that can
            // affect the cut which we need to update later
            double degree = 0;
            count subsetSize = nodeSize(graph, u);

            graph.forNeighborsOf(u, [&](node neighbor, edgeweight ew) { // Calculate degree and cut
                degree += ew;
                if (neighbor != u) {
                    if (S == result[neighbor]) {
                        index z = refined[neighbor];
                        if (z == neighbor) {
                            criticalNodes.push_back(neighbor);
                        }
                        // We don't need to remember the weight of that edge since it's
                        // already saved in cutWeights
                        if (cutWeights.find(z) == cutWeights.end())
                            neighComms.push_back(z); // Keep track of neighbor communities
                        cutWeights[z] += ew;
                    }
                } else {
                    degree += ew;
                }
            });
            if (!refineRSetCondition(cutCtoSminusC[u], degree, subsetSize,
                                     communityVolumes[S] - degree, communitySizes[S] - subsetSize,
                                     communityVolumes[S], communitySizes[S])) {
                continue;
            }

            if (cutWeights.find(u) != cutWeights.end()) {
                // Node has been moved -> not a singleton anymore. Stop.
                continue;
            }

            double singletonScore = scoreCommunity(0.0, degree, 0.0, subsetSize, 0);
            double delta;
            index bestC = none;
            double bestDelta = std::numeric_limits<double>::lowest();
            int idx;
            // Determine Community that yields highest modularity delta
            auto bestCommunity = [&] {
                for (unsigned int j = 0; j < neighComms.size(); j++) {
                    index C = neighComms[j];
                    if (C == none) {
                        continue;
                    }
                    const auto cutWeightIt = cutWeights.find(C);
                    const double cutWeight =
                        cutWeightIt == cutWeights.end() ? 0.0 : cutWeightIt->second;
                    delta = scoreCommunity(cutWeight, degree, refinedVolumes[C], subsetSize,
                                           refinedSizes[C]);

                    if (delta <= singletonScore) {
                        continue;
                    }

                    auto volC = refinedVolumes[C];
                    auto sizeC = refinedSizes[C];
                    if (delta > bestDelta
                        && refineTSetCondition(
                            cutCtoSminusC[C], volC, sizeC, communityVolumes[S] - volC,
                            communitySizes[S] - sizeC, communityVolumes[S], communitySizes[S])) {
                        bestDelta = delta;
                        bestC = C;
                        idx = j;
                    }
                }
            };
            // To update cut values in case neighbors of this node have moved in the meantime
            auto updateCut = [&] {
                for (node &neighbor : criticalNodes) {
                    if (neighbor != none) {
                        index neighborCommunity = refined[neighbor];
                        if (neighborCommunity != neighbor) {
                            if (cutWeights.find(neighborCommunity) == cutWeights.end()) {
                                // remember to clear the vector, this community was not saved
                                // initially since the neighbor moved to it later
                                neighComms.push_back(neighborCommunity);
                            }
                            // cutWeights[Neighbor] is the weight of the edge between Node and
                            // Neighbor, since Neighbor was a singleton
                            const auto neighborCutWeightIt = cutWeights.find(neighbor);
                            const double neighborCutWeight = neighborCutWeightIt == cutWeights.end()
                                                                 ? 0.0
                                                                 : neighborCutWeightIt->second;
                            cutWeights[neighborCommunity] += neighborCutWeight;
                            // Clear cutWeights entry beforehand, so we can "erase" bestC
                            // from the pointers vector by replacing it with "none"
                            cutWeights.erase(neighbor);
                            neighbor = none;
                        }
                    }
                }
            };
            bestCommunity();
            if (bestC == none) {
                continue;
            }
            lockLowerFirst(u, bestC, locks); // avoid deadlocks
            // If this node is no longer a singleton, stop.
            if (singleton[u]) {
                // Target community still contains its "host" node? If not,this community is
                // now empty, choose a new one.
                while (bestC != none && refined[bestC] != bestC) {
                    locks[bestC].unlock();
                    // This makes sure it won't be considered in the next bestCommunity() call
                    neighComms[idx] = none;
                    bestC = none;
                    bestDelta = std::numeric_limits<double>::lowest();
                    updateCut();
                    bestCommunity();
                    if (bestC != none) {
                        if (!locks[bestC].try_lock()) {
                            if (u < bestC) {
                                locks[bestC].lock();
                            } else {
                                // temporarily release lock on current Node to avoid deadlocks
                                locks[u].unlock();
                                lockLowerFirst(u, bestC, locks);
                            }
                            if (!singleton[u]) {
                                locks[u].unlock();
                                locks[bestC].unlock();
                                continue;
                            }
                        }
                    }
                }
                if (bestC == none) {
                    locks[u].unlock(); // bestC was already unlocked in the loop above
                    continue;
                }
                singleton[bestC] = false;
                refined[u] = bestC;

                // This is the only operation that changes `refined` from
                // the singleton partition created at the start of the function.
                anyRefinementChanges.store(true);

                refinedVolumes[bestC] += degree;
                refinedSizes[bestC] += subsetSize;
                updateCut();
                const auto bestCutWeightIt = cutWeights.find(bestC);
                const double bestCutWeight =
                    bestCutWeightIt == cutWeights.end() ? 0.0 : bestCutWeightIt->second;
                cutCtoSminusC[bestC] += cutCtoSminusC[u] - 2 * bestCutWeight;
            }
            locks[bestC].unlock();
            locks[u].unlock();
        }
    }

    refineMadeChanges = anyRefinementChanges.load();

    DEBUG("Ending refinement with ", refined.numberOfSubsets(), " partitions");
    return refined;
}

// Explicit template instantiations
template void ParallelLeidenView::calculateVolumes<Graph>(const Graph &graph);
template void
ParallelLeidenView::calculateVolumes<CoarsenedGraphView>(const CoarsenedGraphView &graph);
template ParallelLeidenView::MoveStats ParallelLeidenView::parallelMove<Graph>(const Graph &graph);
template ParallelLeidenView::MoveStats
ParallelLeidenView::parallelMove<CoarsenedGraphView>(const CoarsenedGraphView &graph);
template Partition ParallelLeidenView::parallelRefine<Graph>(const Graph &graph);
template Partition
ParallelLeidenView::parallelRefine<CoarsenedGraphView>(const CoarsenedGraphView &graph);
template Partition
ParallelLeidenView::parallelRefine<CoarsenedGraphView>(const CoarsenedGraphView &graph,
                                                       bool &refineMadeChanges);

} // namespace NetworKit
