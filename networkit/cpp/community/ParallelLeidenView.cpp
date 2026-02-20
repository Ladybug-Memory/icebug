/*
 * ParallelLeidenView.cpp
 *
 *  Memory-efficient implementation of ParallelLeiden using CoarsenedGraphView
 */

#include <networkit/community/ParallelLeidenView.hpp>
#include <cstdlib>

namespace NetworKit {

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
}

ParallelLeidenView::~ParallelLeidenView() {
    currentCoarsenedView.reset();
    composedMapping.clear();
    composedMapping.shrink_to_fit();
    communityVolumes.clear();
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
        INFO("Using move gain epsilon=", std::setprecision(6), moveGainMarginEpsilon);
        INFO("Using max inner iterations=", maxInnerIterations,
             " | vector oversize=", VECTOR_OVERSIZE);

        // Initialize composed mapping to identity
        composedMapping.clear();
        composedMapping.resize(G->numberOfNodes());
        G->parallelForNodes([&](node u) { composedMapping[u] = u; });

        // Start with the original graph
        const Graph *currentGraph = G;
        currentCoarsenedView.reset();

        Partition refined;

        // Calculate volumes for the current graph
        calculateVolumes(*currentGraph);

        int innerIterations = 0;
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
            const double minGainMargin =
                moveStats.moved ? moveStats.gainMarginMin : 0.0;
            const double maxGainMargin =
                moveStats.moved ? moveStats.gainMarginMax : 0.0;
            const count candidateMoves = moveStats.moved + moveStats.marginalMovesRejected;
            const double rejectedPct = candidateMoves
                                           ? (100.0 * moveStats.marginalMovesRejected
                                              / static_cast<double>(candidateMoves))
                                           : 0.0;
            INFO("Inner iter ", innerIterations, ": moved ", moveStats.moved, " nodes, ",
                 result.numberOfSubsets(), " communities",
                 " | singleton moves=", moveStats.movedToSingleton,
                 " | marginal rejected=", moveStats.marginalMovesRejected,
                 " (", std::setprecision(4), rejectedPct, "%)",
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
            if (currentCoarsenedView) {
                refined = parallelRefine(*currentCoarsenedView);
            } else {
                refined = parallelRefine(*currentGraph);
            }

            handler.assureRunning();

            // Create coarsened view
            if (currentCoarsenedView) {
                Partition originalRefined(G->numberOfNodes());
                originalRefined.setUpperBound(refined.upperBound());

                for (node coarseNode = 0; coarseNode < currentCoarsenedView->numberOfNodes();
                     ++coarseNode) {
                    const auto &originalNodes = currentCoarsenedView->getOriginalNodes(coarseNode);
                    node refinedCommunity = refined[coarseNode];
                    for (node originalNode : originalNodes) {
                        originalRefined[originalNode] = refinedCommunity;
                    }
                }

                ParallelPartitionCoarseningView ppcView(*G, originalRefined);
                ppcView.run();
                auto newCoarsenedView = ppcView.getCoarsenedGraphView();
                auto map = ppcView.getFineToCoarseNodeMapping();
                const auto &oldNodeMapping = currentCoarsenedView->getNodeMapping();

                Partition p(newCoarsenedView->numberOfNodes());
                p.setUpperBound(result.upperBound());

                G->parallelForNodes([map = map, &p, this, &oldNodeMapping](node originalNode) {
                    node newCoarseNode = map[originalNode];
                    node oldCoarseNode = oldNodeMapping[originalNode];
                    p[newCoarseNode] = result[oldCoarseNode];
                });

                // Since each coarsened view is rebuilt from the original graph, this map already
                // represents original -> current-coarse.
                composedMapping = std::move(map);

                result = std::move(p);
                result.compact(true);
                currentCoarsenedView = newCoarsenedView;

            } else {
                ParallelPartitionCoarseningView ppcView(*currentGraph, refined);
                ppcView.run();
                auto newCoarsenedView = ppcView.getCoarsenedGraphView();
                auto map = ppcView.getFineToCoarseNodeMapping();

                Partition p(newCoarsenedView->numberOfNodes());
                p.setUpperBound(result.upperBound());
                currentGraph->parallelForNodes(
                    [map = map, &p, this](node u) { p[map[u]] = result[u]; });

                composedMapping = std::move(map);
                result = std::move(p);
                result.compact(true);
                currentCoarsenedView = newCoarsenedView;
                currentGraph = nullptr;
            }
            
            calculateVolumes(*currentCoarsenedView);

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

    communityVolumes.clear();
    communityVolumes.resize(result.upperBound() + VECTOR_OVERSIZE);
    inverseGraphVolume = 0.0;

    if (graph.isWeighted()) {
        std::vector<double> threadVolumes(omp_get_max_threads());
        graph.parallelForNodes([&](node a) {
            edgeweight ew = graph.weightedDegree(a, true);
#pragma omp atomic
            communityVolumes[result[a]] += ew;
            threadVolumes[omp_get_thread_num()] += ew;
        });
        for (const auto vol : threadVolumes) {
            inverseGraphVolume += vol;
        }
        inverseGraphVolume = 1 / inverseGraphVolume;
    } else {
        inverseGraphVolume = 1.0 / (2 * graph.numberOfEdges());
        graph.parallelForNodes([&](node a) {
#pragma omp atomic
            communityVolumes[result[a]] += graph.weightedDegree(a, true);
        });
    }
    TRACE("Calculating Volumes took " + timer.elapsedTag());
}

void ParallelLeidenView::flattenPartition() {
    auto timer = Aux::Timer();
    timer.start();
    if (composedMapping.empty()) {
        return;
    }
    // composedMapping already contains the composed mapping: original -> final coarse node
    Partition flattenedPartition(G->numberOfNodes());
    flattenedPartition.setUpperBound(result.upperBound());
    G->parallelForNodes([&](node a) { 
        flattenedPartition[a] = result[composedMapping[a]]; 
    });
    flattenedPartition.compact(true);
    result = flattenedPartition;
    composedMapping.clear();
    composedMapping.shrink_to_fit();
    currentCoarsenedView.reset();
    TRACE("Flattening partition took " + timer.elapsedTag());
}

template <typename GraphType>
ParallelLeidenView::MoveStats ParallelLeidenView::parallelMove(const GraphType &graph) {
    DEBUG("Local Moving : ", graph.numberOfNodes(), " Nodes ");
    std::vector<count> moved(omp_get_max_threads(), 0);
    std::vector<count> movedToSingleton(omp_get_max_threads(), 0);
    std::vector<count> marginalMovesRejected(omp_get_max_threads(), 0);
    std::vector<double> gainMarginSum(omp_get_max_threads(), 0.0);
    std::vector<double> gainMarginMin(omp_get_max_threads(), std::numeric_limits<double>::max());
    std::vector<double> gainMarginMax(omp_get_max_threads(), std::numeric_limits<double>::lowest());
    std::vector<count> totalNodesPerThread(omp_get_max_threads(), 0);
    std::atomic_int singleton(0);

    std::vector<std::atomic_bool> inQueue(graph.upperNodeIdBound());
    for (auto &val : inQueue) {
        val.store(false);
    }
    std::queue<std::vector<node>> queue;
    std::mutex qlock;
    std::condition_variable workAvailable;

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
        auto &mt = Aux::Random::getURNG();
        std::vector<node> currentNodes;
        currentNodes.reserve(tshare);
        std::vector<node> newNodes;
        newNodes.reserve(WORKING_SIZE);
        std::vector<double> cutWeights(communityVolumes.capacity());
        std::vector<index> pointers;
        int start = tshare * order[omp_get_thread_num()];
        int end = (1 + order[omp_get_thread_num()]) * tshare;

        for (int i = start; i < end; i++) {
            if (graph.hasNode(i)) {
                currentNodes.push_back(i);
                inQueue[i].store(true);
            }
        }
        if (random)
            std::shuffle(currentNodes.begin(), currentNodes.end(), mt);
#pragma omp barrier
        do {
            handler.assureRunning();
            for (node u : currentNodes) {
                if (resize) {
                    waitingForResize++;
                    while (resize) {
                        std::this_thread::yield();
                    }
                    waitingForResize--;
                }
                cutWeights.resize(vectorSize);
                assert(inQueue[u]);
                index currentCommunity = result[u];
                double maxDelta = std::numeric_limits<double>::lowest();
                index bestCommunity = none;
                double degree = 0;
                for (auto z : pointers) {
                    cutWeights[z] = 0;
                }
                pointers.clear();

                graph.forNeighborsOf(u, [&](node neighbor, edgeweight ew) {
                    index neighborCommunity = result[neighbor];
                    if (cutWeights[neighborCommunity] == 0) {
                        pointers.push_back(neighborCommunity);
                    }
                    if (u == neighbor) {
                        degree += ew;
                    } else {
                        cutWeights[neighborCommunity] += ew;
                    }
                    degree += ew;
                });

                if (pointers.empty())
                    continue;

                for (auto community : pointers) {
                    if (community != currentCommunity) {
                        double delta;
                        delta = modularityDelta(cutWeights[community], degree,
                                                communityVolumes[community]);
                        if (delta > maxDelta) {
                            maxDelta = delta;
                            bestCommunity = community;
                        }
                    }
                }
                double modThreshold = modularityThreshold(
                    cutWeights[currentCommunity], communityVolumes[currentCommunity], degree);

                bool acceptedMove = false;
                bool singletonMove = (0 > maxDelta);
                double gainMargin = 0.0;
                if (singletonMove) {
                    gainMargin = -modThreshold;
                    acceptedMove = (0 > modThreshold);
                } else {
                    gainMargin = maxDelta - modThreshold;
                    acceptedMove = (maxDelta > modThreshold);
                }
                if (acceptedMove && gainMargin <= moveGainMarginEpsilon) {
                    marginalMovesRejected[omp_get_thread_num()]++;
                    acceptedMove = false;
                }

                if (acceptedMove) {
                    const int tid = omp_get_thread_num();
                    moved[tid]++;
                    if (singletonMove) {
                        singleton++;
                        movedToSingleton[tid]++;
                        bestCommunity = upperBound++;
                        if (bestCommunity >= communityVolumes.size()) {
                            bool expected = false;
                            if (resize.compare_exchange_strong(expected, true)) {
                                vectorSize += VECTOR_OVERSIZE;
                                cutWeights.resize(vectorSize);
                                while (waitingForResize < tcount - 1) {
                                    std::this_thread::yield();
                                }
                                communityVolumes.resize(vectorSize);
                                expected = true;
                                resize.compare_exchange_strong(expected, false);
                            } else {
                                waitingForResize++;
                                while (resize) {
                                    std::this_thread::yield();
                                }
                                cutWeights.resize(vectorSize);
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
                    changed = true;
                    bool expected = true;
                    inQueue[u].compare_exchange_strong(expected, false);
                    assert(expected);
                    graph.forNeighborsOf(u, [&](node neighbor, edgeweight) {
                        if (result[neighbor] != bestCommunity && neighbor != u) {
                            expected = false;
                            if (inQueue[neighbor].compare_exchange_strong(expected, true)) {
                                newNodes.push_back(neighbor);
                                if (newNodes.size() == WORKING_SIZE) {
                                    qlock.lock();
                                    queue.emplace(std::move(newNodes));
                                    qlock.unlock();
                                    workAvailable.notify_all();
                                    newNodes.clear();
                                    newNodes.reserve(WORKING_SIZE);
                                }
                                assert(!expected);
                            }
                        }
                    });
                }
            }

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
            } else {
                waitingForNodes++;
                if (waitingForNodes < tcount) {
                    waitingForResize++;
                    while (queue.empty() && waitingForNodes < tcount) {
                        workAvailable.wait(uniqueLock);
                    }
                    if (waitingForNodes < tcount) {
                        std::swap(currentNodes, queue.front());
                        queue.pop();
                        waitingForNodes--;
                        waitingForResize--;
                        continue;
                    }
                }
                uniqueLock.unlock();
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
    count totalMoved = std::accumulate(moved.begin(), moved.end(), (count)0);
    if (Aux::Log::isLogLevelEnabled(Aux::Log::LogLevel::DEBUG)) {
        count totalWorked =
            std::accumulate(totalNodesPerThread.begin(), totalNodesPerThread.end(), (count)0);
        tlx::unused(totalWorked);
        DEBUG("Total worked: ", totalWorked, " Total moved: ", totalMoved,
              " moved to singleton community: ", singleton);
    }
    MoveStats stats;
    stats.moved = totalMoved;
    stats.movedToSingleton =
        std::accumulate(movedToSingleton.begin(), movedToSingleton.end(), static_cast<count>(0));
    stats.marginalMovesRejected = std::accumulate(marginalMovesRejected.begin(),
                                                  marginalMovesRejected.end(),
                                                  static_cast<count>(0));
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
    Partition refined(graph.numberOfNodes());
    refined.allToSingletons();
    DEBUG("Starting refinement with ", result.numberOfSubsets(), " partitions");
    std::vector<uint_fast8_t> singleton(refined.upperBound(), true);
    std::vector<double> cutCtoSminusC(refined.upperBound());
    std::vector<double> refinedVolumes(refined.upperBound());
    std::vector<std::mutex> locks(refined.upperBound());
    std::vector<node> nodes(graph.upperNodeIdBound(), none);

#pragma omp parallel
    {
        std::vector<index> neighComms;
        std::vector<double> cutWeights(refined.upperBound());
        auto &mt = Aux::Random::getURNG();
#pragma omp for
        for (omp_index u = 0; u < static_cast<omp_index>(graph.upperNodeIdBound()); u++) {
            if (graph.hasNode(u)) {
                nodes[u] = u;
                graph.forNeighborsOf(u, [&](node neighbor, edgeweight ew) {
                    if (u != neighbor) {
                        if (result[neighbor] == result[u]) {
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
            if (u == none || !singleton[u]) {
                continue;
            }
            index S = result[u];
            for (auto neighComm : neighComms) {
                if (neighComm != none)
                    cutWeights[neighComm] = 0;
            }

            neighComms.clear();

            std::vector<node> criticalNodes;
            double degree = 0;

            graph.forNeighborsOf(u, [&](node neighbor, edgeweight ew) {
                degree += ew;
                if (neighbor != u) {
                    if (S == result[neighbor]) {
                        index z = refined[neighbor];
                        if (z == neighbor) {
                            criticalNodes.push_back(neighbor);
                        }
                        if (cutWeights[z] == 0)
                            neighComms.push_back(z);
                        cutWeights[z] += ew;
                    }
                } else {
                    degree += ew;
                }
            });
            if (cutCtoSminusC[u] < this->gamma * degree * (communityVolumes[S] - degree)
                                       * inverseGraphVolume) {
                continue;
            }

            if (cutWeights[u] != 0) {
                continue;
            }

            double delta;
            index bestC = none;
            double bestDelta = std::numeric_limits<double>::lowest();
            int idx;
            auto bestCommunity = [&] {
                for (unsigned int j = 0; j < neighComms.size(); j++) {
                    index C = neighComms[j];
                    if (C == none) {
                        continue;
                    }
                    delta = modularityDelta(cutWeights[C], degree, refinedVolumes[C]);

                    if (delta < 0) {
                        continue;
                    }

                    auto volC = refinedVolumes[C];
                    if (delta > bestDelta
                        && cutCtoSminusC[C] >= this->gamma * volC * (communityVolumes[S] - volC)
                                                   * inverseGraphVolume) {
                        bestDelta = delta;
                        bestC = C;
                        idx = j;
                    }
                }
            };
            auto updateCut = [&] {
                for (node &neighbor : criticalNodes) {
                    if (neighbor != none) {
                        index neighborCommunity = refined[neighbor];
                        if (neighborCommunity != neighbor) {
                            if (cutWeights[neighborCommunity] == 0) {
                                neighComms.push_back(neighborCommunity);
                            }
                            cutWeights[neighborCommunity] += cutWeights[neighbor];
                            cutWeights[neighbor] = 0;
                            neighbor = none;
                        }
                    }
                }
            };
            bestCommunity();
            if (bestC == none) {
                continue;
            }
            lockLowerFirst(u, bestC, locks);
            if (singleton[u]) {
                while (bestC != none && refined[bestC] != bestC) {
                    locks[bestC].unlock();
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
                    locks[u].unlock();
                    continue;
                }
                singleton[bestC] = false;
                refined[u] = bestC;
                refinedVolumes[bestC] += degree;
                updateCut();
                cutCtoSminusC[bestC] += cutCtoSminusC[u] - 2 * cutWeights[bestC];
            }
            locks[bestC].unlock();
            locks[u].unlock();
        }
    }

    DEBUG("Ending refinement with ", refined.numberOfSubsets(), " partitions");
    return refined;
}

template void ParallelLeidenView::calculateVolumes<Graph>(const Graph &graph);
template void
ParallelLeidenView::calculateVolumes<CoarsenedGraphView>(const CoarsenedGraphView &graph);
template ParallelLeidenView::MoveStats ParallelLeidenView::parallelMove<Graph>(const Graph &graph);
template ParallelLeidenView::MoveStats
ParallelLeidenView::parallelMove<CoarsenedGraphView>(const CoarsenedGraphView &graph);
template Partition ParallelLeidenView::parallelRefine<Graph>(const Graph &graph);
template Partition
ParallelLeidenView::parallelRefine<CoarsenedGraphView>(const CoarsenedGraphView &graph);

} // namespace NetworKit
