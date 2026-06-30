/*
 * ParallelLeidenView.hpp
 *
 *  Memory-efficient version of ParallelLeiden using CoarsenedGraphView
 */

#ifndef NETWORKIT_COMMUNITY_PARALLEL_LEIDEN_VIEW_HPP_
#define NETWORKIT_COMMUNITY_PARALLEL_LEIDEN_VIEW_HPP_

#include <atomic>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <omp.h>
#include <string>
#include <thread>
#include <tlx/unused.hpp>
#include <networkit/Globals.hpp>
#include <networkit/auxiliary/Parallel.hpp>
#include <networkit/auxiliary/Parallelism.hpp>
#include <networkit/auxiliary/SignalHandling.hpp>
#include <networkit/auxiliary/Timer.hpp>
#include <networkit/coarsening/CoarsenedGraphView.hpp>
#include <networkit/coarsening/ParallelPartitionCoarseningView.hpp>
#include <networkit/community/CommunityDetectionAlgorithm.hpp>
#include <networkit/community/Modularity.hpp>
#include <networkit/community/PLM.hpp>
#include <networkit/community/ParallelLeidenScoringExtension.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/structures/Partition.hpp>

namespace NetworKit {

/**
 * Memory-efficient version of ParallelLeiden that uses CoarsenedGraphView
 * instead of creating new graph structures, significantly reducing memory usage
 * during the coarsening process.
 */
class ParallelLeidenView final : public CommunityDetectionAlgorithm {
public:
    /**
     * @note As reported by Sahu et. al in "GVE-Leiden: Fast Leiden Algorithm for Community
     * Detection in Shared Memory Setting", the current implementation in NetworKit might create a
     * small fraction of disconnected communities. Since this violates the guarantees from the
     * original algorithm, ParallelLeidenView should be used with caution. In addition the
     * modularity value of the resulting partition / clustering can be lower compared to other
     * Leiden implementations and even Louvain.
     *
     * @param graph A networkit graph
     * @param iterations Number of Leiden Iterations to be run
     * @param randomize Randomize node order?
     * @param gamma Resolution parameter
     */
    explicit ParallelLeidenView(const Graph &graph, int iterations = 3, bool randomize = true,
                                double gamma = 1);

    ~ParallelLeidenView();

    void run() override;

    /**
     * Load a shared library that customizes the move-phase scoring metric.
     *
     * The library must export `networkitParallelLeidenCommunityScore`. It may additionally export
     * `networkitParallelLeidenCurrentCommunityThreshold` to replace the default modularity-based
     * stay threshold as well.
     */
    void loadMoveScoringExtension(const std::string &sharedLibraryPath);

    void unloadMoveScoringExtension();

    int VECTOR_OVERSIZE = 10000;

private:
    struct MoveStats {
        count moved = 0;
        count movedToSingleton = 0;
        count marginalMovesRejected = 0;
        double gainMarginSum = 0.0;
        double gainMarginMin = std::numeric_limits<double>::max();
        double gainMarginMax = std::numeric_limits<double>::lowest();
    };

    // Template interface to work with both Graph and CoarsenedGraphView
    template <typename GraphType>
    void calculateVolumes(const GraphType &graph);

    template <typename GraphType>
    MoveStats parallelMove(const GraphType &graph);

    template <typename GraphType>
    Partition parallelRefine(const GraphType &graph);

    template <typename GraphType>
    Partition parallelRefine(const GraphType &graph, bool &refineMadeChanges);

    static count nodeSize(const Graph &graph, node u);

    static count nodeSize(const CoarsenedGraphView &graph, node u);

    static double modularityCommunityScore(double cutD, double degreeV, double volD,
                                           count subsetSize, count sizeD, double gamma,
                                           double inverseGraphVolume) {
        tlx::unused(subsetSize);
        tlx::unused(sizeD);
        return cutD - gamma * degreeV * volD * inverseGraphVolume;
    }

    static double modularityThresholdScore(double cutC, double degreeV, double volC,
                                           count subsetSize, count sizeC, double gamma,
                                           double inverseGraphVolume) {
        tlx::unused(subsetSize);
        tlx::unused(sizeC);
        return cutC - gamma * (volC - degreeV) * degreeV * inverseGraphVolume;
    }

    static bool modularityRefineRSetCondition(double cutWeight, double subsetVolume,
                                              count subsetSize, double targetVolume,
                                              count targetSize, double sourceVolume,
                                              count sourceSize, double gamma,
                                              double inverseGraphVolume) {
        tlx::unused(subsetSize);
        tlx::unused(targetSize);
        tlx::unused(sourceVolume);
        tlx::unused(sourceSize);
        return cutWeight >= gamma * subsetVolume * targetVolume * inverseGraphVolume;
    }

    static bool modularityRefineTSetCondition(double cutWeight, double subsetVolume,
                                              count subsetSize, double targetVolume,
                                              count targetSize, double sourceVolume,
                                              count sourceSize, double gamma,
                                              double inverseGraphVolume) {
        tlx::unused(subsetSize);
        tlx::unused(targetSize);
        tlx::unused(sourceVolume);
        tlx::unused(sourceSize);
        return cutWeight >= gamma * subsetVolume * targetVolume * inverseGraphVolume;
    }

    inline double scoreCommunity(double cutWeight, double degree, double communityVolume,
                                 count subsetSize, count communitySize) const {
        return communityScoreFunction_(cutWeight, degree, communityVolume, subsetSize,
                                       communitySize, gamma, inverseGraphVolume);
    }

    inline double scoreCurrentCommunityThreshold(double cutWeight, double degree,
                                                 double communityVolume, count subsetSize,
                                                 count communitySize) const {
        return currentCommunityThresholdFunction_(cutWeight, degree, communityVolume, subsetSize,
                                                  communitySize, gamma, inverseGraphVolume);
    }

    inline bool refineRSetCondition(double cutWeight, double subsetVolume, count subsetSize,
                                    double targetVolume, count targetSize, double sourceVolume,
                                    count sourceSize) const {
        return refineRSetConditionFunction_(cutWeight, subsetVolume, subsetSize, targetVolume,
                                            targetSize, sourceVolume, sourceSize, gamma,
                                            inverseGraphVolume);
    }

    inline bool refineTSetCondition(double cutWeight, double subsetVolume, count subsetSize,
                                    double targetVolume, count targetSize, double sourceVolume,
                                    count sourceSize) const {
        return refineTSetConditionFunction_(cutWeight, subsetVolume, subsetSize, targetVolume,
                                            targetSize, sourceVolume, sourceSize, gamma,
                                            inverseGraphVolume);
    }

    static inline void lockLowerFirst(index a, index b, std::vector<std::mutex> &locks) {
        if (a < b) {
            locks[a].lock();
            locks[b].lock();
        } else {
            locks[b].lock();
            locks[a].lock();
        }
    }

    void flattenPartition();

    double inverseGraphVolume; // 1/vol(V)

    std::vector<double> communityVolumes;
    std::vector<count> communitySizes;

    static constexpr int WORKING_SIZE = 1000;

    double gamma; // Resolution parameter

    bool changed;

    int numberOfIterations;

    Aux::SignalHandler handler;

    bool random;

    // Current coarsened graph view (only keep current, not all historical)
    std::shared_ptr<CoarsenedGraphView> currentCoarsenedView;

    // Reject moves whose gain margin is too small (numerical-noise churn filter).
    double moveGainMarginEpsilon = 1e-4;

    // Maximum inner iterations per Leiden iteration.
    int maxInnerIterations = 20;

    // Maximum moves per node within a Leiden iteration.
    int maxRequeuesPerNode = 24;

    // Optional convergence stop: minimum relative reduction in community count per inner iter.
    // 0.0 disables this criterion.
    double minCommunityReduction = 0.0;

    void *scoringExtensionHandle_ = nullptr;
    ParallelLeidenCommunityScoreFunction communityScoreFunction_ = &modularityCommunityScore;
    ParallelLeidenCommunityScoreFunction currentCommunityThresholdFunction_ =
        &modularityThresholdScore;
    ParallelLeidenRefineSetConditionFunction refineRSetConditionFunction_ =
        &modularityRefineRSetCondition;
    ParallelLeidenRefineSetConditionFunction refineTSetConditionFunction_ =
        &modularityRefineTSetCondition;
    std::string scoringExtensionPath_;
};

} // namespace NetworKit

#endif // NETWORKIT_COMMUNITY_PARALLEL_LEIDEN_VIEW_HPP_
