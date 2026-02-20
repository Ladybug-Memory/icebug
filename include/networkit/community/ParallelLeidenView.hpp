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

    inline double modularityDelta(double cutD, double degreeV, double volD) const {
        return cutD - gamma * degreeV * volD * inverseGraphVolume;
    };

    inline double modularityThreshold(double cutC, double volC, double degreeV) const {
        return cutC - gamma * (volC - degreeV) * degreeV * inverseGraphVolume;
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

    // Single composed mapping: accumulated incrementally during inner loop
    // Each new mapping is composed into this one, avoiding accumulation
    std::vector<node> composedMapping;

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
};

} // namespace NetworKit

#endif // NETWORKIT_COMMUNITY_PARALLEL_LEIDEN_VIEW_HPP_
