/*
 * ParallelPartitionCoarseningView.hpp
 *
 *  Memory-efficient partition coarsening using views
 */

#ifndef NETWORKIT_COARSENING_PARALLEL_PARTITION_COARSENING_VIEW_HPP_
#define NETWORKIT_COARSENING_PARALLEL_PARTITION_COARSENING_VIEW_HPP_

#include <memory>
#include <networkit/Globals.hpp>
#include <networkit/coarsening/CoarsenedGraphView.hpp>
#include <networkit/structures/Partition.hpp>

namespace NetworKit {

/**
 * @ingroup coarsening
 * Memory-efficient partition coarsening that creates a CoarsenedGraphView
 * instead of a new graph structure, significantly reducing memory usage.
 */
class ParallelPartitionCoarseningView {
public:
    /**
     * Constructor
     * @param G The input graph
     * @param zeta The partition defining how nodes should be grouped
     */
    ParallelPartitionCoarseningView(const Graph &G, const Partition &zeta);

    /**
     * Run the coarsening algorithm
     */
    void run();

    /**
     * Get the coarsened graph view
     * @return Shared pointer to the coarsened graph view
     */
    std::shared_ptr<CoarsenedGraphView> getCoarsenedGraphView() const { return coarsenedView; }

    /**
     * Get the mapping from fine to coarse nodes
     * @return Vector mapping original node IDs to coarsened node IDs
     */
    const std::vector<node> &getFineToCoarseNodeMapping() const;

    /**
     * Check if the algorithm has been run
     */
    bool hasRun() const { return hasRunFlag; }

private:
    const Graph *G;
    const Partition &zeta;
    std::shared_ptr<CoarsenedGraphView> coarsenedView;
    bool hasRunFlag;
};

} /* namespace NetworKit */

#endif // NETWORKIT_COARSENING_PARALLEL_PARTITION_COARSENING_VIEW_HPP_
