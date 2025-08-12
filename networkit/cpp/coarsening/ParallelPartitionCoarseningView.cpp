/*
 * ParallelPartitionCoarseningView.cpp
 *
 *  Implementation of memory-efficient partition coarsening
 */

#include <networkit/auxiliary/Log.hpp>
#include <networkit/auxiliary/Timer.hpp>
#include <networkit/coarsening/ParallelPartitionCoarseningView.hpp>

namespace NetworKit {

ParallelPartitionCoarseningView::ParallelPartitionCoarseningView(const Graph &G,
                                                                 const Partition &zeta)
    : G(&G), zeta(zeta), hasRunFlag(false) {}

void ParallelPartitionCoarseningView::run() {
    Aux::Timer timer;
    timer.start();

    // Create the coarsened graph view directly from the partition
    coarsenedView = std::make_shared<CoarsenedGraphView>(*G, zeta);

    hasRunFlag = true;

    TRACE("ParallelPartitionCoarseningView completed in ", timer.elapsedTag());
}

const std::vector<node> &ParallelPartitionCoarseningView::getFineToCoarseNodeMapping() const {
    if (!hasRunFlag || !coarsenedView) {
        throw std::runtime_error("ParallelPartitionCoarseningView has not been run yet");
    }
    return coarsenedView->getNodeMapping();
}

} /* namespace NetworKit */
