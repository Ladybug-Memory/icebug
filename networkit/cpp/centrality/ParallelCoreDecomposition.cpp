/*
 * ParallelCoreDecomposition.cpp
 *
 * Exact shared-memory k-core decomposition based on Liu et al., SIGMOD 2025.
 */

#include "ParallelCoreDecompositionInternal.hpp"

namespace NetworKit {

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
    const auto result = CoreDecompositionDetail::runWithRecovery(G, scoreData, useSampling);
    restarts = result.restarts;
    maxCore = result.maxCore;

    if (normalized) {
        coreNumbers_.resize(scoreData.size());
        count maximumDegree = 0;
        G.forNodes([&](node u) { maximumDegree = std::max(maximumDegree, G.degree(u)); });
        G.parallelForNodes([&](node u) {
            coreNumbers_[u] = static_cast<index>(scoreData[u]);
            if (maximumDegree > 0)
                scoreData[u] /= maximumDegree;
        });
    }

    hasRun = true;
}

Cover ParallelCoreDecomposition::getCover() const {
    assureFinished();
    Cover result(G.upperNodeIdBound());
    result.setUpperBound(G.upperNodeIdBound());
    G.forNodes([&](node u) {
        for (index core = 0; core <= coreNumber(u); ++core) {
            result.addToSubset(core, u);
        }
    });
    return result;
}

Partition ParallelCoreDecomposition::getPartition() const {
    assureFinished();
    Partition result(G.upperNodeIdBound());
    result.allToSingletons();
    G.forNodes([&](node u) { result.moveToSubset(coreNumber(u), u); });
    return result;
}

index ParallelCoreDecomposition::coreNumber(node u) const {
    return normalized ? coreNumbers_[u] : static_cast<index>(scoreData[u]);
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
    const count n = G.numberOfNodes();
    if (n <= 1) {
        return 0.0;
    }
    return normalized ? 1.0 : static_cast<double>(n - 1);
}

} // namespace NetworKit
