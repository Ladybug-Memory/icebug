/*
 * ParallelCoreDecomposition.hpp
 *
 * Exact shared-memory k-core decomposition based on the SIGMOD 2025 algorithm.
 */

#ifndef NETWORKIT_CENTRALITY_PARALLEL_CORE_DECOMPOSITION_HPP_
#define NETWORKIT_CENTRALITY_PARALLEL_CORE_DECOMPOSITION_HPP_

#include <networkit/centrality/Centrality.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/structures/Cover.hpp>
#include <networkit/structures/Partition.hpp>

namespace NetworKit {

/**
 * @ingroup centrality
 * Computes an exact k-core decomposition in parallel.
 *
 * The implementation is based on the online peeling framework, hierarchical bucketing structure,
 * probabilistic high-degree sampling, and vertical granularity control described by Liu et al. in
 * "Parallel k-Core Decomposition: Theory and Practice" (Proc. ACM Manag. Data 3, 3, SIGMOD
 * 2025). Sampling errors are detected and cause a restart without sampling, making the
 * implementation Las Vegas rather than Monte Carlo.
 *
 * The input graph must be undirected, must not contain self-loops, and must have continuous node
 * ids. Both GraphW and zero-copy CSR-backed GraphR inputs are supported.
 */
class ParallelCoreDecomposition final : public Centrality {
public:
    /**
     * Create a parallel core decomposition for @a G.
     *
     * @param G The undirected input graph.
     * @param normalized If @c true, divide core numbers by the maximum degree.
     * @param useSampling If @c true, use the paper's high-degree sampling optimization.
     */
    ParallelCoreDecomposition(const Graph &G, bool normalized = false, bool useSampling = true);

    /** Compute the core number of every node. */
    void run() override;

    /**
     * Get the nested k-cores as a cover.
     *
     * @return The k-cores as a Cover.
     */
    Cover getCover() const;

    /**
     * Get the k-shells as a partition.
     *
     * @return The k-shells as a Partition.
     */
    Partition getPartition() const;

    /** @return The maximum core number. */
    index maxCoreNumber() const;

    /**
     * Get the number of sampling-error recovery restarts performed by the last run.
     *
     * @return Zero unless a sampling validation failed and the algorithm restarted without
     * sampling.
     */
    count numberOfRestarts() const;

    /** @return The theoretical maximum centrality score. */
    double maximum() override;

private:
    index maxCore{0};
    bool useSampling;
    count restarts{0};
};

} // namespace NetworKit

#endif // NETWORKIT_CENTRALITY_PARALLEL_CORE_DECOMPOSITION_HPP_
