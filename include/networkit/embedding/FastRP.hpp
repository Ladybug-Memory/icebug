/*
 * FastRP.hpp
 *
 * FastRP: Fast and Accurate Network Embeddings via Very Sparse Random Projection,
 * adapted from the reference implementation of the CIKM 2019 paper
 * [https://github.com/GTmac/FastRP/] with the following changes:
 *
 *  - L2 normalization of each power of the (transition) matrix before merging and
 *    of the final embedding.
 *  - Support for node attributes: an optional per-node feature matrix seeds an
 *    additional embedding that is blended into the structural one.
 *  - Robustness for poorly connected graphs: nodes in components smaller than a
 *    threshold are embedded from their attributes alone (when available).
 *  - Full support for (positive) edge weights in the propagation matrix.
 */

#ifndef NETWORKIT_EMBEDDING_FAST_RP_HPP_
#define NETWORKIT_EMBEDDING_FAST_RP_HPP_

#include <cstdint>
#include <vector>

#include <networkit/base/Algorithm.hpp>
#include <networkit/graph/Graph.hpp>

namespace NetworKit {

/// Random projection variant used to build the sketch matrix of FastRP.
enum FastRPProjection {
    FASTRP_SPARSE,  //!< Achlioptas' sparse projection, entries in {-1, 0, +1}
    FASTRP_GAUSSIAN //!< dense Gaussian projection, entries drawn from N(0, 1)
};

/// Matrix propagated through the graph during the iterations.
enum FastRPInputMatrix {
    FASTRP_ADJACENCY, //!< the weighted adjacency matrix A
    FASTRP_TRANSITION //!< the row-normalized transition matrix D^-1 A
};

/**
 * @ingroup embedding
 *
 * FastRP computes structural node embeddings by repeatedly propagating a randomly
 * projected node vector matrix through the graph and combining the results for
 * successive propagation powers with user-provided weights.
 *
 * This implementation differs from the reference FastRP in several ways:
 *
 *  - L2 normalization: when @a normalization is set, every power of the propagation
 *    matrix is row-wise L2-normalized before the weighted merge, and the final
 *    embedding is row-wise L2-normalized as well, so that embedding norms carry no
 *    degree bias and cosine similarity can be used directly.
 *  - Node attributes: an optional per-node feature matrix can be supplied. It is
 *    projected to @a dim dimensions, L2-normalized and blended into the structural
 *    embedding with weight @a featureWeight.
 *  - Poor connectivity: nodes in weakly connected components smaller than
 *    @a fallbackThreshold are considered poorly connected; if node features are
 *    available, such nodes are embedded from their attributes alone, since their
 *    structural neighborhood carries too little signal.
 *  - Edge weights: weighted edges are respected both in the propagation matrix and
 *    in the degree scaling of the projection matrix.
 *
 * @warning The algorithm materializes several dense n x dim matrices; memory usage
 * is O(q * n * dim).
 */
class FastRP final : public Algorithm {

public:
    /**
     * Constructs the FastRP algorithm.
     *
     * @param G                 The graph. Node ids must be contiguous
     *                          (numberOfNodes() == upperNodeIdBound()).
     * @param dim               Dimension of the computed embedding vectors.
     * @param weights           Weight of every propagation power in the final merge;
     *                          the number of powers (iterations) equals
     *                          weights.size(). E.g. {1, 1, 1}.
     * @param normalization     If true, L2-normalize every propagated matrix before
     *                          the merge and the final embedding afterwards.
     * @param alpha             Degree correction exponent applied to the projection
     *                          matrix (entry of node u is scaled by degree(u)^alpha);
     *                          use 0 to disable.
     * @param projection        'sparse' (Achlioptas) or 'gaussian' random projection.
     * @param inputMatrix       Whether to propagate the adjacency matrix or the
     *                          row-normalized transition matrix.
     * @param nodeFeatures      Optional per-node attribute vectors; all vectors must
     *                          have the same length. Empty to disable.
     * @param featureWeight     Weight of the attribute embedding in the final blend.
     * @param fallbackThreshold Nodes in weakly connected components smaller than
     *                          this threshold are embedded from attributes alone
     *                          (if available); use 1 to disable the fallback.
     * @param seed              Seed for the random projection matrix; results are
     *                          deterministic for a fixed seed.
     */
    FastRP(const Graph &G, count dim, const std::vector<double> &weights, bool normalization = true,
           double alpha = -0.5, FastRPProjection projection = FASTRP_SPARSE,
           FastRPInputMatrix inputMatrix = FASTRP_TRANSITION,
           const std::vector<std::vector<double>> &nodeFeatures = {}, double featureWeight = 0.5,
           count fallbackThreshold = 3, uint64_t seed = 42);

    ~FastRP() override = default;

    /**
     * Computes the embedding of every node.
     */
    void run() override;

    /**
     * Returns the embedding vectors, indexed by node id.
     */
    const std::vector<std::vector<double>> &getEmbeddings() const;

private:
    // The graph
    const Graph *G;
    // Embedding dimension
    count dim;
    // Weight of every propagation power in the merge
    std::vector<double> weights;
    // Whether L2 normalization is applied
    bool normalization;
    // Degree correction exponent
    double alpha;
    // Random projection variant
    FastRPProjection projection;
    // Propagated matrix: adjacency or transition
    FastRPInputMatrix inputMatrix;
    // Optional per-node attribute vectors
    std::vector<std::vector<double>> nodeFeatures;
    // Weight of the attribute embedding in the final blend
    double featureWeight;
    // Components smaller than this are considered poorly connected
    count fallbackThreshold;
    // Seed of the random projection matrix
    uint64_t seed;

    // The embedding, one vector per node
    std::vector<std::vector<double>> embeddings;

    /**
     * Propagates @a in through one application of the (weighted) propagation matrix:
     * out[u] = sum over neighbors v of m(u, v) * in[v], where m(u, v) is the edge
     * weight (adjacency) or the edge weight divided by the (weighted) out-degree of
     * u (transition).
     */
    void propagate(const std::vector<double> &in, std::vector<double> &out,
                   const std::vector<double> &invWeightedDegree) const;

    /**
     * Weakly connected components: fills @a compOf (component id per node) and
     * @a compSize (number of nodes per component).
     */
    void weaklyConnectedComponents(std::vector<count> &compOf, std::vector<count> &compSize) const;

    /**
     * Row-wise L2 normalization of a dense row-major matrix with @a n rows of
     * @a dim columns. Rows with norm zero are left unchanged.
     */
    static void l2NormalizeRows(std::vector<double> &matrix, count n, count dim);
};

} /* namespace NetworKit */

#endif // NETWORKIT_EMBEDDING_FAST_RP_HPP_
