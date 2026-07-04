/*
 * PageRank.h
 *
 *  Created on: 19.03.2014
 *      Author: Henning
 */

#ifndef NETWORKIT_CENTRALITY_PAGE_RANK_HPP_
#define NETWORKIT_CENTRALITY_PAGE_RANK_HPP_

#include <atomic>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <networkit/centrality/Centrality.hpp>

namespace NetworKit {

/**
 * @ingroup centrality
 * Compute PageRank as node centrality measure. In the default mode this computation is in line
 * with the original paper "The PageRank citation ranking: Bringing order to the web." by L. Brin et
 * al (1999). In later publications ("PageRank revisited." by M. Brinkmeyer et al. (2005) amongst
 * others) sink-node handling was added for directed graphs in order to comply with the theoretical
 * assumptions by the underlying Markov chain model. This can be activated by setting the matching
 * parameter to true. By default this is disabled, since it is an addition to the original
 * definition.
 *
 * Page-Rank values can also be normalized by post-processed according to "Comparing Apples and
 * Oranges: Normalized PageRank for Evolving Graphs" by Berberich et al. (2007). This decouples
 * the PageRank values from the size of the input graph. To enable this, set the matching parameter
 * to true. Note that sink-node handling is automatically activated if normalization is used.
 *
 * Personalized PageRank (also known as "random walk with restart" or topic-sensitive PageRank) is
 * supported by passing a non-empty teleportation distribution to the constructor. The math is
 * identical to standard PageRank except that the teleportation vector @f$ p @f$ (originally
 * @f$ 1/n @f$) is taken from the user instead of being uniform. Dangling-node mass is also
 * distributed according to @f$ p @f$. See "The PageRank citation ranking: Bringing order to the
 * web." by L. Brin et al. (1999) and "Topic-sensitive PageRank" by Haveliwala (2002).
 *
 * NOTE: There is an inconsistency in the definition in Newman's book (Ch. 7) regarding
 * directed graphs; we follow the verbal description, which requires to sum over the incoming
 * edges (as opposed to outgoing ones).
 */
class PageRank final : public Centrality {

public:
    enum Norm {
        L1_NORM,
        L2_NORM,
        L1Norm = L1_NORM, // this + following added for backwards compatibility
        L2Norm = L2_NORM
    };

    enum SinkHandling { NO_SINK_HANDLING, DISTRIBUTE_SINKS };

    /**
     * Constructs the PageRank class for the Graph @a G
     *
     * @param[in] G Graph to be processed.
     * @param[in] damp Damping factor of the PageRank algorithm.
     * @param[in] tol Error tolerance for PageRank iteration.
     * @param[in] normalized Set to normalize Page-Rank values in order to decouple it from the
     * network size. (default = false)
     * @param[in] distributeSinks Set to distribute PageRank values for sink nodes. (default =
     * NO_SINK_HANDLING)
     * @param[in] personalization Teleportation distribution for personalized PageRank. Pass an
     * empty vector (the default) to recover the original uniform teleportation @f$ 1/n @f$.
     * Otherwise, the vector's length must be at least @c G.upperNodeIdBound(); each entry
     * @c w[u] is the (unnormalized) teleportation weight for node @c u and must be non-negative.
     * The vector is normalized internally so that the resulting distribution sums to one. An
     * exception is thrown if all entries are zero.
     */
    PageRank(const Graph &G, double damp = 0.85, double tol = 1e-8, bool normalized = false,
             SinkHandling distributeSinks = SinkHandling::NO_SINK_HANDLING,
             std::vector<double> personalization = {});

    /**
     * Builds a personalization vector that teleports exclusively to a single @a source node.
     * The returned vector has length @c G.upperNodeIdBound() and sums to one; useful as input
     * for the @c personalization constructor argument of #PageRank.
     *
     * @param G Graph the personalization vector is built for.
     * @param source Node to teleport to.
     * @return Personalization vector of size @c G.upperNodeIdBound() with a single non-zero entry.
     */
    static std::vector<double> personalizationFromSource(const Graph &G, node source);

    /**
     * Builds a personalization vector that teleports uniformly to any node in @a sources.
     * The returned vector has length @c G.upperNodeIdBound() and sums to one.
     *
     * @param G Graph the personalization vector is built for.
     * @param sources Nodes to teleport to (must be non-empty).
     * @return Personalization vector of size @c G.upperNodeIdBound() with uniform non-zero
     * entries on the @a sources.
     */
    static std::vector<double>
    personalizationFromSources(const Graph &G, const std::vector<node> &sources);

    /**
     * Builds a personalization vector from a set of weighted source nodes. Weights must be
     * positive; nodes missing from @a weightedSources receive a weight of zero. The returned
     * vector has length @c G.upperNodeIdBound() and sums to one.
     *
     * @param G Graph the personalization vector is built for.
     * @param weightedSources Pairs of (node, weight) where weight > 0. Must be non-empty and the
     * sum of weights must be positive.
     * @return Personalization vector of size @c G.upperNodeIdBound().
     */
    static std::vector<double>
    personalizationFromWeights(const Graph &G,
                               const std::vector<std::pair<node, double>> &weightedSources);

    /**
     * Computes page rank on the graph passed in constructor.
     */
    void run() override;

    /**
     * Returns the maximum PageRank score
     */
    double maximum() override;

    /**
     * Return the number of iterations performed by the algorithm.
     *
     * @return Number of iterations performed by the algorithm.
     */
    count numberOfIterations() const {
        assureFinished();
        return iterations;
    }

    // Maximum number of iterations allowed
    count maxIterations = std::numeric_limits<count>::max();

    // Norm used as stopping criterion
    Norm norm = Norm::L2_NORM;

private:
    double damp;
    double tol;
    count iterations;
    bool normalized;
    SinkHandling distributeSinks;
    // Per-node teleportation probability, length = upperNodeIdBound(). Sum is 1; empty when the
    // user did not request personalized PageRank (in which case uniform 1/n is used).
    std::vector<double> personalization;
    std::atomic<double> max;
};

} /* namespace NetworKit */

#endif // NETWORKIT_CENTRALITY_PAGE_RANK_HPP_
