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
    // PageRank holds an std::atomic<double> and a const-reference base, both of which make
    // the class non-movable. The static factories therefore construct a PageRank as a
    // prvalue and rely on C++17 guaranteed copy elision for the return; see the private
    // constructor below. Users always go through the public constructor or one of the
    // static factories — no copying/moving of PageRank is supported.
    PageRank(const PageRank &) = delete;
    PageRank &operator=(const PageRank &) = delete;

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
     *
     * @note For very large graphs, prefer #forSource / #forSources / #forWeights, which
     * avoid materializing an @c upperNodeIdBound()-sized personalization vector. This
     * constructor auto-detects sparse personalization (at most #SPARSE_THRESHOLD non-zero
     * entries) and uses the memory-efficient representation internally; the
     * @c forSource-style factories skip even the one-time scan.
     */
    PageRank(const Graph &G, double damp = 0.85, double tol = 1e-8, bool normalized = false,
             SinkHandling distributeSinks = SinkHandling::NO_SINK_HANDLING,
             std::vector<double> personalization = {});

    /**
     * Memory-efficient single-source personalized PageRank (random walk with restart from
     * @a source). Avoids allocating the @c upperNodeIdBound()-sized teleportation vector and
     * uses a fast path in @ref run() that touches only the teleportation target; recommended
     * for very large graphs.
     *
     * @param G Graph to be processed.
     * @param source Node to teleport to (must exist in @a G).
     * @param damp Damping factor (default 0.85).
     * @param tol Error tolerance (default 1e-8).
     * @param normalized Normalize PageRank scores (default false).
     * @param distributeSinks How to handle sink nodes (default NO_SINK_HANDLING).
     */
    static PageRank forSource(const Graph &G, node source, double damp = 0.85, double tol = 1e-8,
                              bool normalized = false,
                              SinkHandling distributeSinks = SinkHandling::NO_SINK_HANDLING);

    /**
     * Memory-efficient @c k-source personalized PageRank with uniform teleportation over
     * @a sources (each source gets weight @c 1/|sources|). Avoids the
     * @c upperNodeIdBound()-sized vector.
     *
     * @param G Graph to be processed.
     * @param sources Non-empty set of teleportation targets.
     */
    static PageRank forSources(const Graph &G, const std::vector<node> &sources, double damp = 0.85,
                               double tol = 1e-8, bool normalized = false,
                               SinkHandling distributeSinks = SinkHandling::NO_SINK_HANDLING);

    /**
     * Memory-efficient @c k-source personalized PageRank with positive per-source weights.
     * The weights are normalized to sum to one. Avoids the @c upperNodeIdBound()-sized vector.
     *
     * @param G Graph to be processed.
     * @param weightedSources Non-empty list of (node, weight) pairs with weight > 0.
     */
    static PageRank forWeights(const Graph &G,
                               const std::vector<std::pair<node, double>> &weightedSources,
                               double damp = 0.85, double tol = 1e-8, bool normalized = false,
                               SinkHandling distributeSinks = SinkHandling::NO_SINK_HANDLING);

    /**
     * Maximum number of non-zero entries in a personalization vector for which the
     * constructor switches to the memory-efficient sparse representation. Above this
     * threshold the dense @c upperNodeIdBound()-sized vector is used. The
     * @c forSource-style factories always use the sparse representation regardless of
     * this value.
     */
    static constexpr count SPARSE_THRESHOLD = 64;

    /**
     * Builds a personalization vector that teleports exclusively to a single @a source node.
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
    static std::vector<double> personalizationFromSources(const Graph &G,
                                                          const std::vector<node> &sources);

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

    /**
     * Constructor that takes the sparse personalization representation directly. Used by
     * the @c forSource / @c forSources / @c forWeights static factories (and by language
     * bindings) to build a PageRank without ever materializing an
     * @c upperNodeIdBound()-sized personalization vector. The @a sparseSpec must sum to 1
     * and contain no duplicate entries; the caller is responsible for validation.
     *
     * @note Prefer the public constructor or one of the @c forX static factories. This
     * overload skips input validation and is mainly intended for binding implementations
     * and for the static factories themselves.
     */
    PageRank(const Graph &G, double damp, double tol, bool normalized, SinkHandling distributeSinks,
             std::vector<std::pair<node, double>> sparseSpec);

private:
    double damp;
    double tol;
    count iterations;
    bool normalized;
    SinkHandling distributeSinks;

    // Personalization storage. Exactly one of three configurations is active at a time:
    //   * sparse.empty() && dense.empty(): uniform teleportation (1/n).
    //   * !sparse.empty(): sparse representation of size k, the number of teleportation
    //     sources. Entries sum to 1. Used when the number of non-zero entries is at most
    //     SPARSE_THRESHOLD, or whenever the user goes through forSource/forSources/forWeights.
    //   * !dense.empty(): dense representation of size upperNodeIdBound(). Entries sum to 1.
    //     Used when the user passes a personalization vector with more than SPARSE_THRESHOLD
    //     non-zero entries to the public constructor.
    std::vector<std::pair<node, double>> sparse;
    std::vector<double> dense;

    std::atomic<double> max;
};

} /* namespace NetworKit */

#endif // NETWORKIT_CENTRALITY_PAGE_RANK_HPP_
