/*
 * SteinerKCore.hpp
 *
 *  Created on: 06.10.26
 *      Author: Ian Chen (ianchen3@illinois.edu)
 */

#ifndef NETWORKIT_SCD_STEINER_K_CORE_HPP_
#define NETWORKIT_SCD_STEINER_K_CORE_HPP_

#include <set>
#include <span>
#include <vector>

#include <networkit/scd/SelectiveCommunityDetector.hpp>

namespace NetworKit {

/**
 * The steiner community expansion algorithm optimizing k-core.
 *
 * This is an implementation of the algorithm proposed in:
 *
 * Chen, I., Yi, H., Sharma, A., Chacko, G., & Warnow, T. (2026).
 * Improved Methods for k-core Community Search.
 * arXiv preprint arXiv:2609.17822.
 * https://arxiv.org/abs/2609.17822
 */
class SteinerKCore : public SelectiveCommunityDetector {
    std::vector<count> ownedCoreness; // for the convenience constructor
    std::span<const count> coreness;

public:
    /**
     * Constructs the Steiner KCore algorithm.
     *
     * @param[in] G The graph to detect communities on
     * @param[in] coreness The pre-computed core decomposition for
          the graph. coreness must be kept alive as long as this is.
     */
    SteinerKCore(const Graph &g, std::span<const count> coreness);

    SteinerKCore(const Graph &g);

    /**
     * Expands a set of seed nodes into a community.
     *
     * @param[in] s The seed nodes
     * @return A community of the seed nodes, or an empty set if they are not connected
     */
    std::set<node> expandOneCommunity(const std::set<node> &s) override;

    /**
     * The community/coreness for each query.
     */
    struct Communities {
        std::vector<std::span<const node>> communities; // empty if the seeds are not connected
        std::vector<count> coreness;                    // none if the seeds are not connected

        Communities() = default;
        Communities(Communities &&) = default;
        Communities &operator=(Communities &&) = default;
        Communities(const Communities &) = delete;
        Communities &operator=(const Communities &) = delete;

    private:
        // builds members/coreness for a set of queries (given in order).
        // interval[q] is query q's component as [first, last] in the circular list cllist;
        // none sentinel represents empty lists.
        Communities(std::span<const node> cllist, std::span<const std::pair<node, node>> interval,
                    std::span<const index> order, std::vector<count> coreness);

        std::vector<node> data; // the buffer that communities view
        friend class SteinerKCore;
    };

    /**
     * Detect a community for each of the given seed sets.
     *
     * @param s The seed sets to find the communities for.
     * @return The communities of @p s, in the order of @p s.
     */
    Communities communities(const std::vector<std::set<node>> &s);

    using SelectiveCommunityDetector::expandOneCommunity;

private:
    static Communities getComms(const std::vector<std::set<node>> &s, count n,
                                std::span<const std::pair<node, node>> forest,
                                std::span<const std::pair<count, std::vector<index>>> levels,
                                std::vector<count> coreness);
};

} // namespace NetworKit

#endif // NETWORKIT_SCD_STEINER_K_CORE_HPP_
