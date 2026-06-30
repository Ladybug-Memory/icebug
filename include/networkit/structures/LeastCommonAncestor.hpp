/*
 * LeastCommonAncestor.hpp
 *
 *  Created on: 06.08.2026
 *      Author: Ian Chen (ianchen3@illinois.edu)
 */

#ifndef NETWORKIT_STRUCTURES_LEAST_COMMON_ANCESTOR_HPP_
#define NETWORKIT_STRUCTURES_LEAST_COMMON_ANCESTOR_HPP_

#include <span>
#include <vector>
#include "networkit/auxiliary/RangeMinimumQuery.hpp"
#include <networkit/Globals.hpp>
#include <networkit/graph/Graph.hpp>

namespace NetworKit {

using index = NetworKit::index;
using count = NetworKit::count;

/**
 * @ingroup structures
 * Least common ancestor for trees.
 * Reduction from LCA to +-1 RMQ, following:
 *
 * Bender, M.A., Farach-Colton, M. (2000).
 * The LCA Problem Revisited.
 * Theoretical Informatics. LATIN 2000.
 * https://doi.org/10.1007/10719839_9
 */
class LeastCommonAncestor {
private:
    const Graph *g;

    Aux::RangeMinimumQuery rmq;
    std::vector<index> position;
    std::vector<node> euler_tour;
    std::vector<int64_t> depths;

public:
    /**
     * Builds LCA from the input tree
     * @param[in] tree Directed graph for the tree.
     */
    LeastCommonAncestor(const Graph &tree, node root);
    LeastCommonAncestor() = default;

    /**
     * Returns the node that is the least common ancestor of all query nodes.
     * @param[in] nodes Nodes.
     *
     * @param[out] least common ancestor.
     */
    node Query(std::span<const node> nodes);
};

} // namespace NetworKit
#endif // NETWORKIT_STRUCTURES_LEAST_COMMON_ANCESTOR_HPP_
