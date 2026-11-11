/*
 * ConcurrentUnionFind.hpp
 *
 *  Created on: 06.06.2026
 *      Author: Ian Chen (ianchen3@illinois.edu)
 */

#ifndef NETWORKIT_STRUCTURES_CONCURRENT_UNION_FIND_HPP_
#define NETWORKIT_STRUCTURES_CONCURRENT_UNION_FIND_HPP_

#include <atomic>
#include <utility>
#include <vector>

#include <networkit/Globals.hpp>
#include <networkit/structures/Partition.hpp>

namespace NetworKit {

/**
 * @ingroup structures
 * Concurrent union find implementation with path halving,
 * immediate parent optimization, compression by rank.
 * We use a (sign) bit to distinguish roots.
 *
 * Concurrent calls to find and merge are linearizable: all calls appear to take effect in a
 * single total order, each at some instant during the call.
 *
 * Optimizations are based off of:
 *
 * Alistarh, Dan, Alexander Fedorov, and Nikita Koval.
 * In search of the fastest concurrent union-find algorithm.
 * arXiv preprint arXiv:1911.06347 (2019).
 * https://doi.org/10.48550/arXiv.1911.06347
 */
class ConcurrentUnionFind {
private:
    using element = int64_t; // signed to distinguish roots
    std::vector<std::atomic<element>> parent;

public:
    /**
     * Create a new set representation with not more than @p max_element elements.
     * Initially every element is in its own set.
     * @param max_element maximum number of elements
     */
    ConcurrentUnionFind(index max_element);

    /**
     * Find the representative to element @u
     * @param u element
     * @return representative of set containing @u
     */
    index find(index u);

    /**
     * Merge the two sets contain @u and @v
     * @param u element u
     * @param v element v
     * @returns whether u and v were previously in different sets
     */
    bool merge(index u, index v);
};

} /* namespace NetworKit */

#endif // NETWORKIT_STRUCTURES_CONCURRENT_UNION_FIND_HPP_
