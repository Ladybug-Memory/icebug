/*
 * ShellStruct.hpp
 *
 *  Created on: 06.05.2026
 *      Author: Ian Chen (ianchen3@illinois.edu)
 */

#ifndef NETWORKIT_SCD_SHELL_STRUCT_HPP_
#define NETWORKIT_SCD_SHELL_STRUCT_HPP_

#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <vector>

#include <networkit/Globals.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/graph/GraphR.hpp>
#include <networkit/scd/SelectiveCommunityDetector.hpp>
#include <networkit/structures/LeastCommonAncestor.hpp>

#include <arrow/array/array_nested.h>
#include <arrow/array/array_primitive.h>
#include <arrow/status.h>

namespace NetworKit {

/**
 * The ShellStruct index for fast online k-core community search.
 * The community of a seed set is the connected component of the k-core containing it, for the
 * largest k in which it is connected.
 *
 * This is an implementation of the algorithm proposed in:
 *
 * Chen, I., Yi, H., Sharma, A., Chacko, G., & Warnow, T. (2026).
 * Improved Methods for k-core Community Search.
 * arXiv preprint arXiv:2609.17822.
 * https://arxiv.org/abs/2609.17822
 */
class ShellStruct : public SelectiveCommunityDetector {

private:
    // Tree nodes are numbered in post-order, so the subtree of t is the nodes
    // [t + 1 - subtreeSize[t], t] and its vertices are one contiguous slice of vertices->values().
    std::shared_ptr<arrow::UInt64Array> assignment;  // vertices -> tree nodes
    std::shared_ptr<arrow::UInt64Array> coreness;    // tree nodes -> coreness
    std::shared_ptr<arrow::LargeListArray> vertices; // tree nodes -> vertices
    std::shared_ptr<arrow::LargeListArray> children; // tree nodes -> child tree nodes

    std::optional<GraphR> tree;
    std::unique_ptr<LeastCommonAncestor> lca; // of tree
    std::vector<count> subtreeSize;           // tree nodes -> size of the subtree

    bool validate;

    // LCA of the tree nodes of the seeds in s. Throws std::out_of_range for invalid seed nodes.
    node treeNodeOf(const std::set<node> &s);

    // Derive tree, lca, and subtreeSize; with verify, first verify invariants of the inputs.
    arrow::Status buildInternal(std::shared_ptr<arrow::UInt64Array> assignment,
                                std::shared_ptr<arrow::UInt64Array> coreness,
                                std::shared_ptr<arrow::LargeListArray> vertices,
                                std::shared_ptr<arrow::LargeListArray> children, bool verify);

    // Prepare the output of Alg. 2 into the final representation and pass it to buildInternal.
    arrow::Status prepareAndBuild(std::vector<node> vertexAssignment,
                                  std::span<const count> treeCoreness,
                                  std::span<const node> treeParent);

    // Write the components and tree tables as Feather files. Returns failures as a Status.
    arrow::Status saveInternal(const std::string &components_path, const std::string &tree_path,
                               const std::string &compression) const;
    // Read the tables written by saveInternal and pass them to buildInternal.
    arrow::Status loadInternal(const std::string &components_path, const std::string &tree_path);

public:
    /**
     * @param[in] g the graph
     * @param[in] validate check the invariants of the files read by load(); has no effect on
     * build().
     */
    ShellStruct(const Graph &g, bool validate = false);

    /**
     * Builds the index, computing the core decomposition of the graph.
     */
    void build();

    /**
     * Builds the index from a precomputed core decomposition.
     *
     * @param[in] coredecomp the core number of every node id; its size must be
     * <code>upperNodeIdBound()</code>.
     */
    void build(std::span<const count> coredecomp);

    /**
     * @param[in] s seed nodes
     *
     * @return the community of the seeds as a set of nodes; empty if there are no seeds.
     */
    std::set<node> expandOneCommunity(const std::set<node> &s) override;

    /**
     * Returns the core number of the community without instantiating it.
     *
     * @param[in] s seed nodes; must not be empty.
     *
     * @return the core number of the community of the seeds.
     */
    count score(const std::set<node> &s);

    /**
     * Returns the community of the seeds without copying it. The span points into the index, so
     * it is invalidated by the next build() or load() and by destroying the index.
     *
     * @param[in] s seed nodes
     *
     * @return the nodes of the community, in no particular order; empty if there are no seeds.
     */
    std::span<const node> community(const std::set<node> &s);

    /**
     * Saves the index to disk as two Feather (Arrow IPC) files.
     *
     * @param[in] components_path the assignment of vertices to tree nodes.
     * @param[in] tree_path the coreness, vertices and children of each tree node.
     * @param[in] compression one of "ZSTD", "LZ4" or "NONE".
     */
    void save(const std::string &components_path, const std::string &tree_path,
              const std::string &compression = "ZSTD") const;

    /**
     * Loads an index saved by save() for the same graph.
     *
     * @param[in] components_path the assignment of vertices to tree nodes.
     * @param[in] tree_path the coreness, vertices and children of each tree node.
     */
    void load(const std::string &components_path, const std::string &tree_path);

    // inherit method from parent class.
    using SelectiveCommunityDetector::expandOneCommunity;
};

} /* namespace NetworKit */
#endif // NETWORKIT_SCD_SHELL_STRUCT_HPP_
