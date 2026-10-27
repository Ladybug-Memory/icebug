/*
 * ShellStruct.cpp
 *
 *  Created on: 06.05.2026
 *      Author: Ian Chen (ianchen3@illinois.edu)
 */

#include <algorithm>
#include <atomic>
#include <initializer_list>
#include <stdexcept>
#include <utility>

#include <networkit/auxiliary/Parallel.hpp>
#include <networkit/auxiliary/Parallelism.hpp>
#include <networkit/centrality/CoreDecomposition.hpp>
#include <networkit/scd/ShellStruct.hpp>
#include <networkit/structures/ConcurrentUnionFind.hpp>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/util/compression.h>

namespace NetworKit {

namespace {

/*
 * Stable parallel counting sort of the indices of keys, after count_sort of ParlayLib:
 *
 * Blelloch, G. E., Anderson, D., & Dhulipala, L. (2020).
 * ParlayLib - A Toolkit for Parallel Algorithms on Shared-Memory Multicore Machines. SPAA 2020.
 * https://github.com/cmuparlay/parlaylib
 *
 * Every key must be below numKeys. Returns the indices grouped by key, each group increasing,
 * and numKeys + 1 offsets: group k is values[offsets[k], offsets[k + 1]).
 */
std::pair<std::vector<index>, std::vector<int64_t>> countSort(std::span<const index> keys,
                                                              count numKeys) {
    const count n = keys.size();
    // ParlayLib's block count keeps the per-block counts to a small fraction of n.
    count numBlocks = 1 + n * sizeof(index) / std::max<count>(numKeys * 500, 5000);
    if (n < 8192 || Aux::getMaxNumberOfThreads() == 1)
        numBlocks = 1;
    const count blockSize = (n + numBlocks - 1) / numBlocks;

    std::vector<int64_t> counts(numBlocks * numKeys, 0);
#pragma omp parallel for schedule(static)
    for (omp_index b = 0; b < static_cast<omp_index>(numBlocks); ++b)
        for (index i = b * blockSize; i < std::min(n, (b + 1) * blockSize); ++i)
            ++counts[b * numKeys + keys[i]];

    std::vector<int64_t> offsets(numKeys + 1);
    int64_t start = 0;
    for (index k = 0; k < numKeys; ++k) {
        offsets[k] = start;
        for (index b = 0; b < numBlocks; ++b)
            start += std::exchange(counts[b * numKeys + k], start);
    }
    offsets[numKeys] = start;

    std::vector<index> values(n);
#pragma omp parallel for schedule(static)
    for (omp_index b = 0; b < static_cast<omp_index>(numBlocks); ++b)
        for (index i = b * blockSize; i < std::min(n, (b + 1) * blockSize); ++i)
            values[counts[b * numKeys + keys[i]]++] = i;

    return {std::move(values), std::move(offsets)};
}

// Post-order ids of a tree whose nodes are numbered children before parents, with the root last.
std::vector<node> postOrder(std::span<const node> parent) {
    const count numNodes = parent.size();
    std::vector<count> size(numNodes, 1);
    for (node t = 0; t + 1 < numNodes; ++t)
        size[parent[t]] += size[t];

    std::vector<node> post(numNodes), start(numNodes, 0);
    for (node t = numNodes; t-- > 0;) {
        if (t + 1 < numNodes)
            start[t] = std::exchange(start[parent[t]], start[parent[t]] + size[t]);
        post[t] = start[t] + size[t] - 1;
    }
    return post;
}

arrow::Result<arrow::ipc::IpcWriteOptions> makeIpcOptions(const std::string &compression) {
    auto opts = arrow::ipc::IpcWriteOptions::Defaults();
    arrow::Compression::type codec;
    if (compression == "ZSTD")
        codec = arrow::Compression::ZSTD;
    else if (compression == "LZ4")
        codec = arrow::Compression::LZ4_FRAME;
    else if (compression == "NONE")
        return opts; // uncompressed default
    else
        return arrow::Status::Invalid("ShellStruct: unknown compression: " + compression);

    ARROW_ASSIGN_OR_RAISE(auto c, arrow::util::Codec::Create(codec));
    opts.codec = std::shared_ptr<arrow::util::Codec>(std::move(c));
    return opts;
}

arrow::Status writeTable(const std::shared_ptr<arrow::Table> &table, const std::string &path,
                         const std::string &compression) {
    ARROW_ASSIGN_OR_RAISE(auto sink, arrow::io::FileOutputStream::Open(path));
    ARROW_ASSIGN_OR_RAISE(auto opts, makeIpcOptions(compression));
    ARROW_ASSIGN_OR_RAISE(auto writer, arrow::ipc::MakeFileWriter(sink, table->schema(), opts));
    ARROW_RETURN_NOT_OK(writer->WriteTable(*table));
    ARROW_RETURN_NOT_OK(writer->Close());
    return sink->Close();
}

arrow::Status readTable(const std::string &path, std::shared_ptr<arrow::Table> *out) {
    ARROW_ASSIGN_OR_RAISE(auto src, arrow::io::ReadableFile::Open(path));
    ARROW_ASSIGN_OR_RAISE(auto reader, arrow::ipc::RecordBatchFileReader::Open(src));
    ARROW_ASSIGN_OR_RAISE(*out, reader->ToTable());
    return arrow::Status::OK();
}

// The named column of table as one array, which must have the given type.
arrow::Result<std::shared_ptr<arrow::Array>>
readColumn(const arrow::Table &table, const std::string &name,
           const std::shared_ptr<arrow::DataType> &type) {
    const auto column = table.GetColumnByName(name);
    if (!column)
        return arrow::Status::Invalid("ShellStruct: missing column '" + name + "'");
    if (!column->type()->Equals(*type))
        return arrow::Status::Invalid("ShellStruct: column '" + name + "' must have type "
                                      + type->ToString());
    if (column->num_chunks() == 0)
        return arrow::MakeEmptyArray(type);
    if (column->num_chunks() == 1)
        return column->chunk(0);
    return arrow::Concatenate(column->chunks(), arrow::default_memory_pool());
}

// The checks of buildInternal that are linear in the size of the index.
arrow::Status verifyColumns(const arrow::UInt64Array &assignment,
                            const arrow::UInt64Array &coreness,
                            const arrow::LargeListArray &vertices,
                            const arrow::LargeListArray &children) {
    const auto &vertexValues = static_cast<const arrow::UInt64Array &>(*vertices.values());
    for (const arrow::Array *array : std::initializer_list<const arrow::Array *>{
             &assignment, &coreness, &vertices, &children, &vertexValues, children.values().get()})
        if (array->null_count() != 0 || array->offset() != 0)
            return arrow::Status::Invalid("ShellStruct: columns must have no nulls or offsets");
    if (vertices.value_offset(0) != 0 || children.value_offset(0) != 0)
        return arrow::Status::Invalid("ShellStruct: list offsets must start at 0");

    const auto n = static_cast<node>(assignment.length());
    std::vector<bool> listed(n, false);
    for (node t = 0; t < static_cast<node>(vertices.length()); ++t)
        for (int64_t j = vertices.value_offset(t); j < vertices.value_offset(t + 1); ++j) {
            const node v = vertexValues.Value(j);
            if (v >= n || listed[v] || assignment.Value(v) != t)
                return arrow::Status::Invalid(
                    "ShellStruct: every vertex must be listed once, under its tree node");
            listed[v] = true;
        }
    return arrow::Status::OK();
}

// The size of the subtree of every tree node. With verify, children must be a post-order tree
// rooted at the last node.
arrow::Result<std::vector<count>> subtreeSizes(const arrow::LargeListArray &children,
                                               bool verify) {
    const auto &childValues = static_cast<const arrow::UInt64Array &>(*children.values());
    const auto numNodes = static_cast<node>(children.length());
    std::vector<count> sizes(numNodes, 1);
    for (node t = 0; t < numNodes; ++t) {
        node next = t;
        for (int64_t j = children.value_offset(t + 1); j-- > children.value_offset(t);) {
            const node c = childValues.Value(j);
            if (verify && (next == 0 || c != next - 1))
                return arrow::Status::Invalid("ShellStruct: children is not a post-order tree");
            next -= sizes[c];
            sizes[t] += sizes[c];
        }
    }
    if (verify && sizes.back() != numNodes)
        return arrow::Status::Invalid("ShellStruct: the last node must be the root");
    return sizes;
}

} // namespace

ShellStruct::ShellStruct(const Graph &g, bool validate)
    : SelectiveCommunityDetector(g), validate(validate) {}

void ShellStruct::build() {
    CoreDecomposition coreDecomposition(*g);
    coreDecomposition.run();
    const std::vector<double> &scores = coreDecomposition.scores();
    build(std::vector<count>(scores.begin(), scores.end()));
}

void ShellStruct::build(std::span<const count> coredecomp) {
    // 3: compute coreness, the core decomposition of G (given)
    const count n = g->upperNodeIdBound();
    if (coredecomp.size() != n)
        throw std::invalid_argument("ShellStruct: coredecomp must have one entry per node id");

    // S_k for every k: the vertices grouped by coreness
    const count maxK =
        n == 0 ? 0 : *Aux::Parallel::max_element(coredecomp.begin(), coredecomp.end());
    const auto [shellNodes, shellOffsets] = countSort(coredecomp, maxK + 1);

    // 4: UF <- UnionFind(V)
    ConcurrentUnionFind<> uf(n);
    // T_{r,k}: the tree node on top of the component of each union-find root r; none otherwise.
    std::vector<node> treeNode(n, none);
    // Tree nodes -> coreness and parent, in creation order.
    std::vector<count> treeCoreness;
    std::vector<node> treeParent;
    // Vertices -> tree nodes.
    std::vector<node> vertexAssignment(n, none);

    // Helper functions: lines 6-10 inside the main-loop.
    // Each is called inside an omp parallel region.

    auto touchedTreeNodes = [&](std::span<const node> shell, count k,
                                std::vector<std::pair<node, node>> &touched) {
        std::vector<std::pair<node, node>> local;
#pragma omp for schedule(guided) nowait
        for (omp_index i = 0; i < static_cast<omp_index>(shell.size()); ++i) {
            g->forNeighborsOf(shell[i], [&](node v) {
                if (coredecomp[v] <= k)
                    return;
                node r = uf.find(v), t;
                std::atomic_ref<node> entry(treeNode[r]);
                // Only the first edge to a component writes, so the others do not contend.
                if (entry.load(std::memory_order_relaxed) != none
                    && (t = entry.exchange(none, std::memory_order_relaxed)) != none)
                    local.emplace_back(r, t);
            });
        }
#pragma omp critical
        touched.insert(touched.end(), local.begin(), local.end());
#pragma omp barrier
    };

    auto unionShell = [&](std::span<const node> shell, count k) {
#pragma omp for schedule(guided)
        for (omp_index i = 0; i < static_cast<omp_index>(shell.size()); ++i) {
            g->forNeighborsOf(shell[i], [&](node v) {
                if (coredecomp[v] >= k)
                    uf.merge(shell[i], v);
            });
        }
    };

    auto createTreeNodes = [&](std::span<const node> shell, count k) {
#pragma omp single
        for (node u : shell) {
            const node r = uf.find(u);
            if (treeNode[r] != none)
                continue;
            treeNode[r] = treeCoreness.size();
            treeCoreness.push_back(k);
            treeParent.push_back(none);
        }
    };

    auto assignShell = [&](std::span<const node> shell) {
#pragma omp for schedule(static)
        for (omp_index i = 0; i < static_cast<omp_index>(shell.size()); ++i)
            vertexAssignment[shell[i]] = treeNode[uf.find(shell[i])];
    };

    auto setParents = [&](std::span<const std::pair<node, node>> touched) {
#pragma omp for schedule(static)
        for (omp_index i = 0; i < static_cast<omp_index>(touched.size()); ++i)
            treeParent[touched[i].second] = treeNode[uf.find(touched[i].first)];
    };

    // 5: for S_k the non-empty k-shell in descending k
    for (count k = maxK; k > 0; --k) {
        const std::span<const node> shell(shellNodes.data() + shellOffsets[k],
                                          shellOffsets[k + 1] - shellOffsets[k]);
        if (shell.empty())
            continue;

        std::vector<std::pair<node, node>> touched;
#pragma omp parallel
        {
            // 6: parinit P_k <- { (r, T_r) | uv in E, u in S_k, coreness(v) > k, r = UF.find(v) }
            touchedTreeNodes(shell, k, touched);

            // 7: parfor uv, u in S_k, coreness(v) >= k: UF.union(u, v)
            unionShell(shell, k);

            // 8: for each distinct UF.find(u), u in S_k: create tree node T_{root,k}
            //    (tree nodes numbered in creation order, so parents after children)
            createTreeNodes(shell, k);

            // 9: parfor u in S_k: assign u to T_{UF.find(u),k}
            assignShell(shell);

            // 10: parfor T in P_k: parent of T <- T_{UF.find(root of T),k}
            setParents(touched);
        }
    }

    // 11: root of coreness 0: parent of every remaining parentless node, holds the vertices of S_0
    const node root = treeCoreness.size();
    std::ranges::replace(treeParent, none, root);
    treeCoreness.push_back(0);
    treeParent.push_back(none);
    for (node u : std::span(shellNodes).first(shellOffsets[1]))
        vertexAssignment[u] = root;

    if (auto status = prepareAndBuild(std::move(vertexAssignment), treeCoreness, treeParent);
        !status.ok())
        throw std::logic_error("ShellStruct: " + status.ToString());
}

arrow::Status ShellStruct::prepareAndBuild(std::vector<node> vertexAssignment,
                                           std::span<const count> treeCoreness,
                                           std::span<const node> treeParent) {
    // post <- the post-order id of each tree node; renumber coreness, parent and assignment by post
    const count numTreeNodes = treeCoreness.size();
    const std::vector<node> post = postOrder(treeParent);
    std::vector<count> postCoreness(numTreeNodes);
    for (node t = 0; t < numTreeNodes; ++t)
        postCoreness[post[t]] = treeCoreness[t];
    std::vector<node> postParent(numTreeNodes - 1);
    for (node t = 0; t + 1 < numTreeNodes; ++t)
        postParent[post[t]] = post[treeParent[t]];
    std::ranges::transform(vertexAssignment, vertexAssignment.begin(),
                           [&](node t) { return post[t]; });
    // vertices <- the vertices grouped by assignment; children <- the tree nodes grouped by parent
    auto [vertexValues, vertexOffsets] = countSort(vertexAssignment, numTreeNodes);
    auto [childValues, childOffsets] = countSort(postParent, numTreeNodes);

    // Lengths are taken first: the vectors are moved into the buffers below.
    const auto numVertices = static_cast<int64_t>(vertexAssignment.size());
    const auto numNodes = static_cast<int64_t>(numTreeNodes);
    const auto listType = arrow::large_list(arrow::uint64());
    auto assignmentArray = std::make_shared<arrow::UInt64Array>(
        numVertices, arrow::Buffer::FromVector(std::move(vertexAssignment)));
    auto corenessArray = std::make_shared<arrow::UInt64Array>(
        numNodes, arrow::Buffer::FromVector(std::move(postCoreness)));
    auto verticesArray = std::make_shared<arrow::LargeListArray>(
        listType, numNodes, arrow::Buffer::FromVector(std::move(vertexOffsets)),
        std::make_shared<arrow::UInt64Array>(numVertices,
                                             arrow::Buffer::FromVector(std::move(vertexValues))));
    auto childrenArray = std::make_shared<arrow::LargeListArray>(
        listType, numNodes, arrow::Buffer::FromVector(std::move(childOffsets)),
        std::make_shared<arrow::UInt64Array>(numNodes - 1,
                                             arrow::Buffer::FromVector(std::move(childValues))));

    return buildInternal(std::move(assignmentArray), std::move(corenessArray),
                         std::move(verticesArray), std::move(childrenArray), false);
}

arrow::Status ShellStruct::buildInternal(std::shared_ptr<arrow::UInt64Array> assignment,
                                         std::shared_ptr<arrow::UInt64Array> coreness,
                                         std::shared_ptr<arrow::LargeListArray> vertices,
                                         std::shared_ptr<arrow::LargeListArray> children,
                                         bool verify) {
    const auto n = static_cast<int64_t>(g->upperNodeIdBound());
    const int64_t numNodes = coreness->length();
    const auto vertexValues = std::static_pointer_cast<arrow::UInt64Array>(vertices->values());
    const auto childValues = std::static_pointer_cast<arrow::UInt64Array>(children->values());
    if (assignment->length() != n)
        return arrow::Status::Invalid("ShellStruct: assignment must have one entry per node id");
    if (numNodes == 0 || vertices->length() != numNodes || children->length() != numNodes)
        return arrow::Status::Invalid(
            "ShellStruct: coreness, vertices and children must have the same nonzero length");
    if (vertices->value_offset(numNodes) - vertices->value_offset(0) != n
        || children->value_offset(numNodes) - children->value_offset(0) != numNodes - 1
        || vertices->value_offset(numNodes) > vertexValues->length()
        || children->value_offset(numNodes) > childValues->length())
        return arrow::Status::Invalid(
            "ShellStruct: the tree must list every vertex and have one edge less than nodes");

    if (verify)
        ARROW_RETURN_NOT_OK(verifyColumns(*assignment, *coreness, *vertices, *children));
    ARROW_ASSIGN_OR_RAISE(auto sizes, subtreeSizes(*children, verify));

    this->assignment = std::move(assignment);
    this->coreness = std::move(coreness);
    this->vertices = std::move(vertices);
    this->children = std::move(children);
    subtreeSize = std::move(sizes);
    tree.emplace(numNodes, true, childValues,
                 std::make_shared<arrow::UInt64Array>(numNodes + 1,
                                                      this->children->value_offsets()));
    lca = std::make_unique<LeastCommonAncestor>(*tree, numNodes - 1);
    return arrow::Status::OK();
}

node ShellStruct::treeNodeOf(const std::set<node> &s) {
    if (*s.rbegin() >= static_cast<node>(assignment->length()))
        throw std::out_of_range("ShellStruct: seed is not a node id");
    std::vector<node> nodes;
    nodes.reserve(s.size());
    for (node u : s)
        nodes.push_back(assignment->Value(u));
    return lca->Query(nodes);
}

std::span<const node> ShellStruct::community(const std::set<node> &s) {
    if (!lca)
        throw std::runtime_error("ShellStruct: call build() or load() first");
    if (s.empty())
        return {};
    const node t = treeNodeOf(s);
    const uint64_t *values =
        std::static_pointer_cast<arrow::UInt64Array>(vertices->values())->raw_values();
    return {values + vertices->value_offset(t + 1 - subtreeSize[t]),
            values + vertices->value_offset(t + 1)};
}

std::set<node> ShellStruct::expandOneCommunity(const std::set<node> &s) {
    const std::span<const node> members = community(s);
    return {members.begin(), members.end()};
}

count ShellStruct::score(const std::set<node> &s) {
    if (!lca)
        throw std::runtime_error("ShellStruct: call build() or load() first");
    if (s.empty())
        throw std::invalid_argument("ShellStruct: score needs at least one seed");
    return coreness->Value(treeNodeOf(s));
}

arrow::Status ShellStruct::saveInternal(const std::string &components_path,
                                        const std::string &tree_path,
                                        const std::string &compression) const {
    {
        auto schema = arrow::schema({arrow::field("assignment", arrow::uint64())});
        auto table = arrow::Table::Make(schema, {assignment});
        ARROW_RETURN_NOT_OK(writeTable(table, components_path, compression));
    }
    {
        auto schema = arrow::schema({
            arrow::field("coreness", arrow::uint64()),
            arrow::field("vertices", arrow::large_list(arrow::uint64())),
            arrow::field("children", arrow::large_list(arrow::uint64())),
        });
        auto table = arrow::Table::Make(schema, {
                                                    std::make_shared<arrow::ChunkedArray>(coreness),
                                                    std::make_shared<arrow::ChunkedArray>(vertices),
                                                    std::make_shared<arrow::ChunkedArray>(children),
                                                });
        ARROW_RETURN_NOT_OK(writeTable(table, tree_path, compression));
    }
    return arrow::Status::OK();
}

arrow::Status ShellStruct::loadInternal(const std::string &components_path,
                                        const std::string &tree_path) {
    std::shared_ptr<arrow::Table> components, treeTable;
    ARROW_RETURN_NOT_OK(readTable(components_path, &components));
    ARROW_RETURN_NOT_OK(readTable(tree_path, &treeTable));

    const auto listType = arrow::large_list(arrow::uint64());
    ARROW_ASSIGN_OR_RAISE(auto assignmentColumn,
                          readColumn(*components, "assignment", arrow::uint64()));
    ARROW_ASSIGN_OR_RAISE(auto corenessColumn, readColumn(*treeTable, "coreness", arrow::uint64()));
    ARROW_ASSIGN_OR_RAISE(auto verticesColumn, readColumn(*treeTable, "vertices", listType));
    ARROW_ASSIGN_OR_RAISE(auto childrenColumn, readColumn(*treeTable, "children", listType));
    return buildInternal(std::static_pointer_cast<arrow::UInt64Array>(assignmentColumn),
                         std::static_pointer_cast<arrow::UInt64Array>(corenessColumn),
                         std::static_pointer_cast<arrow::LargeListArray>(verticesColumn),
                         std::static_pointer_cast<arrow::LargeListArray>(childrenColumn),
                         validate);
}

void ShellStruct::save(const std::string &components_path, const std::string &tree_path,
                       const std::string &compression) const {
    if (!lca)
        throw std::runtime_error("ShellStruct: call build() or load() first");
    if (auto status = saveInternal(components_path, tree_path, compression); !status.ok())
        throw std::runtime_error("ShellStruct: save failed: " + status.ToString());
}

void ShellStruct::load(const std::string &components_path, const std::string &tree_path) {
    if (auto status = loadInternal(components_path, tree_path); !status.ok())
        throw std::runtime_error("ShellStruct: load failed: " + status.ToString());
}

} // namespace NetworKit
