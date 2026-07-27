/*
 * ShellStruct.cpp
 *
 *  Created on: 06.05.2026
 *      Author: Ian Chen (ianchen3@illinois.edu)
 */

#include <iterator>
#include <span>
#include <stdexcept>

#include "networkit/auxiliary/Vector2Arrow.hpp"
#include <networkit/centrality/CoreDecomposition.hpp>
#include <networkit/graph/BFS.hpp>
#include <networkit/scd/SelectiveCommunityDetector.hpp>
#include <networkit/scd/ShellStruct.hpp>
#include <networkit/structures/ConcurrentUnionFind.hpp>
#include <networkit/structures/LeastCommonAncestor.hpp>

#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/type_fwd.h>
#include <arrow/util/compression.h>

namespace NetworKit {

using indptr = int64_t;

template <typename LocalState, typename ProcessFunc, typename CombineFunc>
static void forNeighborsOfSet(const Graph &G, std::span<const node> S, ProcessFunc F,
                              CombineFunc Combine) {
#pragma omp parallel
    {
        LocalState local_state;
#pragma omp for schedule(guided)
        for (omp_index i = 0; i < S.size(); ++i) {
            node v = S[i];
            G.forNeighborsOf(v, [&](node neighbor) { F(v, neighbor, local_state); });
        }
#pragma omp critical
        Combine(local_state);
    }
}

template <typename KeyFunc, typename ValueFunc, typename Index>
static void append_to_csr(std::vector<Index> &indptr, std::vector<node> &indices,
                          std::span<const node> items, size_t num_keys, KeyFunc key_func,
                          ValueFunc val_func) {
    if (num_keys == 0)
        return;

    std::vector<size_t> counts(num_keys, 0);
    for (const auto &item : items)
        counts[key_func(item)]++;

    size_t start_pos = indices.size();
    indices.resize(start_pos + std::size(items));

    std::vector<size_t> write_offsets(num_keys);
    Index current_offset = start_pos;

    for (Index i = 0; i < static_cast<Index>(num_keys); ++i) {
        write_offsets[i] = current_offset;
        current_offset += counts[i];
        indptr.push_back(current_offset);
    }

    for (const auto &item : items) {
        size_t key = key_func(item);
        indices[write_offsets[key]++] = val_func(item);
    }
}

ShellStruct::ShellStruct(const Graph &g) : SelectiveCommunityDetector(g), built(false) {}

void ShellStruct::build() {
    NetworKit::CoreDecomposition coredecomp(*g);
    coredecomp.run();

    const std::vector<double> &scores = coredecomp.scores();
    auto vec = std::vector<uint64_t>();
    vec.reserve(scores.size());
    for (auto s : scores)
        vec.push_back(static_cast<uint64_t>(s));

    build(Aux::vectorToArrow<uint64_t, arrow::UInt64Array>(vec));
}

void ShellStruct::build(const std::shared_ptr<arrow::UInt64Array> &coredecomp) {
    index n = g->upperNodeIdBound();

    index max_k = 0;
    for (node i = 0; i < static_cast<node>(coredecomp->length()); ++i)
        max_k = std::max(max_k, coredecomp->Value(i));

    std::vector<index> shell_indptr(max_k + 2, 0);
    for (node i = 0; i < n; ++i)
        shell_indptr[coredecomp->Value(i)]++;

    index current_offset = 0;
    for (index k = 0; k <= max_k; ++k) {
        index count = shell_indptr[k];
        shell_indptr[k] = current_offset;
        current_offset += count;
    }
    shell_indptr[max_k + 1] = current_offset;

    std::vector<node> shell_nodes(n);
    std::vector<index> insert_offsets = shell_indptr;
    for (node i = 0; i < n; ++i) {
        index k = coredecomp->Value(i);
        shell_nodes[insert_offsets[k]++] = i;
    }

    count next_id = 0;
    std::vector<node> tree_ids(n, none), tree_v, tree_c;
    std::vector<indptr> tree_v_indptr = {0}, tree_c_indptr = {0};
    std::vector<index> coreness_vec, assignment_vec(n, none);
    ConcurrentUnionFind dsu(n);

    std::vector<node> new_ids(n, none);

    for (index k = max_k; k > 0; --k) {
        size_t shell_size = shell_indptr[k + 1] - shell_indptr[k];
        if (shell_size == 0)
            continue;
        std::span<const node> V_k(shell_nodes.data() + shell_indptr[k], shell_size);

        // Phase 2a: Identifies roots of neighboring components in higher k-shells.
        std::vector<node> touched;
        forNeighborsOfSet<std::vector<node>>(
            *g, V_k,
            [&](node /*v*/, node neighbor, std::vector<node> &local_touched) {
                index neighbor_k = coredecomp->Value(neighbor);
                if (neighbor_k > k) {
                    node old_root = dsu.find(neighbor);
                    if (tree_ids[old_root] != none)
                        local_touched.push_back(old_root);
                }
            },
            [&](const std::vector<node> &local_touched) {
                touched.insert(touched.end(), local_touched.begin(), local_touched.end());
            });

        std::sort(touched.begin(), touched.end());
        touched.erase(std::unique(touched.begin(), touched.end()), touched.end());

        // Phase 2b: Contracts k-shell into components.
        forNeighborsOfSet<int>(
            *g, V_k,
            [&](node v, node neighbor, int &) {
                index neighbor_k = coredecomp->Value(neighbor);
                if (neighbor_k >= k)
                    dsu.merge(v, neighbor);
            },
            [&](const int &) {});

        // Phase 3: Assigns new hierarchical tree node IDs to newly formed disjoint components.
        std::vector<node> new_roots;
        for (index i = 0; i < shell_size; ++i) {
            node root = dsu.find(V_k[i]);
            if (new_ids[root] == none) {
                new_roots.push_back(root);
                new_ids[root] = next_id++;
                coreness_vec.push_back(k);
            }
        }

#pragma omp parallel for schedule(static)
        for (omp_index i = 0; i < shell_size; ++i)
            assignment_vec[V_k[i]] = new_ids[dsu.find(V_k[i])];

        // Phases 4: Builds the tree CSR (and the inverse assignment)
        count num_new_trees = new_roots.size();
        count base_tree_id = next_id - num_new_trees;
        append_to_csr(
            tree_v_indptr, tree_v, V_k, num_new_trees,
            [&](node v) { return new_ids[dsu.find(v)] - base_tree_id; }, [&](node v) { return v; });
        append_to_csr(
            tree_c_indptr, tree_c, touched, num_new_trees,
            [&](node v) { return new_ids[dsu.find(v)] - base_tree_id; },
            [&](node v) { return tree_ids[v]; });

        // Phase 5: Cleanup
        for (node old_root : touched)
            tree_ids[old_root] = none;
        for (node root : new_roots) {
            tree_ids[root] = new_ids[root];
            new_ids[root] = none;
        }
    }

    // Phase 6: ensure tree is connected
    size_t shell_size_0 = shell_indptr[1] - shell_indptr[0];
    std::span<const node> V_0(shell_nodes.data() + shell_indptr[0], shell_size_0);
    std::vector<node> forest_roots;
    for (node id : tree_ids)
        if (id != none)
            forest_roots.push_back(id);
    node root = next_id++;
    coreness_vec.push_back(0);

    for (omp_index i = 0; i < shell_size_0; ++i)
        assignment_vec[V_0[i]] = root;

    append_to_csr(
        tree_v_indptr, tree_v, V_0, 1, [&](node) { return 0; }, [&](node v) { return v; });
    append_to_csr(
        tree_c_indptr, tree_c, forest_roots, 1, [&](node) { return 0; },
        [&](node child) { return child; });

    // Final Phase: convert to member types

    treeIndptr = Aux::vectorToArrow<indptr, arrow::UInt64Array>(tree_c_indptr);
    treeIndices = Aux::vectorToArrow<uint64_t, arrow::UInt64Array>(tree_c);
    tree = GraphR(next_id, true, treeIndices, treeIndptr);
    lca = std::make_unique<NetworKit::LeastCommonAncestor>(*tree, root);

    auto offsets_arr = Aux::vectorToArrow<indptr, arrow::Int64Array>(tree_v_indptr);
    auto tree_v_arr = Aux::vectorToArrow<uint64_t, arrow::UInt64Array>(tree_v);
    vertices = std::static_pointer_cast<arrow::LargeListArray>(
        arrow::LargeListArray::FromArrays(*offsets_arr, *tree_v_arr).ValueOrDie());

    assignment = Aux::vectorToArrow<uint64_t, arrow::UInt64Array>(assignment_vec);
    coreness = Aux::vectorToArrow<uint64_t, arrow::UInt64Array>(coreness_vec);

    // mark as built
    built = true;
}

std::set<node> ShellStruct::expandOneCommunity(const std::set<node> &s) {
    if (!built)
        throw std::runtime_error("Need to build() or load() the shell struct");

    std::vector<node> output;
    if (s.empty())
        return std::set<node>(output.begin(), output.end());

    std::vector<node> query;
    query.reserve(s.size());
    for (node u : s)
        query.push_back(assignment->Value(u));
    node ancestor = lca->Query(std::vector<node>(query.begin(), query.end()));

    const uint64_t *raw_v =
        std::static_pointer_cast<arrow::UInt64Array>(vertices->values())->raw_values();
    const indptr *raw_offsets = vertices->raw_value_offsets();

    Traversal::BFSfrom(*tree, ancestor, [&](node u) {
        output.insert(output.end(), raw_v + raw_offsets[u], raw_v + raw_offsets[u + 1]);
    });

    return std::set<node>(output.begin(), output.end());
}

static arrow::Result<arrow::ipc::IpcWriteOptions> makeIpcOptions(const std::string &compression) {
    auto opts = arrow::ipc::IpcWriteOptions::Defaults();
    arrow::Compression::type codec;
    if (compression == "ZSTD")
        codec = arrow::Compression::ZSTD;
    else if (compression == "LZ4")
        codec = arrow::Compression::LZ4_FRAME;
    else if (compression == "NONE")
        return opts; // uncompressed default
    else
        return arrow::Status::Invalid("ShellStruct::save — unknown compression: " + compression);

    ARROW_ASSIGN_OR_RAISE(auto c, arrow::util::Codec::Create(codec));
    opts.codec = std::shared_ptr<arrow::util::Codec>(std::move(c));
    return opts;
}

static arrow::Status writeTable(const std::shared_ptr<arrow::Table> &table, const std::string &path,
                                const std::string &compression) {
    ARROW_ASSIGN_OR_RAISE(auto sink, arrow::io::FileOutputStream::Open(path));
    ARROW_ASSIGN_OR_RAISE(auto opts, makeIpcOptions(compression));
    ARROW_ASSIGN_OR_RAISE(auto writer, arrow::ipc::MakeFileWriter(sink, table->schema(), opts));
    ARROW_RETURN_NOT_OK(writer->WriteTable(*table));
    ARROW_RETURN_NOT_OK(writer->Close());
    return sink->Close();
}

static arrow::Status readTable(const std::string &path, std::shared_ptr<arrow::Table> *out) {
    ARROW_ASSIGN_OR_RAISE(auto src, arrow::io::ReadableFile::Open(path));
    ARROW_ASSIGN_OR_RAISE(auto reader, arrow::ipc::RecordBatchFileReader::Open(src));

    std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
    batches.reserve(reader->num_record_batches());
    for (int i = 0; i < reader->num_record_batches(); ++i) {
        auto result = reader->ReadRecordBatch(i);
        if (!result.ok())
            return result.status();
        batches.push_back(std::move(result).ValueOrDie());
    }

    ARROW_ASSIGN_OR_RAISE(*out, arrow::Table::FromRecordBatches(reader->schema(), batches));
    return arrow::Status::OK();
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
            arrow::field("csr_indptr", arrow::uint64()),
            arrow::field("csr_indices", arrow::uint64()),
        });
        auto indptr = std::static_pointer_cast<arrow::UInt64Array>(
            treeIndptr->Slice(0, treeIndptr->length() - 1));
        arrow::UInt64Builder builder;
        ARROW_RETURN_NOT_OK(builder.AppendValues(treeIndices->raw_values(), treeIndices->length()));
        ARROW_RETURN_NOT_OK(builder.AppendNull());
        std::shared_ptr<arrow::Array> indices;
        ARROW_RETURN_NOT_OK(builder.Finish(&indices));
        auto table = arrow::Table::Make(schema, {
                                                    std::make_shared<arrow::ChunkedArray>(coreness),
                                                    std::make_shared<arrow::ChunkedArray>(vertices),
                                                    std::make_shared<arrow::ChunkedArray>(indptr),
                                                    std::make_shared<arrow::ChunkedArray>(indices),
                                                });
        ARROW_RETURN_NOT_OK(writeTable(table, tree_path, compression));
    }
    return arrow::Status::OK();
}

arrow::Status ShellStruct::loadInternal(const std::string &components_path,
                                        const std::string &tree_path) {
    {
        std::shared_ptr<arrow::Table> table;
        ARROW_RETURN_NOT_OK(readTable(components_path, &table));

        auto col = table->GetColumnByName("assignment");
        if (!col)
            return arrow::Status::Invalid("components file missing 'assignment' column");
        ARROW_ASSIGN_OR_RAISE(auto combined,
                              arrow::Concatenate(col->chunks(), arrow::default_memory_pool()));
        assignment = std::static_pointer_cast<arrow::UInt64Array>(combined);
    }

    {
        std::shared_ptr<arrow::Table> table;
        ARROW_RETURN_NOT_OK(readTable(tree_path, &table));

        auto load_uint64 = [&](const std::string &name,
                               std::shared_ptr<arrow::UInt64Array> &out) -> arrow::Status {
            auto col = table->GetColumnByName(name);
            if (!col)
                return arrow::Status::Invalid("tree file missing '" + name + "' column");
            ARROW_ASSIGN_OR_RAISE(auto combined,
                                  arrow::Concatenate(col->chunks(), arrow::default_memory_pool()));
            out = std::static_pointer_cast<arrow::UInt64Array>(combined);
            return arrow::Status::OK();
        };

        std::shared_ptr<arrow::UInt64Array> indptr, indices;
        ARROW_RETURN_NOT_OK(load_uint64("coreness", coreness));
        ARROW_RETURN_NOT_OK(load_uint64("csr_indptr", indptr));
        ARROW_RETURN_NOT_OK(load_uint64("csr_indices", indices));
        treeIndices =
            std::static_pointer_cast<arrow::UInt64Array>(indices->Slice(0, indices->length() - 1));
        arrow::UInt64Builder builder;
        ARROW_RETURN_NOT_OK(builder.AppendValues(indptr->raw_values(), indptr->length()));
        ARROW_RETURN_NOT_OK(builder.Append(treeIndices->length()));
        ARROW_RETURN_NOT_OK(builder.Finish(&treeIndptr));
        {
            auto col = table->GetColumnByName("vertices");
            if (!col)
                return arrow::Status::Invalid("tree file missing 'vertices' column");
            ARROW_ASSIGN_OR_RAISE(auto combined,
                                  arrow::Concatenate(col->chunks(), arrow::default_memory_pool()));
            vertices = std::static_pointer_cast<arrow::LargeListArray>(combined);
        }

        count n = treeIndptr->length() - 1;
        tree.emplace(n, true, treeIndices, treeIndptr);

        node root = none;
        for (node v = 0; v < n && root == none; ++v)
            if (coreness->Value(v) == 0)
                root = v;

        tree = GraphR(n, true, treeIndices, treeIndptr);
        lca = std::make_unique<NetworKit::LeastCommonAncestor>(*tree, root);
    }

    built = true;
    return arrow::Status::OK();
}

void ShellStruct::save(const std::string &components_path, const std::string &tree_path,
                       const std::string &compression) const {
    if (!built)
        throw std::runtime_error("ShellStruct::save called before build()");

    auto status = saveInternal(components_path, tree_path, compression);
    if (!status.ok())
        throw std::runtime_error("ShellStruct::save failed: " + status.ToString());
}

void ShellStruct::load(const std::string &components_path, const std::string &tree_path) {
    auto status = loadInternal(components_path, tree_path);
    if (!status.ok())
        throw std::runtime_error("ShellStruct::load failed: " + status.ToString());
}

} // namespace NetworKit
