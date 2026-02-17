/*
 * Graph.cpp
 *
 *  Created on: 01.06.2014
 *      Author: Christian Staudt
 *              Klara Reichard <klara.reichard@gmail.com>
 *              Marvin Ritter <marvin.ritter@gmail.com>
 */

#include <cmath>
#include <map>
#include <random>
#include <ranges>
#include <sstream>

#include <networkit/auxiliary/Log.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/graph/GraphTools.hpp>

namespace NetworKit {

/** CONSTRUCTORS **/

Graph::Graph(count n, bool weighted, bool directed, bool edgesIndexed)
    : n(n), m(0), storedNumberOfSelfLoops(0), z(n), omega(0), t(0),

      weighted(weighted), // indicates whether the graph is weighted or not
      directed(directed), // indicates whether the graph is directed or not
      edgesIndexed(edgesIndexed), deletedID(none),
      // edges are not indexed by default

      exists(n, true), usingCSR(false), nodeAttributeMap(this), edgeAttributeMap(this) {}

// CSR constructor for zero-copy Arrow arrays
Graph::Graph(count n, bool directed, std::shared_ptr<arrow::UInt64Array> outIndices,
             std::shared_ptr<arrow::UInt64Array> outIndptr,
             std::shared_ptr<arrow::UInt64Array> inIndices,
             std::shared_ptr<arrow::UInt64Array> inIndptr,
             std::shared_ptr<arrow::DoubleArray> outWeights,
             std::shared_ptr<arrow::DoubleArray> inWeights)
    : n(n), m(0), storedNumberOfSelfLoops(0), z(n), omega(0), t(0),
      weighted(outWeights != nullptr),         // CSR graphs are weighted if weights are provided
      directed(directed), edgesIndexed(false), // CSR graphs don't use edge IDs
      deletedID(none), exists(n, true), outEdgesCSRIndices(outIndices),
      outEdgesCSRIndptr(outIndptr), inEdgesCSRIndices(inIndices), inEdgesCSRIndptr(inIndptr),
      outEdgesCSRWeights(outWeights), inEdgesCSRWeights(inWeights), usingCSR(true),
      nodeAttributeMap(this), edgeAttributeMap(this) {

    // Calculate number of edges from CSR data
    if (outIndices) {
        m = outIndices->length();
    }

    // For directed graphs, we need both in and out CSR arrays
    if (directed && (!inIndices || !inIndptr)) {
        throw std::invalid_argument(
            "Directed CSR graphs require both incoming and outgoing arrays");
    }

    // Validate CSR arrays
    if (outIndptr && static_cast<size_t>(outIndptr->length()) != n + 1) {
        throw std::invalid_argument("outIndptr must have length n+1");
    }
    if (directed && inIndptr && static_cast<size_t>(inIndptr->length()) != n + 1) {
        throw std::invalid_argument("inIndptr must have length n+1");
    }

    // Validate weight arrays if provided
    if (outWeights && outIndices && outWeights->length() != outIndices->length()) {
        throw std::invalid_argument("outWeights must have same length as outIndices");
    }
    if (directed && inWeights && inIndices && inWeights->length() != inIndices->length()) {
        throw std::invalid_argument("inWeights must have same length as inIndices");
    }
}

/** PRIVATE HELPERS **/

index Graph::indexInInEdgeArray(node v, node u) const {
    if (!directed) {
        return indexInOutEdgeArray(v, u);
    }
    throw std::runtime_error("indexInInEdgeArray not implemented for CSR format");
}

index Graph::indexInOutEdgeArray([[maybe_unused]] node u, [[maybe_unused]] node v) const {
    throw std::runtime_error(
        "indexInOutEdgeArray not implemented - use GraphW for vector-based implementation");
}

/** EDGE IDS **/

edgeid Graph::edgeId([[maybe_unused]] node u, [[maybe_unused]] node v) const {
    throw std::runtime_error("edgeId not implemented - use GraphW for vector-based implementation");
}

/** GRAPH INFORMATION **/

edgeweight Graph::computeWeightedDegree(node u, bool inDegree, bool countSelfLoopsTwice) const {
    if (weighted) {
        edgeweight sum = 0.0;
        auto sumWeights = [&](node v, edgeweight w) {
            sum += (countSelfLoopsTwice && u == v) ? 2. * w : w;
        };
        if (inDegree) {
            forInNeighborsOf(u, sumWeights);
        } else {
            forNeighborsOf(u, sumWeights);
        }
        return sum;
    }

    count sum = inDegree ? degreeIn(u) : degreeOut(u);
    auto countSelfLoops = [&](node v) { sum += (u == v); };

    if (countSelfLoopsTwice && numberOfSelfLoops()) {
        if (inDegree) {
            forInNeighborsOf(u, countSelfLoops);
        } else {
            forNeighborsOf(u, countSelfLoops);
        }
    }

    return static_cast<edgeweight>(sum);
}

/** NODE MODIFIERS **/

/** NODE PROPERTIES **/

edgeweight Graph::weightedDegree(node u, bool countSelfLoopsTwice) const {
    return computeWeightedDegree(u, false, countSelfLoopsTwice);
}

edgeweight Graph::weightedDegreeIn(node u, bool countSelfLoopsTwice) const {
    return computeWeightedDegree(u, true, countSelfLoopsTwice);
}

/** EDGE MODIFIERS **/

edgeweight Graph::weight(node u, node v) const {
    // For CSR-based graphs, return default weight of 1.0 if edge exists
    if (hasEdge(u, v)) {
        return 1.0;
    }
    return 0.0; // No edge
}

void Graph::setWeightAtIthNeighbor(Unsafe, node u, index i, [[maybe_unused]] edgeweight ew) {
    // For CSR-based graphs, this operation requires modifying the CSR structure
    // Currently CSR graphs are unweighted, so this operation is not directly supported
    // For weighted graphs, you would need to:
    // 1. Store weights in a separate CSR array
    // 2. Update the weight at the specified position

    if (!usingCSR) {
        throw std::runtime_error("setWeightAtIthNeighbor requires CSR format");
    }

    // Validate indices
    if (u >= z || !outEdgesCSRIndptr) {
        throw std::out_of_range("Invalid node index in setWeightAtIthNeighbor");
    }

    auto start_idx = outEdgesCSRIndptr->Value(u);
    auto end_idx = outEdgesCSRIndptr->Value(u + 1);
    count degree = end_idx - start_idx;

    if (i >= degree) {
        throw std::out_of_range("Invalid neighbor index in setWeightAtIthNeighbor");
    }

    // For now, CSR graphs don't support weights
    // In a full implementation, you would update a separate weights CSR array
    // For unweighted CSR graphs, setting weights doesn't make sense
    throw std::runtime_error("setWeightAtIthNeighbor not supported for unweighted CSR graphs. "
                             "Use GraphW for weighted mutable graphs.");
}

void Graph::setWeightAtIthInNeighbor(Unsafe, node u, index i, [[maybe_unused]] edgeweight ew) {
    // For CSR-based graphs, this operation requires modifying the CSR structure
    // Currently CSR graphs are unweighted, so this operation is not directly supported

    if (!usingCSR) {
        throw std::runtime_error("setWeightAtIthInNeighbor requires CSR format");
    }

    // For directed graphs, use incoming CSR arrays
    if (directed) {
        if (!inEdgesCSRIndptr) {
            throw std::runtime_error("setWeightAtIthInNeighbor requires incoming CSR arrays");
        }

        // Validate indices
        if (u >= z) {
            throw std::out_of_range("Invalid node index in setWeightAtIthInNeighbor");
        }

        auto start_idx = inEdgesCSRIndptr->Value(u);
        auto end_idx = inEdgesCSRIndptr->Value(u + 1);
        count degree = end_idx - start_idx;

        if (i >= degree) {
            throw std::out_of_range("Invalid neighbor index in setWeightAtIthInNeighbor");
        }
    } else {
        // For undirected graphs, incoming neighbors are the same as outgoing neighbors
        // Just redirect to the outgoing neighbor version
        setWeightAtIthNeighbor(Unsafe{}, u, i, ew);
        return;
    }

    // For now, CSR graphs don't support weights
    throw std::runtime_error("setWeightAtIthInNeighbor not supported for unweighted CSR graphs. "
                             "Use GraphW for weighted mutable graphs.");
}

edgeweight Graph::totalEdgeWeight() const noexcept {
    if (weighted)
        return parallelSumForEdges([](node, node, edgeweight ew) { return ew; });
    return numberOfEdges() * defaultEdgeWeight;
}

bool Graph::hasEdge(node u, node v) const {
    if (u >= z || v >= z) {
        return false;
    }

    // Use CSR if available
    if (usingCSR) {
        return hasEdgeCSR(u, v);
    }

    // For vector-based graphs, use virtual dispatch
    return hasEdgeImpl(u, v);
}

bool Graph::hasEdgeImpl(node u, node v) const {
    // CSR-based implementation
    if (usingCSR) {
        return hasEdgeCSR(u, v);
    }
    // Base class doesn't have vector-based implementation
    throw std::runtime_error("hasEdgeImpl not implemented for base Graph class");
}

bool Graph::checkConsistency() const {
    // Base Graph class only supports CSR format
    // For CSR graphs, basic consistency checks are done in constructor
    if (!usingCSR) {
        throw std::runtime_error("checkConsistency for vector-based graphs not supported in base "
                                 "Graph class - use GraphW");
    }

    // For CSR graphs, we can do basic checks
    if (outEdgesCSRIndptr
        && static_cast<int64_t>(outEdgesCSRIndptr->length()) != static_cast<int64_t>(z) + 1) {
        return false;
    }
    if (directed && inEdgesCSRIndptr
        && static_cast<int64_t>(inEdgesCSRIndptr->length()) != static_cast<int64_t>(z) + 1) {
        return false;
    }

    return true; // Basic CSR consistency
}

// CSR helper methods
bool Graph::hasEdgeCSR(node u, node v) const {
    if (!usingCSR || u >= z || v >= z) {
        return false;
    }

    // Use outgoing edges for search
    if (!outEdgesCSRIndices || !outEdgesCSRIndptr) {
        return false;
    }

    auto start_idx = outEdgesCSRIndptr->Value(u);
    auto end_idx = outEdgesCSRIndptr->Value(u + 1);

    // Binary search for neighbor v in sorted adjacency list
    for (auto idx = start_idx; idx < end_idx; ++idx) {
        auto neighbor = outEdgesCSRIndices->Value(idx);
        if (neighbor == v) {
            return true;
        }
        if (neighbor > v) {
            break; // assuming sorted neighbors
        }
    }

    return false;
}

count Graph::degreeCSR(node u, bool incoming) const {
    if (!usingCSR || u >= z) {
        return 0;
    }

    if (incoming && directed) {
        if (!inEdgesCSRIndptr) {
            return 0;
        }
        return inEdgesCSRIndptr->Value(u + 1) - inEdgesCSRIndptr->Value(u);
    } else {
        if (!outEdgesCSRIndptr) {
            return 0;
        }
        return outEdgesCSRIndptr->Value(u + 1) - outEdgesCSRIndptr->Value(u);
    }
}

std::pair<const node *, count> Graph::getCSROutNeighbors(node u) const {
    if (!usingCSR || u >= z || !outEdgesCSRIndices || !outEdgesCSRIndptr) {
        return {nullptr, 0};
    }

    auto start_idx = outEdgesCSRIndptr->Value(u);
    auto end_idx = outEdgesCSRIndptr->Value(u + 1);
    count degree = end_idx - start_idx;

    if (degree == 0) {
        return {nullptr, 0};
    }

    // Return pointer to the beginning of this node's neighbors in the CSR indices array
    const node *neighbors =
        reinterpret_cast<const node *>(outEdgesCSRIndices->raw_values()) + start_idx;
    return {neighbors, degree};
}

std::pair<const node *, count> Graph::getCSRInNeighbors(node u) const {
    if (!usingCSR || u >= z || !directed || !inEdgesCSRIndices || !inEdgesCSRIndptr) {
        // For undirected graphs, incoming neighbors are the same as outgoing neighbors
        if (!directed) {
            return getCSROutNeighbors(u);
        }
        return {nullptr, 0};
    }

    auto start_idx = inEdgesCSRIndptr->Value(u);
    auto end_idx = inEdgesCSRIndptr->Value(u + 1);
    count degree = end_idx - start_idx;

    if (degree == 0) {
        return {nullptr, 0};
    }

    // Return pointer to the beginning of this node's incoming neighbors in the CSR indices array
    const node *neighbors =
        reinterpret_cast<const node *>(inEdgesCSRIndices->raw_values()) + start_idx;
    return {neighbors, degree};
}

void Graph::forEdgesVirtualImpl([[maybe_unused]] bool directed, [[maybe_unused]] bool weighted,
                                [[maybe_unused]] bool hasEdgeIds,
                                std::function<void(node, node, edgeweight, edgeid)> handle) const {
    // CSR-based implementation
    for (node u = 0; u < z; ++u) {
        auto [neighbors, degree] = getCSROutNeighbors(u);
        for (count i = 0; i < degree; ++i) {
            node v = neighbors[i];
            if (!exists[v])
                continue;

            // For undirected graphs, only process edge if u >= v to avoid duplicates
            if (!directed && !useEdgeInIteration<false>(u, v))
                continue;

            // CSR graphs are currently unweighted and don't support edge IDs
            edgeweight w = defaultEdgeWeight;
            edgeid eid = none;
            handle(u, v, w, eid);
        }
    }
}

void Graph::forEdgesOfVirtualImpl(
    node u, [[maybe_unused]] bool directed, [[maybe_unused]] bool weighted,
    [[maybe_unused]] bool hasEdgeIds,
    std::function<void(node, node, edgeweight, edgeid)> handle) const {
    // CSR-based implementation for a single node
    auto [neighbors, degree] = getCSROutNeighbors(u);
    for (count i = 0; i < degree; ++i) {
        node v = neighbors[i];
        if (!exists[v])
            continue;

        // CSR graphs are currently unweighted and don't support edge IDs
        edgeweight w = defaultEdgeWeight;
        edgeid eid = none;
        handle(u, v, w, eid);
    }
}

void Graph::forInEdgesVirtualImpl(
    node u, [[maybe_unused]] bool directed, [[maybe_unused]] bool weighted,
    [[maybe_unused]] bool hasEdgeIds,
    std::function<void(node, node, edgeweight, edgeid)> handle) const {
    // CSR-based implementation for in-edges
    auto [neighbors, degree] = getCSRInNeighbors(u);
    for (count i = 0; i < degree; ++i) {
        node v = neighbors[i];
        if (!exists[v])
            continue;

        // CSR graphs are currently unweighted and don't support edge IDs
        edgeweight w = defaultEdgeWeight;
        edgeid eid = none;
        handle(u, v, w, eid);
    }
}

double Graph::parallelSumForEdgesVirtualImpl(
    [[maybe_unused]] bool directed, [[maybe_unused]] bool weighted,
    [[maybe_unused]] bool hasEdgeIds,
    std::function<double(node, node, edgeweight, edgeid)> handle) const {
    double sum = 0.0;

    // CSR-based parallel implementation
#pragma omp parallel for schedule(guided) reduction(+ : sum)
    for (omp_index u = 0; u < static_cast<omp_index>(z); ++u) {
        if (!exists[u])
            continue;

        auto [neighbors, degree] = getCSROutNeighbors(u);
        if (neighbors == nullptr || degree == 0)
            continue;

        for (count i = 0; i < degree; ++i) {
            node v = neighbors[i];
            if (!exists[v])
                continue;

            // For undirected graphs, only process edge if u >= v to avoid duplicates
            if (!directed && !useEdgeInIteration<false>(u, v))
                continue;

            // CSR graphs are currently unweighted and don't support edge IDs
            edgeweight w = defaultEdgeWeight;
            edgeid eid = none;

            sum += handle(u, v, w, eid);
        }
    }

    return sum;
}

} /* namespace NetworKit */
