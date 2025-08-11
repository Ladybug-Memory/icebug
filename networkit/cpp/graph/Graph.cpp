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
             std::shared_ptr<arrow::UInt64Array> inIndptr)
    : n(n), m(0), storedNumberOfSelfLoops(0), z(n), omega(0), t(0),
      weighted(false),                         // CSR graphs are unweighted for now
      directed(directed), edgesIndexed(false), // CSR graphs don't use edge IDs
      deletedID(none), exists(n, true), outEdgesCSRIndices(outIndices),
      outEdgesCSRIndptr(outIndptr), inEdgesCSRIndices(inIndices), inEdgesCSRIndptr(inIndptr),
      usingCSR(true), nodeAttributeMap(this), edgeAttributeMap(this) {

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
}

/** PRIVATE HELPERS **/

index Graph::indexInInEdgeArray(node v, node u) const {
    // Base Graph class only supports CSR format
    throw std::runtime_error("indexInInEdgeArray not supported in base Graph class - use GraphW "
                             "for vector-based operations");
}

index Graph::indexInOutEdgeArray(node u, node v) const {
    // Base Graph class only supports CSR format
    throw std::runtime_error("indexInOutEdgeArray not supported in base Graph class - use GraphW "
                             "for vector-based operations");
}

/** EDGE IDS **/

edgeid Graph::edgeId(node u, node v) const {
    // Base Graph class only supports CSR format
    throw std::runtime_error(
        "edgeId not supported in base Graph class - use GraphW for vector-based operations");
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
    // Base Graph class only supports CSR format
    throw std::runtime_error(
        "weight method not supported in base Graph class - use GraphW for vector-based operations");
}

void Graph::setWeightAtIthNeighbor(Unsafe, node u, index i, edgeweight ew) {
    // Base Graph class only supports CSR format
    throw std::runtime_error("setWeightAtIthNeighbor not supported in base Graph class - use "
                             "GraphW for mutable operations");
}

void Graph::setWeightAtIthInNeighbor(Unsafe, node u, index i, edgeweight ew) {
    // Base Graph class only supports CSR format
    throw std::runtime_error("setWeightAtIthInNeighbor not supported in base Graph class - use "
                             "GraphW for mutable operations");
}

edgeweight Graph::totalEdgeWeight() const noexcept {
    if (weighted)
        return parallelSumForEdges([](node, node, edgeweight ew) { return ew; });
    return numberOfEdges() * defaultEdgeWeight;
}

bool Graph::hasEdge(node u, node v) const noexcept {
    if (u >= z || v >= z) {
        return false;
    }

    // Use CSR if available
    if (usingCSR) {
        return hasEdgeCSR(u, v);
    }

    // Base Graph class only supports CSR format
    throw std::runtime_error(
        "hasEdge method requires CSR format - non-CSR graphs should use GraphW");
}

bool Graph::checkConsistency() const {
    // Base Graph class only supports CSR format
    // For CSR graphs, basic consistency checks are done in constructor
    if (!usingCSR) {
        throw std::runtime_error("checkConsistency for vector-based graphs not supported in base "
                                 "Graph class - use GraphW");
    }

    // For CSR graphs, we can do basic checks
    if (outEdgesCSRIndptr && outEdgesCSRIndptr->length() != z + 1) {
        return false;
    }
    if (directed && inEdgesCSRIndptr && inEdgesCSRIndptr->length() != z + 1) {
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

} /* namespace NetworKit */
