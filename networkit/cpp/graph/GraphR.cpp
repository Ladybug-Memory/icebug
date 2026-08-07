/*
 * GraphR.cpp
 *
 *  Created on: Feb 8, 2026
 *  Read-only CSR-based graph implementation
 */

#include <networkit/graph/GraphIteration.hpp>
#include <networkit/graph/GraphR.hpp>

namespace NetworKit {

GraphR::GraphR(count n, bool directed, std::shared_ptr<arrow::UInt64Array> outIndices,
               std::shared_ptr<arrow::UInt64Array> outIndptr,
               std::shared_ptr<arrow::UInt64Array> inIndices,
               std::shared_ptr<arrow::UInt64Array> inIndptr,
               std::shared_ptr<arrow::DoubleArray> outWeights,
               std::shared_ptr<arrow::DoubleArray> inWeights)
    : n(n), m(0), storedNumberOfSelfLoops(0), z(n), t(0), weighted(outWeights != nullptr),
      directed(directed), outEdgesCSRIndices(std::move(outIndices)),
      outEdgesCSRIndptr(std::move(outIndptr)), inEdgesCSRIndices(std::move(inIndices)),
      inEdgesCSRIndptr(std::move(inIndptr)), outEdgesCSRWeights(std::move(outWeights)),
      inEdgesCSRWeights(std::move(inWeights)) {

    // An undirected edge occupies a slot in both endpoints' adjacency.
    if (outEdgesCSRIndices) {
        m = directed ? static_cast<count>(outEdgesCSRIndices->length())
                     : static_cast<count>(outEdgesCSRIndices->length()) / 2;
    }

    if (outEdgesCSRIndptr && static_cast<size_t>(outEdgesCSRIndptr->length()) != n + 1) {
        throw std::runtime_error("outIndptr must have length n+1");
    }
    if (inEdgesCSRIndptr && static_cast<size_t>(inEdgesCSRIndptr->length()) != n + 1) {
        throw std::runtime_error("inIndptr must have length n+1");
    }
    if (outEdgesCSRWeights && outEdgesCSRIndices
        && outEdgesCSRWeights->length() != outEdgesCSRIndices->length()) {
        throw std::runtime_error("outWeights must have the same length as outIndices");
    }
    if (inEdgesCSRWeights && inEdgesCSRIndices
        && inEdgesCSRWeights->length() != inEdgesCSRIndices->length()) {
        throw std::runtime_error("inWeights must have the same length as inIndices");
    }
}

bool GraphR::checkConsistency() const {
    if (outEdgesCSRIndptr
        && static_cast<int64_t>(outEdgesCSRIndptr->length()) != static_cast<int64_t>(z) + 1) {
        return false;
    }
    if (directed && inEdgesCSRIndptr
        && static_cast<int64_t>(inEdgesCSRIndptr->length()) != static_cast<int64_t>(z) + 1) {
        return false;
    }
    return true;
}

index GraphR::findInAdjacency(const std::shared_ptr<arrow::UInt64Array> &indptr,
                              const std::shared_ptr<arrow::UInt64Array> &indices, node u,
                              node v) const {
    if (u >= z || !indptr || !indices) {
        return none;
    }

    // The CSR constructor requires sorted adjacency, so this can bisect.
    index low = indptr->Value(u), high = indptr->Value(u + 1);
    while (low < high) {
        const auto mid = low + (high - low) / 2;
        const auto neighbor = indices->Value(mid);
        if (neighbor < v) {
            low = mid + 1;
        } else if (neighbor > v) {
            high = mid;
        } else {
            return mid;
        }
    }
    return none;
}

bool GraphR::hasEdge(node u, node v) const {
    if (v >= z) {
        return false;
    }
    return findInAdjacency(outEdgesCSRIndptr, outEdgesCSRIndices, u, v) != none;
}

count GraphR::degreeCSR(node u, bool incoming) const {
    if (u >= z) {
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

count GraphR::degree(node v) const {
    assert(hasNode(v));
    return degreeCSR(v, false);
}

count GraphR::degreeIn(node v) const {
    assert(hasNode(v));
    return directed ? degreeCSR(v, true) : degreeCSR(v, false);
}

bool GraphR::isIsolated(node v) const {
    if (!hasNode(v))
        throw std::runtime_error("Error, the node does not exist!");
    return degreeCSR(v, false) == 0 && (!directed || degreeCSR(v, true) == 0);
}

edgeweight GraphR::weight(node u, node v) const {
    if (v >= z) {
        return 0.0;
    }
    const index slot = findInAdjacency(outEdgesCSRIndptr, outEdgesCSRIndices, u, v);
    if (slot == none) {
        return 0.0;
    }
    if (weighted && outEdgesCSRWeights) {
        return outEdgesCSRWeights->Value(slot);
    }
    return defaultEdgeWeight;
}

edgeid GraphR::edgeId([[maybe_unused]] node u, [[maybe_unused]] node v) const {
    throw std::runtime_error("edgeId not supported for CSR-based GraphR - use GraphW");
}

std::pair<node, node> GraphR::edgeById([[maybe_unused]] index id) const {
    throw std::runtime_error("edgeById not supported for CSR-based GraphR - use GraphW");
}

std::pair<const node *, count> GraphR::getCSROutNeighbors(node u) const {
    if (u >= z || !outEdgesCSRIndices || !outEdgesCSRIndptr) {
        return {nullptr, 0};
    }

    const auto start_idx = outEdgesCSRIndptr->Value(u);
    const count deg = outEdgesCSRIndptr->Value(u + 1) - start_idx;
    if (deg == 0) {
        return {nullptr, 0};
    }
    return {reinterpret_cast<const node *>(outEdgesCSRIndices->raw_values()) + start_idx, deg};
}

std::pair<const node *, count> GraphR::getCSRInNeighbors(node u) const {
    if (u >= z) {
        return {nullptr, 0};
    }
    if (!directed) {
        return getCSROutNeighbors(u);
    }
    if (!inEdgesCSRIndices || !inEdgesCSRIndptr) {
        return {nullptr, 0};
    }

    const auto start_idx = inEdgesCSRIndptr->Value(u);
    const count deg = inEdgesCSRIndptr->Value(u + 1) - start_idx;
    if (deg == 0) {
        return {nullptr, 0};
    }
    return {reinterpret_cast<const node *>(inEdgesCSRIndices->raw_values()) + start_idx, deg};
}

std::vector<node> GraphR::getNeighborsVector(node u, bool inEdges) const {
    std::pair<const node *, count> neighbors;
    if (inEdges) {
        neighbors = getCSRInNeighbors(u);
    } else {
        neighbors = getCSROutNeighbors(u);
    }

    std::vector<node> result;
    result.reserve(neighbors.second);
    for (count i = 0; i < neighbors.second; ++i) {
        result.push_back(neighbors.first[i]);
    }
    return result;
}

std::pair<std::vector<node>, std::vector<edgeweight>>
GraphR::getNeighborsWithWeightsVector(node u, bool inEdges) const {
    std::pair<const node *, count> neighbors;
    if (inEdges) {
        neighbors = getCSRInNeighbors(u);
    } else {
        neighbors = getCSROutNeighbors(u);
    }

    std::vector<node> nodeVec;
    std::vector<edgeweight> weightVec;
    nodeVec.reserve(neighbors.second);
    weightVec.reserve(neighbors.second);

    // Get the starting index for this node
    index startIdx;
    std::shared_ptr<arrow::DoubleArray> weightsArr;
    if (inEdges && directed && inEdgesCSRIndptr) {
        startIdx = inEdgesCSRIndptr->Value(u);
        weightsArr = inEdgesCSRWeights;
    } else if (inEdges && directed) {
        return {{}, {}};
    } else {
        startIdx = outEdgesCSRIndptr->Value(u);
        weightsArr = outEdgesCSRWeights;
    }

    for (count i = 0; i < neighbors.second; ++i) {
        nodeVec.push_back(neighbors.first[i]);
        // Use actual weight if available, otherwise use default
        if (weighted && weightsArr) {
            weightVec.push_back(weightsArr->Value(startIdx + i));
        } else {
            weightVec.push_back(defaultEdgeWeight);
        }
    }
    return {std::move(nodeVec), std::move(weightVec)};
}

index GraphR::indexInInEdgeArray(node v, node u) const {
    if (!directed) {
        return indexInOutEdgeArray(v, u);
    }
    const index slot = findInAdjacency(inEdgesCSRIndptr, inEdgesCSRIndices, v, u);
    return slot == none ? none : slot - inEdgesCSRIndptr->Value(v);
}

index GraphR::indexInOutEdgeArray(node u, node v) const {
    const index slot = findInAdjacency(outEdgesCSRIndptr, outEdgesCSRIndices, u, v);
    return slot == none ? none : slot - outEdgesCSRIndptr->Value(u);
}

node GraphR::getIthNeighbor(Unsafe, node u, index i) const {
    return outEdgesCSRIndices->Value(outEdgesCSRIndptr->Value(u) + i);
}

node GraphR::getIthNeighbor(node u, index i) const {
    if (!hasNode(u) || i >= degree(u)) {
        return none;
    }
    return getIthNeighbor(Unsafe{}, u, i);
}

node GraphR::getIthInNeighbor(node u, index i) const {
    if (!hasNode(u) || i >= degreeIn(u)) {
        return none;
    }
    if (!directed) {
        return getIthNeighbor(u, i);
    }
    return inEdgesCSRIndices->Value(inEdgesCSRIndptr->Value(u) + i);
}

edgeweight GraphR::getIthNeighborWeight(node u, index i) const {
    if (!hasNode(u) || i >= degree(u)) {
        return nullWeight;
    }
    if (weighted && outEdgesCSRWeights) {
        return outEdgesCSRWeights->Value(outEdgesCSRIndptr->Value(u) + i);
    }
    return defaultEdgeWeight;
}

std::pair<node, edgeweight> GraphR::getIthNeighborWithWeight(node u, index i) const {
    if (!hasNode(u) || i >= degree(u)) {
        return {none, nullWeight};
    }
    const auto start_idx = outEdgesCSRIndptr->Value(u);
    const node v = outEdgesCSRIndices->Value(start_idx + i);
    const edgeweight w = (weighted && outEdgesCSRWeights) ? outEdgesCSRWeights->Value(start_idx + i)
                                                          : defaultEdgeWeight;
    return {v, w};
}

std::pair<node, edgeid> GraphR::getIthNeighborWithId([[maybe_unused]] node u,
                                                     [[maybe_unused]] index i) const {
    throw std::runtime_error(
        "getIthNeighborWithId not supported for CSR-based GraphR - use GraphW");
}

} // namespace NetworKit
