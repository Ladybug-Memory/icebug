/*
 * GraphR.cpp
 *
 *  Created on: Feb 8, 2026
 *  Read-only CSR-based graph implementation
 */

#include <networkit/graph/GraphR.hpp>

namespace NetworKit {

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
    // For CSR-based graphs, return default weight of 1.0 if edge exists
    if (hasEdge(u, v)) {
        return defaultEdgeWeight;
    }
    return 0.0; // No edge
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
        if (exists[neighbors.first[i]]) {
            result.push_back(neighbors.first[i]);
        }
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

    for (count i = 0; i < neighbors.second; ++i) {
        if (exists[neighbors.first[i]]) {
            nodeVec.push_back(neighbors.first[i]);
            // CSR graphs in GraphR don't store weights, all edges have default weight
            weightVec.push_back(defaultEdgeWeight);
        }
    }
    return {std::move(nodeVec), std::move(weightVec)};
}

index GraphR::indexInInEdgeArray(node v, node u) const {
    if (!directed) {
        return indexInOutEdgeArray(v, u);
    }

    // For directed graphs, search in incoming edges CSR
    if (!inEdgesCSRIndptr || !inEdgesCSRIndices) {
        return none;
    }

    auto start_idx = inEdgesCSRIndptr->Value(v);
    auto end_idx = inEdgesCSRIndptr->Value(v + 1);

    for (auto idx = start_idx; idx < end_idx; ++idx) {
        if (inEdgesCSRIndices->Value(idx) == u) {
            return idx - start_idx;
        }
    }
    return none;
}

index GraphR::indexInOutEdgeArray(node u, node v) const {
    if (!outEdgesCSRIndptr || !outEdgesCSRIndices) {
        return none;
    }

    auto start_idx = outEdgesCSRIndptr->Value(u);
    auto end_idx = outEdgesCSRIndptr->Value(u + 1);

    for (auto idx = start_idx; idx < end_idx; ++idx) {
        if (outEdgesCSRIndices->Value(idx) == v) {
            return idx - start_idx;
        }
    }
    return none;
}

edgeid GraphR::edgeId(node u, node v) const {
    throw std::runtime_error("edgeId not supported for CSR-based GraphR - use GraphW");
}

node GraphR::getIthNeighbor(Unsafe, node u, index i) const {
    auto start_idx = outEdgesCSRIndptr->Value(u);
    return outEdgesCSRIndices->Value(start_idx + i);
}

edgeweight GraphR::getIthNeighborWeight(node u, index i) const {
    if (!hasNode(u) || i >= degree(u)) {
        return nullWeight;
    }
    // CSR graphs have uniform weight
    return defaultEdgeWeight;
}

node GraphR::getIthNeighbor(node u, index i) const {
    if (!hasNode(u) || i >= degree(u)) {
        return none;
    }
    auto start_idx = outEdgesCSRIndptr->Value(u);
    return outEdgesCSRIndices->Value(start_idx + i);
}

node GraphR::getIthInNeighbor(node u, index i) const {
    if (!hasNode(u) || i >= degreeIn(u)) {
        return none;
    }
    if (!directed) {
        return getIthNeighbor(u, i);
    }
    auto start_idx = inEdgesCSRIndptr->Value(u);
    return inEdgesCSRIndices->Value(start_idx + i);
}

std::pair<node, edgeweight> GraphR::getIthNeighborWithWeight(node u, index i) const {
    if (!hasNode(u) || i >= degree(u)) {
        return {none, nullWeight};
    }
    auto start_idx = outEdgesCSRIndptr->Value(u);
    return {outEdgesCSRIndices->Value(start_idx + i), defaultEdgeWeight};
}

std::pair<node, edgeid> GraphR::getIthNeighborWithId(node u, index i) const {
    throw std::runtime_error(
        "getIthNeighborWithId not supported for CSR-based GraphR - use GraphW");
}

} // namespace NetworKit
