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

} // namespace NetworKit
