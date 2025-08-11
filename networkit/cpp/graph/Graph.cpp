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
      deletedID(none), exists(n, true), inEdges(0), outEdges(0), // empty vectors since we use CSR
      inEdgeWeights(0), outEdgeWeights(0), inEdgeIds(0), outEdgeIds(0),
      outEdgesCSRIndices(outIndices), outEdgesCSRIndptr(outIndptr), inEdgesCSRIndices(inIndices),
      inEdgesCSRIndptr(inIndptr), usingCSR(true), nodeAttributeMap(this), edgeAttributeMap(this) {

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
    if (!directed) {
        return indexInOutEdgeArray(v, u);
    }
    for (index i = 0; i < inEdges[v].size(); i++) {
        node x = inEdges[v][i];
        if (x == u) {
            return i;
        }
    }
    return none;
}

index Graph::indexInOutEdgeArray(node u, node v) const {
    for (index i = 0; i < outEdges[u].size(); i++) {
        node x = outEdges[u][i];
        if (x == v) {
            return i;
        }
    }
    return none;
}

/** EDGE IDS **/

edgeid Graph::edgeId(node u, node v) const {
    if (!edgesIndexed) {
        throw std::runtime_error("edges have not been indexed - call indexEdges first");
    }

    index i = indexInOutEdgeArray(u, v);

    if (i == none) {
        throw std::runtime_error("Edge does not exist");
    }
    edgeid id = outEdgeIds[u][i];
    return id;
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
    index vi = indexInOutEdgeArray(u, v);
    if (vi == none) {
        return nullWeight;
    } else {
        return weighted ? outEdgeWeights[u][vi] : defaultEdgeWeight;
    }
}

void Graph::setWeightAtIthNeighbor(Unsafe, node u, index i, edgeweight ew) {
    outEdgeWeights[u][i] = ew;
}

void Graph::setWeightAtIthInNeighbor(Unsafe, node u, index i, edgeweight ew) {
    inEdgeWeights[u][i] = ew;
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

    if (!directed && outEdges[u].size() > outEdges[v].size()) {
        return indexInOutEdgeArray(v, u) != none;
    } else if (directed && outEdges[u].size() > inEdges[v].size()) {
        return indexInInEdgeArray(v, u) != none;
    } else {
        return indexInOutEdgeArray(u, v) != none;
    }
}

bool Graph::checkConsistency() const {
    // check for multi-edges
    std::vector<node> lastSeen(z, none);
    bool noMultiEdges = true;
    auto noMultiEdgesDetected = [&noMultiEdges]() { return noMultiEdges; };
    forNodesWhile(noMultiEdgesDetected, [&](node v) {
        forNeighborsOf(v, [&](node u) {
            if (lastSeen[u] == v) {
                noMultiEdges = false;
                DEBUG("Multiedge found between ", u, " and ", v, "!");
            }
            lastSeen[u] = v;
        });
    });

    bool correctNodeUpperbound = (z == outEdges.size()) && ((directed ? z : 0) == inEdges.size())
                                 && ((weighted ? z : 0) == outEdgeWeights.size())
                                 && ((weighted && directed ? z : 0) == inEdgeWeights.size())
                                 && ((edgesIndexed ? z : 0) == outEdgeIds.size())
                                 && ((edgesIndexed && directed ? z : 0) == inEdgeIds.size());

    if (!correctNodeUpperbound)
        DEBUG("Saved node upper bound doesn't actually match the actual node upper bound!");

    count NumberOfOutEdges = 0;
    count NumberOfOutEdgeWeights = 0;
    count NumberOfOutEdgeIds = 0;
    for (index i = 0; i < outEdges.size(); i++) {
        NumberOfOutEdges += outEdges[i].size();
    }
    if (weighted)
        for (index i = 0; i < outEdgeWeights.size(); i++) {
            NumberOfOutEdgeWeights += outEdgeWeights[i].size();
        }
    if (edgesIndexed)
        for (index i = 0; i < outEdgeIds.size(); i++) {
            NumberOfOutEdgeIds += outEdgeIds[i].size();
        }

    count NumberOfInEdges = 0;
    count NumberOfInEdgeWeights = 0;
    count NumberOfInEdgeIds = 0;
    if (directed) {
        for (index i = 0; i < inEdges.size(); i++) {
            NumberOfInEdges += inEdges[i].size();
        }
        if (weighted)
            for (index i = 0; i < inEdgeWeights.size(); i++) {
                NumberOfInEdgeWeights += inEdgeWeights[i].size();
            }
        if (edgesIndexed)
            for (index i = 0; i < inEdgeIds.size(); i++) {
                NumberOfInEdgeIds += inEdgeIds[i].size();
            }
    }

    if (!directed) {
        NumberOfOutEdges = (NumberOfOutEdges + storedNumberOfSelfLoops) / 2;
        if (weighted)
            NumberOfOutEdgeWeights = (NumberOfOutEdgeWeights + storedNumberOfSelfLoops) / 2;
        if (edgesIndexed)
            NumberOfOutEdgeIds = (NumberOfOutEdgeIds + storedNumberOfSelfLoops) / 2;
    }

    bool correctNumberOfEdges = (m == NumberOfOutEdges) && ((directed ? m : 0) == NumberOfInEdges)
                                && ((weighted ? m : 0) == NumberOfOutEdgeWeights)
                                && ((weighted && directed ? m : 0) == NumberOfInEdgeWeights)
                                && ((edgesIndexed ? m : 0) == NumberOfOutEdgeIds)
                                && ((edgesIndexed && directed ? m : 0) == NumberOfInEdgeIds);

    if (!correctNumberOfEdges)
        DEBUG("Saved number of edges is incorrect!");

    return noMultiEdges && correctNodeUpperbound && correctNumberOfEdges;
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

} /* namespace NetworKit */
