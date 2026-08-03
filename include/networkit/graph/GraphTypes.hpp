/*
 * GraphTypes.hpp
 *
 *  Value types shared by every graph class and by the handles over them.
 */

#ifndef NETWORKIT_GRAPH_GRAPH_TYPES_HPP_
#define NETWORKIT_GRAPH_GRAPH_TYPES_HPP_

#include <algorithm>
#include <limits>

#include <networkit/Globals.hpp>

namespace NetworKit {

struct Edge {
    node u, v;

    Edge() : u(none), v(none) {}

    Edge(node _u, node _v, bool sorted = false) {
        u = sorted ? std::min(_u, _v) : _u;
        v = sorted ? std::max(_u, _v) : _v;
    }
};

/**
 * A weighted edge used for the graph constructor with
 * initializer list syntax.
 */
struct WeightedEdge : Edge {
    edgeweight weight;

    // Needed by cython
    WeightedEdge() : Edge(), weight(std::numeric_limits<edgeweight>::max()) {}

    WeightedEdge(node u, node v, edgeweight w) : Edge(u, v), weight(w) {}
};

struct WeightedEdgeWithId : WeightedEdge {
    edgeid eid;

    WeightedEdgeWithId(node u, node v, edgeweight w, edgeid eid)
        : WeightedEdge(u, v, w), eid(eid) {}
};

inline bool operator==(const Edge &e1, const Edge &e2) {
    return e1.u == e2.u && e1.v == e2.v;
}

inline bool operator<(const WeightedEdge &e1, const WeightedEdge &e2) {
    return e1.weight < e2.weight;
}

/// Tag selecting an overload that skips bounds and existence checks.
struct Unsafe {};
static constexpr Unsafe unsafe{};

} // namespace NetworKit

#endif // NETWORKIT_GRAPH_GRAPH_TYPES_HPP_
