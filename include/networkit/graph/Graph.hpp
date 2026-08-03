/*
 * Graph.hpp
 *
 *  Created on: 01.06.2014
 *      Author: Christian Staudt
 *              Klara Reichard <klara.reichard@gmail.com>
 *              Marvin Ritter <marvin.ritter@gmail.com>
 */

#ifndef NETWORKIT_GRAPH_GRAPH_HPP_
#define NETWORKIT_GRAPH_GRAPH_HPP_

/* For transitive includes */
#include <queue>
#include <stack>
#include <unordered_map>
#include <unordered_set>

#include <networkit/graph/GraphR.hpp>
#include <networkit/graph/GraphTypes.hpp>
#include <networkit/graph/GraphW.hpp>
#include <networkit/graph/ReferenceGraphImpl.hpp>

namespace NetworKit {

/**
 * A graph whose concrete type the call site does not name.
 *
 * Non-owning: the referenced graph must outlive the handle. Reads dispatch on the concrete type
 * at compile time, so an algorithm written against Graph iterates as fast as one written against
 * GraphW or GraphR directly.
 */
using Graph = ReferenceGraph;

} // namespace NetworKit

namespace std {
template <>
struct hash<NetworKit::Edge> {
    size_t operator()(const NetworKit::Edge &e) const { return hash_node(e.u) ^ hash_node(e.v); }

    hash<NetworKit::node> hash_node;
};
} // namespace std

#endif // NETWORKIT_GRAPH_GRAPH_HPP_
