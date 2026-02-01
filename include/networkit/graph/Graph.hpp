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

#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <numeric>
#include <omp.h>
#include <queue>
#include <ranges>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <typeindex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <networkit/Globals.hpp>
#include <networkit/auxiliary/ArrayTools.hpp>
#include <networkit/auxiliary/FunctionTraits.hpp>
#include <networkit/auxiliary/Log.hpp>
#include <networkit/auxiliary/Random.hpp>
#include <networkit/graph/Attributes.hpp>
#include <networkit/graph/EdgeIterators.hpp>
#include <networkit/graph/NeighborIterators.hpp>
#include <networkit/graph/NodeIterators.hpp>

#include <tlx/define/deprecated.hpp>

#include <arrow/api.h>
#include <arrow/compute/api.h>

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

struct Unsafe {};
static constexpr Unsafe unsafe{};
} // namespace NetworKit

namespace std {
template <>
struct hash<NetworKit::Edge> {
    size_t operator()(const NetworKit::Edge &e) const { return hash_node(e.u) ^ hash_node(e.v); }

    hash<NetworKit::node> hash_node;
};
} // namespace std

namespace NetworKit {

// forward declaration to randomization/CurveballImpl.hpp
namespace CurveballDetails {
class CurveballMaterialization;
}

/**
 * @ingroup graph
 * A graph (with optional weights) and parallel iterator methods.
 */
class Graph {

protected:
    // graph attributes
    //!< current number of nodes
    count n;
    //!< current number of edges
    count m;

    //!< current number of self loops, edges which have the same origin and
    //!< target
    count storedNumberOfSelfLoops;

    //!< current upper bound of node ids, z will be the id of the next node
    node z;
    //!< current upper bound of edge ids, will be the id of the next edge
    edgeid omega;
    //!< current time step
    count t;

    //!< true if the graph is weighted, false otherwise
    bool weighted;
    //!< true if the graph is directed, false otherwise
    bool directed;
    //!< true if edge ids have been assigned
    bool edgesIndexed;

    //!< true if edge removals should maintain compact edge ids
    bool maintainCompactEdges = false;
    //!< true if edge removals should maintain sorted edge ids
    bool maintainSortedEdges = false;

    //!< saves the ID of the most recently removed edge (if exists)
    edgeid deletedID;

    // per node data
    //!< exists[v] is true if node v has not been removed from the graph
    std::vector<bool> exists;

    // CSR arrays for memory-efficient graph storage (zero-copy from Arrow)
    //!< Arrow array for CSR indices (neighbor node IDs) for outgoing edges
    std::shared_ptr<arrow::UInt64Array> outEdgesCSRIndices;
    //!< Arrow array for CSR indptr (offsets into indices array) for outgoing edges
    std::shared_ptr<arrow::UInt64Array> outEdgesCSRIndptr;
    //!< Arrow array for CSR indices (neighbor node IDs) for incoming edges - only for directed
    //!< graphs
    std::shared_ptr<arrow::UInt64Array> inEdgesCSRIndices;
    //!< Arrow array for CSR indptr (offsets into indices array) for incoming edges - only for
    //!< directed graphs
    std::shared_ptr<arrow::UInt64Array> inEdgesCSRIndptr;
    //!< flag to indicate if CSR arrays are being used instead of vectors
    bool usingCSR;

private:
    AttributeMap<PerNode, Graph> nodeAttributeMap;
    AttributeMap<PerEdge, Graph> edgeAttributeMap;

public:
    auto &nodeAttributes() noexcept { return nodeAttributeMap; }
    const auto &nodeAttributes() const noexcept { return nodeAttributeMap; }
    auto &edgeAttributes() noexcept { return edgeAttributeMap; }
    const auto &edgeAttributes() const noexcept { return edgeAttributeMap; }

    // wrap up some typed attributes for the cython interface:
    //

    auto attachNodeIntAttribute(const std::string &name) {
        nodeAttributes().theGraph = this;
        return nodeAttributes().attach<int>(name);
    }

    auto attachEdgeIntAttribute(const std::string &name) {
        edgeAttributes().theGraph = this;
        return edgeAttributes().attach<int>(name);
    }

    auto attachNodeDoubleAttribute(const std::string &name) {
        nodeAttributes().theGraph = this;
        return nodeAttributes().attach<double>(name);
    }

    auto attachEdgeDoubleAttribute(const std::string &name) {
        edgeAttributes().theGraph = this;
        return edgeAttributes().attach<double>(name);
    }

    auto attachNodeStringAttribute(const std::string &name) {
        nodeAttributes().theGraph = this;
        return nodeAttributes().attach<std::string>(name);
    }

    auto attachEdgeStringAttribute(const std::string &name) {
        edgeAttributes().theGraph = this;
        return edgeAttributes().attach<std::string>(name);
    }

    auto getNodeIntAttribute(const std::string &name) {
        nodeAttributes().theGraph = this;
        return nodeAttributes().get<int>(name);
    }

    auto getEdgeIntAttribute(const std::string &name) {
        edgeAttributes().theGraph = this;
        return edgeAttributes().get<int>(name);
    }

    auto getNodeDoubleAttribute(const std::string &name) {
        nodeAttributes().theGraph = this;
        return nodeAttributes().get<double>(name);
    }

    auto getEdgeDoubleAttribute(const std::string &name) {
        edgeAttributes().theGraph = this;
        return edgeAttributes().get<double>(name);
    }

    auto getNodeStringAttribute(const std::string &name) {
        nodeAttributes().theGraph = this;
        return nodeAttributes().get<std::string>(name);
    }

    auto getEdgeStringAttribute(const std::string &name) {
        edgeAttributes().theGraph = this;
        return edgeAttributes().get<std::string>(name);
    }

    void detachNodeAttribute(std::string const &name) {
        nodeAttributes().theGraph = this;
        nodeAttributes().detach(name);
    }

    void detachEdgeAttribute(std::string const &name) {
        edgeAttributes().theGraph = this;
        edgeAttributes().detach(name);
    }

    using NodeIntAttribute = Attribute<PerNode, Graph, int, false>;
    using NodeDoubleAttribute = Attribute<PerNode, Graph, double, false>;
    using NodeStringAttribute = Attribute<PerNode, Graph, std::string, false>;

    using EdgeIntAttribute = Attribute<PerEdge, Graph, int, false>;
    using EdgeDoubleAttribute = Attribute<PerEdge, Graph, double, false>;
    using EdgeStringAttribute = Attribute<PerEdge, Graph, std::string, false>;

protected:
    /**
     * Returns the index of node u in the array of incoming edges of node v.
     * (for directed graphs inEdges is searched, while for indirected outEdges
     * is searched, which gives the same result as indexInOutEdgeArray).
     */
    index indexInInEdgeArray(node v, node u) const;

    /**
     * Returns the index of node v in the array of outgoing edges of node u.
     */
    index indexInOutEdgeArray(node u, node v) const;

    // CSR helper methods
    /**
     * Get neighbors of node u using CSR format for outgoing edges
     */
    std::pair<const node *, count> getCSROutNeighbors(node u) const;

    /**
     * Get neighbors of node u using CSR format for incoming edges
     */
    std::pair<const node *, count> getCSRInNeighbors(node u) const;

    /**
     * Check if edge (u,v) exists using CSR format
     */
    bool hasEdgeCSR(node u, node v) const;

    /**
     * Get degree of node using CSR format
     */
    count degreeCSR(node u, bool incoming = false) const;

private:
    /**
     * Computes the weighted in/out degree of node @a u.
     *
     * @param u Node.
     * @param inDegree whether to compute the in degree or the out degree.
     * @param countSelfLoopsTwice If set to true, self-loops will be counted twice.
     *
     * @return Weighted in/out degree of node @a u.
     */
    edgeweight computeWeightedDegree(node u, bool inDegree = false,
                                     bool countSelfLoopsTwice = false) const;

    /**
     * Returns the edge weight of the outgoing edge of index i in the outgoing
     * edges of node u
     * @param u The node
     * @param i The index
     * @return The weight of the outgoing edge or defaultEdgeWeight if the graph
     * is unweighted
     */
    template <bool hasWeights>
    inline edgeweight getOutEdgeWeight(node u, index i) const;

    /**
     * Returns the edge weight of the incoming edge of index i in the incoming
     * edges of node u
     *
     * @param u The node
     * @param i The index in the incoming edge array
     * @return The weight of the incoming edge
     */
    template <bool hasWeights>
    inline edgeweight getInEdgeWeight(node u, index i) const;

    /**
     * Returns the edge id of the edge of index i in the outgoing edges of node
     * u
     *
     * @param u The node
     * @param i The index in the outgoing edges
     * @return The edge id
     */
    template <bool graphHasEdgeIds>
    inline edgeid getOutEdgeId(node u, index i) const;

    /**
     * Returns the edge id of the edge of index i in the incoming edges of node
     * u
     *
     * @param u The node
     * @param i The index in the incoming edges of u
     * @return The edge id
     */
    template <bool graphHasEdgeIds>
    inline edgeid getInEdgeId(node u, index i) const;

    /**
     * @brief Returns if the edge (u, v) shall be used in the iteration of all
     * edgesIndexed
     *
     * @param u The source node of the edge
     * @param v The target node of the edge
     * @return If the node shall be used, i.e. if v is not none and in the
     * undirected case if u >= v
     */
    template <bool graphIsDirected>
    inline bool useEdgeInIteration(node u, node v) const;

    /**
     * @brief Implementation of the for loop for outgoing edges of u
     *
     * Note: If all (valid) outgoing edges shall be considered, graphIsDirected
     * needs to be set to true
     *
     * @param u The node
     * @param handle The handle that shall be executed for each edge
     * @return void
     */
    template <bool graphIsDirected, bool hasWeights, bool graphHasEdgeIds, typename L>
    inline void forOutEdgesOfImpl(node u, L handle) const;

    /**
     * @brief Implementation of the for loop for incoming edges of u
     *
     * For undirected graphs, this is the same as forOutEdgesOfImpl but u and v
     * are changed in the handle
     *
     * @param u The node
     * @param handle The handle that shall be executed for each edge
     * @return void
     */
    template <bool graphIsDirected, bool hasWeights, bool graphHasEdgeIds, typename L>
    inline void forInEdgesOfImpl(node u, L handle) const;

    /**
     * @brief Implementation of the for loop for all edges, @see forEdges
     *
     * @param handle The handle that shall be executed for all edges
     * @return void
     */
    template <bool graphIsDirected, bool hasWeights, bool graphHasEdgeIds, typename L>
    inline void forEdgeImpl(L handle) const;

    /**
     * @brief Parallel implementation of the for loop for all edges, @see
     * parallelForEdges
     *
     * @param handle The handle that shall be executed for all edges
     * @return void
     */
    template <bool graphIsDirected, bool hasWeights, bool graphHasEdgeIds, typename L>
    inline void parallelForEdgesImpl(L handle) const;

    /**
     * @brief Summation variant of the parallel for loop for all edges, @see
     * parallelSumForEdges
     *
     * @param handle The handle that shall be executed for all edges
     * @return void
     */
    template <bool graphIsDirected, bool hasWeights, bool graphHasEdgeIds, typename L>
    inline double parallelSumForEdgesImpl(L handle) const;

    /*
     * In the following definition, Aux::FunctionTraits is used in order to only
     * execute lambda functions with the appropriate parameters. The
     * decltype-return type is used for determining the return type of the
     * lambda (needed for summation) but also determines if the lambda accepts
     * the correct number of parameters. Otherwise the return type declaration
     * fails and the function is excluded from overload resolution. Then there
     * are multiple possible lambdas with three (third parameter id or weight)
     * and two (second parameter can be second node id or edge weight for
     * neighbor iterators). This is checked using Aux::FunctionTraits and
     * std::enable_if. std::enable_if only defines the type member when the
     * given bool is true, this bool comes from std::is_same which compares two
     * types. The function traits give either the parameter type or if it is out
     * of bounds they define type as void.
     */

    /**
     * Triggers a static assert error when no other method is chosen. Because of
     * the use of "..." as arguments, the priority of this method is lower than
     * the priority of the other methods. This method avoids ugly and unreadable
     * template substitution error messages from the other declarations.
     */
    template <class F, void * = (void *)0>
    typename Aux::FunctionTraits<F>::result_type edgeLambda(F &, ...) const {
        // the strange condition is used in order to delay the evaluation of the
        // static assert to the moment when this function is actually used
        static_assert(!std::is_same<F, F>::value,
                      "Your lambda does not support the required parameters or the "
                      "parameters have the wrong type.");
        return std::declval<typename Aux::FunctionTraits<F>::result_type>(); // use the correct
                                                                             // return type (this
                                                                             // won't compile)
    }

    /**
     * Calls the given function f if its fourth argument is of the type edgeid
     * and third of type edgeweight Note that the decltype check is not enough
     * as edgeweight can be casted to node and we want to assure that .
     */
    template <class F,
              typename std::enable_if<
                  (Aux::FunctionTraits<F>::arity >= 3)
                  && std::is_same<edgeweight,
                                  typename Aux::FunctionTraits<F>::template arg<2>::type>::value
                  && std::is_same<edgeid, typename Aux::FunctionTraits<F>::template arg<3>::type>::
                      value>::type * = (void *)0>
    auto edgeLambda(F &f, node u, node v, edgeweight ew, edgeid id) const
        -> decltype(f(u, v, ew, id)) {
        return f(u, v, ew, id);
    }

    /**
     * Calls the given function f if its third argument is of the type edgeid,
     * discards the edge weight Note that the decltype check is not enough as
     * edgeweight can be casted to node.
     */
    template <
        class F,
        typename std::enable_if<
            (Aux::FunctionTraits<F>::arity >= 2)
            && std::is_same<edgeid, typename Aux::FunctionTraits<F>::template arg<2>::type>::value
            && std::is_same<node, typename Aux::FunctionTraits<F>::template arg<1>::type>::
                value /* prevent f(v, weight, eid)
                       */
            >::type * = (void *)0>
    auto edgeLambda(F &f, node u, node v, edgeweight, edgeid id) const -> decltype(f(u, v, id)) {
        return f(u, v, id);
    }

    /**
     * Calls the given function f if its third argument is of type edgeweight,
     * discards the edge id Note that the decltype check is not enough as node
     * can be casted to edgeweight.
     */
    template <class F,
              typename std::enable_if<
                  (Aux::FunctionTraits<F>::arity >= 2)
                  && std::is_same<edgeweight, typename Aux::FunctionTraits<F>::template arg<
                                                  2>::type>::value>::type * = (void *)0>
    auto edgeLambda(F &f, node u, node v, edgeweight ew, edgeid /*id*/) const
        -> decltype(f(u, v, ew)) {
        return f(u, v, ew);
    }

    /**
     * Calls the given function f if it has only two arguments and the second
     * argument is of type node, discards edge weight and id Note that the
     * decltype check is not enough as edgeweight can be casted to node.
     */
    template <class F, typename std::enable_if<
                           (Aux::FunctionTraits<F>::arity >= 1)
                           && std::is_same<node, typename Aux::FunctionTraits<F>::template arg<
                                                     1>::type>::value>::type * = (void *)0>
    auto edgeLambda(F &f, node u, node v, edgeweight /*ew*/, edgeid /*id*/) const
        -> decltype(f(u, v)) {
        return f(u, v);
    }

    /**
     * Calls the given function f if it has only two arguments and the second
     * argument is of type edgeweight, discards the first node and the edge id
     * Note that the decltype check is not enough as edgeweight can be casted to
     * node.
     */
    template <class F,
              typename std::enable_if<
                  (Aux::FunctionTraits<F>::arity >= 1)
                  && std::is_same<edgeweight, typename Aux::FunctionTraits<F>::template arg<
                                                  1>::type>::value>::type * = (void *)0>
    auto edgeLambda(F &f, node, node v, edgeweight ew, edgeid /*id*/) const -> decltype(f(v, ew)) {
        return f(v, ew);
    }

    /**
     * Calls the given function f if it has only one argument, discards the
     * first node id, the edge weight and the edge id
     */
    template <class F, void * = (void *)0>
    auto edgeLambda(F &f, node, node v, edgeweight, edgeid) const -> decltype(f(v)) {
        return f(v);
    }

    /**
     * Calls the given BFS handle with distance parameter
     */
    template <class F>
    auto callBFSHandle(F &f, node u, count dist) const -> decltype(f(u, dist)) {
        return f(u, dist);
    }

    /**
     * Calls the given BFS handle without distance parameter
     */
    template <class F>
    auto callBFSHandle(F &f, node u, count) const -> decltype(f(u)) {
        return f(u);
    }

public:
    // For support of API: NetworKit::Graph::NodeIterator
    using NodeIterator = NodeIteratorBase<Graph>;
    // For support of API: NetworKit::Graph::NodeRange
    using NodeRange = NodeRangeBase<Graph>;

    // For support of API: NetworKit::Graph:EdgeIterator
    using EdgeIterator = EdgeTypeIterator<Graph, Edge>;
    // For support of API: NetworKit::Graph:EdgeWeightIterator
    using EdgeWeightIterator = EdgeTypeIterator<Graph, WeightedEdge>;
    // For support of API: NetworKit::Graph:EdgeRange
    using EdgeRange = EdgeTypeRange<Graph, Edge>;
    // For support of API: NetworKit::Graph:EdgeWeightRange
    using EdgeWeightRange = EdgeTypeRange<Graph, WeightedEdge>;

    // For support of API: NetworKit::Graph::NeighborIterator;
    using NeighborIterator = NeighborIteratorBase<std::vector<node>>;
    // For support of API: NetworKit::Graph::NeighborIterator;
    using NeighborWeightIterator =
        NeighborWeightIteratorBase<std::vector<node>, std::vector<edgeweight>>;

    /**
     * Wrapper class to iterate over a range of the neighbors of a node within
     * a for loop.
     */
    template <bool InEdges = false>
    class NeighborRange {
        const Graph *G;
        node u{none};

    public:
        NeighborRange(const Graph &G, node u) : G(&G), u(u) { assert(G.hasNode(u)); };

        NeighborRange() : G(nullptr){};

        NeighborIterator begin() const {
            assert(G);
            // Base Graph class only supports CSR format
            throw std::runtime_error("NeighborRange iterators not supported in base Graph class - "
                                     "use GraphW for mutable operations");
        }

        NeighborIterator end() const {
            assert(G);
            // Base Graph class only supports CSR format
            throw std::runtime_error("NeighborRange iterators not supported in base Graph class - "
                                     "use GraphW for mutable operations");
        }
    };

    using OutNeighborRange = NeighborRange<false>;

    using InNeighborRange = NeighborRange<true>;
    /**
     * Wrapper class to iterate over a range of the neighbors of a node
     * including the edge weights within a for loop.
     * Values are std::pair<node, edgeweight>.
     */
    template <bool InEdges = false>
    class NeighborWeightRange {

        const Graph *G;
        node u{none};

    public:
        NeighborWeightRange(const Graph &G, node u) : G(&G), u(u) { assert(G.hasNode(u)); };

        NeighborWeightRange() : G(nullptr){};

        NeighborWeightIterator begin() const {
            assert(G);
            // Base Graph class only supports CSR format
            throw std::runtime_error("NeighborWeightRange iterators not supported in base Graph "
                                     "class - use GraphW for mutable operations");
        }

        NeighborWeightIterator end() const {
            assert(G);
            // Base Graph class only supports CSR format
            throw std::runtime_error("NeighborWeightRange iterators not supported in base Graph "
                                     "class - use GraphW for mutable operations");
        }
    };

    using OutNeighborWeightRange = NeighborWeightRange<false>;

    using InNeighborWeightRange = NeighborWeightRange<true>;

    /**
     * Create a graph of @a n nodes. The graph has assignable edge weights if @a
     * weighted is set to <code>true</code>. If @a weighted is set to
     * <code>false</code> each edge has edge weight 1.0 and any other weight
     * assignment will be ignored.
     * @param n Number of nodes.
     * @param weighted If set to <code>true</code>, the graph has edge weights.
     * @param directed If set to @c true, the graph will be directed.
     */
    Graph(count n = 0, bool weighted = false, bool directed = false, bool edgesIndexed = false);

    template <class EdgeMerger = std::plus<edgeweight>>
    Graph(const Graph &G, bool weighted, bool directed, bool edgesIndexed = false,
          [[maybe_unused]] EdgeMerger edgeMerger = std::plus<edgeweight>())
        : n(G.n), m(G.m), storedNumberOfSelfLoops(G.storedNumberOfSelfLoops), z(G.z),
          omega(edgesIndexed ? G.omega : 0), t(G.t), weighted(weighted), directed(directed),
          edgesIndexed(edgesIndexed),        // edges are not indexed by default
          exists(G.exists), usingCSR(false), // Base Graph doesn't use CSR by default

          // copy node attribute map
          nodeAttributeMap(G.nodeAttributeMap, this),
          // fill this later
          edgeAttributeMap(this) {

        // Base Graph class only supports CSR format and is immutable
        throw std::runtime_error("Graph template copy constructor not supported in base Graph "
                                 "class - use GraphW for conversions and mutable operations");
    }

    /**
     * Generate a weighted graph from a list of edges. (Useful for small
     * graphs in unit tests that you do not want to read from a file.)
     *
     * @param[in] edges list of weighted edges
     */
    Graph(std::initializer_list<WeightedEdge> edges);

    /**
     * Create a graph from CSR arrays for memory-efficient storage.
     *
     * @param n Number of nodes.
     * @param directed If set to @c true, the graph will be directed.
     * @param outIndices CSR indices array containing neighbor node IDs for outgoing edges
     * @param outIndptr CSR indptr array containing offsets into outIndices for each node
     * @param inIndices CSR indices array containing neighbor node IDs for incoming edges (directed
     * only)
     * @param inIndptr CSR indptr array containing offsets into inIndices for each node (directed
     * only)
     */
    Graph(count n, bool directed, std::vector<node> outIndices, std::vector<index> outIndptr,
          std::vector<node> inIndices = {}, std::vector<index> inIndptr = {});

    /**
     * Create a graph as copy of @a other.
     * @param other The graph to copy.
     */
    Graph(const Graph &other)
        : n(other.n), m(other.m), storedNumberOfSelfLoops(other.storedNumberOfSelfLoops),
          z(other.z), omega(other.omega), t(other.t), weighted(other.weighted),
          directed(other.directed), edgesIndexed(other.edgesIndexed), deletedID(other.deletedID),
          exists(other.exists), outEdgesCSRIndices(other.outEdgesCSRIndices),
          outEdgesCSRIndptr(other.outEdgesCSRIndptr), inEdgesCSRIndices(other.inEdgesCSRIndices),
          inEdgesCSRIndptr(other.inEdgesCSRIndptr), usingCSR(other.usingCSR),
          // call special constructors to copy attribute maps
          nodeAttributeMap(other.nodeAttributeMap, this),
          edgeAttributeMap(other.edgeAttributeMap, this) {

        // Only support copying CSR-based graphs
        if (!other.usingCSR) {
            throw std::runtime_error("Graph copy constructor only supports CSR-based graphs. Use "
                                     "GraphW for vector-based graphs.");
        }
    }

    /** move constructor */
    Graph(Graph &&other) noexcept
        : n(other.n), m(other.m), storedNumberOfSelfLoops(other.storedNumberOfSelfLoops),
          z(other.z), omega(other.omega), t(other.t), weighted(other.weighted),
          directed(other.directed), edgesIndexed(other.edgesIndexed), deletedID(other.deletedID),
          exists(std::move(other.exists)), outEdgesCSRIndices(std::move(other.outEdgesCSRIndices)),
          outEdgesCSRIndptr(std::move(other.outEdgesCSRIndptr)),
          inEdgesCSRIndices(std::move(other.inEdgesCSRIndices)),
          inEdgesCSRIndptr(std::move(other.inEdgesCSRIndptr)), usingCSR(other.usingCSR),
          nodeAttributeMap(std::move(other.nodeAttributeMap)),
          edgeAttributeMap(std::move(other.edgeAttributeMap)) {
        // attributes: set graph pointer to this new graph
        nodeAttributeMap.theGraph = this;
        edgeAttributeMap.theGraph = this;
    };

    virtual ~Graph() = default;

    /**
     * Constructor that creates a graph from Arrow CSR arrays for zero-copy memory efficiency.
     * @param n Number of nodes.
     * @param directed If set to @c true, the graph will be directed.
     * @param outIndices Arrow array containing neighbor node IDs for outgoing edges (CSR indices).
     * @param outIndptr Arrow array containing offsets into outIndices for each node (CSR indptr).
     * @param inIndices Arrow array containing neighbor node IDs for incoming edges (only for
     * directed graphs).
     * @param inIndptr Arrow array containing offsets into inIndices for each node (only for
     * directed graphs).
     */
    Graph(count n, bool directed, std::shared_ptr<arrow::UInt64Array> outIndices,
          std::shared_ptr<arrow::UInt64Array> outIndptr,
          std::shared_ptr<arrow::UInt64Array> inIndices = nullptr,
          std::shared_ptr<arrow::UInt64Array> inIndptr = nullptr);

    /** move assignment operator */
    Graph &operator=(Graph &&other) noexcept {
        std::swap(n, other.n);
        std::swap(m, other.m);
        std::swap(storedNumberOfSelfLoops, other.storedNumberOfSelfLoops);
        std::swap(z, other.z);
        std::swap(omega, other.omega);
        std::swap(t, other.t);
        std::swap(weighted, other.weighted);
        std::swap(directed, other.directed);
        std::swap(edgesIndexed, other.edgesIndexed);
        std::swap(exists, other.exists);
        std::swap(usingCSR, other.usingCSR);
        std::swap(outEdgesCSRIndices, other.outEdgesCSRIndices);
        std::swap(outEdgesCSRIndptr, other.outEdgesCSRIndptr);
        std::swap(inEdgesCSRIndices, other.inEdgesCSRIndices);
        std::swap(inEdgesCSRIndptr, other.inEdgesCSRIndptr);
        std::swap(deletedID, other.deletedID);

        // attributes: set graph pointer to this new graph
        std::swap(nodeAttributeMap, other.nodeAttributeMap);
        std::swap(edgeAttributeMap, other.edgeAttributeMap);
        nodeAttributeMap.theGraph = this;
        edgeAttributeMap.theGraph = this;

        return *this;
    };

    /** copy assignment operator */
    Graph &operator=(const Graph &other) {
        n = other.n;
        m = other.m;
        storedNumberOfSelfLoops = other.storedNumberOfSelfLoops;
        z = other.z;
        omega = other.omega;
        t = other.t;
        weighted = other.weighted;
        directed = other.directed;
        edgesIndexed = other.edgesIndexed;
        exists = other.exists;
        deletedID = other.deletedID;

        // call special constructors to copy attribute maps
        nodeAttributeMap = AttributeMap(other.nodeAttributeMap, this);
        edgeAttributeMap = AttributeMap(other.edgeAttributeMap, this);

        return *this;
    };

    /** EDGE IDS **/

    /**
     * Checks if edges have been indexed
     *
     * @return bool if edges have been indexed
     */
    bool hasEdgeIds() const noexcept { return edgesIndexed; }

    /**
     * Get the id of the given edge.
     */
    edgeid edgeId(node u, node v) const;

    /**
     * Get the Edge (u,v) of the given id. (inverse to edgeId)
     * @note Time complexity of this function is O(n).
     */
    std::pair<node, node> edgeById(index id) const {
        std::pair<node, node> result{none, none};
        bool found = false;

        forNodesWhile([&] { return !found; },
                      [&](node u) {
                          forNeighborsOf(u, [&](node v) {
                              if (!this->isDirected() && v < u)
                                  return;
                              auto uvId = edgeId(u, v);
                              if (uvId == id) {
                                  found = true;
                                  result = std::make_pair(u, v);
                              }
                          });
                      });

        return result;
    }

    /**
     * Get an upper bound for the edge ids in the graph.
     * @return An upper bound for the edge ids.
     */
    index upperEdgeIdBound() const noexcept { return omega; }

    /** GRAPH INFORMATION **/

    /**
     * DEPRECATED: this function will no longer be supported in later releases.
     * Compacts the adjacency arrays by re-using no longer needed slots from
     * deleted edges.
     */

    /**
     * Check if node @a v exists in the graph.
     *
     * @param v Node.
     * @return @c true if @a v exists, @c false otherwise.
     */

    bool hasNode(node v) const noexcept { return (v < z) && this->exists[v]; }

    /**
     * Check if edge (u, v) exists in the graph.
     *
     * @param u First endpoint of edge.
     * @param v Second endpoint of edge.
     * @return @c true if edge exists, @c false otherwise.
     */
    bool hasEdge(node u, node v) const;

    /**
     * Remove adjacent edges satisfying a condition.
     *
     * @param u Node.
     * @param condition A function that takes a node and returns true if the edge should be removed.
     * @param edgesIn Whether to consider incoming edges.
     * @return A pair of (number of removed edges, number of checked edges).
     */
    template <typename Condition>
    std::pair<count, count> removeAdjacentEdges(node u, Condition condition, bool edgesIn = false);

    /** NODE PROPERTIES **/
    /**
     * Returns the number of outgoing neighbors of @a v.
     *
     * @param v Node.
     * @return The number of outgoing neighbors.
     * @note The existence of the node is not checked. Calling this function with a non-existing
     * node results in a segmentation fault. Node existence can be checked by calling hasNode(u).
     */
    virtual count degree(node v) const {
        assert(hasNode(v));
        if (usingCSR) {
            return degreeCSR(v, false);
        }
        // Base Graph only supports CSR, GraphW will override for vector access
        throw std::runtime_error(
            "Base Graph class only supports CSR format. Use GraphW for mutable graphs.");
    }

    /**
     * Get the number of incoming neighbors of @a v.
     *
     * @param v Node.
     * @return The number of incoming neighbors.
     * @note If the graph is not directed, the outgoing degree is returned.
     * @note The existence of the node is not checked. Calling this function with a non-existing
     * node results in a segmentation fault. Node existence can be checked by calling hasNode(u).
     */
    count degreeIn(node v) const {
        assert(hasNode(v));
        if (usingCSR) {
            return directed ? degreeCSR(v, true) : degreeCSR(v, false);
        }
        // Base Graph class only supports CSR format
        throw std::runtime_error("Graph class requires CSR arrays for degree operations");
    }

    /**
     * Get the number of outgoing neighbors of @a v.
     *
     * @param v Node.
     * @return The number of outgoing neighbors.
     * @note The existence of the node is not checked. Calling this function with a non-existing
     * node results in a segmentation fault. Node existence can be checked by calling hasNode(u).
     */
    count degreeOut(node v) const { return degree(v); }

    /**
     * Check whether @a v is isolated, i.e. degree is 0.
     * @param v Node.
     * @return @c true if the node is isolated (= degree is 0)
     */
    bool isIsolated(node v) const {
        if (!exists[v])
            throw std::runtime_error("Error, the node does not exist!");
        if (usingCSR) {
            return degreeCSR(v, false) == 0 && (!directed || degreeCSR(v, true) == 0);
        }
        // Base Graph class only supports CSR format
        throw std::runtime_error("Graph class requires CSR arrays for isolation check");
    }

    /**
     * Returns the weighted degree of @a u.
     *
     * @param u Node.
     * @param countSelfLoopsTwice If set to true, self-loops will be counted twice.
     *
     * @return Weighted degree of @a u.
     */
    edgeweight weightedDegree(node u, bool countSelfLoopsTwice = false) const;

    /**
     * Returns the weighted in-degree of @a u.
     *
     * @param u Node.
     * @param countSelfLoopsTwice If set to true, self-loops will be counted twice.
     *
     * @return Weighted in-degree of @a v.
     */
    edgeweight weightedDegreeIn(node u, bool countSelfLoopsTwice = false) const;

    /**
     * Returns <code>true</code> if this graph supports edge weights other
     * than 1.0.
     * @return <code>true</code> if this graph supports edge weights other
     * than 1.0.
     */
    bool isWeighted() const noexcept { return weighted; }

    /**
     * Return @c true if this graph supports directed edges.
     * @return @c true if this graph supports directed edges.
     */
    bool isDirected() const noexcept { return directed; }

    /**
     * Return <code>true</code> if graph contains no nodes.
     * @return <code>true</code> if graph contains no nodes.
     */
    bool isEmpty() const noexcept { return !n; }

    /**
     * Return the number of nodes in the graph.
     * @return The number of nodes.
     */
    count numberOfNodes() const noexcept { return n; }

    /**
     * Return the number of edges in the graph.
     * @return The number of edges.
     */
    count numberOfEdges() const noexcept { return m; }

    /**
     * Return the number of loops {v,v} in the graph.
     * @return The number of loops.
     */
    count numberOfSelfLoops() const noexcept { return storedNumberOfSelfLoops; }

    /**
     * Get an upper bound for the node ids in the graph.
     * @return An upper bound for the node ids.
     */
    index upperNodeIdBound() const noexcept { return z; }

    /**
     * Returns true if edges are currently being sorted when removeEdge() is called.
     */
    bool getKeepEdgesSorted() const noexcept { return maintainSortedEdges; }

    /**
     * Returns true if edges are currently being compacted when removeEdge() is called.
     */
    bool getMaintainCompactEdges() const noexcept { return maintainCompactEdges; }

    /**
     * Check for invalid graph states, such as multi-edges.
     * @return False if the graph is in invalid state.
     */
    bool checkConsistency() const;

    /* DYNAMICS */

    /**
     * Trigger a time step - increments counter.
     *
     * This method is deprecated and will not be supported in future releases.
     */
    void timeStep() {
        WARN("Graph::timeStep should not be used and will be deprecated in the future.");
        t++;
    }

    /**
     * Get time step counter.
     * @return Time step counter.
     *
     * This method is deprecated and will not be supported in future releases.
     */
    count time() {
        WARN("Graph::time should not be used and will be deprecated in the future.");
        return t;
    }

    /**
     * Return edge weight of edge {@a u,@a v}. Returns 0 if edge does not
     * exist. BEWARE: Running time is \Theta(deg(u))!
     *
     * @param u Endpoint of edge.
     * @param v Endpoint of edge.
     * @return Edge weight of edge {@a u,@a v} or 0 if edge does not exist.
     */
    edgeweight weight(node u, node v) const;

    /**
     * Set the weight to the i-th neighbour of u.
     *
     * @param[in]	u	endpoint of edge
     * @param[in]	i	index of the nexight
     * @param[in]	ew	edge weight
     */
    void setWeightAtIthNeighbor(Unsafe, node u, index i, edgeweight ew);

    /**
     * Set the weight to the i-th incoming neighbour of u.
     *
     * @param[in]	u	endpoint of edge
     * @param[in]	i	index of the nexight
     * @param[in]	ew	edge weight
     */
    void setWeightAtIthInNeighbor(Unsafe, node u, index i, edgeweight ew);

    /* SUMS */

    /**
     * Returns the sum of all edge weights.
     * @return The sum of all edge weights.
     */
    edgeweight totalEdgeWeight() const noexcept;

    /**
     * Return the i-th (outgoing) neighbor of @a u.
     *
     * @param u Node.
     * @param i index; should be in [0, degreeOut(u))
     * @return @a i-th (outgoing) neighbor of @a u, or @c none if no such
     * neighbor exists.
     */
    node getIthNeighbor(Unsafe, [[maybe_unused]] node u, [[maybe_unused]] index i) const {
        // Base Graph class only supports CSR format
        throw std::runtime_error(
            "getIthNeighbor not supported in base Graph class - use GraphW for mutable operations");
    }

    /**
     * Return the weight to the i-th (outgoing) neighbor of @a u.
     *
     * @param u Node.
     * @param i index; should be in [0, degreeOut(u))
     * @return @a edge weight to the i-th (outgoing) neighbor of @a u, or @c +inf if no such
     * neighbor exists.
     */
    edgeweight getIthNeighborWeight(Unsafe, [[maybe_unused]] node u,
                                    [[maybe_unused]] index i) const {
        // Base Graph class only supports CSR format
        throw std::runtime_error("getIthNeighborWeight not supported in base Graph class - use "
                                 "GraphW for mutable operations");
    }

    /**
     * Get an iterable range over the nodes of the graph.
     *
     * @return Iterator range over the nodes of the graph.
     */
    NodeRange nodeRange() const noexcept { return NodeRange(*this); }

    /**
     * Get an iterable range over the edges of the graph.
     *
     * @return Iterator range over the edges of the graph.
     */
    EdgeRange edgeRange() const noexcept { return EdgeRange(*this); }

    /**
     * Get an iterable range over the edges of the graph and their weights.
     *
     * @return Iterator range over the edges of the graph and their weights.
     */
    EdgeWeightRange edgeWeightRange() const noexcept { return EdgeWeightRange(*this); }

    /**
     * Get an iterable range over the neighbors of @a.
     *
     * @param u Node.
     * @return Iterator range over the neighbors of @a u.
     */
    NeighborRange<false> neighborRange(node u) const {
        assert(exists[u]);
        return NeighborRange<false>(*this, u);
    }

    /**
     * Get an iterable range over the neighbors of @a u including the edge
     * weights.
     *
     * @param u Node.
     * @return Iterator range over pairs of neighbors of @a u and corresponding
     * edge weights.
     */
    NeighborWeightRange<false> weightNeighborRange(node u) const {
        assert(isWeighted());
        assert(exists[u]);
        return NeighborWeightRange<false>(*this, u);
    }

    /**
     * Get an iterable range over the in-neighbors of @a.
     *
     * @param u Node.
     * @return Iterator range over pairs of in-neighbors of @a u.
     */
    NeighborRange<true> inNeighborRange(node u) const {
        assert(isDirected());
        assert(exists[u]);
        return NeighborRange<true>(*this, u);
    }

    /**
     * Get an iterable range over the in-neighbors of @a u including the
     * edge weights.
     *
     * @param u Node.
     * @return Iterator range over pairs of in-neighbors of @a u and corresponding
     * edge weights.
     */
    NeighborWeightRange<true> weightInNeighborRange(node u) const {
        assert(isDirected() && isWeighted());
        assert(exists[u]);
        return NeighborWeightRange<true>(*this, u);
    }

    /**
     * Returns the index of node v in the array of outgoing edges of node u.
     *
     * @param u Node
     * @param v Node
     * @return index of node v in the array of outgoing edges of node u.
     */
    index indexOfNeighbor(node u, node v) const { return indexInOutEdgeArray(u, v); }

    /**
     * Return the i-th (outgoing) neighbor of @a u.
     *
     * @param u Node.
     * @param i index; should be in [0, degreeOut(u))
     * @return @a i-th (outgoing) neighbor of @a u, or @c none if no such
     * neighbor exists.
     */
    node getIthNeighbor([[maybe_unused]] node u, [[maybe_unused]] index i) const {
        // Base Graph class only supports CSR format
        throw std::runtime_error(
            "getIthNeighbor not supported in base Graph class - use GraphW for mutable operations");
    }

    /**
     * Return the i-th (incoming) neighbor of @a u.
     *
     * @param u Node.
     * @param i index; should be in [0, degreeIn(u))
     * @return @a i-th (incoming) neighbor of @a u, or @c none if no such
     * neighbor exists.
     */
    node getIthInNeighbor([[maybe_unused]] node u, [[maybe_unused]] index i) const {
        // Base Graph class only supports CSR format
        throw std::runtime_error("getIthInNeighbor not supported in base Graph class - use GraphW "
                                 "for mutable operations");
    }

    /**
     * Return the weight to the i-th (outgoing) neighbor of @a u.
     *
     * @param u Node.
     * @param i index; should be in [0, degreeOut(u))
     * @return @a edge weight to the i-th (outgoing) neighbor of @a u, or @c +inf if no such
     * neighbor exists.
     */
    edgeweight getIthNeighborWeight([[maybe_unused]] node u, [[maybe_unused]] index i) const {
        // Base Graph class only supports CSR format
        throw std::runtime_error("getIthNeighborWeight not supported in base Graph class - use "
                                 "GraphW for mutable operations");
    }

    /**
     * Get i-th (outgoing) neighbor of @a u and the corresponding edge weight.
     *
     * @param u Node.
     * @param i index; should be in [0, degreeOut(u))
     * @return pair: i-th (outgoing) neighbor of @a u and the corresponding
     * edge weight, or @c defaultEdgeWeight if unweighted.
     */
    std::pair<node, edgeweight> getIthNeighborWithWeight([[maybe_unused]] node u,
                                                         [[maybe_unused]] index i) const {
        // Base Graph class only supports CSR format
        throw std::runtime_error("getIthNeighborWithWeight not supported in base Graph class - use "
                                 "GraphW for mutable operations");
    }

    /**
     * Get i-th (outgoing) neighbor of @a u and the corresponding edge weight.
     *
     * @param u Node.
     * @param i index; should be in [0, degreeOut(u))
     * @return pair: i-th (outgoing) neighbor of @a u and the corresponding
     * edge weight, or @c defaultEdgeWeight if unweighted.
     */
    std::pair<node, edgeweight> getIthNeighborWithWeight(Unsafe, [[maybe_unused]] node u,
                                                         [[maybe_unused]] index i) const {
        // Base Graph class only supports CSR format
        throw std::runtime_error("getIthNeighborWithWeight not supported in base Graph class - use "
                                 "GraphW for mutable operations");
    }

    /**
     * Get i-th (outgoing) neighbor of @a u and the corresponding edge id.
     *
     * @param u Node.
     * @param i index; should be in [0, degreeOut(u))
     * @return pair: i-th (outgoing) neighbor of @a u and the corresponding
     * edge id, or @c none if no such neighbor exists.
     */
    std::pair<node, edgeid> getIthNeighborWithId([[maybe_unused]] node u,
                                                 [[maybe_unused]] index i) const {
        // Base Graph class only supports CSR format
        throw std::runtime_error("getIthNeighborWithId not supported in base Graph class - use "
                                 "GraphW for mutable operations");
    }

    /* NODE ITERATORS */

    /**
     * Iterate over all nodes of the graph and call @a handle (lambda
     * closure).
     *
     * @param handle Takes parameter <code>(node)</code>.
     */
    template <typename L>
    void forNodes(L handle) const;

    /**
     * Iterate randomly over all nodes of the graph and call @a handle (lambda
     * closure).
     *
     * @param handle Takes parameter <code>(node)</code>.
     */
    template <typename L>
    void parallelForNodes(L handle) const;

    /** Iterate over all nodes of the graph and call @a handle (lambda
     * closure) as long as @a condition remains true. This allows for breaking
     * from a node loop.
     *
     * @param condition Returning <code>false</code> breaks the loop.
     * @param handle Takes parameter <code>(node)</code>.
     */
    template <typename C, typename L>
    void forNodesWhile(C condition, L handle) const;

    /**
     * Iterate randomly over all nodes of the graph and call @a handle (lambda
     * closure).
     *
     * @param handle Takes parameter <code>(node)</code>.
     */
    template <typename L>
    void forNodesInRandomOrder(L handle) const;

    /**
     * Iterate in parallel over all nodes of the graph and call handler
     * (lambda closure). Using schedule(guided) to remedy load-imbalances due
     * to e.g. unequal degree distribution.
     *
     * @param handle Takes parameter <code>(node)</code>.
     */
    template <typename L>
    void balancedParallelForNodes(L handle) const;

    /**
     * Iterate over all undirected pairs of nodes and call @a handle (lambda
     * closure).
     *
     * @param handle Takes parameters <code>(node, node)</code>.
     */
    template <typename L>
    void forNodePairs(L handle) const;

    /**
     * Iterate over all undirected pairs of nodes in parallel and call @a
     * handle (lambda closure).
     *
     * @param handle Takes parameters <code>(node, node)</code>.
     */
    template <typename L>
    void parallelForNodePairs(L handle) const;

    /* EDGE ITERATORS */

    /**
     * Iterate over all edges of the const graph and call @a handle (lambda
     * closure).
     *
     * @param handle Takes parameters <code>(node, node)</code>, <code>(node,
     * node, edgweight)</code>, <code>(node, node, edgeid)</code> or
     * <code>(node, node, edgeweight, edgeid)</code>.
     */
    template <typename L>
    void forEdges(L handle) const;

    /**
     * Iterate in parallel over all edges of the const graph and call @a
     * handle (lambda closure).
     *
     * @param handle Takes parameters <code>(node, node)</code> or
     * <code>(node, node, edgweight)</code>, <code>(node, node, edgeid)</code>
     * or <code>(node, node, edgeweight, edgeid)</code>.
     */
    template <typename L>
    void parallelForEdges(L handle) const;

    /* NEIGHBORHOOD ITERATORS */

    /**
     * Iterate over all neighbors of a node and call @a handle (lamdba
     * closure).
     *
     * @param u Node.
     * @param handle Takes parameter <code>(node)</code> or <code>(node,
     * edgeweight)</code> which is a neighbor of @a u.
     * @note For directed graphs only outgoing edges from @a u are considered.
     * A node is its own neighbor if there is a self-loop.
     *
     */
    template <typename L>
    void forNeighborsOf(node u, L handle) const;

    /**
     * Iterate over all incident edges of a node and call @a handle (lamdba
     * closure).
     *
     * @param u Node.
     * @param handle Takes parameters <code>(node, node)</code>, <code>(node,
     * node, edgeweight)</code>, <code>(node, node, edgeid)</code> or
     * <code>(node, node, edgeweight, edgeid)</code> where the first node is
     * @a u and the second is a neighbor of @a u.
     * @note For undirected graphs all edges incident to @a u are also
     * outgoing edges.
     */
    template <typename L>
    void forEdgesOf(node u, L handle) const;

    /**
     * Iterate over all neighbors of a node and call handler (lamdba closure).
     * For directed graphs only incoming edges from u are considered.
     */
    template <typename L>
    void forInNeighborsOf(node u, L handle) const;

    /**
     * Iterate over all incoming edges of a node and call handler (lamdba
     * closure).
     * @note For undirected graphs all edges incident to u are also incoming
     * edges.
     *
     * Handle takes parameters (u, v) or (u, v, w) where w is the edge weight.
     */
    template <typename L>
    void forInEdgesOf(node u, L handle) const;

    /* REDUCTION ITERATORS */

    /**
     * Iterate in parallel over all nodes and sum (reduce +) the values
     * returned by the handler
     */
    template <typename L>
    double parallelSumForNodes(L handle) const;

    /**
     * Iterate in parallel over all edges and sum (reduce +) the values
     * returned by the handler
     */
    template <typename L>
    double parallelSumForEdges(L handle) const;
};

/* NODE ITERATORS */

template <typename L>
void Graph::forNodes(L handle) const {
    for (node v = 0; v < z; ++v) {
        if (exists[v]) {
            handle(v);
        }
    }
}

template <typename L>
void Graph::parallelForNodes(L handle) const {
#pragma omp parallel for
    for (omp_index v = 0; v < static_cast<omp_index>(z); ++v) {
        if (exists[v]) {
            handle(v);
        }
    }
}

template <typename C, typename L>
void Graph::forNodesWhile(C condition, L handle) const {
    for (node v = 0; v < z; ++v) {
        if (exists[v]) {
            if (!condition()) {
                break;
            }
            handle(v);
        }
    }
}

template <typename L>
void Graph::forNodesInRandomOrder(L handle) const {
    std::vector<node> randVec;
    randVec.reserve(numberOfNodes());
    forNodes([&](node u) { randVec.push_back(u); });
    std::ranges::shuffle(randVec, Aux::Random::getURNG());
    for (node v : randVec) {
        handle(v);
    }
}

template <typename L>
void Graph::balancedParallelForNodes(L handle) const {
// TODO: define min block size (and test it!)
#pragma omp parallel for schedule(guided)
    for (omp_index v = 0; v < static_cast<omp_index>(z); ++v) {
        if (exists[v]) {
            handle(v);
        }
    }
}

template <typename L>
void Graph::forNodePairs(L handle) const {
    for (node u = 0; u < z; ++u) {
        if (exists[u]) {
            for (node v = u + 1; v < z; ++v) {
                if (exists[v]) {
                    handle(u, v);
                }
            }
        }
    }
}

template <typename L>
void Graph::parallelForNodePairs(L handle) const {
#pragma omp parallel for schedule(guided)
    for (omp_index u = 0; u < static_cast<omp_index>(z); ++u) {
        if (exists[u]) {
            for (node v = u + 1; v < z; ++v) {
                if (exists[v]) {
                    handle(u, v);
                }
            }
        }
    }
}

/* EDGE ITERATORS */

/* HELPERS */

template <typename T>
void erase(node u, index idx, std::vector<std::vector<T>> &vec);
// implementation for weighted == true
template <bool hasWeights>
inline edgeweight Graph::getOutEdgeWeight([[maybe_unused]] node u, [[maybe_unused]] index i) const {
    // Base Graph class only supports CSR format
    throw std::runtime_error(
        "getOutEdgeWeight not supported in base Graph class - use GraphW for mutable operations");
}

// implementation for weighted == false
template <>
inline edgeweight Graph::getOutEdgeWeight<false>(node, index) const {
    return defaultEdgeWeight;
}

// implementation for weighted == true
template <bool hasWeights>
inline edgeweight Graph::getInEdgeWeight([[maybe_unused]] node u, [[maybe_unused]] index i) const {
    // Base Graph class only supports CSR format
    throw std::runtime_error(
        "getInEdgeWeight not supported in base Graph class - use GraphW for mutable operations");
}

// implementation for weighted == false
template <>
inline edgeweight Graph::getInEdgeWeight<false>(node, index) const {
    return defaultEdgeWeight;
}

// implementation for hasEdgeIds == true
template <bool graphHasEdgeIds>
inline edgeid Graph::getOutEdgeId([[maybe_unused]] node u, [[maybe_unused]] index i) const {
    // Base Graph class only supports CSR format
    throw std::runtime_error(
        "getOutEdgeId not supported in base Graph class - use GraphW for mutable operations");
}

// implementation for hasEdgeIds == false
template <>
inline edgeid Graph::getOutEdgeId<false>(node, index) const {
    return none;
}

// implementation for hasEdgeIds == true
template <bool graphHasEdgeIds>
inline edgeid Graph::getInEdgeId([[maybe_unused]] node u, [[maybe_unused]] index i) const {
    // Base Graph class only supports CSR format
    throw std::runtime_error(
        "getInEdgeId not supported in base Graph class - use GraphW for mutable operations");
}

// implementation for hasEdgeIds == false
template <>
inline edgeid Graph::getInEdgeId<false>(node, index) const {
    return none;
}

// implementation for graphIsDirected == true
template <bool graphIsDirected>
inline bool Graph::useEdgeInIteration(node /* u */, node /* v */) const {
    return true;
}

// implementation for graphIsDirected == false
template <>
inline bool Graph::useEdgeInIteration<false>(node u, node v) const {
    return u >= v;
}

template <bool graphIsDirected, bool hasWeights, bool graphHasEdgeIds, typename L>
inline void Graph::forOutEdgesOfImpl(node u, L handle) const {
    if (!exists[u])
        return;

    if (usingCSR) {
        // CSR-based implementation
        auto [neighbors, degree] = getCSROutNeighbors(u);
        if (neighbors == nullptr || degree == 0)
            return;

        for (count i = 0; i < degree; ++i) {
            node v = neighbors[i];
            if (!exists[v])
                continue;

            // For undirected graphs, only process edge if u >= v to avoid duplicates
            if constexpr (!graphIsDirected) {
                if (!useEdgeInIteration<graphIsDirected>(u, v))
                    continue;
            }

            // Get edge weight (currently CSR graphs are unweighted, so use defaultEdgeWeight)
            edgeweight weight = hasWeights ? defaultEdgeWeight : defaultEdgeWeight;

            // Get edge ID (CSR graphs don't currently support edge IDs)
            edgeid eid = graphHasEdgeIds ? none : none;

            // Call the appropriate lambda based on its signature
            edgeLambda(handle, u, v, weight, eid);
        }
    } else {
        // Vector-based graphs should use GraphW
        throw std::runtime_error("forOutEdgesOfImpl not supported for vector-based graphs in base "
                                 "Graph class - use GraphW");
    }
}

template <bool graphIsDirected, bool hasWeights, bool graphHasEdgeIds, typename L>
inline void Graph::forInEdgesOfImpl(node u, L handle) const {
    if (!exists[u])
        return;

    if (usingCSR) {
        if constexpr (graphIsDirected) {
            // For directed graphs, use incoming edges
            auto [neighbors, degree] = getCSRInNeighbors(u);
            if (neighbors == nullptr || degree == 0)
                return;

            for (count i = 0; i < degree; ++i) {
                node v = neighbors[i];
                if (!exists[v])
                    continue;

                // Get edge weight (currently CSR graphs are unweighted)
                edgeweight weight = hasWeights ? defaultEdgeWeight : defaultEdgeWeight;

                // Get edge ID (CSR graphs don't currently support edge IDs)
                edgeid eid = graphHasEdgeIds ? none : none;

                // For incoming edges, v is the source and u is the target
                edgeLambda(handle, v, u, weight, eid);
            }
        } else {
            // For undirected graphs, incoming edges are the same as outgoing edges
            // but we need to swap u and v in the handle call
            auto [neighbors, degree] = getCSROutNeighbors(u);
            if (neighbors == nullptr || degree == 0)
                return;

            for (count i = 0; i < degree; ++i) {
                node v = neighbors[i];
                if (!exists[v])
                    continue;

                // Get edge weight (currently CSR graphs are unweighted)
                edgeweight weight = hasWeights ? defaultEdgeWeight : defaultEdgeWeight;

                // Get edge ID (CSR graphs don't currently support edge IDs)
                edgeid eid = graphHasEdgeIds ? none : none;

                // For undirected graphs, call with v, u (swapped)
                edgeLambda(handle, v, u, weight, eid);
            }
        }
    } else {
        // Vector-based graphs should use GraphW
        throw std::runtime_error("forInEdgesOfImpl not supported for vector-based graphs in base "
                                 "Graph class - use GraphW");
    }
}

template <bool graphIsDirected, bool hasWeights, bool graphHasEdgeIds, typename L>
inline void Graph::forEdgeImpl(L handle) const {
    for (node u = 0; u < z; ++u) {
        forOutEdgesOfImpl<graphIsDirected, hasWeights, graphHasEdgeIds, L>(u, handle);
    }
}

template <bool graphIsDirected, bool hasWeights, bool graphHasEdgeIds, typename L>
inline void Graph::parallelForEdgesImpl(L handle) const {
#pragma omp parallel for schedule(guided)
    for (omp_index u = 0; u < static_cast<omp_index>(z); ++u) {
        forOutEdgesOfImpl<graphIsDirected, hasWeights, graphHasEdgeIds, L>(u, handle);
    }
}

template <bool graphIsDirected, bool hasWeights, bool graphHasEdgeIds, typename L>
inline double Graph::parallelSumForEdgesImpl(L handle) const {
    double sum = 0.0;

    if (usingCSR) {
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
                if constexpr (!graphIsDirected) {
                    if (!useEdgeInIteration<graphIsDirected>(u, v))
                        continue;
                }

                // Get edge weight (currently CSR graphs are unweighted)
                edgeweight weight = hasWeights ? defaultEdgeWeight : defaultEdgeWeight;

                // Get edge ID (CSR graphs don't currently support edge IDs)
                edgeid eid = graphHasEdgeIds ? none : none;

                // Call the lambda and add result to sum
                sum += edgeLambda(handle, u, v, weight, eid);
            }
        }
    } else {
        // Vector-based graphs should use GraphW
        throw std::runtime_error("parallelSumForEdgesImpl not supported for vector-based graphs in "
                                 "base Graph class - use GraphW");
    }

    return sum;
}

template <typename L>
void Graph::forEdges(L handle) const {
    switch (weighted + 2 * directed + 4 * edgesIndexed) {
    case 0: // unweighted, undirected, no edgeIds
        forEdgeImpl<false, false, false, L>(handle);
        break;

    case 1: // weighted,   undirected, no edgeIds
        forEdgeImpl<false, true, false, L>(handle);
        break;

    case 2: // unweighted, directed, no edgeIds
        forEdgeImpl<true, false, false, L>(handle);
        break;

    case 3: // weighted, directed, no edgeIds
        forEdgeImpl<true, true, false, L>(handle);
        break;

    case 4: // unweighted, undirected, with edgeIds
        forEdgeImpl<false, false, true, L>(handle);
        break;

    case 5: // weighted,   undirected, with edgeIds
        forEdgeImpl<false, true, true, L>(handle);
        break;

    case 6: // unweighted, directed, with edgeIds
        forEdgeImpl<true, false, true, L>(handle);
        break;

    case 7: // weighted,   directed, with edgeIds
        forEdgeImpl<true, true, true, L>(handle);
        break;
    }
}

template <typename L>
void Graph::parallelForEdges(L handle) const {
    switch (weighted + 2 * directed + 4 * edgesIndexed) {
    case 0: // unweighted, undirected, no edgeIds
        parallelForEdgesImpl<false, false, false, L>(handle);
        break;

    case 1: // weighted,   undirected, no edgeIds
        parallelForEdgesImpl<false, true, false, L>(handle);
        break;

    case 2: // unweighted, directed, no edgeIds
        parallelForEdgesImpl<true, false, false, L>(handle);
        break;

    case 3: // weighted, directed, no edgeIds
        parallelForEdgesImpl<true, true, false, L>(handle);
        break;

    case 4: // unweighted, undirected, with edgeIds
        parallelForEdgesImpl<false, false, true, L>(handle);
        break;

    case 5: // weighted,   undirected, with edgeIds
        parallelForEdgesImpl<false, true, true, L>(handle);
        break;

    case 6: // unweighted, directed, with edgeIds
        parallelForEdgesImpl<true, false, true, L>(handle);
        break;

    case 7: // weighted,   directed, with edgeIds
        parallelForEdgesImpl<true, true, true, L>(handle);
        break;
    }
}

/* NEIGHBORHOOD ITERATORS */

template <typename L>
void Graph::forNeighborsOf(node u, L handle) const {
    forEdgesOf(u, handle);
}

template <typename L>
void Graph::forEdgesOf(node u, L handle) const {
    if (directed) {
        switch (weighted + 2 * edgesIndexed) {
        case 0: // not weighted, no edge ids
            forOutEdgesOfImpl<true, false, false, L>(u, handle);
            break;

        case 1: // weighted, no edge ids
            forOutEdgesOfImpl<true, true, false, L>(u, handle);
            break;

        case 2: // not weighted, with edge ids
            forOutEdgesOfImpl<true, false, true, L>(u, handle);
            break;

        case 3: // weighted, with edge ids
            forOutEdgesOfImpl<true, true, true, L>(u, handle);
            break;
        }
    } else {
        switch (weighted + 2 * edgesIndexed) {
        case 0: // not weighted, no edge ids
            forOutEdgesOfImpl<false, false, false, L>(u, handle);
            break;

        case 1: // weighted, no edge ids
            forOutEdgesOfImpl<false, true, false, L>(u, handle);
            break;

        case 2: // not weighted, with edge ids
            forOutEdgesOfImpl<false, false, true, L>(u, handle);
            break;

        case 3: // weighted, with edge ids
            forOutEdgesOfImpl<false, true, true, L>(u, handle);
            break;
        }
    }
}

template <typename L>
void Graph::forInNeighborsOf(node u, L handle) const {
    forInEdgesOf(u, handle);
}

template <typename L>
void Graph::forInEdgesOf(node u, L handle) const {
    switch (weighted + 2 * directed + 4 * edgesIndexed) {
    case 0: // unweighted, undirected, no edge ids
        forInEdgesOfImpl<false, false, false, L>(u, handle);
        break;

    case 1: // weighted, undirected, no edge ids
        forInEdgesOfImpl<false, true, false, L>(u, handle);
        break;

    case 2: // unweighted, directed, no edge ids
        forInEdgesOfImpl<true, false, false, L>(u, handle);
        break;

    case 3: // weighted, directed, no edge ids
        forInEdgesOfImpl<true, true, false, L>(u, handle);
        break;

    case 4: // unweighted, undirected, with edge ids
        forInEdgesOfImpl<false, false, true, L>(u, handle);
        break;

    case 5: // weighted, undirected, with edge ids
        forInEdgesOfImpl<false, true, true, L>(u, handle);
        break;

    case 6: // unweighted, directed, with edge ids
        forInEdgesOfImpl<true, false, true, L>(u, handle);
        break;

    case 7: // weighted, directed, with edge ids
        forInEdgesOfImpl<true, true, true, L>(u, handle);
        break;
    }
}

/* REDUCTION ITERATORS */

template <typename L>
double Graph::parallelSumForNodes(L handle) const {
    double sum = 0.0;

#pragma omp parallel for reduction(+ : sum)
    for (omp_index v = 0; v < static_cast<omp_index>(z); ++v) {
        if (exists[v]) {
            sum += handle(v);
        }
    }

    return sum;
}

template <typename L>
double Graph::parallelSumForEdges(L handle) const {
    double sum = 0.0;

    switch (weighted + 2 * directed + 4 * edgesIndexed) {
    case 0: // unweighted, undirected, no edge ids
        sum = parallelSumForEdgesImpl<false, false, false, L>(handle);
        break;

    case 1: // weighted,   undirected, no edge ids
        sum = parallelSumForEdgesImpl<false, true, false, L>(handle);
        break;

    case 2: // unweighted, directed, no edge ids
        sum = parallelSumForEdgesImpl<true, false, false, L>(handle);
        break;

    case 3: // weighted,   directed, no edge ids
        sum = parallelSumForEdgesImpl<true, true, false, L>(handle);
        break;

    case 4: // unweighted, undirected, with edge ids
        sum = parallelSumForEdgesImpl<false, false, true, L>(handle);
        break;

    case 5: // weighted,   undirected, with edge ids
        sum = parallelSumForEdgesImpl<false, true, true, L>(handle);
        break;

    case 6: // unweighted, directed, with edge ids
        sum = parallelSumForEdgesImpl<true, false, true, L>(handle);
        break;

    case 7: // weighted,   directed, with edge ids
        sum = parallelSumForEdgesImpl<true, true, true, L>(handle);
        break;
    }

    return sum;
}

/* EDGE MODIFIERS */

template <typename Condition>
std::pair<count, count> Graph::removeAdjacentEdges([[maybe_unused]] node u,
                                                   [[maybe_unused]] Condition condition,
                                                   [[maybe_unused]] bool edgesIn) {
    // Base Graph class only supports CSR format and is immutable
    throw std::runtime_error("removeAdjacentEdges not supported in base Graph class - use GraphW "
                             "for mutable operations");
}

} /* namespace NetworKit */

#endif // NETWORKIT_GRAPH_GRAPH_HPP_
