#ifndef NETWORKIT_GRAPH_GRAPH_W_HPP_
#define NETWORKIT_GRAPH_GRAPH_W_HPP_

#include <set>
#include <networkit/graph/Graph.hpp>

namespace NetworKit {

/**
 * @ingroup graph
 * A writable graph that extends Graph with mutation operations.
 * This class provides all read operations from Graph plus write operations
 * like addNode, addEdge, removeNode, removeEdge, etc.
 *
 * GraphW uses traditional vector-based adjacency lists for mutable operations,
 * while the base Graph class uses memory-efficient Arrow CSR arrays.
 */
class GraphW final : public Graph {

protected:
    // Vector-based adjacency data structures for mutable operations
    //!< only used for directed graphs, inEdges[v] contains all nodes u that
    //!< have an edge (u, v)
    std::vector<std::vector<node>> inEdges;
    //!< (outgoing) edges, for each edge (u, v) v is saved in outEdges[u] and
    //!< for undirected also u in outEdges[v]
    std::vector<std::vector<node>> outEdges;

    //!< only used for directed graphs, same schema as inEdges
    std::vector<std::vector<edgeweight>> inEdgeWeights;
    //!< same schema (and same order!) as outEdges
    std::vector<std::vector<edgeweight>> outEdgeWeights;

    //!< only used for directed graphs, same schema as inEdges
    std::vector<std::vector<edgeid>> inEdgeIds;
    //!< same schema (and same order!) as outEdges
    std::vector<std::vector<edgeid>> outEdgeIds;

private:
    /**
     * Initialize vector-based data structures based on graph properties
     */
    void initializeVectorStructures() {
        count nodeCount = upperNodeIdBound();

        inEdges.resize(isDirected() ? nodeCount : 0);
        outEdges.resize(nodeCount);
        inEdgeWeights.resize(isWeighted() && isDirected() ? nodeCount : 0);
        outEdgeWeights.resize(isWeighted() ? nodeCount : 0);
        inEdgeIds.resize(hasEdgeIds() && isDirected() ? nodeCount : 0);
        outEdgeIds.resize(hasEdgeIds() ? nodeCount : 0);
    }

public:
    /**
     * Create a graph of @a n nodes. The graph has assignable edge weights if @a
     * weighted is set to <code>true</code>. If @a weighted is set to
     * <code>false</code> each edge has edge weight 1.0 and any other weight
     * assignment will be ignored.
     * @param n Number of nodes.
     * @param weighted If set to <code>true</code>, the graph has edge weights.
     * @param directed If set to @c true, the graph will be directed.
     * @param edgesIndexed If set to @c true, the graph will have indexed edges.
     */
    GraphW(count n = 0, bool weighted = false, bool directed = false, bool edgesIndexed = false)
        : Graph(n, weighted, directed, edgesIndexed), inEdges(directed ? n : 0), outEdges(n),
          inEdgeWeights(weighted && directed ? n : 0), outEdgeWeights(weighted ? n : 0),
          inEdgeIds(edgesIndexed && directed ? n : 0), outEdgeIds(edgesIndexed ? n : 0) {}

    /**
     * Generate a weighted graph from a list of edges. (Useful for small
     * graphs in unit tests that you do not want to read from a file.)
     *
     * @param[in] edges list of weighted edges
     */
    GraphW(std::initializer_list<WeightedEdge> edges);

    /**
     * Create a graph as copy of @a other.
     * @param other The graph to copy.
     */
    GraphW(const GraphW &other)
        : Graph(other), inEdges(other.inEdges), outEdges(other.outEdges),
          inEdgeWeights(other.inEdgeWeights), outEdgeWeights(other.outEdgeWeights),
          inEdgeIds(other.inEdgeIds), outEdgeIds(other.outEdgeIds) {}

    /**
     * Create a graph as copy of @a other.
     * @param other The graph to copy.
     */
    GraphW(const Graph &other) : Graph(other) {
        // Initialize vector structures for writable graph
        initializeVectorStructures();
    }

    /**
     * Create a graph as copy of @a other with modified properties.
     * @param other The graph to copy.
     * @param weighted If set to true, the graph has edge weights.
     * @param directed If set to true, the graph will be directed.
     * @param edgesIndexed If set to true, the graph will have indexed edges.
     */
    template <class EdgeMerger = std::plus<edgeweight>>
    GraphW(const Graph &other, bool weighted, bool directed, bool edgesIndexed = false,
           EdgeMerger edgeMerger = std::plus<edgeweight>())
        : GraphW(other.numberOfNodes(), weighted, directed, edgesIndexed) {

        // Copy all edges using the public API
        if (other.isDirected() == directed) {
            // Same directedness - straightforward copy
            other.forEdges([&](node u, node v, edgeweight w, [[maybe_unused]] edgeid id) {
                addEdge(u, v, weighted ? w : defaultEdgeWeight);
            });
        } else if (other.isDirected() && !directed) {
            // Converting directed to undirected - merge edges
            WARN("Edge attributes are not preserved when converting from directed to undirected "
                 "graphs.");

            std::set<std::pair<node, node>> addedEdges;
            other.forEdges([&](node u, node v, edgeweight w, [[maybe_unused]] edgeid id) {
                std::pair<node, node> edge = {std::min(u, v), std::max(u, v)};
                if (addedEdges.find(edge) == addedEdges.end()) {
                    addEdge(edge.first, edge.second, weighted ? w : defaultEdgeWeight);
                    addedEdges.insert(edge);
                } else if (weighted) {
                    // Merge weights for existing edge
                    edgeweight currentWeight = weight(edge.first, edge.second);
                    setWeight(edge.first, edge.second, edgeMerger(currentWeight, w));
                }
            });
        } else {
            // Converting undirected to directed - add both directions
            WARN("Edge attributes are currently not preserved when converting from undirected to "
                 "directed graphs.");

            other.forEdges([&](node u, node v, edgeweight w, [[maybe_unused]] edgeid id) {
                addEdge(u, v, weighted ? w : defaultEdgeWeight);
                if (u != v) {
                    addEdge(v, u, weighted ? w : defaultEdgeWeight);
                }
            });
        }

        if (edgesIndexed && !other.hasEdgeIds()) {
            indexEdges(true);
        }
    }

    /** move constructor */
    GraphW(GraphW &&other) noexcept
        : Graph(std::move(other)), inEdges(std::move(other.inEdges)),
          outEdges(std::move(other.outEdges)), inEdgeWeights(std::move(other.inEdgeWeights)),
          outEdgeWeights(std::move(other.outEdgeWeights)), inEdgeIds(std::move(other.inEdgeIds)),
          outEdgeIds(std::move(other.outEdgeIds)) {}

    /** move constructor */
    GraphW(Graph &&other) noexcept : Graph(std::move(other)) { initializeVectorStructures(); }

    /** Default destructor */
    ~GraphW() = default;

    /** move assignment operator */
    GraphW &operator=(GraphW &&other) noexcept {
        Graph::operator=(std::move(other));
        inEdges = std::move(other.inEdges);
        outEdges = std::move(other.outEdges);
        inEdgeWeights = std::move(other.inEdgeWeights);
        outEdgeWeights = std::move(other.outEdgeWeights);
        inEdgeIds = std::move(other.inEdgeIds);
        outEdgeIds = std::move(other.outEdgeIds);
        return *this;
    }

    /** move assignment operator */
    GraphW &operator=(Graph &&other) noexcept {
        Graph::operator=(std::move(other));
        initializeVectorStructures();
        return *this;
    }

    /** copy assignment operator */
    GraphW &operator=(const GraphW &other) {
        Graph::operator=(other);
        inEdges = other.inEdges;
        outEdges = other.outEdges;
        inEdgeWeights = other.inEdgeWeights;
        outEdgeWeights = other.outEdgeWeights;
        inEdgeIds = other.inEdgeIds;
        outEdgeIds = other.outEdgeIds;
        return *this;
    }

    /** copy assignment operator */
    GraphW &operator=(const Graph &other) {
        Graph::operator=(other);
        initializeVectorStructures();
        return *this;
    }

    /** EDGE IDS **/

    /**
     * Initially assign integer edge identifiers.
     *
     * @param force Force re-indexing of edges even if they have already been
     * indexed
     */
    void indexEdges(bool force = false);

    /** GRAPH INFORMATION **/

    /**
     * Try to save some memory by shrinking internal data structures of the
     * graph. Only run this once you finished editing the graph. Otherwise it
     * will cause unnecessary reallocation of memory.
     */
    void shrinkToFit();

    /**
     * DEPRECATED: this function will no longer be supported in later releases.
     * Compacts the adjacency arrays by re-using no longer needed slots from
     * deleted edges.
     */
    void TLX_DEPRECATED(compactEdges());

    /**
     * Sorts the outgoing neighbors of a given node according to a user-defined comparison function.
     *
     * @param u The node whose outgoing neighbors will be sorted.
     * @param lambda A binary predicate used to compare two neighbors. The predicate should
     *               take two nodes as arguments and return true if the first node should
     *               precede the second in the sorted order.
     */
    template <typename Lambda>
    void sortNeighbors(node u, Lambda lambda) {
        assert(hasNode(u));
        std::sort(outEdges[u].begin(), outEdges[u].end(), lambda);
        if (isDirected()) {
            std::sort(inEdges[u].begin(), inEdges[u].end(), lambda);
        }
    }

    /**
     * Sorts the adjacency arrays by node id. While the running time is linear
     * this temporarily duplicates the memory.
     */
    void sortEdges();

    /**
     * Sorts the adjacency arrays by a custom criterion.
     *
     * @param lambda Lambda function used to sort the edges. It takes two WeightedEdge
     * e1 and e2 as input parameters, returns true if e1 < e2, false otherwise.
     */
    template <class Lambda>
    void sortEdges(Lambda lambda) {
        parallelForNodes([&](node u) {
            if (isWeighted()) {
                std::vector<WeightedEdge> edges;
                forNeighborsOf(u, [&](node v, edgeweight w) { edges.emplace_back(u, v, w); });
                std::sort(edges.begin(), edges.end(), lambda);

                removePartialOutEdges(unsafe, u);
                for (const auto &edge : edges) {
                    addPartialOutEdge(unsafe, edge.u, edge.v, edge.weight);
                }
            } else {
                std::vector<node> neighbors(outEdges[u]);
                std::sort(neighbors.begin(), neighbors.end(), [&](node v1, node v2) {
                    return lambda(WeightedEdge(u, v1, defaultEdgeWeight),
                                  WeightedEdge(u, v2, defaultEdgeWeight));
                });

                removePartialOutEdges(unsafe, u);
                for (node v : neighbors) {
                    addPartialOutEdge(unsafe, u, v);
                }
            }
        });
    }

    /**
     * Set edge count of the graph to edges.
     * @param edges the edge count of a graph
     */
    void setEdgeCount(Unsafe, count edges) { m = edges; }

    /**
     * Set upper bound of edge count.
     *
     * @param newBound New upper edge id bound.
     */
    void setUpperEdgeIdBound(Unsafe, edgeid newBound) { omega = newBound; }

    /**
     * Set the number of self-loops.
     *
     * @param loops New number of self-loops.
     */
    void setNumberOfSelfLoops(Unsafe, count loops) { storedNumberOfSelfLoops = loops; }

    /* NODE MODIFIERS */

    /**
     * Add a new node to the graph and return it.
     * @return The new node.
     */
    node addNode();

    /**
     * Add numberOfNewNodes new nodes.
     * @param  numberOfNewNodes Number of new nodes.
     * @return The index of the last node added.
     */
    node addNodes(count numberOfNewNodes);

    /**
     * Remove a node @a v and all incident edges from the graph.
     *
     * Incoming as well as outgoing edges will be removed.
     *
     * @param v Node.
     */
    void removeNode(node v);

    /**
     * Removes out-going edges from node @u. If the graph is weighted and/or has edge ids, weights
     * and/or edge ids will also be removed.
     *
     * @param u Node.
     */
    void removePartialOutEdges(Unsafe, node u) {
        assert(hasNode(u));
        outEdges[u].clear();
        if (isWeighted()) {
            outEdgeWeights[u].clear();
        }
        if (hasEdgeIds()) {
            outEdgeIds[u].clear();
        }
    }

    /**
     * Removes in-going edges to node @u. If the graph is weighted and/or has edge ids, weights
     * and/or edge ids will also be removed.
     *
     * @param u Node.
     */
    void removePartialInEdges(Unsafe, node u) {
        assert(hasNode(u));
        inEdges[u].clear();
        if (isWeighted()) {
            inEdgeWeights[u].clear();
        }
        if (hasEdgeIds()) {
            inEdgeIds[u].clear();
        }
    }

    /**
     * Restores a previously deleted node @a v with its previous id in the
     * graph.
     *
     * @param v Node.
     *
     */
    void restoreNode(node v);

    /* EDGE MODIFIERS */

    /**
     * Insert an edge between the nodes @a u and @a v. If the graph is
     * weighted you can optionally set a weight for this edge. The default
     * weight is 1.0. Note: Multi-edges are not supported and will NOT be
     * handled consistently by the graph data structure. It is possible to check
     * for multi-edges by enabling parameter "checkForMultiEdges". If already present,
     * the new edge is not inserted. Enabling this check increases the complexity of the function
     * to O(max(deg(u), deg(v))).
     * @param u Endpoint of edge.
     * @param v Endpoint of edge.
     * @param ew Optional edge weight.
     * @param checkMultiEdge If true, this enables a check for a possible multi-edge.
     * @return @c true if edge has been added, false otherwise (in case checkMultiEdge is set to
     * true and the new edge would have been a multi-edge.)
     */
    bool addEdge(node u, node v, edgeweight ew = defaultEdgeWeight, bool checkMultiEdge = false);

    /**
     * Insert an edge between the nodes @a u and @a v. Unlike the addEdge function, this function
     * does not add any information to v. If the graph is weighted you can optionally set a
     * weight for this edge. The default weight is 1.0. Note: Multi-edges are not supported and will
     * NOT be handled consistently by the graph data structure. It is possible to check
     * for multi-edges by enabling parameter "checkForMultiEdges". If already present,
     * the new edge is not inserted. Enabling this check increases the complexity of the function
     * to O(max(deg(u), deg(v))).
     * @param u Endpoint of edge.
     * @param v Endpoint of edge.
     * @param ew Optional edge weight.
     * @param index Optional edge index.
     * @param checkForMultiEdges If true, this enables a check for a possible multi-edge.
     * @return @c true if edge has been added, false otherwise (in case checkMultiEdge is set to
     * true and the new edge would have been a multi-edge.)
     */
    bool addPartialEdge(Unsafe, node u, node v, edgeweight ew = defaultEdgeWeight,
                        uint64_t index = 0, bool checkForMultiEdges = false);

    /**
     * Insert an in edge between the nodes @a u and @a v in a directed graph. If the graph is
     * weighted you can optionally set a weight for this edge. The default
     * weight is 1.0. Note: Multi-edges are not supported and will NOT be
     * handled consistently by the graph data structure. It is possible to check
     * for multi-edges by enabling parameter "checkForMultiEdges". If already present,
     * the new edge is not inserted. Enabling this check increases the complexity of the function
     * to O(max(deg(u), deg(v))).
     * @param u Endpoint of edge.
     * @param v Endpoint of edge.
     * @param ew Optional edge weight.
     * @param index Optional edge index.
     * @param checkForMultiEdges If true, this enables a check for a possible multi-edge.
     * @return @c true if edge has been added, false otherwise (in case checkMultiEdge is set to
     * true and the new edge would have been a multi-edge.)
     */
    bool addPartialInEdge(Unsafe, node u, node v, edgeweight ew = defaultEdgeWeight,
                          uint64_t index = 0, bool checkForMultiEdges = false);

    /**
     * Insert an out edge between the nodes @a u and @a v in a directed graph. If the graph is
     * weighted you can optionally set a weight for this edge. The default
     * weight is 1.0. Note: Multi-edges are not supported and will NOT be
     * handled consistently by the graph data structure. It is possible to check
     * for multi-edges by enabling parameter "checkForMultiEdges". If already present,
     * the new edge is not inserted. Enabling this check increases the complexity of the function
     * to O(max(deg(u), deg(v))).
     * @param u Endpoint of edge.
     * @param v Endpoint of edge.
     * @param ew Optional edge weight.
     * @param index Optional edge index.
     * @param checkForMultiEdges If true, this enables a check for a possible multi-edge.
     * @return @c true if edge has been added, false otherwise (in case checkMultiEdge is set to
     * true and the new edge would have been a multi-edge.)
     */
    bool addPartialOutEdge(Unsafe, node u, node v, edgeweight ew = defaultEdgeWeight,
                           uint64_t index = 0, bool checkForMultiEdges = false);

    /**
     * If set to true, the ingoing and outgoing adjacency vectors will
     * automatically be updated to maintain a sorting (if it existed before) by performing up to n-1
     * swaps. If the user plans to remove multiple edges, better set it to false and call
     * sortEdges() afterwards to avoid redundant swaps. Default = true.
     */
    void setKeepEdgesSorted(bool sorted = true) { maintainSortedEdges = sorted; }

    /**
     * If set to true, the EdgeIDs will automatically be adjusted,
     * so that no gaps in between IDs exist. If the user plans to remove multiple edges, better set
     * it to false and call indexEdges(force=true) afterwards to avoid redundant re-indexing.
     * Default = true.
     */
    void setMaintainCompactEdges(bool compact = true) { maintainCompactEdges = compact; }

    /**
     * Removes the undirected edge {@a u,@a v}.
     * @param u Endpoint of edge.
     * @param v Endpoint of edge.
     */
    void removeEdge(node u, node v);

    /**
     * Removes all the edges in the graph.
     */
    void removeAllEdges();

    /**
     * Removes edges adjacent to a node according to a specific criterion.
     *
     * @param u The node whose adjacent edges shall be removed.
     * @param condition A function that takes a node as an input and returns a
     * bool. If true the edge (u, v) is removed.
     * @param edgesIn Whether in-going or out-going edges shall be removed.
     * @return std::pair<count, count> The number of removed edges (first) and the number of removed
     * self-loops (second).
     */
    template <typename Condition>
    std::pair<count, count> removeAdjacentEdges(node u, Condition condition, bool edgesIn = false) {
        count removedEdges = 0;
        count removedSelfLoops = 0;

        // For directed graphs, this function is supposed to be called twice: one to remove
        // out-edges, and one to remove in-edges.
        auto &edges_ = edgesIn ? inEdges[u] : outEdges[u];
        for (index vi = 0; vi < edges_.size();) {
            if (condition(edges_[vi])) {
                const auto isSelfLoop = (edges_[vi] == u);
                removedSelfLoops += isSelfLoop;
                removedEdges += !isSelfLoop;
                edges_[vi] = edges_.back();
                edges_.pop_back();
                if (isWeighted()) {
                    auto &weights_ = edgesIn ? inEdgeWeights[u] : outEdgeWeights[u];
                    weights_[vi] = weights_.back();
                    weights_.pop_back();
                }
                if (hasEdgeIds()) {
                    auto &edgeIds_ = edgesIn ? inEdgeIds[u] : outEdgeIds[u];
                    edgeIds_[vi] = edgeIds_.back();
                    edgeIds_.pop_back();
                }
            } else {
                ++vi;
            }
        }

        return {removedEdges, removedSelfLoops};
    }

    /**
     * Removes all self-loops in the graph.
     */
    void removeSelfLoops();

    /**
     * Removes all multi-edges in the graph.
     */
    void removeMultiEdges();

    /**
     * Changes the edges {@a s1, @a t1} into {@a s1, @a t2} and the edge {@a
     * s2,
     * @a t2} into {@a s2, @a t1}.
     *
     * If there are edge weights or edge ids, they are preserved. Note that no
     * check is performed if the swap is actually possible, i.e. does not
     * generate duplicate edges.
     *
     * @param s1 The first source
     * @param t1 The first target
     * @param s2 The second source
     * @param t2 The second target
     */
    void swapEdge(node s1, node t1, node s2, node t2);

    /**
     * Set edge weight of edge {@a u,@a v}. BEWARE: Running time is \Theta(deg(u))!
     *
     * @param u Endpoint of edge.
     * @param v Endpoint of edge.
     * @param ew New edge weight.
     */
    void setWeight(node u, node v, edgeweight ew);

    /**
     * Set edge weight of the @a i-th outgoing edge of node @a u. BEWARE: Running time is constant.
     *
     * @param u Endpoint of edge.
     * @param i Index of the outgoing edge.
     * @param ew New edge weight.
     */
    void setWeightAtIthNeighbor(Unsafe, node u, index i, edgeweight ew);

    /**
     * Set edge weight of the @a i-th incoming edge of node @a u. BEWARE: Running time is constant.
     *
     * @param u Endpoint of edge.
     * @param i Index of the incoming edge.
     * @param ew New edge weight.
     */
    void setWeightAtIthInNeighbor(Unsafe, node u, index i, edgeweight ew);

    /**
     * Increase edge weight of edge {@a u,@a v} by @a ew. BEWARE: Running time is \Theta(deg(u))!
     *
     * @param u Endpoint of edge.
     * @param v Endpoint of edge.
     * @param ew Edge weight increase.
     */
    void increaseWeight(node u, node v, edgeweight ew);

    /**
     * Reserves memory in the node's edge containers for undirected graphs.
     *
     * @param u the node memory should be reserved for
     * @param size the amount of memory to reserve
     *
     * This function is thread-safe if called from different
     * threads on different nodes.
     */
    void preallocateUndirected(node u, size_t size);

    /**
     * Reserves memory in the node's edge containers for directed graphs.
     *
     * @param u the node memory should be reserved for
     * @param inSize the amount of memory to reserve for in edges
     * @param outSize the amount of memory to reserve for out edges
     *
     * This function is thread-safe if called from different
     * threads on different nodes.
     */
    void preallocateDirected(node u, size_t outSize, size_t inSize);

    /**
     * Reserves memory in the node's edge containers for directed graphs.
     *
     * @param u the node memory should be reserved for
     * @param outSize the amount of memory to reserve for out edges
     *
     * This function is thread-safe if called from different
     * threads on different nodes.
     */
    void preallocateDirectedOutEdges(node u, size_t outSize);

    /**
     * Reserves memory in the node's edge containers for directed graphs.
     *
     * @param u the node memory should be reserved for
     * @param inSize the amount of memory to reserve for in edges
     *
     * This function is thread-safe if called from different
     * threads on different nodes.
     */
    void preallocateDirectedInEdges(node u, size_t inSize);

    // Override base class methods to provide vector-based implementations

    /**
     * Returns the number of outgoing neighbors of @a v.
     *
     * @param v Node.
     * @return The number of outgoing neighbors.
     */
    count degree(node v) const override {
        assert(hasNode(v));
        return outEdges[v].size();
    }

    /**
     * Return the i-th (outgoing) neighbor of @a u.
     *
     * @param u Node.
     * @param i index; should be in [0, degreeOut(u))
     * @return @a i-th (outgoing) neighbor of @a u, or @c none if no such
     * neighbor exists.
     */
    node getIthNeighbor(Unsafe, node u, index i) const { return outEdges[u][i]; }

    /**
     * Return the weight to the i-th (outgoing) neighbor of @a u.
     *
     * @param u Node.
     * @param i index; should be in [0, degreeOut(u))
     * @return @a edge weight to the i-th (outgoing) neighbor of @a u, or @c +inf if no such
     * neighbor exists.
     */
    edgeweight getIthNeighborWeight(Unsafe, node u, index i) const {
        return isWeighted() ? outEdgeWeights[u][i] : defaultEdgeWeight;
    }

    /**
     * Return the i-th (outgoing) neighbor of @a u.
     *
     * @param u Node.
     * @param i index; should be in [0, degreeOut(u))
     * @return @a i-th (outgoing) neighbor of @a u, or @c none if no such
     * neighbor exists.
     */
    node getIthNeighbor(node u, index i) const {
        if (!hasNode(u) || i >= outEdges[u].size())
            return none;
        return outEdges[u][i];
    }

    /**
     * Return the i-th (incoming) neighbor of @a u.
     *
     * @param u Node.
     * @param i index; should be in [0, degreeIn(u))
     * @return @a i-th (incoming) neighbor of @a u, or @c none if no such
     * neighbor exists.
     */
    node getIthInNeighbor(node u, index i) const {
        if (!hasNode(u) || i >= inEdges[u].size())
            return none;
        return inEdges[u][i];
    }

    /**
     * Return the weight to the i-th (outgoing) neighbor of @a u.
     *
     * @param u Node.
     * @param i index; should be in [0, degreeOut(u))
     * @return @a edge weight to the i-th (outgoing) neighbor of @a u, or @c +inf if no such
     * neighbor exists.
     */
    edgeweight getIthNeighborWeight(node u, index i) const {
        if (!hasNode(u) || i >= outEdges[u].size())
            return nullWeight;
        return isWeighted() ? outEdgeWeights[u][i] : defaultEdgeWeight;
    }

    /**
     * Get i-th (outgoing) neighbor of @a u and the corresponding edge weight.
     *
     * @param u Node.
     * @param i index; should be in [0, degreeOut(u))
     * @return pair: i-th (outgoing) neighbor of @a u and the corresponding
     * edge weight, or @c defaultEdgeWeight if unweighted.
     */
    std::pair<node, edgeweight> getIthNeighborWithWeight(node u, index i) const {
        if (!hasNode(u) || i >= outEdges[u].size())
            return {none, none};
        return getIthNeighborWithWeight(unsafe, u, i);
    }

    /**
     * Get i-th (outgoing) neighbor of @a u and the corresponding edge weight.
     *
     * @param u Node.
     * @param i index; should be in [0, degreeOut(u))
     * @return pair: i-th (outgoing) neighbor of @a u and the corresponding
     * edge weight, or @c defaultEdgeWeight if unweighted.
     */
    std::pair<node, edgeweight> getIthNeighborWithWeight(Unsafe, node u, index i) const {
        if (!isWeighted())
            return {outEdges[u][i], defaultEdgeWeight};
        return {outEdges[u][i], outEdgeWeights[u][i]};
    }

    /**
     * Get i-th (outgoing) neighbor of @a u and the corresponding edge id.
     *
     * @param u Node.
     * @param i index; should be in [0, degreeOut(u))
     * @return pair: i-th (outgoing) neighbor of @a u and the corresponding
     * edge id, or @c none if no such neighbor exists.
     */
    std::pair<node, edgeid> getIthNeighborWithId(node u, index i) const {
        assert(hasEdgeIds());
        if (!hasNode(u) || i >= outEdges[u].size())
            return {none, none};
        return {outEdges[u][i], outEdgeIds[u][i]};
    }

private:
    // Override template method implementations to use vector-based storage

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
     * @brief Summation variant of the parallel for loop for all edges, @see
     * parallelSumForEdges
     *
     * @param handle The handle that shall be executed for all edges
     * @return void
     */
    template <bool graphIsDirected, bool hasWeights, bool graphHasEdgeIds, typename L>
    inline double parallelSumForEdgesImpl(L handle) const;

public:
    /**
     * Wrapper class to iterate over a range of the neighbors of a node within
     * a for loop.
     */
    template <bool InEdges = false>
    class NeighborRange {
        const GraphW *G;
        node u{none};

    public:
        NeighborRange(const GraphW &G, node u) : G(&G), u(u) { assert(G.hasNode(u)); };

        NeighborRange() : G(nullptr){};

        NeighborIterator begin() const {
            assert(G);
            return InEdges ? NeighborIterator(G->inEdges[u].begin())
                           : NeighborIterator(G->outEdges[u].begin());
        }

        NeighborIterator end() const {
            assert(G);
            return InEdges ? NeighborIterator(G->inEdges[u].end())
                           : NeighborIterator(G->outEdges[u].end());
        }

        // Conversion operator to Graph::NeighborRange for Cython compatibility
        operator typename Graph::NeighborRange<InEdges>() const {
            throw std::runtime_error(
                "Conversion from GraphW::NeighborRange to Graph::NeighborRange not supported - "
                "iterator methods not implemented in base Graph class");
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
        const GraphW *G;
        node u{none};

    public:
        NeighborWeightRange(const GraphW &G, node u) : G(&G), u(u) { assert(G.hasNode(u)); };

        NeighborWeightRange() : G(nullptr){};

        NeighborWeightIterator begin() const {
            assert(G);
            return InEdges
                       ? NeighborWeightIterator(G->inEdges[u].begin(), G->inEdgeWeights[u].begin())
                       : NeighborWeightIterator(G->outEdges[u].begin(),
                                                G->outEdgeWeights[u].begin());
        }

        NeighborWeightIterator end() const {
            assert(G);
            return InEdges
                       ? NeighborWeightIterator(G->inEdges[u].end(), G->inEdgeWeights[u].end())
                       : NeighborWeightIterator(G->outEdges[u].end(), G->outEdgeWeights[u].end());
        }

        // Conversion operator to Graph::NeighborWeightRange for Cython compatibility
        operator typename Graph::NeighborWeightRange<InEdges>() const {
            throw std::runtime_error(
                "Conversion from GraphW::NeighborWeightRange to Graph::NeighborWeightRange not "
                "supported - iterator methods not implemented in base Graph class");
        }
    };

    using OutNeighborWeightRange = NeighborWeightRange<false>;
    using InNeighborWeightRange = NeighborWeightRange<true>;

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
};

// Template method implementations for GraphW

// implementation for weighted == true
template <bool hasWeights>
inline edgeweight GraphW::getOutEdgeWeight(node u, index i) const {
    return outEdgeWeights[u][i];
}

// implementation for weighted == false
template <>
inline edgeweight GraphW::getOutEdgeWeight<false>(node, index) const {
    return defaultEdgeWeight;
}

// implementation for weighted == true
template <bool hasWeights>
inline edgeweight GraphW::getInEdgeWeight(node u, index i) const {
    return inEdgeWeights[u][i];
}

// implementation for weighted == false
template <>
inline edgeweight GraphW::getInEdgeWeight<false>(node, index) const {
    return defaultEdgeWeight;
}

// implementation for hasEdgeIds == true
template <bool graphHasEdgeIds>
inline edgeid GraphW::getOutEdgeId(node u, index i) const {
    return outEdgeIds[u][i];
}

// implementation for hasEdgeIds == false
template <>
inline edgeid GraphW::getOutEdgeId<false>(node, index) const {
    return none;
}

// implementation for hasEdgeIds == true
template <bool graphHasEdgeIds>
inline edgeid GraphW::getInEdgeId(node u, index i) const {
    return inEdgeIds[u][i];
}

// implementation for hasEdgeIds == false
template <>
inline edgeid GraphW::getInEdgeId<false>(node, index) const {
    return none;
}

// implementation for graphIsDirected == true
template <bool graphIsDirected>
inline bool GraphW::useEdgeInIteration(node /* u */, node /* v */) const {
    return true;
}

// implementation for graphIsDirected == false
template <>
inline bool GraphW::useEdgeInIteration<false>(node u, node v) const {
    return u >= v;
}

template <bool graphIsDirected, bool hasWeights, bool graphHasEdgeIds, typename L>
inline void GraphW::forOutEdgesOfImpl(node u, L handle) const {
    for (index i = 0; i < outEdges[u].size(); ++i) {
        node v = outEdges[u][i];

        if (useEdgeInIteration<graphIsDirected>(u, v)) {
            Graph::edgeLambda<L>(handle, u, v, getOutEdgeWeight<hasWeights>(u, i),
                                 getOutEdgeId<graphHasEdgeIds>(u, i));
        }
    }
}

template <bool graphIsDirected, bool hasWeights, bool graphHasEdgeIds, typename L>
inline void GraphW::forInEdgesOfImpl(node u, L handle) const {
    if (graphIsDirected) {
        for (index i = 0; i < inEdges[u].size(); i++) {
            node v = inEdges[u][i];

            Graph::edgeLambda<L>(handle, u, v, getInEdgeWeight<hasWeights>(u, i),
                                 getInEdgeId<graphHasEdgeIds>(u, i));
        }
    } else {
        for (index i = 0; i < outEdges[u].size(); ++i) {
            node v = outEdges[u][i];

            Graph::edgeLambda<L>(handle, u, v, getOutEdgeWeight<hasWeights>(u, i),
                                 getOutEdgeId<graphHasEdgeIds>(u, i));
        }
    }
}

template <bool graphIsDirected, bool hasWeights, bool graphHasEdgeIds, typename L>
inline double GraphW::parallelSumForEdgesImpl(L handle) const {
    double sum = 0.0;

#pragma omp parallel for reduction(+ : sum)
    for (omp_index u = 0; u < static_cast<omp_index>(z); ++u) {
        for (index i = 0; i < outEdges[u].size(); ++i) {
            node v = outEdges[u][i];

            // undirected, do not iterate over edges twice
            // {u, v} instead of (u, v); if v == none, u > v is not fulfilled
            if (useEdgeInIteration<graphIsDirected>(u, v)) {
                sum += Graph::edgeLambda<L>(handle, u, v, getOutEdgeWeight<hasWeights>(u, i),
                                            getOutEdgeId<graphHasEdgeIds>(u, i));
            }
        }
    }

    return sum;
}

} /* namespace NetworKit */

#endif // NETWORKIT_GRAPH_GRAPH_W_HPP_
