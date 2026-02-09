/*
 * GraphR.hpp
 *
 *  Created on: Feb 8, 2026
 *  Read-only CSR-based graph implementation
 */

#ifndef NETWORKIT_GRAPH_GRAPH_R_HPP_
#define NETWORKIT_GRAPH_GRAPH_R_HPP_

#include <networkit/graph/Graph.hpp>

namespace NetworKit {

/**
 * @ingroup graph
 * GraphR - Read-only graph with CSR (Compressed Sparse Row) storage.
 *
 * This class provides a memory-efficient, immutable graph representation using
 * Arrow CSR arrays for zero-copy data sharing. It is optimized for read-only
 * operations and analysis algorithms.
 *
 * For graphs that require mutation (adding/removing nodes or edges), use GraphW instead.
 */
class GraphR : public Graph {
public:
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
    GraphR(count n, bool directed, std::vector<node> outIndices, std::vector<index> outIndptr,
           std::vector<node> inIndices = {}, std::vector<index> inIndptr = {})
        : Graph(n, directed, std::move(outIndices), std::move(outIndptr), std::move(inIndices),
                std::move(inIndptr)) {}

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
    GraphR(count n, bool directed, std::shared_ptr<arrow::UInt64Array> outIndices,
           std::shared_ptr<arrow::UInt64Array> outIndptr,
           std::shared_ptr<arrow::UInt64Array> inIndices = nullptr,
           std::shared_ptr<arrow::UInt64Array> inIndptr = nullptr)
        : Graph(n, directed, std::move(outIndices), std::move(outIndptr), std::move(inIndices),
                std::move(inIndptr)) {}

    /**
     * Copy constructor
     */
    GraphR(const GraphR &other) : Graph(other) {}

    /**
     * Move constructor
     */
    GraphR(GraphR &&other) noexcept : Graph(std::move(other)) {}

    /**
     * Copy assignment
     */
    GraphR &operator=(const GraphR &other) {
        Graph::operator=(other);
        return *this;
    }

    /**
     * Move assignment
     */
    GraphR &operator=(GraphR &&other) noexcept {
        Graph::operator=(std::move(other));
        return *this;
    }

    /** Default destructor */
    ~GraphR() override = default;

    // Implement pure virtual methods from Graph base class

    count degree(node v) const override;
    count degreeIn(node v) const override;
    bool isIsolated(node v) const override;

    /**
     * Return edge weight of edge {@a u,@a v}. Returns 0 if edge does not
     * exist. For CSR graphs, always returns 1.0 if edge exists (unweighted).
     *
     * @param u Endpoint of edge.
     * @param v Endpoint of edge.
     * @return Edge weight of edge {@a u,@a v} or 0 if edge does not exist.
     */
    edgeweight weight(node u, node v) const override;

protected:
    std::vector<node> getNeighborsVector(node u, bool inEdges = false) const override;
    std::pair<std::vector<node>, std::vector<edgeweight>>
    getNeighborsWithWeightsVector(node u, bool inEdges = false) const override;
};

} // namespace NetworKit

#endif // NETWORKIT_GRAPH_GRAPH_R_HPP_
