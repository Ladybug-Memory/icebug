# Icebug Design Document

## Overview

This branch introduces a major architectural refactor of the NetworKit graph library, focusing on:

1. **Immutable/Mutable Graph Separation**: Splitting the monolithic `Graph` class into immutable `Graph` and mutable `GraphW`
2. **CSR Data Structure Migration**: Moving from vector-based adjacency lists to Apache Arrow CSR arrays for memory efficiency
3. **Memory-Efficient Algorithms**: Introducing `CoarsenedGraphView` for algorithms that work with hierarchical graph structures
4. **PyArrow Integration**: Enabling zero-copy graph construction from Python data science ecosystems
5. **Parallel Leiden Algorithm**: A memory-efficient implementation of the Leiden community detection algorithm

## Architecture

### 1. Graph Class Hierarchy

```
Graph (immutable, CSR-based)
  └── GraphW (mutable, vector-based)
```

#### Graph (Base Class - Immutable)

The base `Graph` class has been refactored to be **immutable** and uses **Apache Arrow CSR arrays** for memory-efficient storage:

- **Storage**: Arrow `UInt64Array` for indices and indptr
- **Access Patterns**: Optimized for read-heavy workloads
- **Memory Layout**: CSR format provides better cache locality
- **Interoperability**: Zero-copy construction from Parquet/Arrow formats

**Key Features:**
- Read-only operations: `degree()`, `hasNode()`, `hasEdge()`, `weight()`, iteration methods
- CSR-based iteration: `forNodes()`, `forEdges()`, `forNeighborsOf()`
- Memory-efficient: Contiguous storage, better cache performance
- Arrow integration: Direct construction from Arrow arrays

#### GraphW (Writable Subclass - Mutable)

`GraphW` extends `Graph` with mutation operations using traditional vector-based storage:

- **Storage**: `std::vector<std::vector<node>>` for adjacency lists
- **Operations**: `addNode()`, `addEdge()`, `removeNode()`, `removeEdge()`, `setWeight()`
- **Use Case**: Graph construction, dynamic modifications, algorithm preprocessing
- **Backward Compatibility**: Existing code can use `GraphW` where mutation is needed

**Migration Path:**
```cpp
// Old way (mutating existing graph)
Graph g(n);
g.addEdge(u, v);  // No longer possible with base Graph

// New way (use GraphW for construction)
GraphW gw(n);
gw.addEdge(u, v);
Graph g = gw;  // Convert to immutable for algorithms
```

### 2. CSR Data Structure

The CSR (Compressed Sparse Row) format uses Apache Arrow for memory management:

```cpp
// CSR structure
struct {
    std::shared_ptr<arrow::UInt64Array> outEdgesCSRIndices;  // Neighbor IDs
    std::shared_ptr<arrow::UInt64Array> outEdgesCSRIndptr;   // Row pointers
    // ... similar for in-edges in directed graphs
};
```

**Benefits:**
- **Memory Efficiency**: ~50% less memory for large sparse graphs
- **Cache Performance**: Contiguous memory access patterns
- **Zero-Copy**: Direct use of Arrow arrays from Python/Pandas
- **Parallel Access**: Thread-safe read operations

**Python Integration:**
```python
import pyarrow as pa
import networkit as nk

# Create Arrow arrays
indices = pa.array([1, 2, 0, 2, 1], type=pa.uint64())
indptr = pa.array([0, 2, 3, 5], type=pa.uint64())

# Zero-copy graph construction
g = nk.Graph.fromCSR(n_nodes, directed=False,
                     outIndices=indices, outIndptr=indptr)
```

### 3. CoarsenedGraphView

A memory-efficient view for coarsening operations that avoids creating new graph structures:

```
Original Graph (CSR)  ──▶  CoarsenedGraphView
                                  │
                                  │ (computes on-demand)
                                  ▼
                         Supernode adjacency
```

**How It Works:**
1. Maintains only node-to-supernode mapping (partition)
2. Computes edges on-demand by aggregating original graph edges
3. No actual graph construction - pure view/transform
4. Memory usage: O(n) instead of O(n + m) for explicit coarsening

**Use Case:**
- Leiden/Louvain community detection coarsening phases
- Multilevel graph algorithms
- Any algorithm that repeatedly coarsens/refines

### 4. ParallelLeidenView

Memory-efficient implementation of the Leiden algorithm using `CoarsenedGraphView`:

**Key Differences from Standard ParallelLeiden:**
1. Uses `CoarsenedGraphView` instead of explicit coarsened graphs
2. Templates work with both `Graph` and `CoarsenedGraphView`
3. Significantly lower memory footprint during coarsening
4. Template-based interface for graph-agnostic operations

**Algorithm Phases:**
1. **Local Moving**: Move nodes between communities to maximize modularity
2. **Refinement**: Refine communities for better quality
3. **Coarsening**: Create coarsened view (not graph!) for next iteration

## File Structure

### Core Graph Classes
```
include/networkit/graph/
├── Graph.hpp          # Immutable graph with CSR arrays
├── Graph.cpp          # CSR-based implementation
├── GraphW.hpp         # Mutable graph with vector storage
└── GraphW.cpp         # Vector-based implementation
```

### Coarsening Infrastructure
```
include/networkit/coarsening/
├── CoarsenedGraphView.hpp          # Memory-efficient coarsened view
├── CoarsenedGraphView.cpp          # View implementation
├── ParallelPartitionCoarseningView.hpp  # Parallel coarsening
└── ParallelPartitionCoarseningView.cpp  # Parallel implementation
```

### Algorithm Implementation
```
include/networkit/community/
├── ParallelLeidenView.hpp          # Memory-efficient Leiden
└── ParallelLeidenView.cpp          # Implementation
```

### Python Bindings
```
networkit/
├── graph.pxd          # Cython declarations
├── graph.pyx          # Python bindings for Graph/GraphW
├── community.pyx      # ParallelLeidenView bindings
└── test/
    ├── test_parallel_leiden.py     # Leiden tests
    └── test_arrow_pagerank.py      # Arrow integration tests
```

## Python API

### Graph Construction

```python
import networkit as nk
import pyarrow as pa

# Traditional construction (uses GraphW internally)
g = nk.graph.Graph(n=100)

# From edge list
edges = [(0, 1), (1, 2), (2, 0)]
g = nk.graph.GraphFromEdges(edges)

# Zero-copy from Arrow (immutable)
indices = pa.array([1, 2, 0, 2, 1], type=pa.uint64())
indptr = pa.array([0, 2, 3, 5], type=pa.uint64())
g = nk.graph.Graph.fromCSR(3, False, indices, indptr)

# Directed graph
indices = pa.array([1, 2, 0], type=pa.uint64())
indptr = pa.array([0, 2, 3, 3], type=pa.uint64())
g = nk.graph.Graph.fromCSR(3, True, indices, indptr)
```

### Mutable Operations

```python
# GraphW for construction
from networkit.graph import GraphW

gw = GraphW(n=100, weighted=True, directed=False)
gw.addEdge(0, 1, weight=1.0)
gw.addEdge(1, 2, weight=2.0)

# Convert to immutable Graph for algorithms
g = gw  # Implicit conversion
```

### Algorithms

```python
# Parallel Leiden (memory-efficient)
from networkit.community import ParallelLeidenView

pl = ParallelLeidenView(g, iterations=3, randomize=True, gamma=1.0)
pl.run()
partition = pl.getPartition()

# Number of communities
print(f"Found {partition.numberOfSubsets()} communities")

# Get community of node 0
print(f"Node 0 is in community {partition.subsetOf(0)}")
```

## Memory Management

### Arrow Array Lifetime

**Problem:** When using zero-copy Arrow arrays from Python, the underlying memory can be garbage collected while C++ still holds references.

**Solution:** Global registry keyed by graph ID:

```python
# Internal implementation
_arrow_registry = {}

def fromCSR(n, directed, outIndices, outIndptr):
    g = Graph._fromCSR(n, directed, outIndices, outIndptr)
    # Keep arrays alive
    _arrow_registry[id(g)] = {
        'outIndices': outIndices,
        'outIndptr': outIndptr,
    }
    return g
```

### Ownership Model

```
Python Arrow Array (owner)
       │
       ▼ (shared_ptr)
C++ Graph (reference)
       │
       ▼ (raw pointer)
Algorithm Access (use)
```

## Performance Considerations

### When to Use Graph vs GraphW

| Use Case | Recommended Class | Reason |
|----------|------------------|---------|
| Algorithm execution | `Graph` | CSR is faster for read-heavy ops |
| Graph construction | `GraphW` | Vectors support dynamic modifications |
| Streaming updates | `GraphW` | Mutable operations |
| Large static graphs | `Graph` | Lower memory, better cache |
| Multilevel algorithms | `CoarsenedGraphView` | No memory overhead |

### Memory Usage Comparison

For a graph with n nodes and m edges:

| Format | Memory | Notes |
|--------|--------|-------|
| Vector-based (GraphW) | ~2m × sizeof(node) + overhead | Good for small graphs |
| CSR (Graph) | ~m × sizeof(node) + n × sizeof(offset) | ~40-50% less memory |
| CoarsenedGraphView | O(n) | No edge storage |

## Testing

### Unit Tests
```bash
# C++ tests
./networkit_cpp_tests --gtest_filter="*ParallelLeidenView*"
./networkit_cpp_tests --gtest_filter="*CoarsenedGraphView*"

# Python tests
python -m pytest networkit/test/test_parallel_leiden.py -v
python -m pytest networkit/test/test_arrow_pagerank.py -v
```

### Test Coverage
- Graph/GraphW conversion and compatibility
- CSR construction from Arrow arrays
- ParallelLeidenView correctness vs standard Leiden
- Memory safety with Arrow arrays
- Directed and undirected graph handling

## Migration Guide

### For Algorithm Developers

1. **Read-only algorithms**: No changes needed - use `const Graph&`
2. **Graph modifiers**: Change to `GraphW&` or create new `GraphW`

```cpp
// Old
void myAlgorithm(Graph& g) {
    g.addEdge(u, v);  // Will no longer compile
}

// New - Option 1: Use GraphW explicitly
void myAlgorithm(GraphW& g) {
    g.addEdge(u, v);  // OK
}

// New - Option 2: Create GraphW from Graph
void myAlgorithm(const Graph& g) {
    GraphW gw(g);  // Copy and make writable
    gw.addEdge(u, v);
}
```

### For Python Users

Most existing code will work unchanged. For construction:

```python
# Old (still works)
g = nk.graph.Graph(100)
g.addEdge(0, 1)  # Uses GraphW internally

# New explicit way
from networkit.graph import GraphW
gw = GraphW(100)
gw.addEdge(0, 1)
g = gw.toGraph()  # Convert to immutable
```

