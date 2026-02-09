# Icebug Design Document

## Overview

This branch introduces a major architectural refactor of the NetworKit graph library, focusing on:

1. **Abstract Graph Base Class**: `Graph` is now an abstract base class with pure virtual methods
2. **Polymorphic Graph Types**: Two concrete implementations:
   - `GraphR`: CSR-based (Arrow arrays), read-only, memory-efficient
   - `GraphW`: Vector-based, mutable, supports dynamic modifications
3. **Memory-Efficient Algorithms**: `CoarsenedGraphView` for algorithms that work with hierarchical graph structures
4. **PyArrow Integration**: Enabling zero-copy graph construction from Python data science ecosystems
5. **Parallel Leiden Algorithm**: A memory-efficient implementation of the Leiden community detection algorithm

## Architecture

### 1. Graph Class Hierarchy

```
Graph (abstract base class)
  ├── GraphR (immutable, CSR-based)
  └── GraphW (mutable, vector-based)
```

#### Graph (Abstract Base Class)

The `Graph` class is now an **abstract base class** defining the interface for all graph types:

- **Interface**: Pure virtual methods for all graph operations
- **No Direct Instances**: Cannot instantiate `Graph` directly
- **Polymorphic**: All algorithms work with `const Graph&` and work with both `GraphR` and `GraphW`

**Key Interface Methods:**
- Read operations: `degree()`, `hasNode()`, `hasEdge()`, `weight()`, iteration methods
- Virtual iteration: `forNodes()`, `forEdges()`, `forNeighborsOf()`
- Must be implemented by subclasses

#### GraphR (CSR-Based Implementation - Immutable)

`GraphR` uses **Apache Arrow CSR arrays** for memory-efficient storage:

- **Storage**: Arrow `UInt64Array` for indices and indptr
- **Access Patterns**: Optimized for read-heavy workloads
- **Memory Layout**: CSR format provides better cache locality
- **Interoperability**: Zero-copy construction from Parquet/Arrow formats

**Key Features:**
- Read-only operations (no mutation methods)
- CSR-based iteration: `forNodes()`, `forEdges()`, `forNeighborsOf()`
- Memory-efficient: Contiguous storage, better cache performance
- Arrow integration: Direct construction from Arrow arrays

**Copy Restrictions:**
- Cannot copy `GraphW` to `GraphR` directly
- Only `GraphR` can be copied to `GraphR` (both use CSR)
- To convert `GraphW` to `GraphR`, must iterate edges and call `addEdge()`

#### GraphW (Vector-Based Implementation - Mutable)

`GraphW` extends `Graph` with mutation operations using traditional vector-based storage:

- **Storage**: `std::vector<std::vector<node>>` for adjacency lists
- **Operations**: `addNode()`, `addEdge()`, `removeNode()`, `removeEdge()`, `setWeight()`
- **Use Case**: Graph construction, dynamic modifications, algorithm preprocessing
- **Copy Support**: Can copy `GraphW` to `GraphW` (copies vectors)

**Migration Path:**
```cpp
// Old way (mutating existing graph)
Graph g(n);
g.addEdge(u, v);  // No longer works - Graph is abstract

// New way (use GraphW for construction)
GraphW gw(n);
gw.addEdge(u, v);
// Use gw directly with algorithms (const Graph& accepts GraphW)
```

**Copy Constructor Fix:**
The `GraphW` copy constructor uses a protected `Graph` constructor to bypass the CSR-only check:
```cpp
GraphW(const GraphW &other) : Graph(other, true), ...  // 'true' indicates subclass copy
```

### 2. Iterator Safety

**Important**: The base `Graph::NeighborRange` creates a **new copy** of the neighbor vector in its constructor. This means:

```cpp
// WRONG - Creates two separate vectors, iterators are incompatible
std::vector<node> neighbors(
    G.neighborRange(u).begin(),
    G.neighborRange(u).end()    // Different vector!
);

// CORRECT - Store range first, use same vector
auto range = G.neighborRange(u);
std::vector<node> neighbors(range.begin(), range.end());
```

This caused crashes in `SampledGraphStructuralRandMeasure` when run after other tests due to iterator comparison across different memory locations.

### 3. CSR Data Structure

The CSR (Compressed Sparse Row) format uses Apache Arrow for memory management:

```cpp
// CSR structure in GraphR
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

### 4. CoarsenedGraphView

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

### 5. ParallelLeidenView

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
├── Graph.hpp          # Abstract base class with virtual interface
├── Graph.cpp          # Base class implementations
├── GraphR.hpp         # CSR-based immutable implementation
├── GraphR.cpp         # CSR implementation
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

# Zero-copy from Arrow (creates GraphR)
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

# Use with algorithms directly (const Graph& accepts GraphW)
# No conversion needed!
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
C++ GraphR (reference)
       │
       ▼ (raw pointer)
Algorithm Access (use)
```

### Copy Semantics

| From | To | Method | Notes |
|------|-----|--------|-------|
| GraphW | GraphW | Copy constructor | Copies vectors |
| GraphR | GraphR | Copy constructor | Shares Arrow arrays (immutable) |
| GraphW | GraphR | Not supported | Must iterate and rebuild |
| GraphR | GraphW | Copy constructor | Converts to vectors |

**Note:** `GraphW(other, true)` protected constructor allows subclass copying without CSR check.

## Performance Considerations

### When to Use GraphR vs GraphW

| Use Case | Recommended Class | Reason |
|----------|------------------|---------|
| Algorithm execution | `GraphR` or `GraphW` | Both work via `const Graph&` |
| Graph construction | `GraphW` | Vectors support dynamic modifications |
| Streaming updates | `GraphW` | Mutable operations |
| Large static graphs | `GraphR` | Lower memory, better cache |
| Multilevel algorithms | `CoarsenedGraphView` | No memory overhead |
| Arrow/Pandas interop | `GraphR` | Zero-copy construction |

### Memory Usage Comparison

For a graph with n nodes and m edges:

| Format | Memory | Notes |
|--------|--------|-------|
| Vector-based (GraphW) | ~2m × sizeof(node) + overhead | Good for construction |
| CSR (GraphR) | ~m × sizeof(node) + n × sizeof(offset) | ~40-50% less memory |
| CoarsenedGraphView | O(n) | No edge storage |

## Testing

### Unit Tests
```bash
# C++ tests
./networkit_tests --gtest_filter="*ParallelLeidenView*"
./networkit_tests --gtest_filter="*CoarsenedGraphView*"
./networkit_tests --gtest_filter="*Graph*"

# Python tests
python -m pytest networkit/test/test_parallel_leiden.py -v
python -m pytest networkit/test/test_arrow_pagerank.py -v
```

### Test Coverage
- Graph/GraphW/GraphR conversion and compatibility
- CSR construction from Arrow arrays
- Iterator safety (neighborRange called once)
- ParallelLeidenView correctness vs standard Leiden
- Memory safety with Arrow arrays
- Directed and undirected graph handling
- Copy constructor behavior (GraphW → GraphW, GraphR → GraphR)

## Migration Guide

### For Algorithm Developers

1. **Read-only algorithms**: No changes needed - use `const Graph&`
2. **Graph construction**: Use `GraphW&` for mutation
3. **Copy operations**: Be aware of copy restrictions

```cpp
// Old
void myAlgorithm(Graph& g) {
    g.addEdge(u, v);  // Will no longer compile
}

// New - Option 1: Use GraphW explicitly
void myAlgorithm(GraphW& g) {
    g.addEdge(u, v);  // OK
}

// New - Option 2: Accept any Graph type (polymorphic)
void myAlgorithm(const Graph& g) {
    // Works with GraphR, GraphW, or CoarsenedGraphView
}
```

**Important - Iterator Safety:**
```cpp
// WRONG - May crash or produce garbage
std::vector<node> neighbors(
    G.neighborRange(u).begin(),
    G.neighborRange(u).end()
);

// CORRECT
auto range = G.neighborRange(u);
std::vector<node> neighbors(range.begin(), range.end());
```

### For Python Users

Most existing code will work unchanged:

```python
# Old (still works - uses GraphW)
g = nk.graph.Graph(100)
g.addEdge(0, 1)

# New explicit way (if needed)
from networkit.graph import GraphW
gw = GraphW(100)
gw.addEdge(0, 1)
# Algorithms accept GraphW directly
```

## Known Issues and Fixes

### 1. GraphW Copy Constructor
**Issue:** `GraphW(const GraphW&)` called `Graph(other)` which threw exception for non-CSR graphs.

**Fix:** Added protected `Graph(const Graph&, bool)` constructor that bypasses CSR check for subclass copying.

### 2. NeighborRange Iterator Bug
**Issue:** Calling `G.neighborRange(u).begin()` and `G.neighborRange(u).end()` creates two separate vectors.

**Fix:** Store range in variable before using iterators:
```cpp
auto range = G.neighborRange(u);
std::vector<node> neighbors(range.begin(), range.end());
```

### 3. SampledGraphStructuralRandMeasure Crash
**Issue:** Test crashed with "cannot create std::vector larger than max_size()" when run after other tests.

**Root Cause:** Iterator comparison across different vector allocations.

**Fix:** Applied iterator safety pattern (see above).
