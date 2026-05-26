#!/usr/bin/env python3
"""Tests for Graph.fromIcebugMemGraph constructor."""

import unittest

import pyarrow as pa
import networkit as nk
from icebug_format import IcebugMemGraph


def _make_icebug_graph(edges, n_nodes, *, undirected=False):
    """Build an IcebugMemGraph from a list of (src, dst) tuples."""
    nodes = pa.table({"id": pa.array(list(range(n_nodes)), type=pa.int64())})
    rel = pa.table({
        "source": pa.array([e[0] for e in edges], type=pa.int64()),
        "target": pa.array([e[1] for e in edges], type=pa.int64()),
    })
    return IcebugMemGraph.from_arrow_tables(
        from_node_arrow_table=nodes,
        rel_arrow_table=rel,
        undirected=undirected,
    )


class TestFromIcebugMemGraph(unittest.TestCase):

    def test_basic(self):
        """Basic test of a small directed graph."""
        mem = _make_icebug_graph([(0, 1), (1, 2), (2, 0)], n_nodes=3)
        G = nk.Graph.fromIcebugMemGraph(mem, directed=True)

        self.assertTrue(G.isDirected())
        self.assertEqual(G.numberOfNodes(), 3)
        self.assertEqual(G.numberOfEdges(), 3)
        self.assertTrue(G.hasEdge(0, 1))
        self.assertTrue(G.hasEdge(1, 2))
        self.assertTrue(G.hasEdge(2, 0))
        self.assertFalse(G.hasEdge(1, 0))

    def test_node_count_matches_src_table(self):
        """Number of nodes equals the length of the source node table."""
        mem = _make_icebug_graph([(0, 2), (3, 4)], n_nodes=6)
        G = nk.Graph.fromIcebugMemGraph(mem)
        self.assertEqual(G.numberOfNodes(), 6)

    def test_default_directed_true(self):
        """directed parameter defaults to True."""
        mem = _make_icebug_graph([(0, 1)], n_nodes=2)
        G = nk.Graph.fromIcebugMemGraph(mem)
        self.assertTrue(G.isDirected())

    def test_empty_graph(self):
        """Graph with nodes but no edges."""
        mem = _make_icebug_graph([], n_nodes=5)
        G = nk.Graph.fromIcebugMemGraph(mem)
        self.assertEqual(G.numberOfNodes(), 5)
        self.assertEqual(G.numberOfEdges(), 0)

    def test_plain_icebug_mem_graph_dataclass(self):
        """Passing a hand-built IcebugMemGraph dataclass works."""
        indices = pa.table({"target": pa.array([1, 2, 0], type=pa.uint64())})
        indptr = pa.table({"ptr": pa.array([0, 1, 2, 3], type=pa.uint64())})
        nodes = pa.table({"id": pa.array([0, 1, 2], type=pa.int64())})
        mem = IcebugMemGraph(src=nodes, dest=nodes, indices=indices, indptr=indptr)
        G = nk.Graph.fromIcebugMemGraph(mem)
        self.assertEqual(G.numberOfNodes(), 3)
        self.assertEqual(G.numberOfEdges(), 3)

    def test_neighbors_directed(self):
        """Neighbor lists are correct for a directed graph."""
        mem = _make_icebug_graph([(0, 1), (0, 2), (1, 2)], n_nodes=3)
        G = nk.Graph.fromIcebugMemGraph(mem)

        self.assertEqual(sorted(G.iterNeighbors(0)), [1, 2])
        self.assertEqual(sorted(G.iterNeighbors(1)), [2])
        self.assertEqual(list(G.iterNeighbors(2)), [])

    # ------------------------------------------------------------------
    # Error / validation tests
    # ------------------------------------------------------------------
    def test_missing_src_attribute_raises_type_error(self):
        """An object missing 'src' raises TypeError."""
        class FakeGraph:
            indices = pa.table({"target": pa.array([1], type=pa.uint64())})
            indptr = pa.table({"ptr": pa.array([0, 1], type=pa.uint64())})

        with self.assertRaises(TypeError) as ctx:
            nk.Graph.fromIcebugMemGraph(FakeGraph())
        self.assertIn("src", str(ctx.exception))

    def test_missing_indices_attribute_raises_type_error(self):
        """An object missing 'indices' raises TypeError."""
        nodes = pa.table({"id": pa.array([0, 1], type=pa.int64())})

        class FakeGraph:
            src = nodes
            indptr = pa.table({"ptr": pa.array([0, 0], type=pa.uint64())})

        with self.assertRaises(TypeError) as ctx:
            nk.Graph.fromIcebugMemGraph(FakeGraph())
        self.assertIn("indices", str(ctx.exception))

    def test_missing_indptr_attribute_raises_type_error(self):
        """An object missing 'indptr' raises TypeError."""
        nodes = pa.table({"id": pa.array([0, 1], type=pa.int64())})

        class FakeGraph:
            src = nodes
            indices = pa.table({"target": pa.array([], type=pa.uint64())})

        with self.assertRaises(TypeError) as ctx:
            nk.Graph.fromIcebugMemGraph(FakeGraph())
        self.assertIn("indptr", str(ctx.exception))
    
    def test_src_not_a_table_raises_type_error(self):
        """graph.src that is not a pa.Table raises TypeError."""
        class FakeGraph:
            src = [0, 1]  # wrong type
            indices = pa.table({"target": pa.array([1, 2, 0], type=pa.uint64())})
            indptr = pa.table({"ptr": pa.array([0, 1, 2, 3], type=pa.uint64())})

        with self.assertRaises(TypeError) as ctx:
            nk.Graph.fromIcebugMemGraph(FakeGraph())
        self.assertIn("src", str(ctx.exception))

    def test_indices_not_a_table_raises_type_error(self):
        """graph.indices that is not a pa.Table raises TypeError."""
        nodes = pa.table({"id": pa.array([0, 1], type=pa.int64())})

        class FakeGraph:
            src = nodes
            indices = [1, 2, 0]  # wrong type
            indptr = pa.table({"ptr": pa.array([0, 1, 2, 3], type=pa.uint64())})

        with self.assertRaises(TypeError) as ctx:
            nk.Graph.fromIcebugMemGraph(FakeGraph())
        self.assertIn("indices", str(ctx.exception))

    def test_indptr_not_a_table_raises_type_error(self):
        """graph.indptr that is not a pa.Table raises TypeError."""
        nodes = pa.table({"id": pa.array([0, 1], type=pa.int64())})

        class FakeGraph:
            src = nodes
            indices = pa.table({"target": pa.array([1, 2, 0], type=pa.uint64())})
            indptr = [0, 1, 2, 3]  # wrong type

        with self.assertRaises(TypeError) as ctx:
            nk.Graph.fromIcebugMemGraph(FakeGraph())
        self.assertIn("indptr", str(ctx.exception))

    def test_indices_empty_columns_raises_value_error(self):
        """graph.indices with no columns raises ValueError."""
        nodes = pa.table({"id": pa.array([0, 1, 2], type=pa.int64())})

        class FakeGraph:
            src = nodes
            indices = pa.table({})   # no columns
            indptr = pa.table({"ptr": pa.array([0, 1, 2, 3], type=pa.uint64())})

        with self.assertRaises(ValueError) as ctx:
            nk.Graph.fromIcebugMemGraph(FakeGraph())
        self.assertIn("indices", str(ctx.exception))

    def test_indptr_empty_columns_raises_value_error(self):
        """graph.indptr with no columns raises ValueError."""
        nodes = pa.table({"id": pa.array([0, 1, 2], type=pa.int64())})

        class FakeGraph:
            src = nodes
            indices = pa.table({"target": pa.array([1, 2, 0], type=pa.uint64())})
            indptr = pa.table({})  # no columns

        with self.assertRaises(ValueError) as ctx:
            nk.Graph.fromIcebugMemGraph(FakeGraph())
        self.assertIn("indptr", str(ctx.exception))

    def test_indices_first_column_used_when_no_target(self):
        """When indices has no 'target' column, the first column is used."""
        # Use column name 'neighbor' instead of 'target'
        indices = pa.table({"neighbor": pa.array([1, 2, 0], type=pa.uint64())})
        indptr = pa.table({"ptr": pa.array([0, 1, 2, 3], type=pa.uint64())})
        nodes = pa.table({"id": pa.array([0, 1, 2], type=pa.int64())})
        mem = IcebugMemGraph(src=nodes, dest=nodes, indices=indices, indptr=indptr)
        G = nk.Graph.fromIcebugMemGraph(mem, directed=True)
        self.assertEqual(G.numberOfNodes(), 3)
        self.assertEqual(G.numberOfEdges(), 3)

    def test_indptr_first_column_used_regardless_of_name(self):
        """The first column of indptr is used even when named differently than 'ptr'."""
        indices = pa.table({"target": pa.array([1, 2, 0], type=pa.uint64())})
        indptr = pa.table({"offset": pa.array([0, 1, 2, 3], type=pa.uint64())})
        nodes = pa.table({"id": pa.array([0, 1, 2], type=pa.int64())})
        mem = IcebugMemGraph(src=nodes, dest=nodes, indices=indices, indptr=indptr)
        G = nk.Graph.fromIcebugMemGraph(mem, directed=True)
        self.assertEqual(G.numberOfNodes(), 3)
        self.assertEqual(G.numberOfEdges(), 3)


    # ------------------------------------------------------------------
    # directed=False tests (undirected community-detection use case)
    # ------------------------------------------------------------------

    def test_undirected_from_bidirectional_edges(self):
        """directed=False on a graph built with undirected=True (which stores both (u,v)
        and (v,u), interleaved by source node order) yields an undirected graph with the
        correct edge count (each logical edge counted once)."""
        mem = _make_icebug_graph([(0, 1), (1, 2), (0, 2)], n_nodes=3, undirected=True)
        G = nk.Graph.fromIcebugMemGraph(mem, directed=False)

        self.assertFalse(G.isDirected())
        self.assertEqual(G.numberOfNodes(), 3)
        # 3 logical undirected edges: (0,1), (1,2), (0,2)
        self.assertEqual(G.numberOfEdges(), 3)

    def test_undirected_neighbors(self):
        """Neighbor lists are symmetric when directed=False.  The CSR stores both
        directions interleaved by source, not in forward-then-reverse order."""
        mem = _make_icebug_graph([(0, 1), (0, 2)], n_nodes=3, undirected=True)
        G = nk.Graph.fromIcebugMemGraph(mem, directed=False)

        self.assertFalse(G.isDirected())
        self.assertEqual(sorted(G.iterNeighbors(0)), [1, 2])
        self.assertEqual(sorted(G.iterNeighbors(1)), [0])
        self.assertEqual(sorted(G.iterNeighbors(2)), [0])

    def test_undirected_community_detection(self):
        """PLM community detection runs correctly on an undirected graph built
        from an IcebugMemGraph whose CSR has both directions stored."""
        import networkit.community as nkc
        mem = _make_icebug_graph(
            [(0, 1), (1, 2), (0, 2)], n_nodes=3, undirected=True
        )
        G = nk.Graph.fromIcebugMemGraph(mem, directed=False)

        plm = nkc.PLM(G)
        plm.run()
        partition = plm.getPartition()
        self.assertEqual(partition.numberOfElements(), 3)
        self.assertGreater(partition.numberOfSubsets(), 0)

    def test_default_directed_still_true(self):
        """Default behaviour (directed=True) is unchanged for backward compatibility."""
        mem = _make_icebug_graph([(0, 1), (1, 2)], n_nodes=3)
        G = nk.Graph.fromIcebugMemGraph(mem)
        self.assertTrue(G.isDirected())


if __name__ == "__main__":
    unittest.main()
