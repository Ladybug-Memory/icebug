import unittest

import networkit as nk


class TestInducedSubgraphView(unittest.TestCase):
	def setUp(self):
		# The house graph plus a self-loop on node 0, matching the C++ fixture:
		# edges 3-1, 1-0, 0-2, 2-1, 1-4, 4-3, 3-2, 2-4 in that order, then the loop.
		self.base = nk.graph.Graph(n=5, weighted=True)
		w = 1.0
		for u, v in [(3, 1), (1, 0), (0, 2), (2, 1), (1, 4), (4, 3), (3, 2), (2, 4)]:
			self.base.addEdge(u, v, w, addMissing=True)
			w += 1.0
		self.base.addEdge(0, 0)

	def testViewMatchesSubgraphFromNodes(self):
		subset = [1, 2, 3, 4]
		view = nk.graph.InducedSubgraphView(self.base, subset).asGraph()
		reference = nk.graphtools.subgraphFromNodes(self.base, subset, compact=False)

		self.assertEqual(reference.numberOfNodes(), view.numberOfNodes())
		self.assertEqual(reference.numberOfEdges(), view.numberOfEdges())
		self.assertEqual(reference.numberOfSelfLoops(), view.numberOfSelfLoops())
		self.assertAlmostEqual(reference.totalEdgeWeight(), view.totalEdgeWeight())

		for u in subset:
			self.assertEqual(sorted(reference.iterNeighbors(u)), sorted(view.iterNeighbors(u)))
			self.assertEqual(
				sorted(view.iterNeighborsWeights(u)), sorted(reference.iterNeighborsWeights(u))
			)

	def testReflectsMembershipEdits(self):
		view = nk.graph.InducedSubgraphView(self.base)
		backed = view.asGraph()

		view.addNodes([1, 3])
		self.assertEqual([1, 3], view.getNodeSubset())
		self.assertEqual(1, backed.numberOfEdges()) # only 3-1 spans the subset
		self.assertTrue(backed.hasEdge(1, 3))

		view.addNode(4)
		self.assertEqual(3, backed.numberOfEdges())

		view.removeNodes([1])
		self.assertEqual(1, backed.numberOfEdges())
		self.assertFalse(backed.hasNode(1))

	def testDegreesAndWeights(self):
		view = nk.graph.InducedSubgraphView(self.base, [1, 2, 3, 4]).asGraph()
		self.assertEqual(3, view.degree(1))
		self.assertEqual(6, view.numberOfEdges())
		self.assertAlmostEqual(31.0, view.totalEdgeWeight())
		self.assertFalse(view.hasNode(0))

	def testRunsExistingAlgorithmUnchanged(self):
		subset = [1, 2, 3, 4]
		view = nk.graph.InducedSubgraphView(self.base, subset).asGraph()
		reference = nk.graphtools.subgraphFromNodes(self.base, subset, compact=False)

		kcore = nk.centrality.CoreDecomposition(view)
		kcore.run()
		referenceKcore = nk.centrality.CoreDecomposition(reference)
		referenceKcore.run()

		for u in subset:
			self.assertEqual(referenceKcore.score(u), kcore.score(u))

	def testOverCSRBase(self):
		base = nk.graph.Graph.fromCSR(5, False, [1, 2, 0, 2, 3, 4, 0, 1, 3, 4, 1, 2, 4, 1, 2, 3], [0, 2, 6, 10, 13, 16])
		view = nk.graph.InducedSubgraphView(base, [1, 2, 3]).asGraph()
		self.assertEqual(3, view.numberOfNodes())
		self.assertEqual(3, view.numberOfEdges())

	def testRejectsViewAsBase(self):
		view = nk.graph.InducedSubgraphView(self.base, [1, 2])
		with self.assertRaises(TypeError):
			nk.graph.InducedSubgraphView(view.asGraph())

	def testRejectsUnknownNodes(self):
		with self.assertRaises(RuntimeError):
			nk.graph.InducedSubgraphView(self.base, [99])

	def testRejectsAttributesOnBackedGraph(self):
		backed = nk.graph.InducedSubgraphView(self.base, [1, 2]).asGraph()
		with self.assertRaises(RuntimeError):
			backed.attachNodeAttribute("id", int)
		with self.assertRaises(RuntimeError):
			backed.attachEdgeAttribute("weight", float)

	def testRejectsMutationOfBackedGraph(self):
		backed = nk.graph.InducedSubgraphView(self.base, [1, 2]).asGraph()
		with self.assertRaises(RuntimeError):
			backed.addEdge(1, 2)
		with self.assertRaises(RuntimeError):
			backed.removeNode(1)

	def testRealizeCopiesOut(self):
		view = nk.graph.InducedSubgraphView(self.base, [1, 2, 3, 4])
		materialized = view.realize(compact=False)
		self.assertTrue(materialized.isWeighted())
		self.assertEqual(6, materialized.numberOfEdges())
		# the copy is independent of later membership edits
		view.removeNodes([1])
		self.assertEqual(6, materialized.numberOfEdges())

	def testCompactRealizeHasDenseIds(self):
		view = nk.graph.InducedSubgraphView(self.base, [2, 4])
		materialized = view.realize(compact=True)
		self.assertEqual(2, materialized.numberOfNodes())
		self.assertEqual(2, materialized.upperNodeIdBound())


if __name__ == "__main__":
	unittest.main()
