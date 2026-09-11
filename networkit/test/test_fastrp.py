#!/usr/bin/env python3
import math
import unittest

import networkit as nk


class TestFastRP(unittest.TestCase):

	def setUp(self):
		# Chain graph 0-1-2-3-4
		self.G = nk.Graph(5, weighted=True)
		self.G.addEdge(0, 1)
		self.G.addEdge(1, 2)
		self.G.addEdge(2, 3)
		self.G.addEdge(3, 4)

	def runFastRP(self, G, **kwargs):
		kwargs.setdefault("dim", 8)
		kwargs.setdefault("weights", [1.0, 1.0, 1.0])
		fastrp = nk.embedding.FastRP(G, **kwargs)
		fastrp.run()
		return fastrp.getEmbeddings()

	def testOutputShape(self):
		emb = self.runFastRP(self.G)
		self.assertEqual(len(emb), self.G.numberOfNodes())
		for row in emb:
			self.assertEqual(len(row), 8)
			self.assertTrue(all(math.isfinite(x) for x in row))

	def testL2Normalization(self):
		emb = self.runFastRP(self.G, normalization=True)
		for row in emb:
			norm = math.sqrt(sum(x * x for x in row))
			self.assertAlmostEqual(norm, 1.0, places=6)
		# A non-normalized run must produce norms other than exactly one
		emb = self.runFastRP(self.G, normalization=False, weights=[1.0])
		norms = [math.sqrt(sum(x * x for x in row)) for row in emb]
		self.assertTrue(any(abs(norm - 1.0) > 1e-6 for norm in norms))

	def testDeterministic(self):
		emb1 = self.runFastRP(self.G, seed=42)
		emb2 = self.runFastRP(self.G, seed=42)
		self.assertEqual(emb1, emb2)

	def testNumberOfIterations(self):
		# The number of propagation powers equals len(weights); a single power
		# must produce a valid embedding as well.
		emb = self.runFastRP(self.G, weights=[1.0])
		self.assertEqual(len(emb), 5)
		with self.assertRaises(RuntimeError):
			self.runFastRP(self.G, weights=[])

	def testEdgeWeights(self):
		Gw = nk.Graph(4, weighted=True)
		Gw.addEdge(0, 1, 5.0)
		Gw.addEdge(1, 2, 1.0)
		Gw.addEdge(2, 3, 1.0)
		Gu = nk.Graph(4, weighted=True)
		Gu.addEdge(0, 1, 1.0)
		Gu.addEdge(1, 2, 1.0)
		Gu.addEdge(2, 3, 1.0)
		embWeighted = self.runFastRP(Gw, weights=[1.0, 1.0], inputMatrix="adjacency")
		embUnweighted = self.runFastRP(Gu, weights=[1.0, 1.0], inputMatrix="adjacency")
		self.assertNotEqual(embWeighted, embUnweighted)

	def testProjectionAndInputMatrixOptions(self):
		for projection in ("sparse", "gaussian"):
			for inputMatrix in ("adjacency", "transition"):
				emb = self.runFastRP(self.G, projection=projection, inputMatrix=inputMatrix)
				self.assertEqual(len(emb), 5)
		with self.assertRaises(ValueError):
			self.runFastRP(self.G, projection="bogus")
		with self.assertRaises(ValueError):
			self.runFastRP(self.G, inputMatrix="bogus")

	def testNodeAttributes(self):
		# Graph with three weakly connected components: a chain 0-1-2 (well
		# connected), an edge 3-4 and an isolated node 5 (both poorly connected).
		G = nk.Graph(6, weighted=True)
		G.addEdge(0, 1)
		G.addEdge(1, 2)
		G.addEdge(3, 4)
		features = [[0.5, 0.3], [1.0, 0.5], [0.2, 0.9], [1.0, 1.0], [1.0, 1.0],
				[1.0, 1.0]]

		# Nodes 3, 4, 5 lie in components smaller than the fallback threshold:
		# their embedding must come from the attributes alone, hence nodes with
		# identical attributes get identical embeddings, even if adjacent.
		embFallback = self.runFastRP(G, weights=[1.0, 1.0], nodeFeatures=features,
				featureWeight=0.5, fallbackThreshold=3)
		self.assertEqual(embFallback[3], embFallback[4])
		self.assertEqual(embFallback[3], embFallback[5])

		# With the fallback disabled, the structural embedding of the adjacent
		# nodes 3 and 4 kicks in and changes their attribute-only embedding.
		embStructural = self.runFastRP(G, weights=[1.0, 1.0], nodeFeatures=features,
				featureWeight=0.5, fallbackThreshold=1)
		self.assertNotEqual(embStructural[3], embFallback[3])
		# Well-connected nodes and the isolated node are unaffected by the
		# fallback threshold.
		self.assertEqual(embFallback[0], embStructural[0])
		self.assertEqual(embFallback[5], embStructural[5])

	def testFeatureWeightZeroEqualsNoFeatures(self):
		features = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [8.0, 9.0]]
		embNoFeatures = self.runFastRP(self.G, nodeFeatures=None)
		embZeroWeight = self.runFastRP(self.G, nodeFeatures=features, featureWeight=0.0)
		self.assertEqual(embNoFeatures, embZeroWeight)

	def testFeatureDimensionMismatch(self):
		with self.assertRaises(RuntimeError):
			self.runFastRP(self.G, nodeFeatures=[[1.0], [2.0], [3.0], [4.0], [5.0, 6.0]])
		with self.assertRaises(RuntimeError):
			self.runFastRP(self.G, nodeFeatures=[[1.0], [2.0], [3.0], [4.0]])

	def testUnweightedGraph(self):
		G = nk.Graph(4)
		G.addEdge(0, 1)
		G.addEdge(1, 2)
		G.addEdge(2, 3)
		emb = self.runFastRP(G, weights=[1.0, 1.0])
		self.assertEqual(len(emb), 4)


if __name__ == "__main__":
	unittest.main()
