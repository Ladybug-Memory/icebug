#!/usr/bin/env python3
import unittest

import numpy as np

import networkit as nk


class TestSCD(unittest.TestCase):
    def setUp(self):
        self.G = nk.readGraph("input/PGPgiantcompo.graph", nk.Format.METIS)

    def testApproximatePageRank(self):
        epsilon = 1e-12
        result = dict(nk.scd.ApproximatePageRank(self.G, 0.1, epsilon).run(0))

        result2 = dict(nk.scd.ApproximatePageRank(self.G, 0.1, epsilon).run([0]))

        self.assertGreaterEqual(len(result), 1)
        self.assertEqual(len(result), len(result2))

        for u, score in result.items():
            self.assertTrue(self.G.hasNode(u))
            self.assertGreaterEqual(score, 0)
            self.assertTrue(u in result2)
            self.assertAlmostEqual(score, result2[u])

    def testSCD(self):
        nk.setSeed(42, False)
        seed = 20
        seeds = [seed]
        for name, algo in [
            ("PageRankNibble", nk.scd.PageRankNibble(self.G, 0.1, 1e-12)),
            ("GCE L", nk.scd.GCE(self.G, "L")),
            ("GCE M", nk.scd.GCE(self.G, "M")),
            ("LFM", nk.scd.LFMLocal(self.G, 0.8)),
            ("TwoPhaseL", nk.scd.TwoPhaseL(self.G)),
            ("TCE", nk.scd.TCE(self.G)),
            ("LTE", nk.scd.LocalTightnessExpansion(self.G)),
            ("LocalT", nk.scd.LocalT(self.G)),
            ("Clique", nk.scd.CliqueDetect(self.G)),
        ]:
            result = algo.run(seeds)[seed]

            self.assertGreaterEqual(
                len(result), 1, "{} has empty community".format(name)
            )

            cond = nk.scd.SetConductance(self.G, result).run().getConductance()

            self.assertLessEqual(cond, 0.5, "{} has too large conductance".format(name))

    def testSCDGroundTruth(self):
        G = nk.Graph(5, directed=True)
        G.addEdge(0, 1)
        G.addEdge(2, 1)
        G.addEdge(1, 3)
        G.addEdge(0, 4)
        G.addEdge(3, 2)

        ground = nk.Cover(5)
        ground.allToSingletons()
        ground.addToSubset(1, 2)
        ground.addToSubset(2, 3)

        gt = nk.scd.SCDGroundTruthComparison(G, ground, {1: [1, 2]}, True)
        gt.run()
        self.assertDictEqual(gt.getIndividualJaccard(), {1: 0.5})
        self.assertDictEqual(gt.getIndividualPrecision(), {1: 0.5})
        self.assertDictEqual(gt.getIndividualRecall(), {1: 1.0})
        self.assertDictEqual(gt.getIndividualF1(), {1: 0.6666666666666666})
        self.assertEqual(gt.getAveragePrecision(), 0.5)
        self.assertEqual(gt.getAverageRecall(), 1.0)
        self.assertAlmostEqual(gt.getAverageF1(), 0.667, 3)
        self.assertEqual(gt.getAverageJaccard(), 0.5)

    def testExpandOneCommunity(self):
        nk.setSeed(42, False)
        ground = nk.Cover(self.G.upperNodeIdBound())
        ground.setUpperBound(1)
        for i in range(10):
            ground.addToSubset(0, i)

        bfs = nk.scd.RandomBFS(self.G, ground)
        community = bfs.expandOneCommunity([10, 20, 30, 40, 50])

        self.assertEqual(len(community), 5)
        self.assertRaises(TypeError, bfs.expandOneCommunity(5))

    def testShellStructQuery(self):
        from bisect import bisect_right
        from itertools import accumulate

        def gt_search(cliques, query):
            bounds = list(accumulate(cliques))
            idxs = [bisect_right(bounds, q) for q in query]

            cmin, cmax = min(idxs), max(idxs)
            b = min(cliques[cmin : cmax + 1])

            left = next((i + 1 for i in range(cmin - 1, -1, -1) if cliques[i] < b), 0)
            right = next(
                (i - 1 for i in range(cmax + 1, len(cliques)) if cliques[i] < b),
                len(cliques) - 1,
            )

            vbeg = bounds[left - 1] if left else 0
            return set(range(vbeg, bounds[right]))

        G = self._cliqueChainGraph()
        shell = nk.scd.ShellStruct(G)
        shell.build()

        for u in range(G.numberOfNodes()):
            et_comm = shell.expandOneCommunity({u})
            gt_comm = gt_search([3, 4, 3, 4], {u})
            self.assertEqual(et_comm, gt_comm)

        for u in range(G.numberOfNodes()):
            for v in range(G.numberOfNodes()):
                et_comm = shell.expandOneCommunity({u, v})
                gt_comm = gt_search([3, 4, 3, 4], {u, v})
                self.assertEqual(et_comm, gt_comm)

    def _cliqueChainGraph(self):
        # Cliques of sizes 3, 4, 3, 4, consecutive ones joined by one edge.
        src = [0, 0, 1, 2, 3, 3, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12]
        tgt = [1, 2, 2, 3, 4, 5, 6, 5, 6, 6, 7, 8, 9, 9, 10, 11, 12, 13, 12, 13, 13]
        G = nk.Graph(14, directed=False)
        for s, t in zip(src, tgt):
            G.addEdge(s, t)
        return G

    def testShellStructBuildCoreness(self):
        G = self._cliqueChainGraph()
        cd = nk.centrality.CoreDecomposition(G)
        cd.run()
        coreness = np.array(cd.scores(), dtype=np.uint64)

        computed = nk.scd.ShellStruct(G)
        computed.build()
        given = nk.scd.ShellStruct(G)
        given.build(coreness)
        for u in range(G.numberOfNodes()):
            for v in range(G.numberOfNodes()):
                self.assertEqual(
                    given.expandOneCommunity({u, v}), computed.expandOneCommunity({u, v})
                )

        self.assertRaises(ValueError, given.build, np.empty(0, dtype=np.uint64))

    def testShellStructScoreCommunity(self):
        G = self._cliqueChainGraph()
        shell = nk.scd.ShellStruct(G)
        shell.build()
        for u in range(G.numberOfNodes()):
            for v in range(G.numberOfNodes()):
                self.assertEqual(
                    set(shell.community({u, v}).tolist()), shell.expandOneCommunity({u, v})
                )

        self.assertEqual(shell.score({0, 1}), 2)
        self.assertEqual(shell.score({3, 4}), 3)
        self.assertEqual(len(shell.community(set())), 0)
        self.assertRaises(ValueError, shell.score, set())

    def testSteinerKCoreQuery(self):
        src = [0, 0, 1, 2, 3, 3, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12]
        tgt = [1, 2, 2, 3, 4, 5, 6, 5, 6, 6, 7, 8, 9, 9, 10, 11, 12, 13, 12, 13, 13]
        G = nk.Graph(14, directed=False)
        for s, t in zip(src, tgt):
            G.addEdge(s, t)
        coreness = np.array([2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 3, 3, 3, 3], dtype=np.uint64)

        shell = nk.scd.ShellStruct(G)
        shell.build()
        queries = [{u, v} for u in range(G.numberOfNodes()) for v in range(G.numberOfNodes())]
        expected = [shell.expandOneCommunity(q) for q in queries]

        for steiner in [nk.scd.SteinerKCore(G), nk.scd.SteinerKCore(G, coreness)]:
            members, coreness = steiner.communities(queries)
            self.assertEqual([set(m.tolist()) for m in members], expected)
            self.assertEqual(coreness.tolist(), [shell.score(q) for q in queries])
            self.assertEqual(steiner.expandOneCommunity({0, 13}), expected[13])


if __name__ == "__main__":
    unittest.main()
