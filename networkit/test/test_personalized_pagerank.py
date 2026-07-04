#!/usr/bin/env python3
"""
Tests for personalized PageRank (networkit.centrality.PageRank).

Covers:

* Reference values on hand-solvable graphs (directed and undirected triangle
  with a single teleportation source).
* Behaviour with sink distribution and with multiple / weighted source sets.
* The static ``personalizationFrom*`` factories.
* Validation: invalid input vectors / source sets must raise ``RuntimeError``.
* Backward compatibility: passing no personalization must reproduce standard
  (uniform) PageRank bit-for-bit.
"""

import unittest

import networkit as nk


PR_TOL = 1e-4  # tolerance for hand-derived reference scores


def _build_directed_triangle():
    """0 -> 1 -> 2 -> 0, three nodes, three edges."""
    g = nk.Graph(3, weighted=False, directed=True)
    g.addEdge(0, 1)
    g.addEdge(1, 2)
    g.addEdge(2, 0)
    return g


def _build_undirected_triangle():
    """0 - 1 - 2 - 0, three nodes, three edges."""
    g = nk.Graph(3, weighted=False, directed=False)
    g.addEdge(0, 1)
    g.addEdge(1, 2)
    g.addEdge(2, 0)
    return g


class TestPersonalizedPageRank(unittest.TestCase):
    # --- Reference values on hand-solvable graphs -------------------------

    def test_directed_triangle_single_source(self):
        # Steady state: pr[0] = 0.15 / (1 - 0.85^3) ~= 0.3887,
        #              pr[1] = 0.85 * pr[0]   ~= 0.3304,
        #              pr[2] = 0.85^2 * pr[0] ~= 0.2809.
        g = _build_directed_triangle()
        pers = nk.centrality.PageRank.personalizationFromSource(g, 0)
        pr = nk.centrality.PageRank(
            g, damp=0.85, tol=1e-10, distributeSinks=nk.centrality.SinkHandling.NoSinkHandling,
            personalization=pers,
        )
        pr.run()
        scores = pr.scores()
        self.assertAlmostEqual(scores[0], 0.3887, places=4)
        self.assertAlmostEqual(scores[1], 0.3304, places=4)
        self.assertAlmostEqual(scores[2], 0.2809, places=4)
        self.assertAlmostEqual(sum(scores), 1.0, places=6)

    def test_undirected_triangle_single_source(self):
        # Steady state: pr[0] = 0.15 * 23 / 8.55 = 3.45 / 8.55 ~= 0.40351,
        #              pr[1] = pr[2] = 17/23 * pr[0] ~= 0.29825.
        g = _build_undirected_triangle()
        pers = nk.centrality.PageRank.personalizationFromSource(g, 0)
        pr = nk.centrality.PageRank(g, damp=0.85, tol=1e-10, personalization=pers)
        pr.run()
        scores = pr.scores()
        self.assertAlmostEqual(scores[0], 3.45 / 8.55, places=4)
        self.assertAlmostEqual(scores[1], 17.0 / 23.0 * (3.45 / 8.55), places=4)
        self.assertAlmostEqual(scores[2], 17.0 / 23.0 * (3.45 / 8.55), places=4)
        self.assertAlmostEqual(sum(scores), 1.0, places=6)

    # --- Sanity: PPR actually biases towards the source -------------------

    def test_source_is_emphasized(self):
        g = nk.Graph(4, weighted=False, directed=False)
        for u, v in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            g.addEdge(u, v)

        uniform = nk.centrality.PageRank(g, damp=0.85, tol=1e-10)
        uniform.run()

        pers = nk.centrality.PageRank.personalizationFromSource(g, 0)
        ppr = nk.centrality.PageRank(g, damp=0.85, tol=1e-10, personalization=pers)
        ppr.run()

        u_scores = uniform.scores()
        p_scores = ppr.scores()
        # The source node must receive strictly more mass in PPR.
        self.assertGreater(p_scores[0], u_scores[0])
        # Both distributions must still sum to 1.
        self.assertAlmostEqual(sum(u_scores), 1.0, places=6)
        self.assertAlmostEqual(sum(p_scores), 1.0, places=6)

    # --- Multiple / weighted sources --------------------------------------

    def test_multiple_sources_are_emphasized(self):
        g = nk.Graph(4, weighted=False, directed=False)
        for u, v in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]:
            g.addEdge(u, v)

        pers = nk.centrality.PageRank.personalizationFromSources(g, [0, 1])
        self.assertEqual(len(pers), g.upperNodeIdBound())
        self.assertAlmostEqual(pers[0], 0.5, places=12)
        self.assertAlmostEqual(pers[1], 0.5, places=12)
        self.assertAlmostEqual(pers[2], 0.0, places=12)
        self.assertAlmostEqual(pers[3], 0.0, places=12)

        pr = nk.centrality.PageRank(g, damp=0.85, tol=1e-10, personalization=pers)
        pr.run()
        scores = pr.scores()
        # The two source nodes should each have a higher score than the two
        # non-source nodes.
        self.assertGreater(scores[0], scores[2])
        self.assertGreater(scores[1], scores[2])
        self.assertGreater(scores[0], scores[3])
        self.assertGreater(scores[1], scores[3])
        self.assertAlmostEqual(sum(scores), 1.0, places=6)

    def test_weighted_sources_normalization(self):
        g = _build_undirected_triangle()
        pers = nk.centrality.PageRank.personalizationFromWeights(
            g, [(0, 10.0), (1, 1.0), (2, 1.0)]
        )
        self.assertAlmostEqual(pers[0], 10.0 / 12.0, places=12)
        self.assertAlmostEqual(pers[1], 1.0 / 12.0, places=12)
        self.assertAlmostEqual(pers[2], 1.0 / 12.0, places=12)

        pr = nk.centrality.PageRank(g, damp=0.85, tol=1e-10, personalization=pers)
        pr.run()
        scores = pr.scores()
        # The 10x-weighted source should have the largest score.
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[0], scores[2])
        self.assertAlmostEqual(sum(scores), 1.0, places=6)

    # --- Sinks -------------------------------------------------------------

    def test_personalized_pagerank_with_sinks(self):
        g = nk.Graph(4, weighted=False, directed=True)
        g.addEdge(0, 1)
        g.addEdge(1, 2)
        # Node 3 is a sink (no outgoing edges).
        pers = nk.centrality.PageRank.personalizationFromSource(g, 0)
        pr = nk.centrality.PageRank(
            g, damp=0.85, tol=1e-10,
            distributeSinks=nk.centrality.SinkHandling.DistributeSinks,
            personalization=pers,
        )
        pr.run()
        scores = pr.scores()
        self.assertAlmostEqual(sum(scores), 1.0, places=6)
        # The source still has the highest score.
        self.assertGreaterEqual(scores[0], max(scores[1], scores[2], scores[3]))

    # --- Backward compatibility -------------------------------------------

    def test_default_constructor_matches_standard_pagerank(self):
        # Passing personalization=None must be bit-for-bit equivalent to the
        # original PageRank construction (uniform teleportation 1/n).
        g = _build_directed_triangle()
        a = nk.centrality.PageRank(g, damp=0.85, tol=1e-10)
        a.run()
        b = nk.centrality.PageRank(
            g, damp=0.85, tol=1e-10,
            distributeSinks=nk.centrality.SinkHandling.NoSinkHandling,
            personalization=None,
        )
        b.run()
        for sa, sb in zip(a.scores(), b.scores()):
            self.assertEqual(sa, sb)

    def test_uniform_personalization_matches_standard_pagerank(self):
        # Passing an explicit uniform personalization vector must also reproduce
        # standard PageRank.
        g = _build_directed_triangle()
        n = g.numberOfNodes()
        uniform = [1.0 / n] * g.upperNodeIdBound()
        a = nk.centrality.PageRank(g, damp=0.85, tol=1e-10)
        a.run()
        b = nk.centrality.PageRank(g, damp=0.85, tol=1e-10, personalization=uniform)
        b.run()
        for sa, sb in zip(a.scores(), b.scores()):
            self.assertAlmostEqual(sa, sb, places=10)

    # --- Input validation -------------------------------------------------

    def test_too_small_personalization_raises(self):
        g = _build_undirected_triangle()
        with self.assertRaises(RuntimeError):
            nk.centrality.PageRank(g, damp=0.85, tol=1e-8, personalization=[1.0])

    def test_negative_entry_raises(self):
        g = _build_undirected_triangle()
        z = g.upperNodeIdBound()
        pers = [-1.0] + [0.0] * (z - 1)
        with self.assertRaises(RuntimeError):
            nk.centrality.PageRank(g, damp=0.85, tol=1e-8, personalization=pers)

    def test_all_zero_personalization_raises(self):
        g = _build_undirected_triangle()
        pers = [0.0] * g.upperNodeIdBound()
        with self.assertRaises(RuntimeError):
            nk.centrality.PageRank(g, damp=0.85, tol=1e-8, personalization=pers)

    def test_factory_nonexistent_source_raises(self):
        g = _build_undirected_triangle()
        with self.assertRaises(RuntimeError):
            nk.centrality.PageRank.personalizationFromSource(g, 42)

    def test_factory_empty_sources_raises(self):
        g = _build_undirected_triangle()
        with self.assertRaises(RuntimeError):
            nk.centrality.PageRank.personalizationFromSources(g, [])

    def test_factory_sources_with_unknown_node_raises(self):
        g = _build_undirected_triangle()
        with self.assertRaises(RuntimeError):
            nk.centrality.PageRank.personalizationFromSources(g, [0, 99])

    def test_factory_negative_weight_raises(self):
        g = _build_undirected_triangle()
        with self.assertRaises(RuntimeError):
            nk.centrality.PageRank.personalizationFromWeights(g, [(0, -1.0)])


if __name__ == "__main__":
    unittest.main()
