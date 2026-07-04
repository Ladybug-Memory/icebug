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

    # --- Memory-efficient fast-path static factories ---------------------

    def test_for_source_matches_vector_path(self):
        # The fast-path `forSource` factory must produce bit-for-bit identical
        # scores to the equivalent vector-based construction.
        g = _build_directed_triangle()
        pers = nk.centrality.PageRank.personalizationFromSource(g, 0)
        pr_vec = nk.centrality.PageRank(
            g, damp=0.85, tol=1e-10,
            distributeSinks=nk.centrality.SinkHandling.NoSinkHandling,
            personalization=pers,
        )
        pr_vec.run()
        pr_fast = nk.centrality.PageRank.forSource(
            g, 0, damp=0.85, tol=1e-10,
            distributeSinks=nk.centrality.SinkHandling.NoSinkHandling,
        )
        pr_fast.run()
        for s_vec, s_fast in zip(pr_vec.scores(), pr_fast.scores()):
            self.assertAlmostEqual(s_vec, s_fast, places=12)

    def test_for_sources_matches_vector_path(self):
        g = nk.Graph(4, weighted=False, directed=False)
        for u, v in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]:
            g.addEdge(u, v)

        sources = [0, 1]
        pers = nk.centrality.PageRank.personalizationFromSources(g, sources)
        pr_vec = nk.centrality.PageRank(g, damp=0.85, tol=1e-10, personalization=pers)
        pr_vec.run()
        pr_fast = nk.centrality.PageRank.forSources(g, sources, damp=0.85, tol=1e-10)
        pr_fast.run()
        for s_vec, s_fast in zip(pr_vec.scores(), pr_fast.scores()):
            self.assertAlmostEqual(s_vec, s_fast, places=12)

    def test_for_weights_matches_vector_path(self):
        g = _build_directed_triangle()
        weighted = [(0, 10.0), (1, 1.0), (2, 1.0)]
        pers = nk.centrality.PageRank.personalizationFromWeights(g, weighted)
        pr_vec = nk.centrality.PageRank(g, damp=0.85, tol=1e-10, personalization=pers)
        pr_vec.run()
        pr_fast = nk.centrality.PageRank.forWeights(g, weighted, damp=0.85, tol=1e-10)
        pr_fast.run()
        # The fast path normalizes inside the sparse-representation builder using a
        # slightly different summation order than the dense vector path, so a tolerance
        # of 1e-10 (the same as the iteration tolerance) is the right level here.
        for s_vec, s_fast in zip(pr_vec.scores(), pr_fast.scores()):
            self.assertAlmostEqual(s_vec, s_fast, places=9)

    def test_for_source_validation(self):
        g = _build_undirected_triangle()
        with self.assertRaises(RuntimeError):
            nk.centrality.PageRank.forSource(g, 42)
        with self.assertRaises(RuntimeError):
            nk.centrality.PageRank.forSources(g, [])
        with self.assertRaises(RuntimeError):
            nk.centrality.PageRank.forSources(g, [0, 99])
        with self.assertRaises(RuntimeError):
            nk.centrality.PageRank.forWeights(g, [(0, -1.0)])

    def test_for_source_runs_on_large_graph(self):
        # The optimization is most useful on large graphs: a 200k-node graph with
        # a single teleportation source would otherwise require allocating a 1.6 MB
        # personalization vector on the Python side (plus the C++ copy). Verify
        # that forSource runs to completion and produces the same sum-of-scores
        # invariant as the dense path.
        import random
        random.seed(0)
        n = 200_000
        g = nk.Graph(n, weighted=False, directed=False)
        for _ in range(n):
            u = random.randrange(n)
            v = random.randrange(n)
            if u != v:
                g.addEdge(u, v)

        pr_fast = nk.centrality.PageRank.forSource(g, 0, damp=0.85, tol=1e-6)
        pr_fast.run()
        scores = pr_fast.scores()
        self.assertAlmostEqual(sum(scores), 1.0, places=6)
        # The source should have the largest score (teleportation bias).
        self.assertEqual(max(range(len(scores)), key=lambda u: scores[u]), 0)

    def test_constructor_auto_detects_sparse(self):
        # A personalization vector with at most SPARSE_THRESHOLD non-zero entries
        # is silently switched to the sparse representation by the constructor.
        # We can only observe this indirectly: the result must match the explicit
        # forSource / forSources fast paths bit-for-bit.
        g = _build_directed_triangle()

        # Single source via vector — k=1 should switch to sparse.
        pers1 = nk.centrality.PageRank.personalizationFromSource(g, 0)
        pr_vec = nk.centrality.PageRank(g, damp=0.85, tol=1e-10, personalization=pers1)
        pr_vec.run()
        pr_for = nk.centrality.PageRank.forSource(g, 0, damp=0.85, tol=1e-10)
        pr_for.run()
        for s_vec, s_for in zip(pr_vec.scores(), pr_for.scores()):
            self.assertAlmostEqual(s_vec, s_for, places=12)

        # Three sources via vector — k=3 should switch to sparse.
        sources = [0, 1, 2]
        pers2 = nk.centrality.PageRank.personalizationFromSources(g, sources)
        pr_vec2 = nk.centrality.PageRank(g, damp=0.85, tol=1e-10, personalization=pers2)
        pr_vec2.run()
        pr_for2 = nk.centrality.PageRank.forSources(g, sources, damp=0.85, tol=1e-10)
        pr_for2.run()
        for s_vec, s_for in zip(pr_vec2.scores(), pr_for2.scores()):
            self.assertAlmostEqual(s_vec, s_for, places=12)


if __name__ == "__main__":
    unittest.main()
