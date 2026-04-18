#!/usr/bin/env python3
import unittest

import icebug as ib
import networkit as nk


class TestIcebugAlias(unittest.TestCase):

	def test_top_level_alias(self):
		self.assertIs(ib, nk)

	def test_core_api_available(self):
		graph = ib.Graph(5)
		self.assertEqual(graph.numberOfNodes(), 5)

	def test_submodule_alias(self):
		import icebug.centrality as ib_cent
		import networkit.centrality as nk_cent

		self.assertIs(ib_cent, nk_cent)
