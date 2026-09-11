# distutils: language=c++

from cython.operator import dereference, preincrement

from libc.stdint cimport uint64_t
from libcpp cimport bool as bool_t
from libcpp.vector cimport vector
from libcpp.string cimport string

from .base cimport _Algorithm, Algorithm
from .graph cimport _Graph, Graph
from .structures cimport count

cdef extern from "<networkit/embedding/FastRP.hpp>" namespace "NetworKit":

	cdef enum _FastRPProjection "NetworKit::FastRPProjection":
		FASTRP_SPARSE
		FASTRP_GAUSSIAN

	cdef enum _FastRPInputMatrix "NetworKit::FastRPInputMatrix":
		FASTRP_ADJACENCY
		FASTRP_TRANSITION

	cdef cppclass _FastRP "NetworKit::FastRP"(_Algorithm):
		_FastRP(_Graph G, count dim, vector[double] weights, bool_t normalization,
			double alpha, _FastRPProjection projection, _FastRPInputMatrix inputMatrix,
			vector[vector[double]] nodeFeatures, double featureWeight,
			count fallbackThreshold, uint64_t seed) except +
		vector[vector[double]] &getEmbeddings() except +

cdef class FastRP(Algorithm):
	"""
	FastRP(G, dim=128, weights=(1, 1, 1), normalization=True, alpha=-0.5,
	      projection='sparse', inputMatrix='transition', nodeFeatures=None,
	      featureWeight=0.5, fallbackThreshold=3, seed=42)

	Algorithm to extract node embeddings with the FastRP algorithm according to
	[https://arxiv.org/pdf/1607.00653v1.pdf], with the following changes:

	- L2 normalization of every propagation power before the weighted merge and of
	  the final embedding (if `normalization` is set).
	- Support for node attributes: `nodeFeatures` supplies one attribute vector per
	  node which is projected, L2-normalized and blended into the structural
	  embedding with weight `featureWeight`.
	- Robustness for poorly connected graphs: nodes in weakly connected components
	  smaller than `fallbackThreshold` are embedded from their attributes alone
	  (if attributes are available).
	- Support for edge weights.

	The number of propagation powers equals `len(weights)`.

	Parameters
	----------
	G : networkit.Graph
		The graph. Node ids must be contiguous.
	dim : int
		The dimension of the calculated embedding.
	weights : list(float)
		Weight of every propagation power in the final merge, e.g. [1, 1, 1].
	normalization : bool
		Whether to L2-normalize the propagated matrices and the final embedding.
	alpha : float
		Degree correction exponent applied to the projection matrix; 0 disables it.
	projection : str
		Random projection variant: 'sparse' or 'gaussian'.
	inputMatrix : str
		Matrix to propagate: 'adjacency' or 'transition' (row-normalized adjacency).
	nodeFeatures : list(list(float))
		Optional per-node attribute vectors, all of equal length; None disables.
	featureWeight : float
		Weight of the attribute embedding in the final blend.
	fallbackThreshold : int
		Components smaller than this are embedded from attributes alone (if
		available); 1 disables the fallback.
	seed : int
		Seed of the random projection matrix; results are deterministic for a
		fixed seed.
	"""

	cdef Graph _G

	def __cinit__(self, Graph G, dim=128, weights=(1.0, 1.0, 1.0), normalization=True,
			alpha=-0.5, projection='sparse', inputMatrix='transition', nodeFeatures=None,
			featureWeight=0.5, fallbackThreshold=3, seed=42):
		cdef vector[double] _weights
		for w in weights:
			_weights.push_back(w)

		cdef _FastRPProjection _projection
		if projection == 'sparse':
			_projection = FASTRP_SPARSE
		elif projection == 'gaussian':
			_projection = FASTRP_GAUSSIAN
		else:
			raise ValueError("projection must be 'sparse' or 'gaussian'")

		cdef _FastRPInputMatrix _inputMatrix
		if inputMatrix == 'adjacency':
			_inputMatrix = FASTRP_ADJACENCY
		elif inputMatrix == 'transition':
			_inputMatrix = FASTRP_TRANSITION
		else:
			raise ValueError("inputMatrix must be 'adjacency' or 'transition'")

		cdef vector[vector[double]] _features
		if nodeFeatures is not None:
			for row in nodeFeatures:
				_features.push_back(row)

		self._G = G
		self._this = new _FastRP(dereference(G._view()), dim, _weights, normalization,
			alpha, _projection, _inputMatrix, _features, featureWeight,
			fallbackThreshold, seed)

	def getEmbeddings(self):
		"""
		getEmbeddings()

		Returns the embedding vector of every node, indexed by node id.

		Returns
		-------
		list(list(float))
			A vector containing embedding vectors of all nodes.
		"""
		return (<_FastRP*>(self._this)).getEmbeddings()
cdef extern from "<networkit/embedding/Node2Vec.hpp>":

	cdef cppclass _Node2Vec "NetworKit::Node2Vec"(_Algorithm):
		_Node2Vec(_Graph G, double P, double Q, count L, count N, count D) except +
		vector[vector[float]] &getFeatures() except +

cdef class Node2Vec(Algorithm):
	""" 
	Node2Vec(G, P, Q, L, N, D)

	Algorithm to extract features from the graph with the node2vec(word2vec)
	algorithm according to [https://arxiv.org/pdf/1607.00653v1.pdf].

	Node2Vec learns embeddings for nodes in a graph by optimizing a neighborhood preserving
	objective. In order to achieve this, biased random walks are initiated for every node and the
	result is of probabilistic nature. Several input parameters control the specific behavior of
	the random walks. Amongst others Node2Vec is able to produce embeddings for visualization
	(D=2 or D=3) and machine learning (D=128 [default]). Both directed and undirected graphs
	withouth isolated nodes are supported.

	Note
	---- 
	This algorithm could take a lot of time on large networks (many nodes).
 
	Parameters
	----------
	G : networkit.Graph
		The graph.
	P : float
		The ratio for returning to the previous node on a walk.
		For P > max(Q,1) it is less likely to sample an already-visited node in the following two steps.
		For P < min(Q,1) it is more likely to sample an already-visited node in the following two steps.
	Q : float
		The ratio for the direction of the next step
		For Q > 1 the random walk is biased towards nodes close to the previous one.
		For Q < 1 the random walk is biased towards nodes which are further away from the previous one. 
	L : int
		The walk length.
	N : int
		The number of walks per node.
	D: int
		The dimension of the calculated embedding. 
	"""

	cdef Graph _G
 
	def __cinit__(self, Graph G, P=1, Q=1, L=80, N=10, D=128):
		self._G = G
		self._this = new _Node2Vec(dereference(G._view()), P, Q, L, N, D)

	def getFeatures(self):
		"""
		getFeatures()

		Returns all feature vectors

		Returns
		-------
		list(list(float))
			A vector containing feature vectors of all nodes
		"""
		return (<_Node2Vec*>(self._this)).getFeatures()
