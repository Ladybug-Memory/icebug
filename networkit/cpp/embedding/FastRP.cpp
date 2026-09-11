/*
 * FastRP.cpp
 *
 * Implementation of FastRP with L2 normalization, node attribute support, a
 * connectivity fallback and edge weight support. See FastRP.hpp.
 */

#include <networkit/embedding/FastRP.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

#include <networkit/auxiliary/SignalHandling.hpp>

namespace NetworKit {

FastRP::FastRP(const Graph &G, count dim, const std::vector<double> &weights, bool normalization,
               double alpha, FastRPProjection projection, FastRPInputMatrix inputMatrix,
               const std::vector<std::vector<double>> &nodeFeatures, double featureWeight,
               count fallbackThreshold, uint64_t seed)
    : G(&G), dim(dim), weights(weights), normalization(normalization), alpha(alpha),
      projection(projection), inputMatrix(inputMatrix), nodeFeatures(nodeFeatures),
      featureWeight(featureWeight), fallbackThreshold(fallbackThreshold), seed(seed) {

    if (weights.empty())
        throw std::runtime_error("FastRP: at least one weight is required.");
    if (dim == 0)
        throw std::runtime_error("FastRP: the embedding dimension must be positive.");
    if (G.numberOfNodes() != G.upperNodeIdBound())
        throw std::runtime_error("FastRP: node ids must be contiguous.");
    if (featureWeight < 0.0)
        throw std::runtime_error("FastRP: featureWeight must be non-negative.");
    if (fallbackThreshold == 0)
        throw std::runtime_error("FastRP: fallbackThreshold must be at least one.");
    if (!nodeFeatures.empty()) {
        if (nodeFeatures.size() != G.upperNodeIdBound())
            throw std::runtime_error("FastRP: exactly one feature vector per node is required.");
        const count f = nodeFeatures[0].size();
        for (const auto &row : nodeFeatures) {
            if (row.size() != f)
                throw std::runtime_error("FastRP: all feature vectors must have equal length.");
        }
    }
}

void FastRP::propagate(const std::vector<double> &in, std::vector<double> &out,
                       const std::vector<double> &invWeightedDegree) const {
    const count n = G->upperNodeIdBound();
    const bool transition = (inputMatrix == FASTRP_TRANSITION);
    G->parallelForNodes([&](node u) {
        double *outRow = &out[u * dim];
        std::fill(outRow, outRow + dim, 0.0);
        const double m = transition ? invWeightedDegree[u] : 1.0;
        auto accumulate = [&](node v, double w) {
            const double weight = m * w;
            const double *inRow = &in[v * dim];
            for (count k = 0; k < dim; ++k)
                outRow[k] += weight * inRow[k];
        };
        if (G->isWeighted()) {
            for (auto [v, w] : G->weightNeighborRange(u))
                accumulate(v, w);
        } else {
            for (node v : G->neighborRange(u))
                accumulate(v, 1.0);
        }
    });
}

void FastRP::weaklyConnectedComponents(std::vector<count> &compOf,
                                       std::vector<count> &compSize) const {
    const count n = G->upperNodeIdBound();
    constexpr count NONE = static_cast<count>(-1);
    compOf.assign(n, NONE);
    compSize.clear();
    std::vector<node> stack;
    for (node s = 0; s < n; ++s) {
        if (compOf[s] != NONE)
            continue;
        const count c = compSize.size();
        count size = 0;
        compOf[s] = c;
        stack.push_back(s);
        while (!stack.empty()) {
            const node u = stack.back();
            stack.pop_back();
            ++size;
            auto visit = [&](node v) {
                if (compOf[v] == NONE) {
                    compOf[v] = c;
                    stack.push_back(v);
                }
            };
            for (node v : G->neighborRange(u))
                visit(v);
            if (G->isDirected())
                for (node v : G->inNeighborRange(u))
                    visit(v);
        }
        compSize.push_back(size);
    }
}

void FastRP::l2NormalizeRows(std::vector<double> &matrix, count n, count dim) {
#pragma omp parallel for
    for (omp_index u = 0; u < static_cast<omp_index>(n); ++u) {
        double *row = &matrix[u * dim];
        double norm = 0.0;
        for (count k = 0; k < dim; ++k)
            norm += row[k] * row[k];
        if (norm > 0.0) {
            norm = std::sqrt(norm);
            for (count k = 0; k < dim; ++k)
                row[k] /= norm;
        }
    }
}

void FastRP::run() {
    Aux::SignalHandler handler;
    const count n = G->upperNodeIdBound();
    const count q = weights.size();

    embeddings.clear();
    if (n == 0) {
        hasRun = true;
        return;
    }

    // (Inverse) weighted out-degree of every node.
    std::vector<double> invDeg(n, 0.0);
    std::vector<double> deg(n, 0.0);
    G->forNodes([&](node u) {
        const double d = G->isWeighted() ? G->weightedDegree(u) : static_cast<double>(G->degree(u));
        deg[u] = d;
        invDeg[u] = d > 0.0 ? 1.0 / d : 0.0;
    });

    // Weakly connected components, for the poor-connectivity fallback.
    std::vector<count> compOf;
    std::vector<count> compSize;
    weaklyConnectedComponents(compOf, compSize);

    // Node features are used only if they are non-empty and carry positive weight.
    const bool useFeatures =
        !nodeFeatures.empty() && !nodeFeatures[0].empty() && featureWeight > 0.0;

    // Structural weight per node: nodes of tiny components rely on their attributes.
    std::vector<double> structuralWeight(n, 1.0);
    if (useFeatures && fallbackThreshold > 1) {
        G->forNodes([&](node u) {
            if (compSize[compOf[u]] < fallbackThreshold)
                structuralWeight[u] = 0.0;
        });
    }

    std::mt19937_64 rng(seed);
    const double projScale = 1.0 / std::sqrt(static_cast<double>(dim));
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::normal_distribution<double> gaussian(0.0, 1.0);

    // Random projection matrix (n x dim), rows scaled by degree(u)^alpha.
    std::vector<double> projectionMatrix(n * dim, 0.0);
    for (node u = 0; u < n; ++u) {
        double *row = &projectionMatrix[u * dim];
        for (count k = 0; k < dim; ++k) {
            switch (projection) {
            case FASTRP_SPARSE: {
                const double r = uniform(rng);
                row[k] = (r < 1.0 / 6.0 ? 1.0 : (r < 2.0 / 6.0 ? -1.0 : 0.0)) * projScale;
                break;
            }
            case FASTRP_GAUSSIAN:
                row[k] = gaussian(rng) * projScale;
                break;
            }
        }
        if (alpha != 0.0) {
            const double degScale = deg[u] > 0.0 ? std::pow(deg[u], alpha) : 0.0;
            for (count k = 0; k < dim; ++k)
                row[k] *= degScale;
        }
    }

    // Attribute embedding: project the (row-wise L2-normalized) feature matrix into
    // dim dimensions with a deterministic sparse projection, then L2-normalize.
    std::vector<double> attributeEmbedding;
    if (useFeatures) {
        const count f = nodeFeatures[0].size();
        std::vector<double> featureProjection(f * dim, 0.0);
        for (count j = 0; j < f; ++j) {
            double *row = &featureProjection[j * dim];
            for (count k = 0; k < dim; ++k) {
                const double r = uniform(rng);
                row[k] = (r < 1.0 / 6.0 ? 1.0 : (r < 2.0 / 6.0 ? -1.0 : 0.0)) * projScale;
            }
        }

        attributeEmbedding.assign(n * dim, 0.0);
        G->forNodes([&](node u) {
            const std::vector<double> &features = nodeFeatures[u];
            double norm = 0.0;
            for (const double value : features)
                norm += value * value;
            norm = norm > 0.0 ? std::sqrt(norm) : 1.0;
            double *outRow = &attributeEmbedding[u * dim];
            for (count j = 0; j < f; ++j) {
                const double scaled = features[j] / norm;
                const double *projRow = &featureProjection[j * dim];
                for (count k = 0; k < dim; ++k)
                    outRow[k] += scaled * projRow[k];
            }
        });
        l2NormalizeRows(attributeEmbedding, n, dim);
    }

    // Propagation chain: U_1 = M * (D^alpha * R), U_i = M * U_(i-1), merged with
    // the per-power weights (after optional row-wise L2 normalization).
    std::vector<double> prev(n * dim, 0.0);
    std::vector<double> cur(n * dim, 0.0);
    std::vector<double> acc(n * dim, 0.0);
    propagate(projectionMatrix, cur, invDeg);
    projectionMatrix.clear();
    projectionMatrix.shrink_to_fit();

    for (count i = 1; i <= q; ++i) {
        handler.assureRunning();
        if (normalization)
            l2NormalizeRows(cur, n, dim);
        const double w = weights[i - 1];
#pragma omp parallel for
        for (omp_index u = 0; u < static_cast<omp_index>(n); ++u) {
            const double *curRow = &cur[u * dim];
            double *accRow = &acc[u * dim];
            for (count k = 0; k < dim; ++k)
                accRow[k] += w * curRow[k];
        }
        if (i < q) {
            std::swap(prev, cur);
            propagate(prev, cur, invDeg);
        }
    }

    // Final blend of the structural and attribute embeddings, followed by the
    // final L2 normalization.
    embeddings.assign(n, std::vector<double>(dim, 0.0));
    G->parallelForNodes([&](node u) {
        std::vector<double> &emb = embeddings[u];
        const double *accRow = &acc[u * dim];
        const double sw = structuralWeight[u];
        const double fw = useFeatures ? featureWeight : 0.0;
        const double *attrRow = useFeatures ? &attributeEmbedding[u * dim] : nullptr;
        for (count k = 0; k < dim; ++k)
            emb[k] = sw * accRow[k] + (fw > 0.0 ? fw * attrRow[k] : 0.0);
        if (normalization) {
            double norm = 0.0;
            for (count k = 0; k < dim; ++k)
                norm += emb[k] * emb[k];
            if (norm > 0.0) {
                norm = std::sqrt(norm);
                for (count k = 0; k < dim; ++k)
                    emb[k] /= norm;
            }
        }
    });

    hasRun = true;
}

const std::vector<std::vector<double>> &FastRP::getEmbeddings() const {
    assureFinished();
    return embeddings;
}

} /* namespace NetworKit */
