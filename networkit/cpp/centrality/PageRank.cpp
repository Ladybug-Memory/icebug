/*
 * PageRank.cpp
 *
 *  Created on: 19.03.2014
 *      Authors: Henning Meyerhenke
 *               Fabian Brandt-Tumescheit <brandtfa@hu-berlin.de>
 */

#include <networkit/auxiliary/NumericTools.hpp>
#include <networkit/auxiliary/Parallel.hpp>
#include <networkit/auxiliary/SignalHandling.hpp>
#include <networkit/centrality/PageRank.hpp>

#include <sstream>
#include <stdexcept>

namespace NetworKit {

namespace {

// Validates a personalization vector in a single O(z) pass. On success, fills @a normalized
// (must already have size z) with the input divided by its sum, and reports the count of
// non-zero entries. Throws @c std::runtime_error on invalid input.
void validateAndNormalizeInto(const std::vector<double> &raw, count z,
                              std::vector<double> &normalized, count &nonZeroCount) {
    if (raw.size() < z) {
        std::ostringstream oss;
        oss << "Personalization vector has size " << raw.size()
            << " but the graph has upperNodeIdBound() == " << z
            << "; the vector must cover every node id.";
        throw std::runtime_error(oss.str());
    }
    normalized.assign(z, 0.0);
    double sum = 0.0;
    nonZeroCount = 0;
    for (count u = 0; u < z; ++u) {
        if (raw[u] < 0.0) {
            std::ostringstream oss;
            oss << "Personalization weight for node " << u << " is negative (" << raw[u] << ").";
            throw std::runtime_error(oss.str());
        }
        if (raw[u] > 0.0) {
            normalized[u] = raw[u];
            sum += raw[u];
            ++nonZeroCount;
        }
    }
    if (sum <= 0.0) {
        throw std::runtime_error(
            "Personalization vector sums to zero; at least one node must have a positive weight.");
    }
    const double inv = 1.0 / sum;
    for (count u = 0; u < z; ++u) {
        normalized[u] *= inv;
    }
}

} // namespace

PageRank::PageRank(const Graph &G, double damp, double tol, bool normalized,
                   SinkHandling distributeSinks, std::vector<double> personalization)
    : Centrality(G, true), damp(damp), tol(tol), normalized(normalized),
      distributeSinks(distributeSinks) {
    if (personalization.empty()) {
        // Uniform teleportation: leave both sparse and dense empty.
        return;
    }
    const count z = G.upperNodeIdBound();
    std::vector<double> normalizedP;
    count nonZeroCount = 0;
    validateAndNormalizeInto(personalization, z, normalizedP, nonZeroCount);

    if (nonZeroCount <= SPARSE_THRESHOLD) {
        // Build the sparse representation. Memory is O(k) instead of O(z).
        sparse.reserve(nonZeroCount);
        for (count u = 0; u < z; ++u) {
            if (normalizedP[u] > 0.0) {
                sparse.emplace_back(u, normalizedP[u]);
            }
        }
    } else {
        // Too many non-zero entries; keep the dense representation.
        dense = std::move(normalizedP);
    }
}

PageRank::PageRank(const Graph &G, double damp, double tol, bool normalized,
                   SinkHandling distributeSinks, std::vector<std::pair<node, double>> sparseSpec)
    : Centrality(G, true), damp(damp), tol(tol), normalized(normalized),
      distributeSinks(distributeSinks), sparse(std::move(sparseSpec)) {}

PageRank PageRank::forSource(const Graph &G, node source, double damp, double tol, bool normalized,
                             SinkHandling distributeSinks) {
    if (!G.hasNode(source)) {
        throw std::runtime_error("Personalization source node does not exist in the graph.");
    }
    // Build a prvalue that the caller's storage is constructed into (C++17 guaranteed
    // copy elision — PageRank is not movable).
    std::vector<std::pair<node, double>> spec;
    spec.emplace_back(source, 1.0);
    return PageRank(G, damp, tol, normalized, distributeSinks, std::move(spec));
}

PageRank PageRank::forSources(const Graph &G, const std::vector<node> &sources, double damp,
                              double tol, bool normalized, SinkHandling distributeSinks) {
    if (sources.empty()) {
        throw std::runtime_error("Personalization source set must be non-empty.");
    }
    for (node s : sources) {
        if (!G.hasNode(s)) {
            throw std::runtime_error("Personalization source node does not exist in the graph.");
        }
    }
    std::vector<std::pair<node, double>> spec;
    spec.reserve(sources.size());
    const double w = 1.0 / static_cast<double>(sources.size());
    for (node s : sources) {
        spec.emplace_back(s, w);
    }
    return PageRank(G, damp, tol, normalized, distributeSinks, std::move(spec));
}

PageRank PageRank::forWeights(const Graph &G,
                              const std::vector<std::pair<node, double>> &weightedSources,
                              double damp, double tol, bool normalized,
                              SinkHandling distributeSinks) {
    if (weightedSources.empty()) {
        throw std::runtime_error("Personalization weighted source set must be non-empty.");
    }
    for (const auto &entry : weightedSources) {
        if (!G.hasNode(entry.first)) {
            throw std::runtime_error("Personalization source node does not exist in the graph.");
        }
        if (entry.second < 0.0) {
            std::ostringstream oss;
            oss << "Personalization weight for node " << entry.first << " is negative ("
                << entry.second << ").";
            throw std::runtime_error(oss.str());
        }
    }
    // Build a per-node weight vector and run the standard validator / normalizer. We can't
    // skip this even though the inputs are already "weights" because we need to enforce
    // sum > 0 and the non-negativity check.
    std::vector<double> raw(G.upperNodeIdBound(), 0.0);
    for (const auto &entry : weightedSources) {
        raw[entry.first] = entry.second;
    }
    std::vector<double> normalizedP;
    count nonZeroCount = 0;
    validateAndNormalizeInto(raw, G.upperNodeIdBound(), normalizedP, nonZeroCount);

    std::vector<std::pair<node, double>> spec;
    spec.reserve(nonZeroCount);
    for (count u = 0; u < G.upperNodeIdBound(); ++u) {
        if (normalizedP[u] > 0.0) {
            spec.emplace_back(u, normalizedP[u]);
        }
    }
    return PageRank(G, damp, tol, normalized, distributeSinks, std::move(spec));
}

std::vector<double> PageRank::personalizationFromSource(const Graph &G, node source) {
    if (!G.hasNode(source)) {
        throw std::runtime_error("Personalization source node does not exist in the graph.");
    }
    std::vector<double> p(G.upperNodeIdBound(), 0.0);
    p[source] = 1.0;
    return p;
}

std::vector<double> PageRank::personalizationFromSources(const Graph &G,
                                                         const std::vector<node> &sources) {
    if (sources.empty()) {
        throw std::runtime_error("Personalization source set must be non-empty.");
    }
    for (node s : sources) {
        if (!G.hasNode(s)) {
            throw std::runtime_error("Personalization source node does not exist in the graph.");
        }
    }
    std::vector<double> p(G.upperNodeIdBound(), 0.0);
    const double w = 1.0 / static_cast<double>(sources.size());
    for (node s : sources) {
        p[s] = w;
    }
    return p;
}

std::vector<double>
PageRank::personalizationFromWeights(const Graph &G,
                                     const std::vector<std::pair<node, double>> &weightedSources) {
    if (weightedSources.empty()) {
        throw std::runtime_error("Personalization weighted source set must be non-empty.");
    }
    std::vector<double> p(G.upperNodeIdBound(), 0.0);
    for (const auto &entry : weightedSources) {
        if (!G.hasNode(entry.first)) {
            throw std::runtime_error("Personalization source node does not exist in the graph.");
        }
        if (entry.second < 0.0) {
            std::ostringstream oss;
            oss << "Personalization weight for node " << entry.first << " is negative ("
                << entry.second << ").";
            throw std::runtime_error(oss.str());
        }
        p[entry.first] = entry.second;
    }
    // Reuse the validator / normalizer.
    std::vector<double> normalizedP;
    count nonZeroCount = 0;
    validateAndNormalizeInto(p, G.upperNodeIdBound(), normalizedP, nonZeroCount);
    (void)nonZeroCount;
    return normalizedP;
}

void PageRank::run() {
    Aux::SignalHandler handler;
    const auto n = G.numberOfNodes();
    const auto z = G.upperNodeIdBound();
    const bool hasSparse = !sparse.empty();
    const bool hasDense = !dense.empty();
    const double teleportBase = 1.0 - damp;

    // Initialize scores from the personalization distribution. Sparse stores only the k
    // non-zero entries, so the rest stay at zero; dense and uniform read from a single value
    // (dense[u] or 1/n) per node.
    scoreData.resize(z, 0.0);
    if (hasDense) {
        G.parallelForNodes([&](const node u) { scoreData[u] = dense[u]; });
    } else if (hasSparse) {
        for (const auto &entry : sparse) {
            scoreData[entry.first] = entry.second;
        }
    } else {
        const double uniformInit = 1.0 / static_cast<double>(n);
        G.parallelForNodes([&](const node u) { scoreData[u] = uniformInit; });
    }
    std::vector<double> pr = scoreData;

    std::vector<double> deg(z, 0.0);
    G.parallelForNodes([&](const node u) { deg[u] = static_cast<double>(G.weightedDegree(u)); });

    std::vector<node> sinks;
    if (G.isDirected() && ((distributeSinks == SinkHandling::DISTRIBUTE_SINKS) || normalized)) {
        // Reserve the worst case (every node is a sink) so push_back never reallocates.
        // In typical web-like graphs ~10-30% of nodes are sinks, so the over-allocation is
        // modest and bounded by 8·n bytes.
        sinks.reserve(n);
        G.forNodes([&](const node u) {
            if (G.degree(u) == 0) {
                sinks.push_back(u);
            }
        });
    }
    count nSinks = sinks.size();

    iterations = 0;

    auto sumL1Norm = [&](const node u) { return std::abs(scoreData[u] - pr[u]); };

    auto sumL2Norm = [&](const node u) {
        const auto d = scoreData[u] - pr[u];
        return d * d;
    };

    auto converged([&]() {
        if (iterations >= maxIterations) {
            return true;
        }

        if (norm == Norm::L2_NORM) {
            return std::sqrt(G.parallelSumForNodes(sumL2Norm)) <= tol;
        }

        return G.parallelSumForNodes(sumL1Norm) <= tol;
    });

    // Hot-loop body split into three specialized variants so the personalization cost is
    // O(1) per node in the sparse case (a tight loop over k entries, k <= SPARSE_THRESHOLD)
    // and the dense/uniform cases keep their original single-multiply form.
    auto stepSparse = [&](const node u) {
        pr[u] = 0.0;
        G.forInEdgesOf(u, [&](const node, const node v, const edgeweight w) {
            pr[u] += scoreData[v] * w / deg[v];
        });
        pr[u] *= damp;
        for (const auto &entry : sparse) {
            if (u == entry.first) {
                pr[u] += teleportBase * entry.second;
                break;
            }
        }
    };
    auto stepDense = [&](const node u) {
        pr[u] = 0.0;
        G.forInEdgesOf(u, [&](const node, const node v, const edgeweight w) {
            pr[u] += scoreData[v] * w / deg[v];
        });
        pr[u] *= damp;
        pr[u] += teleportBase * dense[u];
    };
    auto stepUniform = [&](const node u) {
        pr[u] = 0.0;
        G.forInEdgesOf(u, [&](const node, const node v, const edgeweight w) {
            pr[u] += scoreData[v] * w / deg[v];
        });
        pr[u] *= damp;
        pr[u] += teleportBase / static_cast<double>(n);
    };

    bool isConverged = false;
    do {
        handler.assureRunning();
        if (hasSparse) {
            G.balancedParallelForNodes(stepSparse);
        } else if (hasDense) {
            G.balancedParallelForNodes(stepDense);
        } else {
            G.balancedParallelForNodes(stepUniform);
        }

        // For directed graphs sink-handling is needed to fulfill |pr| == 1 in each step. Otherwise
        // probability mass would be leaked, creating wrong results. For this, we redistribute
        // each sink's mass in proportion to the personalization distribution p. With uniform p
        // this is equivalent to distributing 1/n per node. With sparse p we touch only the k
        // sources, so the cost is O(k) per iteration rather than O(n).
        if (G.isDirected() && ((distributeSinks == SinkHandling::DISTRIBUTE_SINKS) || normalized)) {
            double totalSinkPr = 0.0;
#pragma omp parallel for reduction(+ : totalSinkPr)
            for (omp_index i = 0; i < static_cast<omp_index>(nSinks); i++) {
                totalSinkPr += scoreData[sinks[i]];
            }
            const double totalSinkContrib = damp * totalSinkPr;
            if (hasSparse) {
                for (const auto &entry : sparse) {
                    pr[entry.first] += totalSinkContrib * entry.second;
                }
            } else if (hasDense) {
                G.balancedParallelForNodes(
                    [&](const node u) { pr[u] += totalSinkContrib * dense[u]; });
            } else {
                G.balancedParallelForNodes(
                    [&](const node u) { pr[u] += totalSinkContrib / static_cast<double>(n); });
            }
        }

        ++iterations;
        isConverged = converged();
        std::swap(pr, scoreData);
    } while (!isConverged);

    handler.assureRunning();

    // Post-processing for normalized PageRank. The normalization rescales by a uniform factor
    // and therefore applies to personalized PageRank unchanged; it just sets the (constant)
    // target value the average score should take.
    if (normalized) {
        double normFactor;
        if (G.isDirected()) {
            // Calculate sum of dangling Nodes for normalization
            double sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
            for (omp_index i = 0; i < static_cast<omp_index>(nSinks); i++) {
                sum += scoreData[sinks[i]];
            }
            normFactor = (1.0 / static_cast<double>(n)) * ((1.0 - damp) + (damp * sum));
        } else {
            normFactor = (1.0 - damp) / static_cast<double>(n);
        }
        G.parallelForNodes([&](const node u) { scoreData[u] /= normFactor; });

        // Post-processing for non-normalized PageRank
    } else {
        if (G.isDirected() && distributeSinks == SinkHandling::NO_SINK_HANDLING) {
            // In case no sink handling was done, make sure that |pr| == 1
            const auto sum = G.parallelSumForNodes([&](const node u) { return scoreData[u]; });
            G.parallelForNodes([&](const node u) { scoreData[u] /= sum; });
        }
    }
    // calculate the maxium
    max = scoreData[0];
    G.balancedParallelForNodes([&](const node u) { Aux::Parallel::atomic_max(max, scoreData[u]); });
    hasRun = true;
}

double PageRank::maximum() {
    return max.load(std::memory_order_relaxed);
}

} /* namespace NetworKit */
