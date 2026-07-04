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

// Validates and normalizes a user-supplied personalization vector. The returned vector has
// length @a z and its entries sum to 1. Throws @c std::runtime_error on invalid input.
std::vector<double> normalizePersonalization(const std::vector<double> &raw, count z) {
    if (raw.size() < z) {
        std::ostringstream oss;
        oss << "Personalization vector has size " << raw.size()
            << " but the graph has upperNodeIdBound() == " << z
            << "; the vector must cover every node id.";
        throw std::runtime_error(oss.str());
    }
    double sum = 0.0;
    for (count u = 0; u < z; ++u) {
        if (raw[u] < 0.0) {
            std::ostringstream oss;
            oss << "Personalization weight for node " << u << " is negative (" << raw[u] << ").";
            throw std::runtime_error(oss.str());
        }
        sum += raw[u];
    }
    if (sum <= 0.0) {
        throw std::runtime_error(
            "Personalization vector sums to zero; at least one node must have a positive weight.");
    }
    std::vector<double> p(z, 0.0);
    const double inv = 1.0 / sum;
    for (count u = 0; u < z; ++u) {
        p[u] = raw[u] * inv;
    }
    return p;
}

} // namespace

PageRank::PageRank(const Graph &G, double damp, double tol, bool normalized,
                   SinkHandling distributeSinks, std::vector<double> personalization)
    : Centrality(G, true), damp(damp), tol(tol), normalized(normalized),
      distributeSinks(distributeSinks),
      personalization(personalization.empty()
                          ? std::vector<double>()
                          : normalizePersonalization(personalization, G.upperNodeIdBound())) {}

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
    return normalizePersonalization(p, G.upperNodeIdBound());
}

void PageRank::run() {
    Aux::SignalHandler handler;
    const auto n = G.numberOfNodes();
    const auto z = G.upperNodeIdBound();

    // Per-node teleportation probability p[u], summing to 1. When the user did not request
    // personalized PageRank we use the uniform distribution 1/n.
    std::vector<double> personalizationProb;
    if (personalization.empty()) {
        personalizationProb.assign(n, 1.0 / static_cast<double>(n));
    } else {
        personalizationProb = personalization;
    }

    scoreData.resize(z, 0.0);
    G.parallelForNodes([&](const node u) { scoreData[u] = personalizationProb[u]; });
    std::vector<double> pr = scoreData;

    std::vector<double> deg(z, 0.0);
    G.parallelForNodes([&](const node u) { deg[u] = static_cast<double>(G.weightedDegree(u)); });

    std::vector<node> sinks;
    if (G.isDirected() && ((distributeSinks == SinkHandling::DISTRIBUTE_SINKS) || normalized)) {
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

    bool isConverged = false;
    do {
        handler.assureRunning();
        G.balancedParallelForNodes([&](const node u) {
            pr[u] = 0.0;
            G.forInEdgesOf(u, [&](const node u, const node v, const edgeweight w) {
                // note: inconsistency in definition in Newman's book (Ch. 7) regarding
                // directed graphs we follow the verbal description, which requires to
                // sum over the incoming edges
                pr[u] += scoreData[v] * w / deg[v];
            });
            pr[u] *= damp;
            // Personalized teleportation: (1 - damp) * p[u]. For the default uniform case this
            // collapses to (1 - damp) / n, recovering the original PageRank iteration.
            pr[u] += (1.0 - damp) * personalizationProb[u];
        });

        // For directed graphs sink-handling is needed to fulfill |pr| == 1 in each step. Otherwise
        // probability mass would be leaked, creating wrong results. For this, we redistribute
        // each sink's mass to all nodes in proportion to the personalization distribution p.
        // This is described amongst others in "PageRank revisited." by M. Brinkmeyer et al.
        // (2005). For uniform p this is equivalent to distributing uniformly (1/n per node).
        if (G.isDirected() && ((distributeSinks == SinkHandling::DISTRIBUTE_SINKS) || normalized)) {
            double totalSinkPr = 0.0;
#pragma omp parallel for reduction(+ : totalSinkPr)
            for (omp_index i = 0; i < static_cast<omp_index>(nSinks); i++) {
                totalSinkPr += scoreData[sinks[i]];
            }
            const double totalSinkContrib = damp * totalSinkPr;
            G.balancedParallelForNodes(
                [&](const node u) { pr[u] += totalSinkContrib * personalizationProb[u]; });
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
