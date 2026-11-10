/*
 * SteinerKCore.cpp
 *
 *  Created on: 06.10.26
 *      Author: Ian Chen (ianchen3@illinois.edu)
 */

#include <algorithm>
#include <bit>
#include <limits>
#include <numeric>
#include <omp.h>
#include <ranges>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <networkit/auxiliary/HierarchicalBucketPQ.hpp>
#include <networkit/centrality/CoreDecomposition.hpp>
#include <networkit/scd/SteinerKCore.hpp>
#include <networkit/structures/ConcurrentUnionFind.hpp>
#include <networkit/structures/UnionFind.hpp>

namespace NetworKit {

namespace {

using HBS = Aux::HierarchicalBucketPQ;
// query -> \# of seeds of the query in a component
using CountMap = std::unordered_map<index, count>;

// Merge the batch into its largest map, pushing the queries it connects into R.
CountMap mergeSmallerToLarger(const std::vector<std::set<node>> &s, std::vector<CountMap> &&batch,
                              std::vector<index> &R) {
    const auto largest =
        std::max_element(batch.begin(), batch.end(),
                         [](const CountMap &a, const CountMap &b) { return a.size() < b.size(); });
    CountMap into = std::move(*largest);
    for (auto from = batch.begin(); from != batch.end(); ++from) {
        if (from == largest)
            continue;
        for (const auto &[q, c] : *from) {
            count &total = into[q];
            total += c;
            if (total == s[q].size()) {
                R.push_back(q);
                into.erase(q);
            }
        }
    }
    return into;
}

} // namespace

SteinerKCore::Communities::Communities(std::span<const node> cllist,
                                       std::span<const std::pair<node, node>> interval,
                                       std::span<const index> order, std::vector<count> coreness)
    : coreness(std::move(coreness)) {
    // build shared buffer of all output vertices
    std::vector<index> pos(cllist.size(), none);
    for (index q : order) {
        auto [x, last] = interval[q];
        if (pos[x] != none)
            continue;
        do {
            pos[x] = data.size();
            data.push_back(x);
        } while (std::exchange(x, cllist[x]) != last);
    }

    // assign elements to the Communities output
    communities.resize(interval.size());
    for (index q = 0; q < interval.size(); ++q)
        if (const auto [first, last] = interval[q]; first != none)
            communities[q] =
                std::span<const node>(data.data() + pos[first], pos[last] + 1 - pos[first]);
}

// The augmented union-find of the paper's supplement.
// Components are in an interval of an underlying list, avoiding copies of shared vertices.
SteinerKCore::Communities SteinerKCore::getComms(
    const std::vector<std::set<node>> &s, count n, std::span<const std::pair<node, node>> forest,
    std::span<const std::pair<count, std::vector<index>>> levels, std::vector<count> coreness) {
    // query -> (first, last) vertex of its component's interval in the list (when ready)
    std::vector<std::pair<node, node>> interval(s.size(), {none, none});

    // replay the merges of forest, recording each ready query's interval
    UnionFind uf(n);
    std::vector<node> cllist(n); // circular linked list
    std::iota(cllist.begin(), cllist.end(), 0);
    index i = 0;
    for (const auto &[end, R] : levels) {
        for (; i < end; ++i) {
            std::swap(cllist[uf.find(forest[i].first)], cllist[uf.find(forest[i].second)]);
            uf.merge(forest[i].first, forest[i].second);
        }
        for (index q : R) {
            const node r = uf.find(*s[q].begin());
            interval[q] = {cllist[r], r};
        }
    }

    // queries by decreasing component size, so that nested intervals share their parent's buffer
    std::vector<index> order;
    for (index q : levels | std::views::reverse | std::views::values | std::views::join)
        order.push_back(q);
    return Communities(cllist, interval, order, std::move(coreness));
}

SteinerKCore::SteinerKCore(const Graph &g, std::span<const count> coreness)
    : SelectiveCommunityDetector(g), coreness(coreness) {}

SteinerKCore::SteinerKCore(const Graph &g) : SelectiveCommunityDetector(g) {
    CoreDecomposition cd(g);
    cd.run();
    const std::vector<double> &scores = cd.scores();
    ownedCoreness = std::vector<count>(scores.begin(), scores.end());
    coreness = ownedCoreness;
}

std::set<node> SteinerKCore::expandOneCommunity(const std::set<node> &s) {
    const Communities c = communities({s});
    return std::set<node>(c.communities[0].begin(), c.communities[0].end());
}

SteinerKCore::Communities SteinerKCore::communities(const std::vector<std::set<node>> &s) {
    if (s.empty())
        return {};
    const count n = g->upperNodeIdBound();
    if (n > std::numeric_limits<HBS::Value>::max())
        throw std::invalid_argument("SteinerKCore: too many node ids for the priority queue");
    for (const auto &q : s)
        if (!q.empty() && *q.rbegin() >= n)
            throw std::out_of_range("SteinerKCore: seed is not a node id");

    // 3: UF <- UnionFind(V), PQ <- { coreness(q) -> q | q in Q_i }
    const count numThreads = static_cast<count>(omp_get_max_threads());
    ConcurrentUnionFind uf(n);

    // Keys are flipped so that the min-PQ extracts the highest coreness first.
    count maxKey = 0;
    for (const auto &q : s)
        for (node u : q)
            maxKey = std::max(maxKey, coreness[u]);
    auto flip = [&](count k) { return maxKey - k; };
    HBS pq(n, numThreads, static_cast<count>(std::bit_width(maxKey)), 1);

    // thread -> (v, u) for each merge in the current round that joined two sets
    std::vector<std::vector<std::pair<node, node>>> merges(numThreads);
    // (v, u) for each merge that joined two sets, grouped by level in descending order
    std::vector<std::pair<node, node>> forest;
    // level -> (size of forest at its end, R), for the levels whose R is not empty
    std::vector<std::pair<count, std::vector<index>>> levels;
    // root -> (query -> seeds of the query in the root's component)
    std::unordered_map<node, CountMap> counts;
    // round -> all query indices that are connected
    std::vector<std::vector<index>> ready(maxKey + 1);
    for (index i = 0; i < s.size(); ++i) {
        for (node u : s[i]) {
            pq.insert(flip(coreness[u]), static_cast<HBS::Value>(u));
            if (s[i].size() == 1)
                ready[coreness[u]].push_back(i);
            else
                ++counts[u][i];
        }
    }

    // query -> coreness (when ready)
    std::vector<count> queryCoreness(s.size(), none);
    if (pq.empty())
        return getComms(s, n, forest, levels, std::move(queryCoreness));

    HBS::Key round;
    std::span<const HBS::Value> frontier;
    std::tie(round, frontier) = pq.extractMin();

    count completed = 0;
    // 4: for each k in descending order, until every query is answered or no vertex is left
    while (completed < s.size() && !frontier.empty()) {
        const count k = flip(round);
        // 6: while PQ.GetMaxKey() = k: v <- PQ.PopMaxValue()
        while (!frontier.empty() && flip(round) == k) {
#pragma omp parallel for schedule(guided)
            for (omp_index i = 0; i < static_cast<omp_index>(frontier.size()); ++i) {
                // 7: Seen <- Seen + {v} is the extraction from the pq
                const node v = frontier[i];
                const count tid = static_cast<count>(omp_get_thread_num());
                // 8: for u in N(v)
                g->forNeighborsOf(v, [&](node u) {
                    // 9: if coreness(u) >= k and UF.Union(u, v)
                    if (coreness[u] >= k && uf.merge(v, u))
                        merges[tid].emplace_back(v, u);
                    // 11-12: if u not in Seen: PQ <- PQ + { min(k, coreness(u)) -> u }
                    pq.insert(flip(std::min(k, coreness[u])), static_cast<HBS::Value>(u), tid);
                });
            }
            if (pq.empty())
                frontier = {};
            else
                std::tie(round, frontier) = pq.extractMin();
        }

        // 9-10: if Q_i becomes connected from edge uv: add Q_i to R
        std::vector<index> R = std::move(ready[k]);
        std::unordered_map<node, std::vector<CountMap>> groups;
        for (auto &[x, m] : counts)
            groups[uf.find(x)].push_back(std::move(m));
        counts.clear();
        for (auto &[r, batch] : groups)
            if (CountMap m = mergeSmallerToLarger(s, std::move(batch), R); !m.empty())
                counts.emplace(r, std::move(m));

        // 13: output C_i <- UF.ComponentOf(Q_i) for each Q_i in R
        // The level of R is recorded; its members are laid out after the last round.
        for (auto &threadMerges : merges) {
            forest.insert(forest.end(), threadMerges.begin(), threadMerges.end());
            threadMerges.clear();
        }
        for (index q : R)
            queryCoreness[q] = k;
        completed += R.size();
        if (!R.empty())
            levels.emplace_back(forest.size(), std::move(R));
    }

    return getComms(s, n, forest, levels, std::move(queryCoreness));
}

} // namespace NetworKit
