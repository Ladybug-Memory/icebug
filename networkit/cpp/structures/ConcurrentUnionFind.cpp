#include <atomic>
#include <omp.h>
#include <networkit/Globals.hpp>
#include <networkit/structures/ConcurrentUnionFind.hpp>

namespace NetworKit {

ConcurrentUnionFind::ConcurrentUnionFind(index max_element) : parent(max_element) {
#pragma omp parallel for schedule(static)
    for (omp_index i = 0; i < max_element; ++i) {
        parent[i].store(-1, std::memory_order_relaxed);
    }
}

index ConcurrentUnionFind::find(index i) {
    while (1) {
        element p = parent[i].load();
        if (p < 0)
            return i;

        // path halving
        element gp = parent[p].load();
        if (gp < 0)
            return p;

        parent[i].store((element)gp, std::memory_order_release);
        i = gp;
    }
}

bool ConcurrentUnionFind::merge(index u, index v) {
    element pu = parent[u].load();
    element pv = parent[v].load();

    // immediate parent check
    if (pu == pv && pu >= 0) {
        return false;
    }

    while (1) {
        element root_u = (element)find(u);
        element root_v = (element)find(v);

        if (root_u == root_v) {
            return false;
        }

        // both elements should be roots
        element val_u = parent[root_u].load();
        element val_v = parent[root_v].load();
        if (val_u >= 0 || val_v >= 0) {
            continue;
        }

        // merge into larger rank (more negative)
        if (val_u > val_v || (val_u == val_v && root_u < root_v)) {
            std::swap(root_u, root_v);
            std::swap(val_u, val_v);
        }

        element expected = val_v;
        if (parent[root_v].compare_exchange_weak(expected, root_u)) {
            // increment rank
            if (val_u == val_v) {
                element expected_rank = val_u;
                while (expected_rank < 0
                       && !parent[root_u].compare_exchange_weak(expected_rank, expected_rank - 1)) {
                }
            }
            return true;
        }
    }
}

} /* namespace NetworKit */
