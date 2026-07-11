/*
 * FlatMap.tpp
 *
 *  Created on: 07.11.2026
 *      Author: Ian Chen (ianchen3@illinois.edu)
 */

#ifndef NETWORKIT_AUXILIARY_FLAT_MAP_TPP_
#define NETWORKIT_AUXILIARY_FLAT_MAP_TPP_

#include <algorithm>
#include <cassert>
#include <numeric>
#include <stdexcept>

namespace Aux {

namespace detail {
template <class C>
bool is_sorted_unique(const C &c) {
    return std::adjacent_find(c.begin(), c.end(),
                              [](const auto &a, const auto &b) { return !(a < b); })
           == c.end();
}
} // namespace detail

template <class Key, class T>
auto flat_map<Key, T>::rank_of(const key_type &v) const -> size_type {
    return static_cast<size_type>(std::lower_bound(c.keys.begin(), c.keys.end(), v)
                                  - c.keys.begin());
}

// Merge sorted-unique (bk, bv) into (c.keys, c.values); existing keys win.
template <class Key, class T>
void flat_map<Key, T>::merge_sorted_vectors(key_container_type &bk, mapped_container_type &bv) {
    assert(detail::is_sorted_unique(bk));
    assert(bk.size() == bv.size());
    if (bk.empty())
        return;

    size_type i = 0, j = 0, nnew = 0; // count keys not already present
    while (j < bk.size()) {
        if (i == c.keys.size() || bk[j] < c.keys[i]) {
            ++nnew;
            ++j;
        } else if (c.keys[i] < bk[j]) {
            ++i;
        } else {
            ++i;
            ++j;
        }
    }
    if (nnew == 0)
        return;

    const size_type oldk = c.keys.size();
    c.keys.resize(oldk + nnew);
    c.values.resize(oldk + nnew);

    difference_type ri = static_cast<difference_type>(oldk) - 1;
    difference_type rj = static_cast<difference_type>(bk.size()) - 1;
    difference_type w = static_cast<difference_type>(oldk + nnew) - 1;
    while (rj >= 0) {
        if (ri >= 0 && bk[rj] < c.keys[ri]) { // old is larger
            c.keys[w] = std::move(c.keys[ri]);
            c.values[w] = std::move(c.values[ri]);
            --ri;
            --w;
        } else if (ri >= 0 && !(c.keys[ri] < bk[rj])) { // equal: old wins
            c.keys[w] = std::move(c.keys[ri]);
            c.values[w] = std::move(c.values[ri]);
            --ri;
            --rj;
            --w;
        } else { // batch is larger
            c.keys[w] = std::move(bk[rj]);
            c.values[w] = std::move(bv[rj]);
            --rj;
            --w;
        }
    }
    assert(w == ri); // remaining prefix already in place
    assert(detail::is_sorted_unique(c.keys));
}

// ---- construct ----

template <class Key, class T>
flat_map<Key, T>::flat_map(key_container_type keys, mapped_container_type values) {
    assert(keys.size() == values.size());
    std::vector<size_type> perm(keys.size());
    std::iota(perm.begin(), perm.end(), size_type{0});
    std::stable_sort(perm.begin(), perm.end(),
                     [&](size_type a, size_type b) { return keys[a] < keys[b]; });
    c.keys.reserve(keys.size());
    c.values.reserve(values.size());
    for (size_type p : perm) { // apply perm; keep first of equals
        if (!c.keys.empty() && !(c.keys.back() < keys[p]))
            continue;
        c.keys.push_back(std::move(keys[p]));
        c.values.push_back(std::move(values[p]));
    }
}

template <class Key, class T>
flat_map<Key, T>::flat_map(sorted_unique_t, key_container_type keys, mapped_container_type values)
    : c{std::move(keys), std::move(values)} {
    assert(c.keys.size() == c.values.size());
    assert(detail::is_sorted_unique(c.keys));
}

// ---- element access ----

template <class Key, class T>
auto flat_map<Key, T>::operator[](const key_type &v) -> mapped_type & {
    size_type i = rank_of(v);
    if (i == c.keys.size() || v < c.keys[i]) {
        c.keys.insert(c.keys.begin() + static_cast<difference_type>(i), v);
        c.values.insert(c.values.begin() + static_cast<difference_type>(i), mapped_type{});
    }
    return c.values[i];
}

template <class Key, class T>
auto flat_map<Key, T>::at(const key_type &v) -> mapped_type & {
    size_type i = rank_of(v);
    if (i == c.keys.size() || v < c.keys[i])
        throw std::out_of_range("flat_map::at");
    return c.values[i];
}

template <class Key, class T>
auto flat_map<Key, T>::at(const key_type &v) const -> const mapped_type & {
    size_type i = rank_of(v);
    if (i == c.keys.size() || v < c.keys[i])
        throw std::out_of_range("flat_map::at");
    return c.values[i];
}

// ---- modifiers ----

template <class Key, class T>
auto flat_map<Key, T>::insert(const value_type &kv) -> std::pair<iterator, bool> {
    size_type i = rank_of(kv.first);
    if (i < c.keys.size() && !(kv.first < c.keys[i]))
        return {begin() + static_cast<difference_type>(i), false};
    c.keys.insert(c.keys.begin() + static_cast<difference_type>(i), kv.first);
    c.values.insert(c.values.begin() + static_cast<difference_type>(i), kv.second);
    return {begin() + static_cast<difference_type>(i), true};
}

template <class Key, class T>
template <std::input_iterator It>
void flat_map<Key, T>::insert(It first, It last) {
    key_container_type bk;
    mapped_container_type bv;
    for (; first != last; ++first) {
        value_type kv = *first;
        bk.push_back(std::move(kv.first));
        bv.push_back(std::move(kv.second));
    }
    std::vector<size_type> perm(bk.size());
    std::iota(perm.begin(), perm.end(), size_type{0});
    std::stable_sort(perm.begin(), perm.end(),
                     [&](size_type a, size_type b) { return bk[a] < bk[b]; });
    key_container_type sk;
    mapped_container_type sv;
    sk.reserve(bk.size());
    sv.reserve(bv.size());
    for (size_type p : perm) { // dedupe within batch: first wins
        if (!sk.empty() && !(sk.back() < bk[p]))
            continue;
        sk.push_back(std::move(bk[p]));
        sv.push_back(std::move(bv[p]));
    }
    merge_sorted_vectors(sk, sv);
}

template <class Key, class T>
template <std::input_iterator It>
void flat_map<Key, T>::insert(sorted_unique_t, It first, It last) {
    key_container_type bk;
    mapped_container_type bv;
    for (; first != last; ++first) {
        value_type kv = *first;
        bk.push_back(std::move(kv.first));
        bv.push_back(std::move(kv.second));
    }
    merge_sorted_vectors(bk, bv);
}

template <class Key, class T>
auto flat_map<Key, T>::extract() && -> containers {
    containers out = std::move(c);
    c.keys.clear();
    c.values.clear();
    return out;
}

template <class Key, class T>
void flat_map<Key, T>::replace(key_container_type &&keys, mapped_container_type &&values) {
    assert(keys.size() == values.size());
    assert(detail::is_sorted_unique(keys));
    c.keys = std::move(keys);
    c.values = std::move(values);
}

template <class Key, class T>
auto flat_map<Key, T>::erase(const key_type &v) -> size_type {
    size_type i = rank_of(v);
    if (i == c.keys.size() || v < c.keys[i])
        return 0;
    c.keys.erase(c.keys.begin() + static_cast<difference_type>(i));
    c.values.erase(c.values.begin() + static_cast<difference_type>(i));
    return 1;
}

template <class Key, class T>
auto flat_map<Key, T>::erase(const_iterator pos) -> iterator {
    difference_type i = pos - cbegin();
    c.keys.erase(c.keys.begin() + i);
    c.values.erase(c.values.begin() + i);
    return begin() + i;
}

template <class Key, class T>
void flat_map<Key, T>::clear() noexcept {
    c.keys.clear();
    c.values.clear();
}

// ---- lookup ----

template <class Key, class T>
auto flat_map<Key, T>::find(const key_type &v) -> iterator {
    size_type i = rank_of(v);
    if (i == c.keys.size() || v < c.keys[i])
        return end();
    return begin() + static_cast<difference_type>(i);
}

template <class Key, class T>
auto flat_map<Key, T>::find(const key_type &v) const -> const_iterator {
    size_type i = rank_of(v);
    if (i == c.keys.size() || v < c.keys[i])
        return end();
    return begin() + static_cast<difference_type>(i);
}

template <class Key, class T>
bool flat_map<Key, T>::contains(const key_type &v) const {
    size_type i = rank_of(v);
    return i < c.keys.size() && !(v < c.keys[i]);
}

template <class Key, class T>
auto flat_map<Key, T>::lower_bound(const key_type &v) -> iterator {
    return begin() + static_cast<difference_type>(rank_of(v));
}

template <class Key, class T>
auto flat_map<Key, T>::lower_bound(const key_type &v) const -> const_iterator {
    return begin() + static_cast<difference_type>(rank_of(v));
}

// ---- free functions ----

template <class Key, class T, class Pred>
typename flat_map<Key, T>::size_type erase_if(flat_map<Key, T> &m, Pred pred) {
    auto &ks = m.c.keys;
    auto &vs = m.c.values;
    typename flat_map<Key, T>::size_type w = 0;
    for (typename flat_map<Key, T>::size_type i = 0; i < ks.size(); ++i) {
        typename flat_map<Key, T>::const_reference r{ks[i], vs[i]};
        if (pred(r))
            continue;
        if (w != i) {
            ks[w] = std::move(ks[i]);
            vs[w] = std::move(vs[i]);
        }
        ++w;
    }
    auto removed = ks.size() - w;
    ks.resize(w);
    vs.resize(w);
    return removed;
}

} // namespace Aux

#endif // NETWORKIT_AUXILIARY_FLAT_MAP_TPP_
