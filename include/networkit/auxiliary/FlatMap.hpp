/*
 * FlatMap.hpp
 *
 *  Created on: 07.11.2026
 *      Author: Ian Chen (ianchen3@illinois.edu)
 */

#ifndef NETWORKIT_AUXILIARY_FLAT_MAP_HPP_
#define NETWORKIT_AUXILIARY_FLAT_MAP_HPP_

#include <cassert>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace Aux {

/**
 * @ingroup Aux
 * A subset of std::flat_map in the C++23 standard.
 *
 * Resources:
 * https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p0429r6.pdf
 * https://en.cppreference.com/cpp/header/flat_map
 *
 */

struct sorted_unique_t {
    explicit sorted_unique_t() = default;
};
inline constexpr sorted_unique_t sorted_unique{};

template <class Key, class T>
class flat_map {
public:
    using key_container_type = std::vector<Key>;
    using mapped_container_type = std::vector<T>;
    using key_type = Key;
    using mapped_type = T;
    using value_type = std::pair<key_type, mapped_type>;
    using reference = std::pair<const key_type &, mapped_type &>;
    using const_reference = std::pair<const key_type &, const mapped_type &>;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    struct containers {
        key_container_type keys;
        mapped_container_type values;
    };

private:
    containers c;

    template <bool Const>
    class basic_iterator {
        using KPtr = const Key *;
        using VPtr = std::conditional_t<Const, const T *, T *>;
        KPtr k = nullptr;
        VPtr v = nullptr;

        friend flat_map;
        template <bool>
        friend class basic_iterator;
        basic_iterator(KPtr k_, VPtr v_) : k(k_), v(v_) {}

    public:
        using iterator_concept = std::random_access_iterator_tag;
        using iterator_category = std::input_iterator_tag;
        using value_type = flat_map::value_type;
        using reference = std::conditional_t<Const, const_reference, flat_map::reference>;
        using difference_type = flat_map::difference_type;

        basic_iterator() = default;
        basic_iterator(const basic_iterator<!Const> &o)
            requires Const
            : k(o.k), v(o.v) {}

        reference operator*() const { return {*k, *v}; }
        reference operator[](difference_type n) const { return {k[n], v[n]}; }

        struct arrow_proxy {
            reference r;
            reference *operator->() { return &r; }
        };
        arrow_proxy operator->() const { return {**this}; }

        basic_iterator &operator++() {
            ++k;
            ++v;
            return *this;
        }
        basic_iterator operator++(int) {
            auto t = *this;
            ++*this;
            return t;
        }
        basic_iterator &operator--() {
            --k;
            --v;
            return *this;
        }
        basic_iterator operator--(int) {
            auto t = *this;
            --*this;
            return t;
        }
        basic_iterator &operator+=(difference_type n) {
            k += n;
            v += n;
            return *this;
        }
        basic_iterator &operator-=(difference_type n) {
            k -= n;
            v -= n;
            return *this;
        }

        friend basic_iterator operator+(basic_iterator i, difference_type n) { return i += n; }
        friend basic_iterator operator+(difference_type n, basic_iterator i) { return i += n; }
        friend basic_iterator operator-(basic_iterator i, difference_type n) { return i -= n; }
        friend difference_type operator-(basic_iterator a, basic_iterator b) { return a.k - b.k; }
        friend bool operator==(basic_iterator a, basic_iterator b) { return a.k == b.k; }
        friend auto operator<=>(basic_iterator a, basic_iterator b) { return a.k <=> b.k; }
    };

    size_type rank_of(const key_type &v) const;
    void merge_sorted_vectors(key_container_type &bk, mapped_container_type &bv);

public:
    using iterator = basic_iterator<false>;
    using const_iterator = basic_iterator<true>;

    // construct
    flat_map() = default;
    flat_map(key_container_type keys, mapped_container_type values);
    flat_map(sorted_unique_t, key_container_type keys, mapped_container_type values);

    // element access
    mapped_type &operator[](const key_type &v);
    mapped_type &at(const key_type &v);
    const mapped_type &at(const key_type &v) const;

    // iterators
    iterator begin() noexcept { return {c.keys.data(), c.values.data()}; }
    const_iterator begin() const noexcept { return {c.keys.data(), c.values.data()}; }
    iterator end() noexcept { return begin() + static_cast<difference_type>(size()); }
    const_iterator end() const noexcept { return begin() + static_cast<difference_type>(size()); }
    const_iterator cbegin() const noexcept { return begin(); }
    const_iterator cend() const noexcept { return end(); }

    // capacity
    [[nodiscard]] bool empty() const noexcept { return c.keys.empty(); }
    size_type size() const noexcept { return c.keys.size(); }

    // modifiers
    std::pair<iterator, bool> insert(const value_type &kv);
    template <std::input_iterator It>
    void insert(It first, It last);
    template <std::input_iterator It>
    void insert(sorted_unique_t, It first, It last);
    containers extract() &&;
    void replace(key_container_type &&keys, mapped_container_type &&values);

    size_type erase(const key_type &v);
    iterator erase(const_iterator pos);
    void clear() noexcept;

    // lookup
    iterator find(const key_type &v);
    const_iterator find(const key_type &v) const;
    bool contains(const key_type &v) const;
    iterator lower_bound(const key_type &v);
    const_iterator lower_bound(const key_type &v) const;

    // observers
    const key_container_type &keys() const noexcept { return c.keys; }
    const mapped_container_type &values() const noexcept { return c.values; }

    template <class K, class V, class Pred>
    friend typename flat_map<K, V>::size_type erase_if(flat_map<K, V> &m, Pred pred);
};

template <class Key, class T, class Pred>
typename flat_map<Key, T>::size_type erase_if(flat_map<Key, T> &m, Pred pred);

} /* namespace Aux */

#endif // NETWORKIT_AUXILIARY_FLAT_MAP_HPP_

#include "FlatMap.tpp"
