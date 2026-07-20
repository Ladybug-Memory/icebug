/*
 * FlatMapGTest.cpp
 *
 *  Created on: 19.07.2026
 *      Author: Ian Chen (ianchen3@illinois.edu)
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include <networkit/auxiliary/FlatMap.hpp>

namespace NetworKit {

class FlatMapGTest : public testing::Test {};

// ============================================================================
// Construction & Capacity
// ============================================================================

TEST_F(FlatMapGTest, testDefaultConstruction) {
    Aux::flat_map<int, double> m;
    EXPECT_TRUE(m.empty());
    EXPECT_EQ(m.size(), 0u);
    EXPECT_EQ(m.begin(), m.end());
    EXPECT_EQ(m.cbegin(), m.cend());
}

TEST_F(FlatMapGTest, testSortedUniqueConstruction) {
    Aux::flat_map<int, std::string>::containers data{{1, 3, 5}, {"a", "b", "c"}};
    Aux::flat_map<int, std::string> m(Aux::sorted_unique, std::move(data.keys),
                                      std::move(data.values));

    EXPECT_EQ(m.size(), 3u);
    EXPECT_EQ(m.at(1), "a");
    EXPECT_EQ(m.at(3), "b");
    EXPECT_EQ(m.at(5), "c");
}

TEST_F(FlatMapGTest, testUnsortedConstruction) {
    // keys 5,3,1 are out of order; 3 appears twice
    Aux::flat_map<int, std::string>::containers data{{5, 3, 1, 3}, {"e", "d", "a", "c"}};
    Aux::flat_map<int, std::string> m(std::move(data.keys), std::move(data.values));

    EXPECT_EQ(m.size(), 3u); // duplicate 3 deduped; first occurrence wins
    EXPECT_EQ(m.at(1), "a");
    EXPECT_EQ(m.at(3), "d"); // first occurrence wins
    EXPECT_EQ(m.at(5), "e");
}

// ============================================================================
// Element Access
// ============================================================================

TEST_F(FlatMapGTest, testOperatorSquareBrackets) {
    Aux::flat_map<int, double> m;
    m[1] = 1.5;
    m[3] = 3.5;
    m[2] = 2.5; // inserted in middle

    EXPECT_EQ(m.size(), 3u);
    EXPECT_DOUBLE_EQ(m[1], 1.5);
    EXPECT_DOUBLE_EQ(m[2], 2.5);
    EXPECT_DOUBLE_EQ(m[3], 3.5);
}

TEST_F(FlatMapGTest, testAtAccess) {
    Aux::flat_map<int, double> m;
    m[10] = 100.0;

    EXPECT_DOUBLE_EQ(m.at(10), 100.0);
    EXPECT_THROW(m.at(42), std::out_of_range);

    const auto &cm = m;
    EXPECT_DOUBLE_EQ(cm.at(10), 100.0);
    EXPECT_THROW(cm.at(42), std::out_of_range);
}

// ============================================================================
// Modifiers — Single Insert
// ============================================================================

TEST_F(FlatMapGTest, testInsertSingle) {
    Aux::flat_map<int, std::string> m;

    auto [it, inserted] = m.insert({1, "one"});
    EXPECT_TRUE(inserted);
    EXPECT_EQ(it->first, 1);
    EXPECT_EQ(it->second, "one");
    EXPECT_EQ(m.size(), 1u);

    // Insert existing key — should fail
    auto [it2, inserted2] = m.insert({1, "ONE"});
    EXPECT_FALSE(inserted2);
    EXPECT_EQ(it2->first, 1);
    EXPECT_EQ(it2->second, "one"); // unchanged
    EXPECT_EQ(m.size(), 1u);
}

TEST_F(FlatMapGTest, testInsertMultipleKeysInOrder) {
    Aux::flat_map<int, int> m;
    for (int i = 0; i < 100; ++i) {
        auto [it, inserted] = m.insert({i, i * 2});
        EXPECT_TRUE(inserted);
        EXPECT_EQ(it->second, i * 2);
    }
    EXPECT_EQ(m.size(), 100u);
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(m.at(i), i * 2);
    }
}

TEST_F(FlatMapGTest, testInsertMultipleKeysReverseOrder) {
    Aux::flat_map<int, int> m;
    for (int i = 99; i >= 0; --i) {
        auto [it, inserted] = m.insert({i, i * 3});
        EXPECT_TRUE(inserted);
        EXPECT_EQ(it->second, i * 3);
    }
    EXPECT_EQ(m.size(), 100u);
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(m.at(i), i * 3);
    }
}

// ============================================================================
// Modifiers — Range Insert
// ============================================================================

TEST_F(FlatMapGTest, testRangeInsert) {
    Aux::flat_map<int, std::string> m;
    m[1] = "one";
    m[4] = "four";

    std::vector<std::pair<int, std::string>> batch = {{2, "two"}, {3, "three"}, {5, "five"}};
    m.insert(batch.begin(), batch.end());

    EXPECT_EQ(m.size(), 5u);
    EXPECT_EQ(m.at(1), "one"); // existing unchanged
    EXPECT_EQ(m.at(2), "two");
    EXPECT_EQ(m.at(3), "three");
    EXPECT_EQ(m.at(4), "four"); // existing unchanged
    EXPECT_EQ(m.at(5), "five");
}

TEST_F(FlatMapGTest, testRangeInsertWithOverlap) {
    Aux::flat_map<int, std::string> m;
    m[1] = "original_one";
    m[3] = "original_three";

    // batch has an existing key (1) and a new one (2) — existing should win
    std::vector<std::pair<int, std::string>> batch = {{1, "new_one"}, {2, "two"}};
    m.insert(batch.begin(), batch.end());

    EXPECT_EQ(m.size(), 3u);
    EXPECT_EQ(m.at(1), "original_one"); // existing wins
    EXPECT_EQ(m.at(2), "two");
    EXPECT_EQ(m.at(3), "original_three");
}

TEST_F(FlatMapGTest, testRangeInsertSortedUnique) {
    Aux::flat_map<int, int> m;
    m[2] = 20;

    std::vector<std::pair<int, int>> batch = {{1, 10}, {3, 30}};
    m.insert(Aux::sorted_unique, batch.begin(), batch.end());

    EXPECT_EQ(m.size(), 3u);
    EXPECT_EQ(m.at(1), 10);
    EXPECT_EQ(m.at(2), 20);
    EXPECT_EQ(m.at(3), 30);
}

TEST_F(FlatMapGTest, testRangeInsertSortedUniqueWithDuplicates) {
    Aux::flat_map<int, int> m;
    m[1] = 100;

    std::vector<std::pair<int, int>> batch = {{1, 200}, {2, 300}};
    m.insert(Aux::sorted_unique, batch.begin(), batch.end());

    EXPECT_EQ(m.size(), 2u);
    EXPECT_EQ(m.at(1), 100); // existing wins
    EXPECT_EQ(m.at(2), 300);
}

// ============================================================================
// Modifiers — Erase
// ============================================================================

TEST_F(FlatMapGTest, testEraseByKey) {
    Aux::flat_map<int, double> m;
    m[1] = 1.0;
    m[2] = 2.0;
    m[3] = 3.0;

    EXPECT_EQ(m.erase(2), 1u);
    EXPECT_EQ(m.size(), 2u);
    EXPECT_FALSE(m.contains(2));
    EXPECT_TRUE(m.contains(1));
    EXPECT_TRUE(m.contains(3));

    // Erase non-existent key
    EXPECT_EQ(m.erase(42), 0u);
    EXPECT_EQ(m.size(), 2u);
}

TEST_F(FlatMapGTest, testEraseByIterator) {
    Aux::flat_map<int, int> m;
    m[1] = 10;
    m[2] = 20;
    m[3] = 30;

    auto it = m.find(2);
    auto next = m.erase(it);
    EXPECT_EQ(next->first, 3);
    EXPECT_EQ(next->second, 30);
    EXPECT_EQ(m.size(), 2u);
    EXPECT_FALSE(m.contains(2));

    // Erase first element
    next = m.erase(m.begin());
    EXPECT_EQ(next->first, 3);
    EXPECT_EQ(next->second, 30);
    EXPECT_EQ(m.size(), 1u);
}

TEST_F(FlatMapGTest, testEraseFirstAndLast) {
    Aux::flat_map<int, int> m;
    m[1] = 10;
    m[2] = 20;
    m[3] = 30;

    // Erase first
    m.erase(m.begin());
    EXPECT_EQ(m.size(), 2u);
    EXPECT_EQ(m.begin()->first, 2);

    // Erase last
    auto it = m.find(3);
    m.erase(it);
    EXPECT_EQ(m.size(), 1u);
    EXPECT_EQ(m.begin()->first, 2);
}

TEST_F(FlatMapGTest, testEraseAllElements) {
    Aux::flat_map<int, int> m;
    m[1] = 10;
    m[2] = 20;

    m.erase(m.find(1));
    EXPECT_EQ(m.size(), 1u);
    m.erase(m.find(2));
    EXPECT_TRUE(m.empty());
    EXPECT_EQ(m.begin(), m.end());
}

// ============================================================================
// Modifiers — Clear
// ============================================================================

TEST_F(FlatMapGTest, testClear) {
    Aux::flat_map<int, double> m;
    m[1] = 1.1;
    m[2] = 2.2;
    m.clear();

    EXPECT_TRUE(m.empty());
    EXPECT_EQ(m.size(), 0u);
    EXPECT_EQ(m.begin(), m.end());
    EXPECT_FALSE(m.contains(1));
    EXPECT_FALSE(m.contains(2));
}

// ============================================================================
// Lookup
// ============================================================================

TEST_F(FlatMapGTest, testFind) {
    Aux::flat_map<int, std::string> m;
    m[5] = "five";

    auto it = m.find(5);
    EXPECT_NE(it, m.end());
    EXPECT_EQ(it->first, 5);
    EXPECT_EQ(it->second, "five");

    EXPECT_EQ(m.find(42), m.end());

    // Const version
    const auto &cm = m;
    auto cit = cm.find(5);
    EXPECT_NE(cit, cm.cend());
    EXPECT_EQ(cit->second, "five");
    EXPECT_EQ(cm.find(42), cm.cend());
}

TEST_F(FlatMapGTest, testContains) {
    Aux::flat_map<int, double> m;
    m[0] = 0.0;
    m[100] = 100.0;

    EXPECT_TRUE(m.contains(0));
    EXPECT_TRUE(m.contains(100));
    EXPECT_FALSE(m.contains(-1));
    EXPECT_FALSE(m.contains(50));
}

TEST_F(FlatMapGTest, testLowerBound) {
    Aux::flat_map<int, int> m;
    m[10] = 100;
    m[20] = 200;
    m[30] = 300;

    // Exact match
    auto it = m.lower_bound(20);
    EXPECT_EQ(it->first, 20);

    // Between keys — returns next higher
    it = m.lower_bound(15);
    EXPECT_EQ(it->first, 20);

    // Past the end
    it = m.lower_bound(40);
    EXPECT_EQ(it, m.end());

    // Before first
    it = m.lower_bound(5);
    EXPECT_EQ(it->first, 10);
}

// ============================================================================
// Modifiers — Extract & Replace
// ============================================================================

TEST_F(FlatMapGTest, testExtractAndReplace) {
    Aux::flat_map<int, double> m;
    m[1] = 1.5;
    m[2] = 2.5;

    auto containers = std::move(m).extract();
    EXPECT_EQ(containers.keys.size(), 2u);
    EXPECT_EQ(containers.values.size(), 2u);
    EXPECT_TRUE(m.empty()); // post-extract

    // Replace with new data
    m.replace(std::move(containers.keys), std::move(containers.values));
    EXPECT_EQ(m.size(), 2u);
    EXPECT_DOUBLE_EQ(m.at(1), 1.5);
    EXPECT_DOUBLE_EQ(m.at(2), 2.5);
}

TEST_F(FlatMapGTest, testReplaceValidatesSortedUnique) {
    Aux::flat_map<int, int> m;
    // replace with empty — valid
    std::vector<int> empty_keys;
    std::vector<int> empty_values;
    m.replace(std::move(empty_keys), std::move(empty_values));
    EXPECT_TRUE(m.empty());
}

// ============================================================================
// Modifiers — erase_if
// ============================================================================

TEST_F(FlatMapGTest, testEraseIf) {
    Aux::flat_map<int, int> m;
    for (int i = 0; i < 10; ++i)
        m[i] = i * 10;

    // Erase all even keys
    auto removed = Aux::erase_if(m, [](const auto &kv) { return kv.first % 2 == 0; });

    EXPECT_EQ(removed, 5u);
    EXPECT_EQ(m.size(), 5u);
    for (int i = 0; i < 10; ++i) {
        if (i % 2 == 0)
            EXPECT_FALSE(m.contains(i));
        else
            EXPECT_TRUE(m.contains(i));
    }
}

TEST_F(FlatMapGTest, testEraseIfNothingRemoved) {
    Aux::flat_map<int, int> m;
    m[1] = 10;
    m[2] = 20;

    auto removed = Aux::erase_if(m, [](const auto &) { return false; });
    EXPECT_EQ(removed, 0u);
    EXPECT_EQ(m.size(), 2u);
}

TEST_F(FlatMapGTest, testEraseIfAllRemoved) {
    Aux::flat_map<int, int> m;
    m[1] = 10;
    m[2] = 20;

    auto removed = Aux::erase_if(m, [](const auto &) { return true; });
    EXPECT_EQ(removed, 2u);
    EXPECT_TRUE(m.empty());
}

// ============================================================================
// Iterators
// ============================================================================

TEST_F(FlatMapGTest, testIteratorTraversal) {
    Aux::flat_map<int, int> m;
    m[1] = 100;
    m[2] = 200;
    m[3] = 300;

    std::vector<int> keys_seen;
    std::vector<int> values_seen;
    for (auto it = m.begin(); it != m.end(); ++it) {
        keys_seen.push_back(it->first);
        values_seen.push_back(it->second);
    }

    EXPECT_EQ(keys_seen, (std::vector<int>{1, 2, 3}));
    EXPECT_EQ(values_seen, (std::vector<int>{100, 200, 300}));
}

TEST_F(FlatMapGTest, testIteratorRandomAccess) {
    Aux::flat_map<int, int> m;
    for (int i = 0; i < 5; ++i)
        m[i] = i * 10;

    auto it = m.begin();

    // operator[]
    EXPECT_EQ(it[0].first, 0);
    EXPECT_EQ(it[0].second, 0);
    EXPECT_EQ(it[3].first, 3);
    EXPECT_EQ(it[3].second, 30);

    // operator+
    auto it3 = it + 3;
    EXPECT_EQ(it3->first, 3);
    EXPECT_EQ(it3->second, 30);

    // operator-
    auto it_back = it3 - 2;
    EXPECT_EQ(it_back->first, 1);

    // difference
    EXPECT_EQ(m.end() - m.begin(), 5);

    // operator+=(n), operator-=(n)
    auto it2 = m.begin();
    it2 += 2;
    EXPECT_EQ(it2->first, 2);
    it2 -= 1;
    EXPECT_EQ(it2->first, 1);
}

TEST_F(FlatMapGTest, testIteratorComparison) {
    Aux::flat_map<int, int> m;
    m[1] = 10;
    m[2] = 20;

    auto a = m.begin();
    auto b = m.begin();
    EXPECT_EQ(a, b);
    EXPECT_FALSE(a != b);

    ++b;
    EXPECT_NE(a, b);
    EXPECT_LT(a, b);
    EXPECT_LE(a, b);
    EXPECT_GT(b, a);
    EXPECT_GE(b, a);
}

TEST_F(FlatMapGTest, testIteratorPrePostIncrement) {
    Aux::flat_map<int, int> m;
    m[1] = 10;
    m[2] = 20;

    auto it = m.begin();
    EXPECT_EQ(it->first, 1);

    auto old = it++;
    EXPECT_EQ(old->first, 1);
    EXPECT_EQ(it->first, 2);

    auto &ref = ++it;
    EXPECT_EQ(ref, m.end());
}

TEST_F(FlatMapGTest, testIteratorPrePostDecrement) {
    Aux::flat_map<int, int> m;
    m[1] = 10;
    m[2] = 20;
    m[3] = 30;

    auto it = m.end();
    --it;
    EXPECT_EQ(it->first, 3);
    auto old = it--;
    EXPECT_EQ(old->first, 3);
    EXPECT_EQ(it->first, 2);
}

TEST_F(FlatMapGTest, testReverseTraversal) {
    Aux::flat_map<int, int> m;
    m[1] = 10;
    m[2] = 20;
    m[3] = 30;

    std::vector<int> keys;
    for (auto it = m.end(); it != m.begin();) {
        --it;
        keys.push_back(it->first);
    }
    EXPECT_EQ(keys, (std::vector<int>{3, 2, 1}));
}

TEST_F(FlatMapGTest, testConstIterator) {
    Aux::flat_map<int, std::string> m;
    m[1] = "a";
    m[2] = "b";

    const auto &cm = m;
    auto cit = cm.cbegin();
    EXPECT_EQ(cit->first, 1);
    EXPECT_EQ(cit->second, "a");
    ++cit;
    EXPECT_EQ(cit->second, "b");

    // Const iterator from find
    auto found = cm.find(2);
    EXPECT_NE(found, cm.cend());
    EXPECT_EQ(found->second, "b");

    // Non-const to const conversion
    Aux::flat_map<int, std::string>::const_iterator it = m.begin();
    EXPECT_EQ(it->first, 1);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(FlatMapGTest, testEmptyMapOperations) {
    Aux::flat_map<int, double> m;

    EXPECT_TRUE(m.empty());
    EXPECT_EQ(m.find(1), m.end());
    EXPECT_FALSE(m.contains(1));
    EXPECT_EQ(m.erase(1), 0u);
    EXPECT_EQ(m.lower_bound(1), m.end());
    EXPECT_THROW(m.at(1), std::out_of_range);
    EXPECT_NO_THROW(m.clear());
}

TEST_F(FlatMapGTest, testSingleElementMap) {
    Aux::flat_map<int, int> m;
    m[42] = 100;

    EXPECT_EQ(m.size(), 1u);
    EXPECT_EQ(m.at(42), 100);
    EXPECT_EQ(m.find(42)->second, 100);
    EXPECT_EQ(m.begin()->first, 42);

    // Erase the only element
    m.erase(m.begin());
    EXPECT_TRUE(m.empty());
}

TEST_F(FlatMapGTest, testLargeInsertionInterleaved) {
    // Insert keys in a pattern that exercises the backward merge
    Aux::flat_map<int, int> m;

    for (int i = 0; i < 50; ++i)
        m[i] = i;

    // Insert a batch where some keys overlap
    std::vector<std::pair<int, int>> batch;
    for (int i = 25; i < 75; ++i)
        batch.emplace_back(i, i * 10); // 25-49 overlap with existing (existing should win)
    m.insert(batch.begin(), batch.end());

    EXPECT_EQ(m.size(), 75u);
    for (int i = 0; i < 25; ++i)
        EXPECT_EQ(m.at(i), i); // existing, unchanged
    for (int i = 25; i < 50; ++i)
        EXPECT_EQ(m.at(i), i); // existing wins over batch's i*10
    for (int i = 50; i < 75; ++i)
        EXPECT_EQ(m.at(i), i * 10); // batch inserted
}

TEST_F(FlatMapGTest, testStringKeys) {
    Aux::flat_map<std::string, int> m;
    m["alpha"] = 1;
    m["beta"] = 2;
    m["gamma"] = 3;

    EXPECT_EQ(m.size(), 3u);
    EXPECT_EQ(m.at("alpha"), 1);
    EXPECT_EQ(m.at("beta"), 2);
    EXPECT_EQ(m.at("gamma"), 3);

    auto it = m.find("beta");
    EXPECT_NE(it, m.end());
    EXPECT_EQ(it->second, 2);
    EXPECT_EQ(m.find("delta"), m.end());
}

TEST_F(FlatMapGTest, testArrowOperator) {
    Aux::flat_map<int, std::string> m;
    m[1] = "hello";

    auto it = m.begin();
    // operator-> returns arrow_proxy; access through ->
    EXPECT_EQ(it->first, 1);
    EXPECT_EQ(it->second, "hello");
}

TEST_F(FlatMapGTest, testKeysAndValuesObservers) {
    Aux::flat_map<int, std::string> m;
    m[3] = "three";
    m[1] = "one";
    m[2] = "two";

    const auto &ks = m.keys();
    const auto &vs = m.values();

    EXPECT_EQ(ks.size(), 3u);
    EXPECT_EQ(vs.size(), 3u);

    // Should be sorted by key
    EXPECT_EQ(ks[0], 1);
    EXPECT_EQ(ks[1], 2);
    EXPECT_EQ(ks[2], 3);

    EXPECT_EQ(vs[0], "one");
    EXPECT_EQ(vs[1], "two");
    EXPECT_EQ(vs[2], "three");
}

TEST_F(FlatMapGTest, testIteratorValueMutation) {
    // Iterator should allow mutation of values (but not keys)
    Aux::flat_map<int, int> m;
    m[1] = 100;

    auto it = m.begin();
    // it->first is const, so this won't compile:
    // it->first = 42;
    it->second = 999;

    EXPECT_EQ(m.at(1), 999);
}

TEST_F(FlatMapGTest, testReferenceSemantics) {
    Aux::flat_map<int, int> m;
    m[1] = 10;

    // operator* returns reference (pair of references)
    auto ref = *m.begin();
    ref.second = 20;
    EXPECT_EQ(m.at(1), 20);
}

} // namespace NetworKit
