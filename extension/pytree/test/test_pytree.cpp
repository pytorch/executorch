/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/pytree/pytree.h>

#include <gtest/gtest.h>
#include <string>

using ::executorch::extension::pytree::ContainerHandle;
using ::executorch::extension::pytree::Key;
using ::executorch::extension::pytree::Kind;
using ::executorch::extension::pytree::unflatten;

using Leaf = int32_t;

TEST(PyTreeTest, List) {
  Leaf items[2] = {11, 12};
  std::string spec = "L2#1#1($,$)";
  auto c = unflatten(spec, items);
  ASSERT_TRUE(c.isList());
  ASSERT_EQ(c.size(), 2);

  const auto& child0 = c[0];
  const auto& child1 = c[1];

  ASSERT_TRUE(child0.isLeaf());
  ASSERT_TRUE(child1.isLeaf());
  ASSERT_EQ(child0, 11);
  ASSERT_EQ(child1, 12);
}

TEST(PyTreeTest, Tuple) {
  std::string spec = "T1#1($)";
  Leaf items[1] = {11};
  auto c = unflatten(spec, items);
  ASSERT_TRUE(c.isTuple());
  ASSERT_EQ(c.size(), 1);

  const auto& child0 = c[0];

  ASSERT_TRUE(child0.isLeaf());
  ASSERT_EQ(child0, 11);
}

TEST(PyTreeTest, Dict) {
  std::string spec = "D2#1#1('key0':$,'key1':$)";
  Leaf items[2] = {11, 12};
  auto c = unflatten(spec, items);
  ASSERT_TRUE(c.isDict());
  ASSERT_EQ(c.size(), 2);

  const auto& key0 = c.key(0);
  const auto& key1 = c.key(1);

  ASSERT_TRUE(key0 == Key("key0"));
  ASSERT_TRUE(key1 == Key("key1"));

  const auto& child0 = c[0];
  const auto& child1 = c[1];
  ASSERT_TRUE(child0.isLeaf());
  ASSERT_TRUE(child1.isLeaf());
  ASSERT_EQ(child0, 11);
  ASSERT_EQ(child1, 12);

  const auto& ckey0 = c.at("key0");
  ASSERT_EQ(child0, ckey0);

  ASSERT_EQ(c.at("key0"), 11);
  ASSERT_EQ(c.at("key1"), 12);
}

TEST(PyTreeTest, Leaf) {
  std::string spec = "$";
  Leaf items[2] = {11};
  auto c = unflatten(spec, items);
  ASSERT_TRUE(c.isLeaf());
  ASSERT_EQ(c, 11);
}

TEST(PyTreeTest, DictWithList) {
  std::string spec = "D2#2#1('key0':L2#1#1($,$),'key1':$)";
  Leaf items[3] = {11, 12, 13};
  auto c = unflatten(spec, items);
  ASSERT_TRUE(c.isDict());
  ASSERT_EQ(c.size(), 2);

  const auto& key0 = c.key(0);
  const auto& key1 = c.key(1);

  ASSERT_TRUE(key0 == Key("key0"));
  ASSERT_TRUE(key1 == Key("key1"));

  const auto& child1 = c[1];
  ASSERT_TRUE(child1.isLeaf());
  ASSERT_EQ(child1, 13);

  const auto& list = c[0];
  ASSERT_TRUE(list.isList());
  ASSERT_EQ(list.size(), 2);

  const auto& list_child0 = list[0];
  const auto& list_child1 = list[1];

  ASSERT_TRUE(list_child0.isLeaf());
  ASSERT_TRUE(list_child1.isLeaf());

  ASSERT_EQ(list_child0, 11);
  ASSERT_EQ(list_child1, 12);
}

TEST(PyTreeTest, DictKeysStrInt) {
  std::string spec = "D4#1#1#1#1('key0':$,1:$,23:$,123:$)";
  Leaf items[4] = {11, 12, 13, 14};
  auto c = unflatten(spec, items);
  ASSERT_TRUE(c.isDict());
  ASSERT_EQ(c.size(), 4);

  const auto& key0 = c.key(0);
  const auto& key1 = c.key(1);

  ASSERT_TRUE(key0 == Key("key0"));
  ASSERT_TRUE(key1 == Key(1));

  const auto& child0 = c[0];
  const auto& child1 = c[1];
  ASSERT_TRUE(child0.isLeaf());
  ASSERT_TRUE(child1.isLeaf());
  ASSERT_EQ(child0, 11);
  ASSERT_EQ(child1, 12);

  const auto& ckey0 = c.at("key0");
  ASSERT_EQ(child0, ckey0);

  ASSERT_EQ(c.at(1), 12);
  ASSERT_EQ(c.at(23), 13);
  ASSERT_EQ(c.at(123), 14);
}

TEST(pytree, FlattenDict) {
  Leaf items[3] = {11, 12, 13};
  auto c = ContainerHandle<Leaf>(Kind::Dict, 3);
  c[0] = &items[0];
  c[1] = &items[1];
  c[2] = &items[2];
  c.key(0) = 0;
  c.key(1) = Key("key_1");
  c.key(2) = Key("key_2");
  auto p = flatten(c);
  const auto& leaves = p.first;
  ASSERT_EQ(leaves.size(), 3);
  for (size_t i = 0; i < 3; ++i) {
    ASSERT_EQ(*leaves[i], items[i]);
  }
}

TEST(pytree, FlattenNestedDict) {
  Leaf items[5] = {11, 12, 13, 14, 15};
  auto c = ContainerHandle<Leaf>(Kind::Dict, 3);
  auto c2 = ContainerHandle<Leaf>(Kind::Dict, 3);
  c2[0] = &items[2];
  c2[1] = &items[3];
  c2[2] = &items[4];
  c2.key(0) = Key("c2_key_0");
  c2.key(1) = Key("c2_key_1");
  c2.key(2) = Key("c2_key_2");

  c[0] = &items[0];
  c[1] = &items[1];
  c[2] = std::move(c2);
  c.key(0) = 0;
  c.key(1) = Key("key_1");
  c.key(2) = Key("key_2");

  auto p = flatten(c);
  const auto& leaves = p.first;
  ASSERT_EQ(leaves.size(), 5);
  for (size_t i = 0; i < 5; ++i) {
    ASSERT_EQ(*leaves[i], items[i]);
  }
}
