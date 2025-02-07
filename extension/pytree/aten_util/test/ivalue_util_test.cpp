/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/pytree/aten_util/ivalue_util.h>
#include <gtest/gtest.h>

using executorch::extension::flatten;
using executorch::extension::is_same;
using executorch::extension::unflatten;

std::vector<at::Tensor> makeExampleTensors(size_t N) {
  std::vector<at::Tensor> tensors;
  for (int i = 0; i < N; ++i) {
    tensors.push_back(at::randn({2, 3, 5}));
  }
  return tensors;
}

struct TestCase {
  c10::IValue ivalue;
  std::vector<at::Tensor> tensors;
};

TestCase makeExampleListOfTensors() {
  auto tensors = makeExampleTensors(3);
  auto list = c10::List<at::Tensor>{
      tensors[0],
      tensors[1],
      tensors[2],
  };
  return {list, tensors};
}

TestCase makeExampleTupleOfTensors() {
  auto tensors = makeExampleTensors(3);
  auto tuple = std::make_tuple(tensors[0], tensors[1], tensors[2]);
  return {tuple, tensors};
}

TestCase makeExampleDictOfTensors() {
  auto tensors = makeExampleTensors(3);
  auto dict = c10::Dict<std::string, at::Tensor>();
  dict.insert("x", tensors[0]);
  dict.insert("y", tensors[1]);
  dict.insert("z", tensors[2]);
  return {dict, tensors};
}

TestCase makeExampleComposite() {
  auto tensors = makeExampleTensors(8);

  c10::IValue list = c10::List<at::Tensor>{
      tensors[1],
      tensors[2],
  };

  auto inner_dict1 = c10::Dict<std::string, at::Tensor>();
  inner_dict1.insert("x", tensors[3]);
  inner_dict1.insert("y", tensors[4]);

  auto inner_dict2 = c10::Dict<std::string, at::Tensor>();
  inner_dict2.insert("z", tensors[5]);
  inner_dict2.insert("w", tensors[6]);

  auto dict = c10::Dict<std::string, c10::Dict<std::string, at::Tensor>>();
  dict.insert("a", inner_dict1);
  dict.insert("b", inner_dict2);

  return {{std::make_tuple(tensors[0], list, dict, tensors[7])}, tensors};
}

void testFlatten(const TestCase& testcase) {
  auto ret = flatten(testcase.ivalue);
  ASSERT_TRUE(is_same(ret.first, testcase.tensors));
}

TEST(IValueFlattenTest, ListOfTensor) {
  testFlatten(makeExampleListOfTensors());
}

TEST(IValueFlattenTest, TupleOfTensor) {
  testFlatten(makeExampleTupleOfTensors());
}

TEST(IValueFlattenTest, DictOfTensor) {
  testFlatten(makeExampleDictOfTensors());
}

TEST(IValueFlattenTest, Composite) {
  testFlatten(makeExampleComposite());
}

void testUnflatten(const TestCase& testcase) {
  // first we flatten the IValue
  auto ret = flatten(testcase.ivalue);

  // then we unflatten it
  c10::IValue unflattened = unflatten(ret.first, ret.second);

  // and see if we got the same IValue back
  ASSERT_TRUE(is_same(unflattened, testcase.ivalue));
}

TEST(IValueUnflattenTest, ListOfTensor) {
  testUnflatten(makeExampleListOfTensors());
}

TEST(IValueUnflattenTest, TupleOfTensor) {
  testUnflatten(makeExampleTupleOfTensors());
}

TEST(IValueUnflattenTest, DictOfTensor) {
  testUnflatten(makeExampleDictOfTensors());
}

TEST(IValueUnflattenTest, Composite) {
  testUnflatten(makeExampleComposite());
}
