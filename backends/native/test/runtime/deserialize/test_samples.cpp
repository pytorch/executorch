// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/deserialize/Samples.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <executorch/backends/native/runtime/deserialize/test/PackageTestData.h>
#include <gtest/gtest.h>

namespace ptn {
namespace {

using testing::make_safetensors;
using testing::make_zip;

std::vector<uint8_t> bytes_of(const std::string& text) {
  return {text.begin(), text.end()};
}

// A zip holding samples/forward/golden/ with `index` and a safetensors blob
// of `header` over `data`.
std::vector<uint8_t> make_set(
    const std::string& index,
    const std::string& header,
    const std::string& data) {
  std::vector<uint8_t> tensors = make_safetensors(header, data.size());
  std::memcpy(
      tensors.data() + tensors.size() - data.size(), data.data(), data.size());
  return make_zip(
      {{"program.ptg", {}},
       {"samples/forward/golden/index.json", bytes_of(index)},
       {"samples/forward/golden/data.safetensors", tensors}});
}

const std::string kTwoCallHeader =
    R"({"samples[0].calls[0].in[0]":{"dtype":"U8","shape":[1,2],"data_offsets":[0,2]},)"
    R"("samples[0].calls[0].out[0]":{"dtype":"F32","shape":[1],"data_offsets":[2,6]},)"
    R"("samples[0].calls[0].intermediates.h":{"dtype":"U8","shape":[1],"data_offsets":[6,7]},)"
    R"("samples[0].calls[1].in[0]":{"dtype":"U8","shape":[1,1],"data_offsets":[7,8]},)"
    R"("samples[1].calls[0].in[0]":{"dtype":"U8","shape":[1],"data_offsets":[8,9]}})";
const std::string kTwoCallData =
    std::string("ab") + std::string("\x00\x00\x80\x3f", 4) + "hcd";

SampleSet read(const std::vector<uint8_t>& archive) {
  return read_sample_set(ZipReader::open(archive), "forward", "golden");
}

// cppcheck-suppress-begin syntaxError
TEST(SamplesTest, ReadsCallsInOrderWithResolvedTolerance) {
  const std::string index =
      R"({"version":"1.7","description":"d","source":"eager",
    "tol":[0.001,0.002],"unknown":0,"samples":[
      {"tol":[0.0001,0.0002],"calls":[
        {"in":["samples[0].calls[0].in[0]"],"out":["samples[0].calls[0].out[0]"],
         "intermediates":{"h":"samples[0].calls[0].intermediates.h"}},
        {"in":["samples[0].calls[1].in[0]"],"tol":[0,0],"unknown":0}]},
      {"calls":[{"in":["samples[1].calls[0].in[0]"]}]}]})";

  const SampleSet set = read(make_set(index, kTwoCallHeader, kTwoCallData));

  EXPECT_EQ(set.description, "d");
  EXPECT_EQ(set.source, "eager");
  ASSERT_EQ(set.samples.size(), 2);
  ASSERT_EQ(set.samples[0].calls.size(), 2);
  const SampleCall& prefill = set.samples[0].calls[0];
  ASSERT_EQ(prefill.inputs.size(), 1);
  EXPECT_EQ(prefill.inputs[0].dtype, kByte);
  EXPECT_EQ(prefill.inputs[0].sizes, (std::vector<int64_t>{1, 2}));
  EXPECT_EQ(prefill.inputs[0].bytes, bytes_of("ab"));
  ASSERT_TRUE(prefill.outputs);
  ASSERT_EQ(prefill.outputs->size(), 1);
  EXPECT_EQ((*prefill.outputs)[0].dtype, kFloat);
  float one = 0;
  std::memcpy(&one, (*prefill.outputs)[0].bytes.data(), sizeof(one));
  EXPECT_EQ(one, 1.0f);
  EXPECT_EQ(prefill.intermediates.at("h").bytes, bytes_of("h"));
  ASSERT_TRUE(prefill.tol);
  EXPECT_EQ(prefill.tol->atol, 0.0001);
  EXPECT_EQ(prefill.tol->rtol, 0.0002);
  const SampleCall& decode = set.samples[0].calls[1];
  EXPECT_EQ(decode.inputs[0].bytes, bytes_of("c"));
  EXPECT_FALSE(decode.outputs);
  ASSERT_TRUE(decode.tol);
  EXPECT_EQ(decode.tol->atol, 0.0);
  ASSERT_TRUE(set.samples[1].calls[0].tol);
  EXPECT_EQ(set.samples[1].calls[0].tol->atol, 0.001);
}

TEST(SamplesTest, LeavesToleranceUnsetWhenNoLevelHasOne) {
  const SampleSet set = read(make_set(
      R"({"version":"1.0","samples":[{"calls":[{"in":["samples[0].calls[0].in[0]"]}]}]})",
      R"({"samples[0].calls[0].in[0]":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}})",
      "a"));

  EXPECT_FALSE(set.samples[0].calls[0].tol);
}

TEST(SamplesTest, RejectsMalformedSets) {
  const std::string one_tensor =
      R"({"samples[0].calls[0].in[0]":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}})";
  const std::string one_call =
      R"("samples":[{"calls":[{"in":["samples[0].calls[0].in[0]"]}]}])";
  const std::string call0 = R"("in":["samples[0].calls[0].in[0]"])";
  const auto with_call = [&](const std::string& fields) {
    return R"({"version":"1.0","samples":[{"calls":[{)" + call0 + "," + fields +
        "}]}]}";
  };
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"[]", "must contain a JSON object"},
      {R"({"version":)", "invalid index.json"},
      {R"({"version":"2.0",)" + one_call + "}", "unsupported version"},
      {R"({"version":1,)" + one_call + "}", "unsupported version"},
      {R"({"version":"1",)" + one_call + "}", "unsupported version"},
      {R"({"version":"1.01",)" + one_call + "}", "unsupported version"},
      {R"({"version":"1.x",)" + one_call + "}", "unsupported version"},
      {R"({"version":"1.0","samples":[]})", "samples must be a non-empty list"},
      {R"({"version":"1.0","samples":[{"calls":[]}]})",
       "samples[0] must be an object with a non-empty 'calls' list"},
      {R"({"version":"1.0","samples":[{"in":["samples[0].calls[0].in[0]"]}]})",
       "samples[0] must be an object with a non-empty 'calls' list"},
      {R"({"version":"1.0","samples":[{"calls":[{"in":["samples[0].in[0]"]}]}]})",
       "key samples[0].in[0] must be samples[0].calls[0].in[0]"},
      {R"({"version":"1.0","samples":[{"calls":[{"in":["samples[0].calls[0].in[0]",1]}]}]})",
       "samples[0].calls[0].in[1] must be a tensor"},
      {R"({"version":"1.0","samples":[{"calls":[{"in":[["samples[0].calls[0].in[0]"]]}]}]})",
       "samples[0].calls[0].in[0] must be a tensor"},
      {R"({"version":"1.0","samples":[{"calls":[{"in":"samples[0].calls[0].in[0]"}]}]})",
       "samples[0].calls[0].in must be a list"},
      {R"({"version":"1.0","samples":[{"calls":[{"in":[]}]}]})",
       "unreferenced tensor samples[0].calls[0].in[0]"},
      {R"({"version":"1.0","tol":[1],)" + one_call + "}",
       "tol must be [atol, rtol]"},
      {R"({"version":"1.0","tol":[-1,0],)" + one_call + "}",
       "tol must be two finite numbers"},
      {R"({"version":"1.0","tol":[true,0],)" + one_call + "}",
       "tol must be two finite numbers"},
      {R"({"version":"1.0","description":1,)" + one_call + "}",
       "description must be a string"},
      {R"({"version":"1.0","source":null,)" + one_call + "}",
       "source must be a string"},
      {with_call(R"("out":"samples[0].calls[0].out[0]")"),
       "samples[0].calls[0].out must be a list"},
      {with_call(R"("out":["samples[0].calls[0].out[0]"])"),
       "no tensor named samples[0].calls[0].out[0]"},
      {with_call(R"("intermediates":[])"),
       "samples[0].calls[0].intermediates must be an object"},
      {with_call(
           R"("intermediates":{"a.b":"samples[0].calls[0].intermediates.a.b"})"),
       "bad intermediate a.b in samples[0].calls[0]"},
      {with_call(R"("intermediates":{"h":1})"),
       "samples[0].calls[0].intermediates.h must be a tensor"},
      {with_call(R"("tol":[0,-1])"), "tol must be two finite numbers"},
  };
  for (const auto& [index, message] : cases) {
    SCOPED_TRACE(index);
    try {
      read(make_set(index, one_tensor, "a"));
      ADD_FAILURE() << "expected an exception";
    } catch (const std::runtime_error& error) {
      EXPECT_NE(std::string(error.what()).find(message), std::string::npos)
          << error.what();
    }
  }
}
// cppcheck-suppress-end syntaxError

} // namespace
} // namespace ptn
