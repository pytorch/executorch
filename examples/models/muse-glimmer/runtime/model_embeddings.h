/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/examples/models/muse-glimmer/runtime/embedding_materializer.h>
#include <executorch/extension/module/module.h>

#include <initializer_list>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace executorch::extension::llm {

namespace muse_glimmer_abi {
// Exact dtype, contiguous shape and byte size of a method input or output.
bool tensor_abi(
    const runtime::TensorInfo& info,
    aten::ScalarType dtype,
    std::initializer_list<int64_t> shape);
// Non-empty, contiguous and host-resident.
bool dense_tensor(const aten::Tensor& tensor);
// Every delegate of the method is `backend`.
runtime::Error check_backends(
    const runtime::MethodMeta& meta,
    const char* backend);
// Metadata can be read during create(), but no backend may load off-engine.
runtime::Error check_constant_method(Module& module, const char* name);
// Checks vision_encoder's host-preprocessing ABI without loading it. Returns
// the soft tokens one image may contribute.
runtime::Result<int64_t> check_vision_abi(
    Module& module,
    int32_t hidden,
    aten::ScalarType dtype,
    int max_patches,
    const char* backend);
} // namespace muse_glimmer_abi

// The program's embed_text and, when the spec has vision, vision_encoder:
// prepared slices become one host [1, rows, hidden] tensor of decoder inputs.
class MuseGlimmerModelEmbeddings final {
 public:
  MuseGlimmerModelEmbeddings(
      std::shared_ptr<const MuseGlimmerPreparationSpec> spec,
      std::string pos_embed_path);

  runtime::Error initialize(Module& module);
  bool accepts(const batching::PreparedInput& input) const {
    return materializer_.accepts(input);
  }
  // Text rows run through embed_text in one call; image rows reuse each image's
  // encoding across the slices of one prompt.
  runtime::Result<TensorPtr> materialize(
      Module& module,
      const std::vector<batching::Input>& inputs);

 private:
  std::shared_ptr<const MuseGlimmerPreparationSpec> spec_;
  MuseGlimmerMaterializer materializer_;
  std::string pos_embed_path_;
  std::mutex vision_mutex_;
  std::unique_ptr<MuseGlimmerVisionRuntime> vision_;
};

} // namespace executorch::extension::llm
