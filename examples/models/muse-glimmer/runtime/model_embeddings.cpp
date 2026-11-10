/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/examples/models/muse-glimmer/runtime/model_embeddings.h>

#include <executorch/extension/llm/runner/model_metadata.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>

#include <algorithm>
#include <limits>

namespace executorch::extension::llm {
using runtime::Error;

namespace muse_glimmer_abi {
bool tensor_abi(
    const runtime::TensorInfo& info,
    aten::ScalarType dtype,
    std::initializer_list<int64_t> shape) {
  if (info.scalar_type() != dtype || info.sizes().size() != shape.size() ||
      info.dim_order().size() != shape.size()) {
    return false;
  }
  size_t bytes = runtime::elementSize(dtype);
  size_t dim = 0;
  for (int64_t size : shape) {
    if (size <= 0 || size != info.sizes()[dim] ||
        info.dim_order()[dim] != dim ||
        static_cast<uint64_t>(size) >
            std::numeric_limits<size_t>::max() / bytes) {
      return false;
    }
    bytes *= static_cast<size_t>(size);
    ++dim;
  }
  return info.nbytes() == bytes;
}

bool dense_tensor(const aten::Tensor& tensor) {
  int64_t stride = 1;
  for (int64_t dim = tensor.dim() - 1; dim >= 0; --dim) {
    if (tensor.size(dim) <= 0 ||
        (tensor.size(dim) != 1 && tensor.strides()[dim] != stride) ||
        stride > std::numeric_limits<int64_t>::max() / tensor.size(dim)) {
      return false;
    }
    stride *= tensor.size(dim);
  }
  return tensor.device().is_cpu() && tensor.const_data_ptr() != nullptr;
}

Error check_backends(const runtime::MethodMeta& meta, const char* backend) {
  if (meta.num_backends() == 0) {
    return Error::InvalidProgram;
  }
  for (size_t i = 0; i < meta.num_backends(); ++i) {
    ET_ASSIGN_OR_RETURN(name, meta.get_backend_name(i));
    if (std::string(name) != backend) {
      return Error::NotSupported;
    }
  }
  return Error::Ok;
}

Error check_constant_method(Module& module, const char* name) {
  ET_ASSIGN_OR_RETURN(meta, module.method_meta(name));
  if (meta.num_inputs() != 0 || meta.num_outputs() != 1 ||
      meta.num_backends() != 0 || meta.num_instructions() != 0) {
    ET_LOG(Error, "Muse Glimmer metadata %s must be constant", name);
    return Error::InvalidProgram;
  }
  return Error::Ok;
}

runtime::Result<int64_t> check_vision_abi(
    Module& module,
    int32_t hidden,
    aten::ScalarType dtype,
    int max_patches,
    const char* backend) {
  ET_CHECK_OK_OR_RETURN_ERROR(
      check_constant_method(module, "get_max_vision_patches"));
  ET_ASSIGN_OR_RETURN(
      published_patches,
      detail::read_required_positive_int(module, "get_max_vision_patches"));
  if (published_patches < 4 ||
      published_patches > std::numeric_limits<int32_t>::max()) {
    return Error::InvalidProgram;
  }
  // Export bounds patches in complete 2x2 downsampling groups, but publishes
  // the original limit (which need not be divisible by four).
  const int64_t patches = (published_patches / 4) * 4;
  ET_ASSIGN_OR_RETURN(meta, module.method_meta("vision_encoder"));
  if (meta.num_inputs() != 9 || meta.num_outputs() != 1) {
    return Error::InvalidProgram;
  }
  ET_CHECK_OK_OR_RETURN_ERROR(check_backends(meta, backend));
  // Host preprocessing's nine-input ABI; inspect it without loading vision.
  for (size_t i = 0; i < 9; ++i) {
    ET_ASSIGN_OR_RETURN(input, meta.input_tensor_meta(i));
    bool valid = false;
    if (i == 0) {
      valid = tensor_abi(input, aten::ScalarType::Float, {1, patches, 1176});
    } else if (i == 1) {
      // Positional interpolation is always BF16, even for an FP16 decoder.
      valid = tensor_abi(input, aten::ScalarType::BFloat16, {1, patches, 1536});
    } else if (i == 2 || i == 3) {
      valid = tensor_abi(input, aten::ScalarType::Float, {patches, 48});
    } else if (i == 6 || i == 7) {
      valid =
          tensor_abi(input, aten::ScalarType::Bool, {1, 1, patches, patches});
    } else {
      valid = tensor_abi(input, aten::ScalarType::Long, {patches});
    }
    if (!valid) {
      return Error::InvalidProgram;
    }
  }
  ET_ASSIGN_OR_RETURN(output, meta.output_tensor_meta(0));
  if (!tensor_abi(output, dtype, {1, patches / 4, hidden})) {
    return Error::InvalidProgram;
  }
  return std::min<int64_t>(patches, max_patches) / 4;
}
} // namespace muse_glimmer_abi

MuseGlimmerModelEmbeddings::MuseGlimmerModelEmbeddings(
    std::shared_ptr<const MuseGlimmerPreparationSpec> spec,
    std::string pos_embed_path)
    : spec_(std::move(spec)),
      materializer_(spec_),
      pos_embed_path_(std::move(pos_embed_path)) {}

Error MuseGlimmerModelEmbeddings::initialize(Module& module) {
  ET_CHECK_OK_OR_RETURN_ERROR(module.load_method("embed_text"));
  if (spec_->has_vision) {
    ET_CHECK_OK_OR_RETURN_ERROR(module.load_method("vision_encoder"));
    MuseGlimmerVisionRuntimeConfig config;
    config.module = &module;
    config.execution_mutex = &vision_mutex_;
    config.pos_embed_path = pos_embed_path_;
    config.activation_dtype = spec_->activation_dtype;
    config.expected_hidden_dim = spec_->hidden_dim;
    config.max_image_tokens = spec_->max_soft_tokens;
    config.max_encoded_bytes = spec_->image_limits.max_encoded_bytes;
    config.max_image_dimension = spec_->image_limits.max_image_dimension;
    config.max_image_pixels = spec_->image_limits.max_image_pixels;
    vision_ = std::make_unique<MuseGlimmerVisionRuntime>(std::move(config));
  }
  return Error::Ok;
}

runtime::Result<TensorPtr> MuseGlimmerModelEmbeddings::materialize(
    Module& module,
    const std::vector<batching::Input>& inputs) {
  return materializer_.materialize(
      inputs,
      [&](const MuseGlimmerRGBImage& image)
          -> runtime::Result<PreparedMuseGlimmerImage> {
        if (!vision_) {
          return Error::InvalidState;
        }
        return vision_->prepare_decoded_image(
            image.rgb.data(), image.width, image.height);
      },
      [&](const std::vector<int64_t>& tokens) -> runtime::Result<TensorPtr> {
        auto input = make_tensor_ptr(
            {1, static_cast<aten::SizesType>(tokens.size())}, tokens);
        ET_ASSIGN_OR_RETURN(
            outputs,
            module.execute("embed_text", std::vector<runtime::EValue>{input}));
        if (outputs.size() != 1 || !outputs[0].isTensor() ||
            !muse_glimmer_abi::dense_tensor(outputs[0].toTensor())) {
          return Error::InvalidProgram;
        }
        // The materializer copies these rows before another method can reuse
        // the module's shared output arena.
        return make_tensor_ptr(outputs[0].toTensor());
      });
}

} // namespace executorch::extension::llm
