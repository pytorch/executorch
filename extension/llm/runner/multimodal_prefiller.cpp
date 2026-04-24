/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Generic encoder prefiller that handles multimodal inputs (text, image and
// audio (to be implemented)) to prefill the KV cache of a multimodal LLM.
// @lint-ignore-every CLANGTIDY facebook-hte-Deprecated

#include <executorch/extension/llm/runner/constants.h>
#include <executorch/extension/llm/runner/multimodal_prefiller.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/tensor/tensor.h>

namespace executorch::extension::llm {

MultimodalPrefiller::MultimodalPrefiller(
    Module* module,
    MultimodalDecoderRunner* decoder_runner,
    Tokenizer* tokenizer,
    IOManager* io_manager)
    : module_(module),
      text_decoder_runner_(decoder_runner),
      tokenizer_(tokenizer),
      io_manager_(io_manager) {}

/**
 * Prefill an LLM Module with the given multimodal input.
 * @param input The multimodal input (text, image or audio) to the multimodal
 * LLM.
 * @param start_pos The starting position in KV cache of the input in the LLM
 * @return logits of the prefill.
 */
Result<uint64_t> MultimodalPrefiller::prefill(
    const MultimodalInput& input,
    int64_t& start_pos,
    int32_t bos,
    int32_t eos) {
  // 1. Run encoder model.
  // pli_ids is populated per-branch and passed to text_decoder as the 3rd
  // input (Approach C PLI). Image/audio use modality placeholder IDs;
  // text uses real token IDs. Ignored when the pte has only 2 inputs.
  std::vector<int64_t> pli_ids;
  ::executorch::runtime::EValue encoder_output;
  if (input.is_image()) {
    const Image& image = input.get_image();

    auto method_meta_result = module_->method_meta(kVisionEncoderMethod);
    ET_CHECK_OK_OR_RETURN_ERROR(
        method_meta_result.error(),
        "Failed to get method_meta for %s",
        kVisionEncoderMethod);
    auto method_meta = method_meta_result.get();

    ET_CHECK_OR_RETURN_ERROR(
        method_meta.num_inputs() > 0,
        InvalidArgument,
        "Image encoder should have at least 1 input");
    auto input_meta_result = method_meta.input_tensor_meta(0);
    ET_CHECK_OK_OR_RETURN_ERROR(
        input_meta_result.error(), "Cannot get input tensor meta at index 0");
    auto input_meta = input_meta_result.get();
    auto expected_dtype = input_meta.scalar_type();

    if (expected_dtype == ::executorch::aten::ScalarType::Float) {
      ET_CHECK_OR_RETURN_ERROR(
          image.is_float(),
          InvalidArgument,
          "Model expects float image data, but image has uint8_t data.");
    } else if (expected_dtype == ::executorch::aten::ScalarType::Byte) {
      ET_CHECK_OR_RETURN_ERROR(
          image.is_uint8(),
          InvalidArgument,
          "Model expects uint8_t image data, but image has float data.");
    } else if (expected_dtype == ::executorch::aten::ScalarType::BFloat16) {
      ET_CHECK_OR_RETURN_ERROR(
          image.is_float(),
          InvalidArgument,
          "Model expects BFloat16 data, we need to take image in float32 type and convert afterwards. But now image has uint8_t data.");
    } else {
      ET_CHECK_OR_RETURN_ERROR(
          false,
          NotSupported,
          "Unsupported image encoder input dtype: %s",
          ::executorch::runtime::toString(expected_dtype));
    }

    // The model might expect a 4D tensor (NCHW), but toTensor() returns a 3D
    // tensor (CHW). Add a batch dimension of 1 if needed.
    auto expected_dims = input_meta.sizes();
    auto image_tensor_result =
        image.toTensor(/*with_batch*/ expected_dims.size() == 4);
    ET_CHECK_OK_OR_RETURN_ERROR(
        image_tensor_result.error(), "Failed to convert image to tensor");
    auto image_tensor = image_tensor_result.get();

    if (expected_dtype == ::executorch::aten::ScalarType::BFloat16) {
      // Convert to bfloat16 for model input
      auto image_tensor_return = convert_to_bfloat16(image_tensor);
      ET_CHECK_OK_OR_RETURN_ERROR(
          image_tensor_return.error(),
          "Failed to convert image tensor to bfloat16");
      image_tensor = image_tensor_return.get();
    }

    ET_LOG(
        Info,
        "Image tensor dim: %zu, dtype: %s",
        image_tensor->dim(),
        ::executorch::runtime::toString(image_tensor->scalar_type()));
    // Run image encoder
    auto image_encoder_result =
        module_->execute(kVisionEncoderMethod, image_tensor);
    ET_CHECK_OK_OR_RETURN_ERROR(image_encoder_result.error());
    auto image_encoder_outputs = image_encoder_result.get();

    encoder_output = image_encoder_outputs[0];
    // HF Gemma4Model.forward sets image positions to pad_token_id (0) before PLI
    // lookup — NOT the image sentinel. Per HF line 2215:
    //   llm_input_ids[multimodal_mask] = self.config.text_config.pad_token_id
    //   per_layer_inputs = get_per_layer_inputs(llm_input_ids, llm_inputs_embeds)
    int64_t n_soft = encoder_output.toTensor().size(1);
    pli_ids.assign(n_soft, 0LL);  // pad_token_id = 0
  } else if (input.is_audio()) {
    const Audio& audio = input.get_audio();

    auto method_meta_result = module_->method_meta(kAudioEncoderMethod);
    ET_CHECK_OK_OR_RETURN_ERROR(
        method_meta_result.error(),
        "Failed to get method_meta for %s",
        kAudioEncoderMethod);
    auto method_meta = method_meta_result.get();

    ET_CHECK_OR_RETURN_ERROR(
        method_meta.num_inputs() > 0,
        InvalidArgument,
        "Audio encoder should have at least 1 input");
    auto input_meta_result = method_meta.input_tensor_meta(0);
    ET_CHECK_OK_OR_RETURN_ERROR(
        input_meta_result.error(), "Cannot get input tensor meta at index 0");
    auto input_meta = input_meta_result.get();
    auto expected_dtype = input_meta.scalar_type();

    // Create tensor with original dtype
    auto audio_tensor_result = audio.toTensor();
    ET_CHECK_OK_OR_RETURN_ERROR(
        audio_tensor_result.error(), "Failed to convert audio to tensor");
    auto audio_tensor = audio_tensor_result.get();

    // Convert to expected dtype if needed
    if (audio_tensor->scalar_type() != expected_dtype) {
      if (expected_dtype == ::executorch::aten::ScalarType::BFloat16) {
        // Convert to bfloat16
        auto convert_result = convert_to_bfloat16(audio_tensor);
        ET_CHECK_OK_OR_RETURN_ERROR(
            convert_result.error(),
            "Failed to convert audio tensor to bfloat16");
        audio_tensor = convert_result.get();
      } else {
        ET_CHECK_OR_RETURN_ERROR(
            false,
            NotSupported,
            "Unsupported audio encoder input dtype: %s. Expecting %s",
            ::executorch::runtime::toString(audio_tensor->scalar_type()),
            ::executorch::runtime::toString(expected_dtype));
      }
    }

    ET_LOG(
        Info,
        "Audio tensor dim: %zu, dtype: %s",
        audio_tensor->dim(),
        ::executorch::runtime::toString(audio_tensor->scalar_type()));

    // Run audio encoder
    auto audio_encoder_result =
        module_->execute(kAudioEncoderMethod, audio_tensor);
    if (audio_encoder_result.error() != ::executorch::runtime::Error::Ok) {
      return ::executorch::runtime::Error::Internal;
    }
    auto audio_encoder_outputs = audio_encoder_result.get();

    encoder_output = audio_encoder_outputs[0];
    // Same as image: HF uses pad_token_id (0) for all soft-token positions in PLI.
    int64_t n_soft = encoder_output.toTensor().size(1);
    pli_ids.assign(n_soft, 0LL);  // pad_token_id = 0
  } else if (input.is_text() || input.is_tokens()) {
    std::vector<uint64_t> tokens;
    if (input.is_text()) {
      auto& text = input.get_text();
      auto encode_result = tokenizer_->encode(text, bos, eos);
      if (!encode_result.ok()) {
        ET_LOG(
            Error,
            "Tokenizers error code %d",
            static_cast<uint32_t>(encode_result.error()));
        return ::executorch::runtime::Error::InvalidArgument;
      }
      tokens = std::move(*encode_result);
    } else {
      tokens = input.get_tokens();
    }

    auto text_tensor = executorch::extension::from_blob(
        tokens.data(),
        {1, static_cast<aten::SizesType>(tokens.size())},
        ::executorch::aten::ScalarType::Long);

    // Run text encoder (token embeddings)
    auto token_embedding_result =
        module_->execute(kTokenEmbeddingMethod, text_tensor);
    ET_CHECK_OK_OR_RETURN_ERROR(token_embedding_result.error());
    auto token_embedding_outputs = token_embedding_result.get();

    encoder_output = token_embedding_outputs[0];
    // PLI for text: use the actual token IDs (cast uint64_t → int64_t).
    pli_ids.reserve(tokens.size());
    for (uint64_t tok : tokens) pli_ids.push_back(static_cast<int64_t>(tok));
  } else {
    ET_LOG(Error, "Unsupported input type");
    // For any other input types, return error
    return ::executorch::runtime::Error::NotSupported;
  }

  // 2. Run decoder model for prefill.

  // Get expected shape of cache position tensor, which should be the second
  // argument

  int64_t seq_len = encoder_output.toTensor().size(1);
  if (seq_len == 0) {
    ET_LOG(Error, "The encoder returned an empty output.");
    return ::executorch::runtime::Error::InvalidState;
  }
  std::vector<int64_t> cache_positions;

  auto cache_position_result = populate_start_pos_or_cache_position(
      module_, start_pos, cache_positions, seq_len, kTextModelMethod);
  ET_CHECK_OK_OR_RETURN_ERROR(cache_position_result.error());
  auto cache_position_tensor = cache_position_result.get();

  // Approach C: detect 3-input text_decoder (Gemma4 PLI).
  // pli_ids are built per-branch above:
  //   text  → real token IDs (matches HF: PLI uses actual token IDs for text)
  //   image → 0 (pad_token_id, matches HF line 2215)
  //   audio → 0 (pad_token_id, same convention)
  // Falls back to 2-input silently for ptes without PLI (num_inputs < 3).
  //
  // Also detect optional specialized "prefill" method (qwen3_5_moe pattern,
  // exported via gemma4 export.py --split-text-decoder). When present we
  // call it for the batched prompt instead of the unified text_decoder so
  // backends can specialize the prefill graph (tensor-core matmuls, etc.).
  //
  // Cache the detection so we only query method_meta once per runner.
  if (!pli_detected_) {
    auto dec_meta = module_->method_meta(kTextModelMethod);
    size_t n_inputs = dec_meta.ok() ? (*dec_meta).num_inputs() : 0;
    has_pli_input_ = n_inputs >= 3;
    // Probe for the specialized prefill method. method_names() returns the
    // set of registered method names; check membership rather than calling
    // load_method (which would error-log on a miss).
    auto names_res = module_->method_names();
    if (names_res.ok() && names_res.get().count("prefill")) {
      auto pre_load = module_->load_method("prefill");
      if (pre_load == ::executorch::runtime::Error::Ok) {
        has_prefill_method_ = true;
      }
    }
    pli_detected_ = true;
    ET_LOG(Info, "MultimodalPrefiller: PLI %s, prefill method %s "
                 "(text_decoder num_inputs=%zu)",
           has_pli_input_ ? "enabled" : "disabled",
           has_prefill_method_ ? "present" : "absent",
           n_inputs);
  }
  ET_LOG(Debug, "text_decoder has_pli=%d pli_ids=%zu",
         (int)has_pli_input_, pli_ids.size());

  const char* method_name =
      has_prefill_method_ ? "prefill" : kTextModelMethod;
  auto run_text_decoder = [&]() {
    if (has_pli_input_ && !pli_ids.empty()) {
      // Pass real/placeholder token IDs as the 3rd input for Approach C PLI.
      auto pli_t = ::executorch::extension::from_blob(
          pli_ids.data(),
          {1, static_cast<int>(pli_ids.size())},
          ::executorch::aten::ScalarType::Long);
      return module_->execute(
          method_name,
          {encoder_output, cache_position_tensor, *pli_t});
    }
    return module_->execute(
        method_name, {encoder_output, cache_position_tensor});
  };
  auto prefill_result = run_text_decoder();
  if (prefill_result.error() != ::executorch::runtime::Error::Ok) {
    return prefill_result.error();
  }
  // Check if prefill_outputs is empty, if it is return error and log that the
  // specified encoder returned empty results when used to prefill decoder.
  auto prefill_outputs = prefill_result.get();
  if (prefill_outputs.empty()) {
    ET_LOG(
        Error, "Encoder returned empty results when used to prefill decoder");
    return ::executorch::runtime::Error::InvalidState;
  }
  auto outputs_res = prefill_outputs[0].toTensor();

  // Update start_pos, tracking the current cache position.
  start_pos += seq_len;

  return static_cast<uint64_t>(
      text_decoder_runner_->logits_to_token(outputs_res));
}

/**
 * Load the Module for encoder prefill purpose.
 * @return The error code.
 */
::executorch::runtime::Error MultimodalPrefiller::load() {
  if (is_method_loaded()) {
    return ::executorch::runtime::Error::Ok;
  }
  // token_embeddings and text_model have to show up in method names.
  ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(kTokenEmbeddingMethod));
  ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(kTextModelMethod));

  auto method_names_result = module_->method_names();
  ET_CHECK_OK_OR_RETURN_ERROR(
      method_names_result.error(), "Failed to get method names");
  std::unordered_set<std::string> methods = method_names_result.get();

  // Load image_encoder method if exists.
  if (methods.find(kVisionEncoderMethod) != methods.end()) {
    ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(kVisionEncoderMethod));
  }

  if (methods.find(kAudioEncoderMethod) != methods.end()) {
    ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(kAudioEncoderMethod));
  }

  return ::executorch::runtime::Error::Ok;
}

/**
 * Check if the required methods in the Module is loaded.
 * @return True if the Module is loaded, false otherwise.
 */
bool MultimodalPrefiller::is_method_loaded() {
  ::executorch::runtime::Result<std::unordered_set<std::string>> methods_res =
      module_->method_names();
  if (!module_->is_method_loaded(kTokenEmbeddingMethod)) {
    return false;
  }
  if (!module_->is_method_loaded(kTextModelMethod)) {
    return false;
  }
  if (methods_res.error() != ::executorch::runtime::Error::Ok) {
    ET_CHECK_MSG(false, "Failed to get method names");
  }
  std::unordered_set<std::string> methods = methods_res.get();
  if (methods.find(kVisionEncoderMethod) != methods.end()) {
    return module_->is_method_loaded(kVisionEncoderMethod);
  }
  return true;
}

} // namespace executorch::extension::llm
