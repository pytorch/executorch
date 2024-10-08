/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple diffusion runner that includes preprocessing and post processing
// logic. The module takes in a string as input and emites a tensor as output.

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <executorch/extension/module/module.h>

namespace example {

class Runner {
 public:
  explicit Runner(
      const std::vector<std::string>& models_path,
      const int num_time_steps,
      const float guidance_scale,
      const float text_encoder_output_scale,
      const int text_encoder_output_offset,
      const float unet_input_latent_scale,
      const int unet_input_latent_offset,
      const float unet_input_text_emb_scale,
      const float unet_input_text_emb_offset,
      const float unet_output_scale,
      const int unet_output_offset,
      const float vae_input_scale,
      const int vae_input_offset,
      const float vae_output_scale,
      const int vae_output_offset,
      const std::string output_path,
      const bool fix_latents);

  struct Stats {
    // Scaling factor for timestamps - in this case, we use ms.
    const long SCALING_FACTOR_UNITS_PER_SECOND = 1000;
    // Time stamps for the different stages of the execution
    // model_load_start_ms: Model loading time
    long model_load_start_ms;
    long model_load_end_ms;

    // tokenizer loading time
    long tokenizer_load_start_ms = 0;
    long tokenizer_load_end_ms = 0;

    // tokenizer parsing time
    long tokenizer_parsing_start_ms = 0;
    long tokenizer_parsing_end_ms = 0;

    // Total time to run generate
    long generate_start_ms = 0;
    long generate_end_ms = 0;

    // text encoder execution time
    long text_encoder_execution_time = 0;

    // Unet aggregation execution time over n steps for cond + uncond
    long unet_aggregate_execution_time = 0;

    // UNet aggregation post processing time over n steps for cond + uncond.
    // This is the time from processing unet's output until feeding it into the
    // next iteration.
    long unet_aggregate_post_processing_time = 0;

    // VAE execution time
    long vae_execution_time = 0;
  };

  bool is_loaded() const;
  executorch::runtime::Error load();
  executorch::runtime::Error init_tokenizer(const std::string& vocab_json_path);
  executorch::runtime::Error print_performance();
  std::vector<int> tokenize(std::string prompt);
  std::vector<float> gen_latent_from_file();
  std::vector<float> gen_random_latent(float sigma);
  void step(
      const std::vector<float>& model_output,
      const std::vector<float>& sigmas,
      std::vector<float>& sample,
      std::vector<float>& prev_sample,
      int step_index);
  std::vector<executorch::runtime::Result<executorch::runtime::MethodMeta>>
  get_methods_meta();
  std::vector<float> get_time_steps();
  std::vector<float> get_sigmas(const std::vector<float>& time_steps);
  void scale_model_input(
      const std::vector<float>& vec,
      std::vector<float>& latent_model_input,
      float sigma);
  executorch::runtime::Error parse_input_list(std::string& path);
  executorch::runtime::Error generate(std::string prompt);
  void quant_tensor(
      const std::vector<float>& fp_vec,
      std::vector<uint16_t>& quant_vec,
      float scale,
      int offset);
  void dequant_tensor(
      const std::vector<uint16_t>& quant_vec,
      std::vector<float>& fp_vec,
      float scale,
      int offset);

 private:
  Stats stats_;
  std::vector<std::unique_ptr<executorch::extension::Module>> modules_;
  std::vector<std::vector<uint16_t>> time_emb_list_;
  std::unordered_map<std::string, int32_t> vocab_to_token_map_;

  std::string output_path_;
  int num_time_steps_;
  float guidance_scale_;
  float text_encoder_output_scale_;
  int text_encoder_output_offset_;
  float unet_input_latent_scale_;
  int unet_input_latent_offset_;
  float unet_input_text_emb_scale_;
  int unet_input_text_emb_offset_;
  float unet_output_scale_;
  int unet_output_offset_;
  float vae_input_scale_;
  int vae_input_offset_;
  float vae_output_scale_;
  int vae_output_offset_;
  const float beta_start_ = 0.00085;
  const float beta_end_ = 0.012;
  const int num_train_timesteps_ = 1000;
  const int max_tokens_ = 77;
  const bool fix_latents_ = false;
};

} // namespace example
