/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple stable diffusion runner that includes preprocessing and post
// processing logic. The module takes in a string as input and emits a tensor as
// output.

#include <executorch/examples/qualcomm/qaihub_scripts/stable_diffusion/runner/runner.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/tensor/tensor.h>

#include <ctime>
#include <fstream>
#include <random>
#include <regex>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/platform/log.h>

using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::extension::llm::time_in_ms;
using executorch::runtime::Error;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;

namespace example {

Runner::Runner(
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
    const bool fix_latents)
    : num_time_steps_(num_time_steps),
      guidance_scale_(guidance_scale),
      text_encoder_output_scale_(text_encoder_output_scale),
      text_encoder_output_offset_(text_encoder_output_offset),
      unet_input_latent_scale_(unet_input_latent_scale),
      unet_input_latent_offset_(unet_input_latent_offset),
      unet_input_text_emb_scale_(unet_input_text_emb_scale),
      unet_input_text_emb_offset_(unet_input_text_emb_offset),
      unet_output_scale_(unet_output_scale),
      unet_output_offset_(unet_output_offset),
      vae_input_scale_(vae_input_scale),
      vae_input_offset_(vae_input_offset),
      vae_output_scale_(vae_output_scale),
      vae_output_offset_(vae_output_offset),
      output_path_(output_path),
      fix_latents_(fix_latents) {
  for (int i = 0; i < models_path.size(); i++) {
    modules_.push_back(std::make_unique<Module>(
        models_path[i], Module::LoadMode::MmapUseMlockIgnoreErrors));
    ET_LOG(Info, "creating module: model_path=%s", models_path[i].c_str());
  }
}

std::vector<Result<MethodMeta>> Runner::get_methods_meta() {
  std::vector<Result<MethodMeta>> methods_meta;
  for (std::unique_ptr<Module>& module : modules_) {
    methods_meta.emplace_back(module->method_meta("forward"));
  }
  return methods_meta;
}

bool Runner::is_loaded() const {
  bool loaded = true;
  for (const std::unique_ptr<Module>& module : modules_) {
    loaded &= module->is_loaded();
  }
  return loaded;
}

Error Runner::load() {
  if (is_loaded()) {
    return Error::Ok;
  }
  stats_.model_load_start_ms = time_in_ms();
  for (auto& module : modules_) {
    ET_CHECK_OK_OR_RETURN_ERROR(module->load_method("forward"));
  }
  stats_.model_load_end_ms = time_in_ms();
  return Error::Ok;
}

Error Runner::parse_input_list(std::string& path) {
  // Fill in data for input
  std::ifstream input_list(path);
  time_emb_list_.reserve(num_time_steps_);
  ET_CHECK_MSG(input_list.is_open(), "Input list error opening file");
  std::string time_emb_file;
  for (int i = 0; i < num_time_steps_; i++) {
    std::getline(input_list, time_emb_file);
    std::ifstream is;
    is.open(time_emb_file, std::ios::binary);
    is.seekg(0, std::ios::end);
    size_t filesize = is.tellg();
    is.seekg(0, std::ios::beg);
    std::vector<uint16_t> time_emb;
    time_emb.resize(filesize / sizeof(uint16_t));
    is.read(reinterpret_cast<char*>(time_emb.data()), filesize);
    time_emb_list_.push_back(time_emb);
  }
  return Error::Ok;
}

Error Runner::init_tokenizer(const std::string& vocab_json_path) {
  ET_LOG(Info, "Loading Tokenizer from json");
  stats_.tokenizer_load_start_ms = time_in_ms();
  std::ifstream fin(vocab_json_path);
  auto update_map = [this](std::string& target, std::regex& re) {
    std::smatch sm;
    std::regex_search(target, sm, re);
    // replace special character, please extend this if any cornor case found
    std::string text = sm[1];
    std::unordered_map<std::string, std::regex> post_process = {
        {"\"", std::regex(R"(\\\")")},
        {" ", std::regex(R"(</w>)")},
        {"\\", std::regex(R"(\\\\)")}};
    for (auto& p : post_process) {
      text = std::regex_replace(text, p.second, p.first);
    }
    vocab_to_token_map_[text] = std::stoi(sm[2]);
  };

  if (fin.is_open()) {
    std::string line, text;
    while (getline(fin, line)) {
      text += line;
    }
    fin.close();

    std::regex re_anchor(R"(\d,\")");
    std::regex re_pattern(R"(\{?\"(.*)\":([\d]+)\}?)");
    auto begin = std::sregex_iterator(text.begin(), text.end(), re_anchor);
    auto end = std::sregex_iterator();
    size_t pos = 0;
    for (std::sregex_iterator iter = begin; iter != end; ++iter) {
      std::smatch match;
      size_t len = iter->position() - pos + 1;
      std::string target = text.substr(pos, len);
      update_map(target, re_pattern);
      pos = iter->position() + 1;
    }
    // process last vocabulary
    std::string target = text.substr(pos);
    update_map(target, re_pattern);
  }
  stats_.tokenizer_load_end_ms = time_in_ms();
  return Error::Ok;
}

std::vector<int> Runner::tokenize(std::string prompt) {
  std::string bos("<|startoftext|>"), eos("<|endoftext|>");
  std::vector<std::string> vocabs;
  vocabs.reserve(max_tokens_);
  std::vector<int32_t> tokens(1, vocab_to_token_map_[bos]);

  // pretokenize
  // ref: https://github.com/monatis/clip.cpp
  //      https://huggingface.co/openai/clip-vit-base-patch32
  std::string text;
  std::regex re(
      R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)");
  std::smatch sm;
  while (std::regex_search(prompt, sm, re)) {
    for (auto& v : sm) {
      vocabs.push_back(v);
    }
    prompt = sm.suffix();
  }
  for (std::string& v : vocabs) {
    std::string word = (v[0] == ' ') ? v.substr(1) : v;
    word += " ";
    auto iter = vocab_to_token_map_.find(word);
    if (iter != vocab_to_token_map_.end()) {
      tokens.push_back(iter->second);
      continue;
    }
    for (int i = 0; i < v.size(); ++i) {
      for (int j = v.size() - 1; j >= i; --j) {
        std::string token = v.substr(i, j - 1 + 1);
        auto iter = vocab_to_token_map_.find(token);
        if (iter != vocab_to_token_map_.end()) {
          tokens.push_back(iter->second);
          i = j + 1;
          break;
        } else if (j == i) {
          ET_LOG(Error, "unknown token found: %s", token.c_str());
        }
      }
    }
  }
  tokens.push_back(vocab_to_token_map_[eos]);
  return tokens;
}

std::vector<float> Runner::gen_latent_from_file() {
  std::vector<float> tensor_vector;
  std::ifstream file("latents.raw", std::ios::binary);
  if (!file.is_open()) {
    ET_LOG(Error, "Error opening file!");
    return tensor_vector;
  }

  // Read the tensor data
  float value;
  while (file.read(reinterpret_cast<char*>(&value), sizeof(float))) {
    tensor_vector.push_back(value);
  }
  file.close();
  return tensor_vector;
}

std::vector<float> Runner::gen_random_latent(float sigma) {
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::normal_distribution<float> dist{0.0f, 1.0f};

  constexpr int latent_size = 1 * 64 * 64 * 4;
  std::vector<float> random_vector(latent_size);

  for (float& value : random_vector) {
    value = dist(mersenne_engine) * sigma;
  }
  return random_vector;
}

std::vector<float> Runner::get_time_steps() {
  std::vector<float> time_steps(num_time_steps_);
  for (int i = 0; i < num_time_steps_; ++i) {
    time_steps[i] = (num_train_timesteps_ - 1) *
        (1.0f - static_cast<float>(i) / (num_time_steps_ - 1));
  }
  return time_steps;
}

std::vector<float> Runner::get_sigmas(const std::vector<float>& time_steps) {
  float start = std::sqrt(beta_start_);
  float end = std::sqrt(beta_end_);
  std::vector<float> betas(num_train_timesteps_);
  float step = (end - start) / (num_train_timesteps_ - 1);
  for (int i = 0; i < num_train_timesteps_; ++i) {
    float value = start + i * step;
    betas[i] = 1 - (value * value);
  }

  std::vector<float> alphas_cumprod(num_train_timesteps_);
  float cumprod = 1.0;
  for (int i = 0; i < num_train_timesteps_; ++i) {
    cumprod *= betas[i];
    alphas_cumprod[i] = cumprod;
  }

  std::vector<float> sigmas(num_train_timesteps_);
  for (int i = 0; i < num_train_timesteps_; ++i) {
    sigmas[i] = std::sqrt((1.0 - alphas_cumprod[i]) / alphas_cumprod[i]);
  }

  std::vector<float> res(time_steps.size());
  for (size_t i = 0; i < time_steps.size(); ++i) {
    float index =
        static_cast<float>(i) * (sigmas.size() - 1) / (time_steps.size() - 1);
    size_t lower_index = static_cast<size_t>(std::floor(index));
    size_t upper_index = static_cast<size_t>(std::ceil(index));

    float weight = index - lower_index;
    res[i] =
        (1.0 - weight) * sigmas[lower_index] + weight * sigmas[upper_index];
  }
  std::reverse(res.begin(), res.end());
  res.push_back(0);

  return res;
}

void Runner::scale_model_input(
    const std::vector<float>& latents,
    std::vector<float>& latent_model_input,
    float sigma) {
  for (int i = 0; i < latents.size(); i++) {
    latent_model_input[i] = (latents[i] / std::sqrt(sigma * sigma + 1));
  }
}

void Runner::quant_tensor(
    const std::vector<float>& fp_vec,
    std::vector<uint16_t>& quant_vec,
    float scale,
    int offset) {
  offset = abs(offset);
  for (int i = 0; i < fp_vec.size(); i++) {
    quant_vec[i] = static_cast<uint16_t>((fp_vec[i] / scale) + offset);
  }
}

void Runner::dequant_tensor(
    const std::vector<uint16_t>& quant_vec,
    std::vector<float>& fp_vec,
    float scale,
    int offset) {
  offset = abs(offset);
  for (int i = 0; i < quant_vec.size(); i++) {
    fp_vec[i] = (quant_vec[i] - offset) * scale;
  }
}

// Using the same algorithm as EulerDiscreteScheduler in python.
void Runner::step(
    const std::vector<float>& model_output,
    const std::vector<float>& sigmas,
    std::vector<float>& sample,
    std::vector<float>& prev_sample,
    int step_index) {
  float sigma = sigmas[step_index];
  float dt = sigmas[step_index + 1] - sigma;

  for (int i = 0; i < sample.size(); ++i) {
    float sigma_hat = sample[i] - (sigma * model_output[i]);
    prev_sample[i] = (sample[i] - sigma_hat) / sigma;
    prev_sample[i] = sample[i] + (prev_sample[i] * dt);
  }
  sample = prev_sample;
}

Error Runner::generate(std::string prompt) {
  ET_LOG(Info, "Start generating");
  stats_.generate_start_ms = time_in_ms();

  // Start tokenize
  stats_.tokenizer_parsing_start_ms = time_in_ms();
  std::vector<int32_t> cond_tokens = tokenize(prompt);
  cond_tokens.resize(max_tokens_);
  std::vector<int32_t> uncond_tokens = tokenize("");
  uncond_tokens.resize(max_tokens_);
  stats_.tokenizer_parsing_end_ms = time_in_ms();

  std::vector<Result<MethodMeta>> method_metas = get_methods_meta();

  MethodMeta encoder_method_meta = method_metas[0].get();
  // Initialize text_encoder input tensors: cond/uncond tokenized_input[1,77]
  auto cond_tokens_tensor = from_blob(
      cond_tokens.data(),
      {1, 77},
      encoder_method_meta.input_tensor_meta(0)->scalar_type());
  auto uncond_tokens_tensor = from_blob(
      uncond_tokens.data(),
      {1, 77},
      encoder_method_meta.input_tensor_meta(0)->scalar_type());
  // Initialize text_encoder output tensors: cond/uncond embedding[1, 77, 1024]
  constexpr int emb_size = 1 * 77 * 1024;
  std::vector<uint16_t> cond_emb_vec(emb_size);
  std::vector<uint16_t> uncond_emb_vec(emb_size);
  std::vector<float> fp_emb_vec(emb_size);
  auto cond_emb_tensor = from_blob(
      cond_emb_vec.data(),
      {1, 77, 1024},
      encoder_method_meta.output_tensor_meta(0)->scalar_type());
  auto uncond_emb_tensor = from_blob(
      uncond_emb_vec.data(),
      {1, 77, 1024},
      encoder_method_meta.output_tensor_meta(0)->scalar_type());
  modules_[0]->set_output(cond_emb_tensor);
  long encoder_start = time_in_ms();
  auto cond_res = modules_[0]->forward(cond_tokens_tensor);
  stats_.text_encoder_execution_time += (time_in_ms() - encoder_start);
  modules_[0]->set_output(uncond_emb_tensor);
  encoder_start = time_in_ms();
  auto uncond_res = modules_[0]->forward(uncond_tokens_tensor);
  stats_.text_encoder_execution_time += (time_in_ms() - encoder_start);

  // Initialize unet parameters
  MethodMeta unet_method_meta = method_metas[1].get();
  std::vector<float> time_steps = get_time_steps();
  std::vector<float> sigmas = get_sigmas(time_steps);
  float max_sigma = *std::max_element(sigmas.begin(), sigmas.end());
  std::vector<float> latent;
  if (fix_latents_) {
    latent = gen_latent_from_file();
  } else {
    latent = gen_random_latent(max_sigma);
  }
  std::vector<float> prev_sample(latent.size());

  // Initialize unet input tensors
  //  1. latent[1,64,64,4]
  //  2. time_embedding[1,1280]
  //  3. cond/uncond embedding[1,77,1024]
  std::vector<uint16_t> latent_model_input(latent.size());
  std::vector<float> fp_latent_model_input(latent.size());
  auto latent_tensor = from_blob(
      latent_model_input.data(),
      {1, 64, 64, 4},
      unet_method_meta.input_tensor_meta(0)->scalar_type());
  std::vector<TensorPtr> time_emb_tensors;
  time_emb_tensors.reserve(num_time_steps_);
  for (auto step_index = 0; step_index < num_time_steps_; step_index++) {
    time_emb_tensors.emplace_back(from_blob(
        time_emb_list_[step_index].data(),
        {1, 1280},
        unet_method_meta.input_tensor_meta(1)->scalar_type()));
  }
  // requantize text encoders output
  dequant_tensor(
      cond_emb_vec,
      fp_emb_vec,
      text_encoder_output_scale_,
      text_encoder_output_offset_);
  quant_tensor(
      fp_emb_vec,
      cond_emb_vec,
      unet_input_text_emb_scale_,
      unet_input_text_emb_offset_);
  dequant_tensor(
      uncond_emb_vec,
      fp_emb_vec,
      text_encoder_output_scale_,
      text_encoder_output_offset_);
  quant_tensor(
      fp_emb_vec,
      uncond_emb_vec,
      unet_input_text_emb_scale_,
      unet_input_text_emb_offset_);

  // Initialize unet output tensors: text/uncond noise_pred[1,64,64,4]
  std::vector<uint16_t> noise_pred_text(latent.size());
  std::vector<uint16_t> noise_pred_uncond(latent.size());
  std::vector<float> fp_noise_pred_text(noise_pred_text.size());
  std::vector<float> fp_noise_pred_uncond(noise_pred_uncond.size());
  auto noise_pred_text_tensor = from_blob(
      noise_pred_text.data(),
      {1, 64, 64, 4},
      unet_method_meta.output_tensor_meta(0)->scalar_type());
  auto noise_pred_uncond_tensor = from_blob(
      noise_pred_uncond.data(),
      {1, 64, 64, 4},
      unet_method_meta.output_tensor_meta(0)->scalar_type());

  // Execute unet
  for (int step_index = 0; step_index < num_time_steps_; step_index++) {
    long start_post_process = time_in_ms();
    scale_model_input(latent, fp_latent_model_input, sigmas[step_index]);

    quant_tensor(
        fp_latent_model_input,
        latent_model_input,
        unet_input_latent_scale_,
        unet_input_latent_offset_);

    stats_.unet_aggregate_post_processing_time +=
        (time_in_ms() - start_post_process);
    modules_[1]->set_output(noise_pred_text_tensor);
    long start_unet_execution = time_in_ms();
    auto cond_res = modules_[1]->forward(
        {latent_tensor, time_emb_tensors[step_index], cond_emb_tensor});
    stats_.unet_aggregate_execution_time +=
        (time_in_ms() - start_unet_execution);
    modules_[1]->set_output(noise_pred_uncond_tensor);
    start_unet_execution = time_in_ms();
    auto uncond_res = modules_[1]->forward(
        {latent_tensor,
         time_emb_tensors[step_index],
         uncond_emb_tensor}); // results in noise_pred_uncond_vec
    stats_.unet_aggregate_execution_time +=
        (time_in_ms() - start_unet_execution);

    // start unet post processing
    start_post_process = time_in_ms();

    dequant_tensor(
        noise_pred_text,
        fp_noise_pred_text,
        unet_output_scale_,
        unet_output_offset_);
    dequant_tensor(
        noise_pred_uncond,
        fp_noise_pred_uncond,
        unet_output_scale_,
        unet_output_offset_);

    for (int i = 0; i < fp_noise_pred_text.size(); i++) {
      fp_noise_pred_text[i] = fp_noise_pred_uncond[i] +
          guidance_scale_ * (fp_noise_pred_text[i] - fp_noise_pred_uncond[i]);
    }
    step(fp_noise_pred_text, sigmas, latent, prev_sample, step_index);
    stats_.unet_aggregate_post_processing_time +=
        (time_in_ms() - start_post_process);
  }

  // Start VAE
  MethodMeta vae_method_meta = method_metas[2].get();
  // Initialize vae input tensor : latent[1,64,64,4]
  std::vector<uint16_t> vae_input(latent.size());
  auto vae_input_tensor = from_blob(
      vae_input.data(),
      {1, 64, 64, 4},
      vae_method_meta.input_tensor_meta(0)->scalar_type());
  // Intialize vae output tensor: output[1,512,512,3]
  constexpr int image_size = 1 * 512 * 512 * 3;
  std::vector<uint16_t> q_out(image_size);
  std::vector<float> out(image_size);
  auto output_tensor = from_blob(
      q_out.data(),
      {1, 512, 512, 3},
      vae_method_meta.output_tensor_meta(0)->scalar_type());

  quant_tensor(latent, vae_input, vae_input_scale_, vae_input_offset_);

  modules_[2]->set_output(output_tensor);
  long start_vae_execution = time_in_ms();
  auto vae_res = modules_[2]->forward(vae_input_tensor);
  stats_.vae_execution_time = (time_in_ms() - start_vae_execution);
  stats_.generate_end_ms = time_in_ms();

  // Dequant uint16 output to fp32 output
  dequant_tensor(q_out, out, vae_output_scale_, vae_output_offset_);

  // Saving outputs
  auto output_file_name = output_path_ + "/output_0_0.raw";
  std::ofstream fout(output_file_name.c_str(), std::ios::binary);
  fout.write(
      reinterpret_cast<const char*>(out.data()), out.size() * sizeof(float));
  fout.close();

  return Error::Ok;
}

Error Runner::print_performance() {
  ET_LOG(Info, "\tTotal Number of steps:\t\t\t\t%d", num_time_steps_);

  ET_LOG(
      Info,
      "\tTokenizer Load Time:\t\t\t\t%f (seconds)",
      ((double)(stats_.tokenizer_load_end_ms - stats_.tokenizer_load_start_ms) /
       stats_.SCALING_FACTOR_UNITS_PER_SECOND));

  ET_LOG(
      Info,
      "\tModel Load Time:\t\t\t\t%f (seconds)",
      ((double)(stats_.model_load_end_ms - stats_.model_load_start_ms) /
       stats_.SCALING_FACTOR_UNITS_PER_SECOND));

  ET_LOG(
      Info,
      "\tGenerate Time(Tokenize + Encoder + UNet + VAE):\t%f (seconds)",
      ((double)(stats_.generate_end_ms - stats_.generate_start_ms) /
       stats_.SCALING_FACTOR_UNITS_PER_SECOND));

  ET_LOG(
      Info,
      "\tTokenize Time:\t\t\t\t\t%f (seconds)",
      ((double)(stats_.tokenizer_parsing_end_ms -
                stats_.tokenizer_parsing_start_ms) /
       stats_.SCALING_FACTOR_UNITS_PER_SECOND));

  ET_LOG(
      Info,
      "\tText Encoder Execution Time:\t\t\t%f (seconds)",
      ((double)(stats_.text_encoder_execution_time) /
       stats_.SCALING_FACTOR_UNITS_PER_SECOND));

  ET_LOG(
      Info,
      "\tUnet Aggregate (Cond + Uncond) Execution Time:\t%f (seconds)",
      ((double)stats_.unet_aggregate_execution_time /
       (stats_.SCALING_FACTOR_UNITS_PER_SECOND)));

  ET_LOG(
      Info,
      "\tUnet Average Execution Time:\t\t\t%f (seconds)",
      ((double)(stats_.unet_aggregate_execution_time / (num_time_steps_ * 2)) /
       (stats_.SCALING_FACTOR_UNITS_PER_SECOND)));

  ET_LOG(
      Info,
      "\tUnet Aggregate Post-Processing Time:\t\t%f (seconds)",
      ((double)(stats_.unet_aggregate_post_processing_time) /
       stats_.SCALING_FACTOR_UNITS_PER_SECOND));

  ET_LOG(
      Info,
      "\tUnet Average Post-Processing Time:\t\t%f (seconds)",
      ((double)(stats_.unet_aggregate_post_processing_time /
                (num_time_steps_ * 2)) /
       (stats_.SCALING_FACTOR_UNITS_PER_SECOND)));

  ET_LOG(
      Info,
      "\tVAE Execution Time:\t\t\t\t%f (seconds)",
      ((double)(stats_.vae_execution_time) /
       stats_.SCALING_FACTOR_UNITS_PER_SECOND));
  return Error::Ok;
}

} // namespace example
