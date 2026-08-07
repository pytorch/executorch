/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/gemma4/runner/gemma4_spec_runner.h>

#include <executorch/backends/webgpu/runner/webgpu_model_loader.h>
#include <executorch/backends/webgpu/runtime/WebGPUDevice.h>
#include <executorch/extension/tensor/tensor.h>

#include <cstdlib>
#include <deque>
#include <exception>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string_view>
#include <utility>

namespace executorch::examples::gemma4 {
namespace {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::backends::webgpu::compare_and_set_default_webgpu_context;
using ::executorch::backends::webgpu::create_webgpu_context;
using ::executorch::backends::webgpu::destroy_webgpu_context;
using ::executorch::backends::webgpu::get_explicit_default_webgpu_context;
using ::executorch::backends::webgpu::load_webgpu_model;
using ::executorch::backends::webgpu::WebGPUContext;
using ::executorch::backends::webgpu::WebGPUModelLoadSpec;
using ::executorch::extension::make_tensor_ptr;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;
using ::executorch::runtime::MethodMeta;
using ::executorch::runtime::Result;
using ::executorch::runtime::Tag;
using ::executorch::runtime::TensorInfo;

bool verify_context(const WebGPUContext* context) {
  return context != nullptr && get_explicit_default_webgpu_context() == context;
}

bool shape_is(const Tensor& tensor, std::initializer_list<int32_t> expected) {
  if (tensor.dim() != static_cast<ssize_t>(expected.size())) {
    return false;
  }
  size_t index = 0;
  for (int32_t dimension : expected) {
    if (tensor.size(index++) != dimension) {
      return false;
    }
  }
  return true;
}

bool tensor_shape_is(
    const TensorInfo& tensor,
    std::initializer_list<int32_t> expected) {
  const auto sizes = tensor.sizes();
  if (sizes.size() != expected.size()) {
    return false;
  }
  size_t index = 0;
  for (int32_t dimension : expected) {
    if (sizes[index++] != dimension) {
      return false;
    }
  }
  return true;
}

bool all_tensor_tags(const MethodMeta& meta) {
  for (size_t index = 0; index < meta.num_inputs(); ++index) {
    auto tag = meta.input_tag(index);
    if (!tag.ok() || tag.get() != Tag::Tensor) {
      return false;
    }
  }
  for (size_t index = 0; index < meta.num_outputs(); ++index) {
    auto tag = meta.output_tag(index);
    if (!tag.ok() || tag.get() != Tag::Tensor) {
      return false;
    }
  }
  return true;
}

bool method_contract_is(
    const MethodMeta& meta,
    const Gemma4SpecRunnerConfig& config) {
  if (std::string_view(meta.name()) != config.method_name ||
      meta.num_inputs() != 4 || meta.num_outputs() != 5 ||
      meta.num_backends() != 1 || meta.num_instructions() != 1 ||
      !all_tensor_tags(meta)) {
    return false;
  }
  auto backend = meta.get_backend_name(0);
  if (!backend.ok() || std::string_view(backend.get()) != "VulkanBackend") {
    return false;
  }

  auto input_ids = meta.input_tensor_meta(0);
  auto input_positions = meta.input_tensor_meta(1);
  auto is_round = meta.input_tensor_meta(2);
  auto donor_length = meta.input_tensor_meta(3);
  if (!input_ids.ok() || !input_positions.ok() || !is_round.ok() ||
      !donor_length.ok() || input_ids->scalar_type() != ScalarType::Long ||
      input_positions->scalar_type() != ScalarType::Long ||
      is_round->scalar_type() != ScalarType::Long ||
      donor_length->scalar_type() != ScalarType::Long ||
      input_ids->sizes().size() != 2 || input_ids->sizes()[0] != 1 ||
      input_ids->sizes()[1] != config.max_input_length ||
      input_positions->sizes().size() != 1 ||
      input_positions->sizes()[0] != input_ids->sizes()[1] ||
      !tensor_shape_is(is_round.get(), {1}) ||
      !tensor_shape_is(donor_length.get(), {1, 1})) {
    return false;
  }

  auto candidates = meta.output_tensor_meta(0);
  auto target_greedy = meta.output_tensor_meta(1);
  auto output_matches = meta.output_tensor_meta(2);
  auto output_bonus = meta.output_tensor_meta(3);
  auto state_probe = meta.output_tensor_meta(4);
  return candidates.ok() && target_greedy.ok() && output_matches.ok() &&
      output_bonus.ok() && state_probe.ok() &&
      candidates->scalar_type() == ScalarType::Long &&
      target_greedy->scalar_type() == ScalarType::Long &&
      output_matches->scalar_type() == ScalarType::Long &&
      output_bonus->scalar_type() == ScalarType::Long &&
      state_probe->scalar_type() == ScalarType::Float &&
      tensor_shape_is(candidates.get(), {1, 2}) &&
      tensor_shape_is(target_greedy.get(), {1, 3}) &&
      tensor_shape_is(output_matches.get(), {1}) &&
      tensor_shape_is(output_bonus.get(), {1, 1}) &&
      tensor_shape_is(state_probe.get(), {1, 1});
}

bool request_fits_capacity(
    size_t prompt_depth,
    size_t token_budget,
    size_t speculative_tail,
    int64_t capacity) {
  if (capacity <= 0) {
    return false;
  }
  const auto limit = static_cast<uint64_t>(capacity);
  const auto prompt = static_cast<uint64_t>(prompt_depth);
  const auto tokens = static_cast<uint64_t>(token_budget);
  const auto tail = static_cast<uint64_t>(speculative_tail);
  return prompt <= limit && tokens <= limit - prompt &&
      tail <= limit - prompt - tokens;
}

bool position_range_fits_capacity(
    int64_t start_position,
    size_t count,
    int64_t capacity) {
  return start_position >= 0 && capacity >= start_position &&
      static_cast<uint64_t>(count) <=
      static_cast<uint64_t>(capacity - start_position);
}

#ifdef WGPU_BACKEND_ENABLE_PROFILING
void append_json_string(std::ostringstream& output, const std::string& value) {
  output << '"';
  for (const unsigned char character : value) {
    switch (character) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (character < 0x20) {
          constexpr char kHex[] = "0123456789abcdef";
          output << "\\u00" << kHex[character >> 4] << kHex[character & 0x0f];
        } else {
          output << static_cast<char>(character);
        }
    }
  }
  output << '"';
}

std::string serialize_profile_json(
    bool timestamp_supported,
    bool fresh,
    bool valid,
    const std::vector<::executorch::backends::webgpu::ShaderDuration>&
        durations,
    uint64_t execute_generation,
    uint64_t context_generation,
    uint64_t querypool_generation) {
  std::ostringstream output;
  output << std::fixed << std::setprecision(9)
         << "{\"schemaVersion\":1,\"supported\":"
         << (timestamp_supported ? "true" : "false")
         << ",\"fresh\":" << (fresh ? "true" : "false")
         << ",\"valid\":" << (valid ? "true" : "false")
         << ",\"context_generation\":" << context_generation
         << ",\"querypool_generation\":" << querypool_generation
         << ",\"execute_generation\":" << execute_generation;
  if (!timestamp_supported || !fresh || !valid) {
    output << ",\"total_kernel_ms\":0,\"pass_span_ms\":0,"
              "\"interpass_gap_ms\":0,\"perop\":[]}";
    return output.str();
  }

  struct Aggregate {
    uint64_t nanoseconds = 0;
    uint64_t calls = 0;
  };
  std::map<std::string, Aggregate> per_op;
  uint64_t total_nanoseconds = 0;
  uint64_t first_start = 0;
  uint64_t last_end = 0;
  bool have_timestamp = false;
  for (const auto& duration : durations) {
    auto& aggregate = per_op
        [duration.kernel_name.empty() ? std::string("dispatch")
                                      : duration.kernel_name];
    aggregate.nanoseconds += duration.execution_duration_ns;
    ++aggregate.calls;
    total_nanoseconds += duration.execution_duration_ns;
    if (!have_timestamp) {
      first_start = duration.start_time_ns;
      last_end = duration.end_time_ns;
      have_timestamp = true;
    } else {
      first_start = std::min(first_start, duration.start_time_ns);
      last_end = std::max(last_end, duration.end_time_ns);
    }
  }
  const uint64_t span_nanoseconds =
      have_timestamp && last_end > first_start ? last_end - first_start : 0;
  const uint64_t gap_nanoseconds = span_nanoseconds > total_nanoseconds
      ? span_nanoseconds - total_nanoseconds
      : 0;
  constexpr double kNanosecondsPerMillisecond = 1.0e6;
  output << ",\"total_kernel_ms\":"
         << total_nanoseconds / kNanosecondsPerMillisecond
         << ",\"pass_span_ms\":"
         << span_nanoseconds / kNanosecondsPerMillisecond
         << ",\"interpass_gap_ms\":"
         << gap_nanoseconds / kNanosecondsPerMillisecond << ",\"perop\":[";
  bool first = true;
  for (const auto& [name, aggregate] : per_op) {
    if (!first) {
      output << ',';
    }
    first = false;
    output << "{\"op\":";
    append_json_string(output, name);
    output << ",\"ms\":" << aggregate.nanoseconds / kNanosecondsPerMillisecond
           << ",\"calls\":" << aggregate.calls << '}';
  }
  output << "]}";
  return output.str();
}
#endif

#ifndef WGPU_BACKEND_ENABLE_PROFILING
std::string unsupported_profile_json(
    uint64_t execute_generation,
    uint64_t context_generation) {
  return "{\"schemaVersion\":1,\"supported\":false,"
         "\"fresh\":false,\"valid\":false,"
         "\"context_generation\":" +
      std::to_string(context_generation) +
      ",\"querypool_generation\":0,"
      "\"execute_generation\":" +
      std::to_string(execute_generation) +
      ",\"total_kernel_ms\":0,\"pass_span_ms\":0,"
      "\"interpass_gap_ms\":0,\"perop\":[]}";
}
#endif

} // namespace

Error validate_gemma4_spec_request(
    const Gemma4SpecRunnerConfig& config,
    const std::vector<int64_t>& prompt_ids,
    size_t token_budget,
    const std::vector<int64_t>& stop_tokens) {
  constexpr size_t kSpeculativeTail = 2;
  if (config.vocab_size <= 0 || config.max_input_length <= 0 ||
      config.max_input_length > std::numeric_limits<int32_t>::max() ||
      config.target_capacity <= 0 || config.donor_capacity <= 0 ||
      prompt_ids.empty() || token_budget == 0) {
    return Error::InvalidArgument;
  }
  const auto valid_token = [&config](int64_t token) {
    return token >= 0 && token < config.vocab_size;
  };
  if (!std::all_of(prompt_ids.begin(), prompt_ids.end(), valid_token) ||
      !std::all_of(stop_tokens.begin(), stop_tokens.end(), valid_token)) {
    return Error::InvalidArgument;
  }

  const size_t speculative_tail = token_budget > 1 ? kSpeculativeTail : 0;
  if ((token_budget > 1 && prompt_ids.size() < 2) ||
      !request_fits_capacity(
          prompt_ids.size(),
          token_budget,
          speculative_tail,
          config.target_capacity) ||
      !request_fits_capacity(
          prompt_ids.size(),
          token_budget,
          speculative_tail,
          config.donor_capacity)) {
    return Error::InvalidArgument;
  }
  return Error::Ok;
}

class Gemma4SpecRunner::Impl final {
 public:
  explicit Impl(Gemma4SpecRunnerConfig runner_config)
      : config(std::move(runner_config)) {}

  bool valid_token(int64_t token) const {
    return token >= 0 && token < config.vocab_size;
  }

  void clear_controller_state() {
    buffered.clear();
    incremental_prefill.clear();
    execute_count = 0;
    accepted_drafts = 0;
    next_position = -1;
    last_emitted = -1;
  }

  void arm_profile() {
    consumed_profile_generation = profile_generation;
  }

  void clear_profile_binding() {
#ifdef WGPU_BACKEND_ENABLE_PROFILING
    has_profile_binding = false;
    bound_profile_generation = 0;
    bound_querypool_generation = 0;
#endif
  }

  bool enable_profile_environment() {
    if (profile_environment_owned) {
      return true;
    }
    const char* current = std::getenv("WEBGPU_TIMESTAMP_QUERY");
    previous_profile_environment =
        current != nullptr ? std::optional<std::string>(current) : std::nullopt;
    if (setenv("WEBGPU_TIMESTAMP_QUERY", "1", 1) != 0) {
      previous_profile_environment.reset();
      return false;
    }
    profile_environment_owned = true;
    return true;
  }

  void restore_profile_environment() {
    if (!profile_environment_owned) {
      return;
    }
    if (previous_profile_environment.has_value()) {
      (void)setenv(
          "WEBGPU_TIMESTAMP_QUERY", previous_profile_environment->c_str(), 1);
    } else {
      (void)unsetenv("WEBGPU_TIMESTAMP_QUERY");
    }
    previous_profile_environment.reset();
    profile_environment_owned = false;
  }

  Gemma4SpecRunnerConfig config;
  std::unique_ptr<WebGPUContext> context;
  std::unique_ptr<extension::Module> module;
  std::deque<int64_t> buffered;
  std::vector<int64_t> incremental_prefill;
  size_t execute_count = 0;
  size_t accepted_drafts = 0;
  int64_t next_position = -1;
  int64_t last_emitted = -1;
  bool method_fresh = false;
  bool method_healthy = false;
  bool profile_enabled = false;
  bool profile_environment_owned = false;
  std::optional<std::string> previous_profile_environment;
  uint64_t context_generation = 0;
  uint64_t profile_generation = 0;
  uint64_t consumed_profile_generation = 0;
#ifdef WGPU_BACKEND_ENABLE_PROFILING
  bool has_profile_binding = false;
  uint64_t bound_profile_generation = 0;
  uint64_t bound_querypool_generation = 0;
#endif
};

Gemma4SpecRunner::Gemma4SpecRunner(Gemma4SpecRunnerConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

Gemma4SpecRunner::~Gemma4SpecRunner() {
  (void)unload();
}

Error Gemma4SpecRunner::load(
    const std::string& pte_path,
    std::vector<std::string> ptd_paths,
    Gemma4SpecLoadMode load_mode) {
  if (pte_path.empty() || impl_->config.vocab_size <= 0 ||
      impl_->config.max_input_length <= 0 ||
      impl_->config.max_input_length > std::numeric_limits<int32_t>::max() ||
      impl_->config.target_capacity <= 0 || impl_->config.donor_capacity <= 0 ||
      impl_->config.method_name != "k2_round" || ptd_paths.size() != 3) {
    return Error::InvalidArgument;
  }

  const bool acquired_context = impl_->context == nullptr;
  if (acquired_context) {
    try {
      impl_->context = std::make_unique<WebGPUContext>(create_webgpu_context());
    } catch (const std::exception&) {
      impl_->context.reset();
      return Error::Internal;
    }
    if (!compare_and_set_default_webgpu_context(
            nullptr, impl_->context.get())) {
      destroy_webgpu_context(*impl_->context);
      impl_->context.reset();
      impl_->method_fresh = false;
      impl_->method_healthy = false;
      return Error::InvalidState;
    }
  } else if (!verify_context(impl_->context.get())) {
    impl_->method_fresh = false;
    impl_->method_healthy = false;
    return Error::InvalidState;
  }
  const auto release_acquired_context = [&]() {
    if (!acquired_context) {
      return true;
    }
    const bool released =
        compare_and_set_default_webgpu_context(impl_->context.get(), nullptr);
    destroy_webgpu_context(*impl_->context);
    impl_->context.reset();
    return released;
  };

  WebGPUModelLoadSpec spec;
  spec.pte_path = pte_path;
  spec.ptd_paths = std::move(ptd_paths);
  spec.required_methods = {impl_->config.method_name};
  spec.load_mode = load_mode == Gemma4SpecLoadMode::File
      ? extension::Module::LoadMode::File
      : extension::Module::LoadMode::Mmap;
  auto loaded = load_webgpu_model(std::move(spec));
  if (!loaded.ok()) {
    return release_acquired_context() ? loaded.error() : Error::InvalidState;
  }

  auto next_module = std::move(loaded.get());
  auto methods = next_module->method_names();
  auto meta = next_module->method_meta(impl_->config.method_name);
  if (!methods.ok() || methods->size() != 1 ||
      methods->count(impl_->config.method_name) != 1 || !meta.ok() ||
      !method_contract_is(meta.get(), impl_->config)) {
    if (next_module->is_method_loaded(impl_->config.method_name)) {
      (void)next_module->unload_method(impl_->config.method_name);
    }
    next_module.reset();
    return release_acquired_context() ? Error::InvalidProgram
                                      : Error::InvalidState;
  }
  if (!verify_context(impl_->context.get())) {
    (void)next_module->unload_method(impl_->config.method_name);
    next_module.reset();
    (void)release_acquired_context();
    return Error::InvalidState;
  }

  if (impl_->module != nullptr &&
      impl_->module->is_method_loaded(impl_->config.method_name) &&
      !impl_->module->unload_method(impl_->config.method_name)) {
    (void)next_module->unload_method(impl_->config.method_name);
    impl_->method_fresh = false;
    impl_->method_healthy = false;
    return Error::Internal;
  }
  impl_->module = std::move(next_module);
  impl_->clear_controller_state();
  impl_->method_fresh = true;
  impl_->method_healthy = true;
  impl_->profile_enabled = false;
  if (acquired_context) {
    ++impl_->context_generation;
  }
  impl_->profile_generation = 0;
  impl_->arm_profile();
  impl_->clear_profile_binding();
  impl_->restore_profile_environment();
  return Error::Ok;
}

Error Gemma4SpecRunner::reset() {
  if (impl_->module == nullptr ||
      !impl_->module->is_method_loaded(impl_->config.method_name) ||
      impl_->context == nullptr || !verify_context(impl_->context.get())) {
    impl_->method_healthy = false;
    impl_->method_fresh = false;
    impl_->clear_controller_state();
    return Error::InvalidState;
  }
  impl_->arm_profile();
  impl_->clear_profile_binding();
  impl_->clear_controller_state();
  if (!impl_->module->unload_method(impl_->config.method_name)) {
    impl_->method_healthy = false;
    impl_->method_fresh = false;
    return Error::Internal;
  }
  const Error error = impl_->module->load_method(impl_->config.method_name);
  impl_->method_healthy =
      error == Error::Ok && verify_context(impl_->context.get());
  impl_->method_fresh = impl_->method_healthy;
  return error != Error::Ok
      ? error
      : (impl_->method_healthy ? Error::Ok : Error::InvalidState);
}

Error Gemma4SpecRunner::unload() {
  if (impl_->context != nullptr && !verify_context(impl_->context.get())) {
    return Error::InvalidState;
  }
  impl_->profile_enabled = false;
  impl_->restore_profile_environment();
  bool method_unloaded = true;
  if (impl_->module != nullptr) {
    if (impl_->module->is_method_loaded(impl_->config.method_name)) {
      method_unloaded = impl_->module->unload_method(impl_->config.method_name);
    }
    impl_->module.reset();
  }
  impl_->clear_profile_binding();
  impl_->method_fresh = false;
  impl_->method_healthy = false;
  impl_->clear_controller_state();
  if (impl_->context != nullptr) {
    if (!compare_and_set_default_webgpu_context(
            impl_->context.get(), nullptr)) {
      return Error::InvalidState;
    }
    destroy_webgpu_context(*impl_->context);
    impl_->context.reset();
  }
  return method_unloaded ? Error::Ok : Error::Internal;
}

bool Gemma4SpecRunner::is_loaded() const {
  return impl_->module != nullptr && impl_->method_healthy &&
      impl_->context != nullptr && verify_context(impl_->context.get());
}

Result<Gemma4K2Output> Gemma4SpecRunner::execute(
    const std::vector<int64_t>& input_ids,
    const std::vector<int64_t>& input_positions,
    bool is_round,
    int64_t donor_length) {
  if (!is_loaded() || impl_->context == nullptr ||
      !verify_context(impl_->context.get())) {
    impl_->method_healthy = false;
    return Error::InvalidState;
  }
  if (input_ids.empty() || input_ids.size() != input_positions.size() ||
      donor_length < 0 || donor_length > impl_->config.donor_capacity ||
      !position_range_fits_capacity(
          input_positions.front(),
          input_positions.size(),
          impl_->config.target_capacity) ||
      !position_range_fits_capacity(
          input_positions.front(),
          input_positions.size(),
          impl_->config.donor_capacity)) {
    return Error::InvalidArgument;
  }
  if ((!is_round &&
       input_ids.size() >
           static_cast<size_t>(impl_->config.max_input_length)) ||
      (is_round && input_ids.size() != 3)) {
    return Error::InvalidArgument;
  }
  for (size_t index = 0; index < input_ids.size(); ++index) {
    if (!impl_->valid_token(input_ids[index]) || input_positions[index] < 0 ||
        (index > 0 &&
         input_positions[index] != input_positions[index - 1] + 1)) {
      return Error::InvalidArgument;
    }
  }
  const int64_t start_position = input_positions.front();
  if ((is_round && (donor_length != start_position || donor_length < 2)) ||
      (!is_round && start_position == 0 && donor_length != 2) ||
      (!is_round && start_position > 0 && donor_length != start_position)) {
    return Error::InvalidArgument;
  }

  auto ids =
      make_tensor_ptr({1, static_cast<int32_t>(input_ids.size())}, input_ids);
  auto positions = make_tensor_ptr(
      {static_cast<int32_t>(input_positions.size())}, input_positions);
  auto round = make_tensor_ptr({1}, std::vector<int64_t>{is_round ? 1 : 0});
  auto donor = make_tensor_ptr({1, 1}, std::vector<int64_t>{donor_length});

  impl_->clear_profile_binding();
  impl_->method_fresh = false;
  auto execution = impl_->module->execute(
      impl_->config.method_name,
      {EValue(ids), EValue(positions), EValue(round), EValue(donor)});
  if (!verify_context(impl_->context.get())) {
    impl_->method_healthy = false;
    return Error::InvalidState;
  }
  if (!execution.ok()) {
    impl_->method_healthy = false;
    return execution.error();
  }
  if (execution->size() != 5) {
    impl_->method_healthy = false;
    return Error::InvalidProgram;
  }
  for (const EValue& value : *execution) {
    if (!value.isTensor()) {
      impl_->method_healthy = false;
      return Error::InvalidType;
    }
  }

  const Tensor& candidates = execution->at(0).toTensor();
  const Tensor& target = execution->at(1).toTensor();
  const Tensor& matches = execution->at(2).toTensor();
  const Tensor& bonus = execution->at(3).toTensor();
  const Tensor& probe = execution->at(4).toTensor();
  if (candidates.scalar_type() != ScalarType::Long ||
      target.scalar_type() != ScalarType::Long ||
      matches.scalar_type() != ScalarType::Long ||
      bonus.scalar_type() != ScalarType::Long ||
      probe.scalar_type() != ScalarType::Float ||
      !shape_is(candidates, {1, 2}) || !shape_is(target, {1, 3}) ||
      !shape_is(matches, {1}) || !shape_is(bonus, {1, 1}) ||
      !shape_is(probe, {1, 1})) {
    impl_->method_healthy = false;
    return Error::InvalidProgram;
  }

  Gemma4K2Output output;
  const int64_t* candidate_data = candidates.const_data_ptr<int64_t>();
  const int64_t* target_data = target.const_data_ptr<int64_t>();
  output.candidates = {candidate_data[0], candidate_data[1]};
  output.target_greedy = {target_data[0], target_data[1], target_data[2]};
  output.match_count = matches.const_data_ptr<int64_t>()[0];
  output.bonus = bonus.const_data_ptr<int64_t>()[0];
  output.state_probe = probe.const_data_ptr<float>()[0];

  for (int64_t token : output.candidates) {
    if (!impl_->valid_token(token)) {
      impl_->method_healthy = false;
      return Error::InvalidProgram;
    }
  }
  for (int64_t token : output.target_greedy) {
    if (!impl_->valid_token(token)) {
      impl_->method_healthy = false;
      return Error::InvalidProgram;
    }
  }

  if (is_round) {
    const auto decision = reconcile_gemma4_k2(
        output, start_position, 3, {}, impl_->config.vocab_size);
    if (!decision.valid) {
      impl_->method_healthy = false;
      return Error::InvalidProgram;
    }
  } else if (
      output.match_count != 0 ||
      output.target_greedy[0] != output.target_greedy[1] ||
      output.target_greedy[1] != output.target_greedy[2] ||
      output.bonus != output.target_greedy[0] ||
      !impl_->valid_token(output.bonus) || !std::isfinite(output.state_probe)) {
    impl_->method_healthy = false;
    return Error::InvalidProgram;
  }
  ++impl_->execute_count;
  ++impl_->profile_generation;
#ifdef WGPU_BACKEND_ENABLE_PROFILING
  if (impl_->profile_enabled && impl_->context->timestamp_supported &&
      impl_->context->querypool != nullptr) {
    impl_->has_profile_binding = true;
    impl_->bound_profile_generation = impl_->profile_generation;
    impl_->bound_querypool_generation =
        impl_->context->querypool->result_generation();
  }
#endif
  return output;
}

Result<int64_t> Gemma4SpecRunner::prefill(
    const std::vector<int64_t>& input_ids,
    int64_t start_position) {
  if (input_ids.empty() ||
      input_ids.size() > static_cast<size_t>(impl_->config.max_input_length) ||
      !position_range_fits_capacity(
          start_position, input_ids.size(), impl_->config.target_capacity) ||
      !position_range_fits_capacity(
          start_position, input_ids.size(), impl_->config.donor_capacity) ||
      !std::all_of(input_ids.begin(), input_ids.end(), [this](int64_t token) {
        return impl_->valid_token(token);
      })) {
    return Error::InvalidArgument;
  }
  if (start_position == 0) {
    impl_->buffered.clear();
    impl_->incremental_prefill.clear();
    if (!impl_->method_fresh) {
      const Error error = reset();
      if (error != Error::Ok) {
        return error;
      }
    }
  } else if (start_position != impl_->next_position) {
    return Error::InvalidArgument;
  }
  std::vector<int64_t> positions;
  positions.reserve(input_ids.size());
  for (size_t index = 0; index < input_ids.size(); ++index) {
    positions.push_back(start_position + static_cast<int64_t>(index));
  }
  auto output = execute(
      input_ids, positions, false, start_position == 0 ? 2 : start_position);
  if (!output.ok()) {
    return output.error();
  }
  impl_->next_position =
      start_position + static_cast<int64_t>(input_ids.size());
  impl_->last_emitted = output->bonus;
  return output->bonus;
}

Error Gemma4SpecRunner::prefill_step(int64_t token, int64_t position) {
  const int64_t max_incremental_position = std::min(
      {impl_->config.max_input_length,
       impl_->config.target_capacity,
       impl_->config.donor_capacity});
  if (!is_loaded() || !impl_->valid_token(token) || position < 0 ||
      max_incremental_position <= 1 ||
      position >= max_incremental_position - 1) {
    return Error::InvalidArgument;
  }
  if (position == 0) {
    impl_->incremental_prefill.clear();
  }
  if (static_cast<size_t>(position) != impl_->incremental_prefill.size()) {
    impl_->incremental_prefill.clear();
    return Error::InvalidArgument;
  }
  impl_->incremental_prefill.push_back(token);
  return Error::Ok;
}

Result<int64_t> Gemma4SpecRunner::step(
    int64_t seed_token,
    int64_t seed_position) {
  const int64_t capacity =
      std::min(impl_->config.target_capacity, impl_->config.donor_capacity);
  if (!is_loaded() || !impl_->valid_token(seed_token) || seed_position < 1 ||
      seed_position >= capacity) {
    return Error::InvalidArgument;
  }
  if (!impl_->incremental_prefill.empty()) {
    if (static_cast<size_t>(seed_position) !=
            impl_->incremental_prefill.size() ||
        seed_position >= impl_->config.max_input_length) {
      impl_->incremental_prefill.clear();
      return Error::InvalidArgument;
    }
    impl_->incremental_prefill.push_back(seed_token);
    auto prompt = std::move(impl_->incremental_prefill);
    impl_->incremental_prefill.clear();
    return prefill(prompt, 0);
  }
  if (seed_position != impl_->next_position ||
      seed_token != impl_->last_emitted) {
    return Error::InvalidArgument;
  }
  if (!impl_->buffered.empty()) {
    const int64_t token = impl_->buffered.front();
    impl_->buffered.pop_front();
    impl_->last_emitted = token;
    ++impl_->next_position;
    return token;
  }

  if (!position_range_fits_capacity(
          seed_position, 3, impl_->config.target_capacity) ||
      !position_range_fits_capacity(
          seed_position, 3, impl_->config.donor_capacity)) {
    return Error::InvalidArgument;
  }

  auto output = execute(
      {seed_token, 0, 0},
      {seed_position, seed_position + 1, seed_position + 2},
      true,
      seed_position);
  if (!output.ok()) {
    return output.error();
  }
  const auto decision = reconcile_gemma4_k2(
      output.get(), seed_position, 3, {}, impl_->config.vocab_size);
  if (!decision.valid || decision.committed.empty()) {
    impl_->method_healthy = false;
    return Error::InvalidProgram;
  }
  impl_->accepted_drafts += decision.accepted_drafts;
  for (size_t index = 1; index < decision.committed.size(); ++index) {
    impl_->buffered.push_back(decision.committed[index]);
  }
  impl_->last_emitted = decision.committed.front();
  ++impl_->next_position;
  return decision.committed.front();
}

Result<Gemma4SpecTrace> Gemma4SpecRunner::generate(
    const std::vector<int64_t>& prompt_ids,
    size_t token_budget,
    const std::vector<int64_t>& stop_tokens) {
  const Error request_error = validate_gemma4_spec_request(
      impl_->config, prompt_ids, token_budget, stop_tokens);
  if (request_error != Error::Ok) {
    return request_error;
  }

  int64_t start_position = 0;
  int64_t prefill_token = -1;
  while (start_position < static_cast<int64_t>(prompt_ids.size())) {
    const int64_t count = std::min<int64_t>(
        impl_->config.max_input_length,
        static_cast<int64_t>(prompt_ids.size()) - start_position);
    std::vector<int64_t> chunk(
        prompt_ids.begin() + start_position,
        prompt_ids.begin() + start_position + count);
    auto result = prefill(chunk, start_position);
    if (!result.ok()) {
      return result.error();
    }
    prefill_token = result.get();
    start_position += count;
  }

  Gemma4SpecTrace trace;
  trace.prefill_token = prefill_token;
  if (std::find(stop_tokens.begin(), stop_tokens.end(), prefill_token) !=
      stop_tokens.end()) {
    trace.stop_token = prefill_token;
  } else {
    trace.tokens.push_back(prefill_token);
  }

  int64_t seed = prefill_token;
  while (!trace.stop_token.has_value() && trace.tokens.size() < token_budget) {
    const int64_t round_start = start_position;
    auto output = execute(
        {seed, 0, 0},
        {round_start, round_start + 1, round_start + 2},
        true,
        round_start);
    if (!output.ok()) {
      return output.error();
    }
    const auto decision = reconcile_gemma4_k2(
        output.get(),
        round_start,
        token_budget - trace.tokens.size(),
        stop_tokens,
        impl_->config.vocab_size);
    if (!decision.valid) {
      impl_->method_healthy = false;
      return Error::InvalidProgram;
    }
    trace.tokens.insert(
        trace.tokens.end(),
        decision.committed.begin(),
        decision.committed.end());
    trace.discarded_tokens += decision.discarded.size();
    trace.accepted_drafts += decision.accepted_drafts;
    trace.rounds.push_back(decision);
    if (decision.stopped) {
      trace.stop_token = decision.stop_token;
    }
    start_position = decision.next_position;
    seed = decision.next_seed;
  }
  impl_->accepted_drafts += trace.accepted_drafts;
  impl_->next_position = start_position;
  impl_->last_emitted = seed;
  trace.execute_count = impl_->execute_count;
  return trace;
}

void Gemma4SpecRunner::set_profiling_enabled(bool enabled) {
  impl_->arm_profile();
  if (impl_->profile_enabled != enabled) {
    impl_->clear_profile_binding();
  }
#ifdef WGPU_BACKEND_ENABLE_PROFILING
  if (enabled) {
    impl_->profile_enabled = impl_->enable_profile_environment();
  } else {
    impl_->profile_enabled = false;
    impl_->restore_profile_environment();
  }
#else
  (void)enabled;
  impl_->profile_enabled = false;
  impl_->restore_profile_environment();
#endif
}

std::string Gemma4SpecRunner::profile_json() {
#ifdef WGPU_BACKEND_ENABLE_PROFILING
  const bool supported =
      impl_->context != nullptr && impl_->context->timestamp_supported;
  const auto* querypool =
      impl_->context != nullptr ? impl_->context->querypool.get() : nullptr;
  const uint64_t querypool_generation =
      querypool != nullptr ? querypool->result_generation() : 0;
  const bool binding_current = impl_->profile_enabled &&
      impl_->has_profile_binding &&
      impl_->bound_profile_generation == impl_->profile_generation &&
      impl_->bound_querypool_generation == querypool_generation;
  const bool fresh = binding_current &&
      impl_->consumed_profile_generation != impl_->profile_generation;
  const bool valid =
      binding_current && querypool != nullptr && querypool->results_valid();
  if (fresh) {
    impl_->consumed_profile_generation = impl_->profile_generation;
  }
  const std::vector<::executorch::backends::webgpu::ShaderDuration>
      empty_durations;
  const auto& durations =
      querypool != nullptr ? querypool->results() : empty_durations;
  return serialize_profile_json(
      supported,
      fresh,
      valid,
      durations,
      impl_->profile_generation,
      impl_->context_generation,
      querypool_generation);
#else
  return unsupported_profile_json(
      impl_->profile_generation, impl_->context_generation);
#endif
}

size_t Gemma4SpecRunner::execute_count() const {
  return impl_->execute_count;
}

size_t Gemma4SpecRunner::accepted_drafts() const {
  return impl_->accepted_drafts;
}

size_t Gemma4SpecRunner::buffered_tokens() const {
  return impl_->buffered.size();
}

} // namespace executorch::examples::gemma4
