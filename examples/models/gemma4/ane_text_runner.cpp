/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @lint-ignore-every CLANGTIDY facebook-hte-Deprecated
 */

#include <gflags/gflags.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/log.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#endif

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

namespace {
bool readArgsFromFile(
    const std::string& filename,
    std::vector<std::string>& args,
    std::vector<char*>& argsAsCstr,
    int& argc,
    char**& argv) {
  args.clear();
  std::ifstream inputFile(filename);
  if (!inputFile.is_open()) {
    return false;
  }
  std::string line;
  while (std::getline(inputFile, line)) {
    auto not_empty =
        std::any_of(line.begin(), line.end(), [](unsigned char c) {
          return !std::isspace(c);
        });
    if (not_empty) {
      std::string unescaped;
      unescaped.reserve(line.size());
      for (size_t i = 0; i < line.size(); ++i) {
        if (line[i] == '\\' && i + 1 < line.size()) {
          if (line[i + 1] == 'n') {
            unescaped.push_back('\n');
            ++i;
          } else if (line[i + 1] == '\\') {
            unescaped.push_back('\\');
            ++i;
          } else {
            unescaped.push_back(line[i]);
          }
        } else {
          unescaped.push_back(line[i]);
        }
      }
      args.push_back(unescaped);
    }
  }
  inputFile.close();
  argc = args.size();
  argsAsCstr.resize(argc);
  for (int i = 0; i < argc; i++) {
    argsAsCstr[i] = const_cast<char*>(args[i].c_str());
  }
  argv = argsAsCstr.data();
  return true;
}
} // namespace

using executorch::extension::Module;
using executorch::extension::from_blob;
using executorch::extension::zeros;
using executorch::runtime::EValue;
using executorch::runtime::Error;

using Clock = std::chrono::steady_clock;

DEFINE_string(
    model_path,
    "/tmp/gemma4_backbone_coreml.pte",
    "CoreML ANE PTE model path.");
DEFINE_string(
    tokenizer_path,
    "/tmp/tokenizer.model",
    "Tokenizer model path.");
DEFINE_string(
    prompt,
    "What is the capital of France?",
    "Prompt for generation.");
DEFINE_int32(cpu_threads, 4, "Number of CPU threads.");
DEFINE_int32(
    seq_len,
    32,
    "Static sequence length matching the exported model. "
    "Every call passes exactly [1, seq_len] input_ids and [seq_len] input_pos.");
DEFINE_string(
    method_name,
    "forward",
    "Method name to execute in the PTE model.");

std::pair<double, double> getMemoryFootprintMB() {
#if defined(__APPLE__)
  task_vm_info_data_t vm_info;
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  kern_return_t kr =
      task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vm_info, &count);
  if (kr == KERN_SUCCESS) {
    double phys_mb =
        static_cast<double>(vm_info.phys_footprint) / (1024.0 * 1024.0);
    double peak_mb = -1.0;
    if (count >= TASK_VM_INFO_COUNT) {
      // NOLINTNEXTLINE(clang-analyzer-core.UndefinedBinaryOperatorResult)
      peak_mb = static_cast<double>(vm_info.ledger_phys_footprint_peak) /
          (1024.0 * 1024.0);
    }
    return {phys_mb, peak_mb};
  }
#endif
  return {-1.0, -1.0};
}

void emitMemoryRow(const std::string& phase) {
  auto [phys_mb, peak_mb] = getMemoryFootprintMB();
  printf("MEMORY_ROW\t%s\t%.2f\t%.2f\n", phase.c_str(), phys_mb, peak_mb);
  fflush(stdout);
}

static double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

#if defined(__APPLE__) && defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE == 1
extern "C" int32_t benchmark_main(int32_t argc, char** argv);
extern "C" int32_t benchmark_main(int32_t argc, char** argv) {
#else
int32_t main(int32_t argc, char** argv) {
#endif
#if defined(__APPLE__) && defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE == 1
  const char* home = std::getenv("HOME");
  std::string home_dir = home ? home : "";
#else
  std::string home_dir(""); // NOLINT(readability-redundant-string-init)
#endif

  std::vector<std::string> args_string;
  std::vector<char*> args_cstr;
  const std::string args_filename = home_dir + "/tmp/args.txt";
  if (argc < 2) {
    auto success =
        readArgsFromFile(args_filename, args_string, args_cstr, argc, argv);
    ET_LOG(Info, "readArgsFromFile success=%d", success);
  }
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const std::string model_path = home_dir + FLAGS_model_path;
  const std::string tokenizer_path = home_dir + FLAGS_tokenizer_path;

#if defined(ET_USE_THREADPOOL)
  uint32_t num_threads = FLAGS_cpu_threads <= 0
      ? ::executorch::extension::cpuinfo::get_num_performant_cores()
      : static_cast<uint32_t>(FLAGS_cpu_threads);
  if (num_threads == 0) {
    num_threads = std::min(std::thread::hardware_concurrency(), 8u);
  }
  ET_LOG(Info, "Setting threadpool to %d threads", (int)num_threads);
  ::executorch::extension::threadpool::get_threadpool()
      ->_unsafe_reset_threadpool(num_threads);
#endif

  ET_LOG(
      Info,
      "gemma_ane_prefill: model_path=%s, tokenizer_path=%s, prompt=%s, "
      "method=%s, cpu_threads=%d",
      model_path.c_str(),
      tokenizer_path.c_str(),
      FLAGS_prompt.c_str(),
      FLAGS_method_name.c_str(),
      FLAGS_cpu_threads);

  emitMemoryRow("before_load");

  // Load model
  auto load_start = Clock::now();
  Module module(model_path, Module::LoadMode::Mmap);
  auto err = module.load_method(FLAGS_method_name);
  if (err != Error::Ok) {
    ET_LOG(
        Error,
        "Failed to load method '%s'",
        FLAGS_method_name.c_str());
    return 1;
  }
  double load_ms = elapsed_ms(load_start);
  ET_LOG(Info, "Model loaded in %.1f ms", load_ms);

  emitMemoryRow("after_load");

  // Load tokenizer
  auto tokenizer =
      executorch::extension::llm::load_tokenizer(tokenizer_path);
  if (!tokenizer) {
    ET_LOG(Error, "Failed to load tokenizer");
    return 1;
  }
  ET_LOG(Info, "Tokenizer loaded");

  // Tokenize prompt
  static constexpr int64_t kBosId = 2;
  static constexpr int64_t kTurnStartId = 105;
  static constexpr int64_t kTurnEndId = 106;

  auto user_res = tokenizer->encode("user\n", 0, 0);
  ET_CHECK_MSG(user_res.ok(), "Failed to encode user token");
  auto prompt_res = tokenizer->encode(FLAGS_prompt, 0, 0);
  ET_CHECK_MSG(prompt_res.ok(), "Failed to encode prompt");
  auto newline_res = tokenizer->encode("\n", 0, 0);
  ET_CHECK_MSG(newline_res.ok(), "Failed to encode newline token");
  auto model_res = tokenizer->encode("model\n", 0, 0);
  ET_CHECK_MSG(model_res.ok(), "Failed to encode model token");

  std::vector<int64_t> input_ids;
  input_ids.push_back(kBosId);
  input_ids.push_back(kTurnStartId);
  for (auto t : user_res.get())
    input_ids.push_back(static_cast<int64_t>(t));
  for (auto t : prompt_res.get())
    input_ids.push_back(static_cast<int64_t>(t));
  input_ids.push_back(kTurnEndId);
  for (auto t : newline_res.get())
    input_ids.push_back(static_cast<int64_t>(t));
  input_ids.push_back(kTurnStartId);
  for (auto t : model_res.get())
    input_ids.push_back(static_cast<int64_t>(t));

  int32_t num_prompt_tokens = static_cast<int32_t>(input_ids.size());
  const int32_t sl = FLAGS_seq_len;
  ET_LOG(
      Info,
      "Prompt tokenized: %d tokens, static seq_len: %d",
      num_prompt_tokens,
      sl);

  // Build static-shape input tensors matching the exported model:
  //   input 0: input_ids [1, seq_len] int32
  //   input 1: input_pos [seq_len]    int64
  // Shapes are static — pad with 0 if prompt is shorter, truncate if longer.
  std::vector<int32_t> input_ids_i32(sl, 0);
  int32_t tokens_to_copy = std::min(num_prompt_tokens, sl);
  for (int32_t i = 0; i < tokens_to_copy; ++i)
    input_ids_i32[i] = static_cast<int32_t>(input_ids[i]);
  auto input_ids_tensor = from_blob(
      input_ids_i32.data(), {1, sl}, executorch::aten::ScalarType::Int);

  std::vector<int64_t> positions(sl);
  for (int32_t i = 0; i < sl; ++i)
    positions[i] = i;
  auto input_pos =
      from_blob(positions.data(), {sl}, executorch::aten::ScalarType::Long);

  if (num_prompt_tokens > sl) {
    ET_LOG(
        Info,
        "Prompt truncated from %d to %d tokens",
        num_prompt_tokens,
        sl);
  }

  // Prefill: 2 flat inputs — input_ids [1, seq_len] int32, input_pos [seq_len] int64
  ET_LOG(Info, "Running prefill (%d tokens, seq_len=%d)...", tokens_to_copy, sl);
  auto prefill_start = Clock::now();
  auto prefill_result = module.execute(
      FLAGS_method_name,
      {EValue(input_ids_tensor), EValue(input_pos)});
  double prefill_ms = elapsed_ms(prefill_start);

  if (!prefill_result.ok()) {
    ET_LOG(Error, "Prefill failed");
    return 1;
  }
  ET_LOG(Info, "Prefill completed in %.1f ms", prefill_ms);

  emitMemoryRow("after_inference");

  double prefill_s = prefill_ms / 1000.0;
  double prefill_tps = sl / prefill_s;

  printf(
      "LATENCY_ROW\t%d\t%.4f\t%.4f\t%.2f\t%.2f\t%d\t%d\t%.4f\n",
      0,
      prefill_s,
      0.0,
      prefill_tps,
      0.0,
      sl,
      0,
      prefill_s);
  fflush(stdout);

  auto [phys_mb, peak_mb] = getMemoryFootprintMB();
  printf("\n========== BENCHMARK SUMMARY ==========\n");
  printf("Mode:               prefill-only\n");
  printf("Method:             %s\n", FLAGS_method_name.c_str());
  printf("Prompt tokens:      %d\n", sl);
  printf("Model load:         %.4f s\n", load_ms / 1000.0);
  printf("Prefill latency:    %.4f s\n", prefill_s);
  printf("Prefill tok/s:      %.2f\n", prefill_tps);
  printf("Current memory:     %.2f MB\n", phys_mb);
  printf("Peak memory:        %.2f MB\n", peak_mb);
  printf("========== END SUMMARY ==========\n");
  fflush(stdout);

  const std::string bench_done_path = home_dir + "/tmp/BENCH_DONE";
  std::ofstream bench_done_file(bench_done_path);
  if (bench_done_file.is_open()) {
    bench_done_file << "done" << std::endl;
    bench_done_file.close();
    ET_LOG(Info, "BENCH_DONE marker written to %s", bench_done_path.c_str());
  }

  return 0;
}
