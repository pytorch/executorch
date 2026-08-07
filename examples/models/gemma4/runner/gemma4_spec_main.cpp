/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/gemma4/runner/gemma4_spec_runner.h>

#include <charconv>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using ::executorch::examples::gemma4::Gemma4SpecLoadMode;
using ::executorch::examples::gemma4::Gemma4SpecRunner;
using ::executorch::examples::gemma4::Gemma4SpecRunnerConfig;
using ::executorch::examples::gemma4::validate_gemma4_spec_request;
using ::executorch::runtime::Error;

struct Arguments {
  std::string pte;
  std::vector<std::string> ptd;
  std::vector<int64_t> prompt_ids;
  std::vector<int64_t> stop_tokens;
  size_t max_new_tokens = 0;
  Gemma4SpecLoadMode load_mode = Gemma4SpecLoadMode::File;
};

bool parse_int64(std::string_view value, int64_t* output) {
  const char* begin = value.data();
  const char* end = begin + value.size();
  const auto result = std::from_chars(begin, end, *output);
  return result.ec == std::errc() && result.ptr == end;
}

bool parse_size(std::string_view value, size_t* output) {
  const char* begin = value.data();
  const char* end = begin + value.size();
  const auto result = std::from_chars(begin, end, *output);
  return result.ec == std::errc() && result.ptr == end && *output > 0;
}

bool parse_token_list(std::string_view value, std::vector<int64_t>* output) {
  size_t start = 0;
  while (start <= value.size()) {
    const size_t end = value.find(',', start);
    const std::string_view token = value.substr(start, end - start);
    int64_t parsed = -1;
    if (token.empty() || !parse_int64(token, &parsed)) {
      return false;
    }
    output->push_back(parsed);
    if (end == std::string_view::npos) {
      break;
    }
    start = end + 1;
  }
  return true;
}

bool parse_arguments(int argc, char** argv, Arguments* arguments) {
  for (int index = 1; index < argc; ++index) {
    const std::string_view option(argv[index]);
    if (option == "--mmap") {
      arguments->load_mode = Gemma4SpecLoadMode::Mmap;
      continue;
    }
    if (index + 1 >= argc) {
      return false;
    }
    const std::string_view value(argv[++index]);
    if (option == "--pte") {
      arguments->pte = value;
    } else if (option == "--ptd") {
      arguments->ptd.emplace_back(value);
    } else if (option == "--prompt-ids") {
      if (!parse_token_list(value, &arguments->prompt_ids)) {
        return false;
      }
    } else if (option == "--stop-token") {
      int64_t token = -1;
      if (!parse_int64(value, &token)) {
        return false;
      }
      arguments->stop_tokens.push_back(token);
    } else if (option == "--max-new-tokens") {
      if (!parse_size(value, &arguments->max_new_tokens)) {
        return false;
      }
    } else {
      return false;
    }
  }
  return !arguments->pte.empty() && arguments->ptd.size() == 3 &&
      !arguments->prompt_ids.empty() && arguments->max_new_tokens > 0;
}

void print_usage(const char* program) {
  std::cerr << "Usage: " << program
            << " --pte MODEL.pte --ptd A.ptd --ptd B.ptd --ptd C.ptd"
               " --prompt-ids ID[,ID...] --max-new-tokens N"
               " [--stop-token ID] [--mmap]\n";
}

} // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    print_usage(argv[0]);
    return 0;
  }
  Arguments arguments;
  if (!parse_arguments(argc, argv, &arguments)) {
    print_usage(argv[0]);
    return 2;
  }

  const Gemma4SpecRunnerConfig config;
  if (validate_gemma4_spec_request(
          config,
          arguments.prompt_ids,
          arguments.max_new_tokens,
          arguments.stop_tokens) != Error::Ok) {
    std::cerr << "Invalid prompt, token budget, stop token, or capacity\n";
    return 3;
  }

  Gemma4SpecRunner runner(config);
  const Error load_error =
      runner.load(arguments.pte, std::move(arguments.ptd), arguments.load_mode);
  if (load_error != Error::Ok) {
    std::cerr << "Failed to load `k2_round`: "
              << static_cast<uint32_t>(load_error) << '\n';
    return 4;
  }
  auto trace = runner.generate(
      arguments.prompt_ids, arguments.max_new_tokens, arguments.stop_tokens);
  if (!trace.ok()) {
    std::cerr << "Generation failed: " << static_cast<uint32_t>(trace.error())
              << '\n';
    (void)runner.unload();
    return 5;
  }
  for (size_t index = 0; index < trace->tokens.size(); ++index) {
    std::cout << (index == 0 ? "" : " ") << trace->tokens[index];
  }
  std::cout << '\n';
  return runner.unload() == Error::Ok ? 0 : 6;
}
