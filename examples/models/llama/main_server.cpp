/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// OpenAI-compatible HTTP server wrapping the ExecuTorch Llama runner.
// Exposes POST /v1/chat/completions (streaming & non-streaming).

#include <executorch/examples/models/llama/runner/runner.h>
#include <gflags/gflags.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#ifdef ET_EVENT_TRACER_ENABLED
#include <executorch/devtools/etdump/etdump_flatcc.h>
#endif

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

using json = nlohmann::json;

// ── flags (same as main.cpp) ────────────────────────────────────────────────

DEFINE_string(
    model_path,
    "llama3_2.pte",
    "Model serialized in flatbuffer format.");

DEFINE_string(
    data_paths,
    "",
    "Comma-separated data files for the model.");

DEFINE_string(tokenizer_path, "tokenizer.model", "Tokenizer stuff.");

DEFINE_double(
    temperature,
    0.8f,
    "Temperature; Default is 0.8f. 0 = greedy argmax sampling.");

DEFINE_int32(
    seq_len,
    4000,
    "Total number of tokens to generate (prompt + output).");

DEFINE_int32(
    cpu_threads,
    -1,
    "Number of CPU threads for inference. -1 = heuristic.");

DEFINE_bool(warmup, false, "Whether to run a warmup run.");

DEFINE_string(host, "0.0.0.0", "Server listen address.");
DEFINE_int32(port, 8080, "Server listen port.");
DEFINE_string(model_name, "llama", "Model name returned in API responses.");

// ── helpers ─────────────────────────────────────────────────────────────────

static std::vector<std::string> parseStringList(const std::string& input) {
  std::vector<std::string> result;
  if (input.empty())
    return result;
  std::stringstream ss(input);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item.erase(0, item.find_first_not_of(" \t"));
    item.erase(item.find_last_not_of(" \t") + 1);
    if (!item.empty())
      result.push_back(item);
  }
  return result;
}

static std::string generate_id() {
  static std::atomic<uint64_t> counter{0};
  return "chatcmpl-" + std::to_string(counter.fetch_add(1));
}

static int64_t unix_timestamp() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

static std::string messages_to_prompt(const json& messages) {
  std::string prompt;
  for (const auto& msg : messages) {
    std::string role = msg.value("role", "");
    std::string content = msg.value("content", "");
    if (role == "system") {
      prompt += "System: " + content + "\n";
    } else if (role == "user") {
      prompt += "User: " + content + "\n";
    } else if (role == "assistant") {
      prompt += "Assistant: " + content + "\n";
    } else {
      prompt += content + "\n";
    }
  }
  prompt += "Assistant:";
  return prompt;
}

// ── main ────────────────────────────────────────────────────────────────────

int32_t main(int32_t argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Same setup as main.cpp
  const char* model_path = FLAGS_model_path.c_str();
  std::vector<std::string> data_paths = parseStringList(FLAGS_data_paths);
  const char* tokenizer_path = FLAGS_tokenizer_path.c_str();
  float temperature = FLAGS_temperature;
  int32_t seq_len = FLAGS_seq_len;
  int32_t cpu_threads = FLAGS_cpu_threads;

#if defined(ET_USE_THREADPOOL)
  uint32_t num_performant_cores = cpu_threads == -1
      ? ::executorch::extension::cpuinfo::get_num_performant_cores()
      : static_cast<uint32_t>(cpu_threads);
  ET_LOG(
      Info, "Resetting threadpool with num threads = %d", num_performant_cores);
  if (num_performant_cores > 0) {
    ::executorch::extension::threadpool::get_threadpool()
        ->_unsafe_reset_threadpool(num_performant_cores);
  }
#endif

#ifdef ET_EVENT_TRACER_ENABLED
  auto etdump_gen_ptr = std::make_unique<executorch::etdump::ETDumpGen>();
#endif

  // Create llama runner (same as main.cpp)
  std::unique_ptr<::executorch::extension::llm::TextLLMRunner> runner =
      example::create_llama_runner(
          model_path,
          tokenizer_path,
          data_paths,
          temperature,
#ifdef ET_EVENT_TRACER_ENABLED
          std::move(etdump_gen_ptr)
#else
          nullptr
#endif
      );

  if (runner == nullptr) {
    ET_LOG(Error, "Failed to create llama runner");
    return 1;
  }

  if (FLAGS_warmup) {
    auto error = runner->warmup("warmup", /*max_new_tokens=*/seq_len);
    if (error != executorch::runtime::Error::Ok) {
      ET_LOG(Error, "Failed to warmup llama runner");
      return 1;
    }
  }

  // Serialise access to the runner (one request at a time).
  std::mutex runner_mutex;

  // ── HTTP server ──────────────────────────────────────────────────────────

  httplib::Server svr;

  // Health check
  svr.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
    json body = {
        {"object", "list"},
        {"data",
         {{{"id", FLAGS_model_name},
           {"object", "model"},
           {"owned_by", "executorch"}}}}};
    res.set_content(body.dump(), "application/json");
  });

  // POST /v1/chat/completions
  svr.Post(
      "/v1/chat/completions",
      [&](const httplib::Request& req, httplib::Response& res) {
        json request;
        try {
          request = json::parse(req.body);
        } catch (const json::parse_error& e) {
          res.status = 400;
          json err = {
              {"error",
               {{"message", std::string("Invalid JSON: ") + e.what()},
                {"type", "invalid_request_error"}}}};
          res.set_content(err.dump(), "application/json");
          return;
        }

        if (!request.contains("messages") || !request["messages"].is_array()) {
          res.status = 400;
          json err = {
              {"error",
               {{"message", "Missing or invalid 'messages' field."},
                {"type", "invalid_request_error"}}}};
          res.set_content(err.dump(), "application/json");
          return;
        }

        std::string prompt = messages_to_prompt(request["messages"]);
        bool stream = request.value("stream", false);
        float req_temperature =
            request.value("temperature", static_cast<float>(temperature));
        int32_t max_tokens = request.value("max_tokens", -1);

        // Build generation config — same logic as main.cpp
        executorch::extension::llm::GenerationConfig config;
        config.temperature = req_temperature;
        if (max_tokens > 0) {
          config.max_new_tokens = max_tokens;
        } else {
          config.seq_len = seq_len;
        }

        printf(
            "[request] prompt=%zu chars, stream=%s, temp=%.2f, seq_len=%d, max_new_tokens=%d\n",
            prompt.size(),
            stream ? "true" : "false",
            req_temperature,
            config.seq_len,
            config.max_new_tokens);
        printf("[prompt] %s\n", prompt.c_str());
        fflush(stdout);

        std::string id = generate_id();
        int64_t created = unix_timestamp();
        std::string model_name = FLAGS_model_name;

        std::lock_guard<std::mutex> lock(runner_mutex);
        runner->reset();

        if (stream) {
          // ── Server-Sent Events streaming ─────────────────────────────────
          // Generate all tokens first, collecting SSE payloads, then send.
          // This avoids dangling-reference issues with httplib's chunked
          // content provider and matches main.cpp's generate-then-return flow.
          std::string sse_body;

          auto token_cb = [&](const std::string& token) {
            json chunk = {
                {"id", id},
                {"object", "chat.completion.chunk"},
                {"created", created},
                {"model", model_name},
                {"choices",
                 {{{"index", 0},
                   {"delta", {{"content", token}}},
                   {"finish_reason", nullptr}}}}};
            sse_body += "data: " + chunk.dump() + "\n\n";
          };

          auto error = runner->generate(prompt, config, token_cb);

          // Final chunk with finish_reason = "stop"
          json final_chunk = {
              {"id", id},
              {"object", "chat.completion.chunk"},
              {"created", created},
              {"model", model_name},
              {"choices",
               {{{"index", 0},
                 {"delta", json::object()},
                 {"finish_reason", "stop"}}}}};
          sse_body += "data: " + final_chunk.dump() + "\n\n";
          sse_body += "data: [DONE]\n\n";

          res.set_content(sse_body, "text/event-stream");
        } else {
          // ── Non-streaming response ───────────────────────────────────────
          std::string full_response;
          int32_t completion_tokens = 0;

          auto token_cb = [&](const std::string& token) {
            full_response += token;
            completion_tokens++;
          };

          auto error = runner->generate(prompt, config, token_cb);

          json body = {
              {"id", id},
              {"object", "chat.completion"},
              {"created", created},
              {"model", model_name},
              {"choices",
               {{{"index", 0},
                 {"message",
                  {{"role", "assistant"}, {"content", full_response}}},
                 {"finish_reason", "stop"}}}},
              {"usage",
               {{"prompt_tokens", -1},
                {"completion_tokens", completion_tokens},
                {"total_tokens", -1}}}};
          res.set_content(body.dump(), "application/json");
        }
      });

  printf("============================================\n");
  printf("  ExecuTorch Llama Server\n");
  printf("============================================\n");
  printf("Model path:     %s\n", FLAGS_model_path.c_str());
  printf("Tokenizer path: %s\n", FLAGS_tokenizer_path.c_str());
  printf("Model name:     %s\n", FLAGS_model_name.c_str());
  printf("Temperature:    %.2f\n", static_cast<float>(FLAGS_temperature));
  printf("Seq len:        %d\n", FLAGS_seq_len);
  printf("CPU threads:    %d\n", FLAGS_cpu_threads);
  printf("Warmup:         %s\n", FLAGS_warmup ? "true" : "false");
  printf("--------------------------------------------\n");
  printf("Listening on %s:%d\n", FLAGS_host.c_str(), FLAGS_port);
  printf("Endpoints:\n");
  printf("  GET  /v1/models\n");
  printf("  POST /v1/chat/completions\n");
  printf("============================================\n");
  fflush(stdout);

  if (!svr.listen(FLAGS_host, FLAGS_port)) {
    ET_LOG(Error, "Failed to start server");
    return 1;
  }

  return 0;
}
