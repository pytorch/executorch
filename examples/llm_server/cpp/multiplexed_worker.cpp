/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/llm_server/cpp/multiplexed_worker.h>
#include <executorch/examples/llm_server/cpp/multiplexed_worker_test.h>

#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <iterator>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

namespace executorch::examples::llm_server {
namespace {
namespace serving = extension::llm::serving;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using Id = std::uint64_t;
using testing::Checkpoint;

const char* error_code(serving::ErrorCode code) {
  switch (code) {
    case serving::ErrorCode::InvalidArgument:
      return "invalid_argument";
    case serving::ErrorCode::NotReady:
      return "not_ready";
    case serving::ErrorCode::SessionNotFound:
      return "session_not_found";
    case serving::ErrorCode::SessionBusy:
      return "session_busy";
    case serving::ErrorCode::CapacityExceeded:
      return "capacity_exhausted";
    case serving::ErrorCode::Internal:
      return "internal";
  }
  return "internal";
}

Json failure(const char* code, const std::string& message) {
  return {{"error", message}, {"code", code}};
}

Id unsigned_integer(const Json& value, bool positive = false) {
  if (!(value.is_number_unsigned() ||
        (value.is_number_integer() && value.get<std::int64_t>() >= 0))) {
    throw std::invalid_argument("expected unsigned integer");
  }
  auto result = value.get<Id>();
  if (positive && result == 0) {
    throw std::invalid_argument("request IDs must be positive");
  }
  return result;
}

void fields(const Json& object, std::initializer_list<const char*> allowed) {
  if (!object.is_object()) {
    throw std::invalid_argument("expected object");
  }
  for (auto it = object.begin(); it != object.end(); ++it) {
    bool known = false;
    for (auto name : allowed) {
      known |= it.key() == name;
    }
    if (!known) {
      throw std::invalid_argument("unsupported field: " + it.key());
    }
  }
}

std::string session_key(const Json& request) {
  const auto& key = request.at("session_id");
  if (!key.is_string() || key.get_ref<const std::string&>().empty() ||
      key.get_ref<const std::string&>().size() > 1024) {
    throw std::invalid_argument(
        "session_id must be a nonempty string of at most 1024 bytes");
  }
  return key.get<std::string>();
}

extension::llm::EncodedImage image_input(
    const Json& value,
    std::size_t max_bytes) {
  fields(value, {"mime_type", "data"});
  auto mime = value.at("mime_type").get<std::string>();
  if (mime != "image/jpeg" && mime != "image/png") {
    throw std::invalid_argument(
        "image MIME type must be image/jpeg or image/png");
  }
  const auto& encoded = value.at("data").get_ref<const std::string&>();
  if (encoded.empty() || encoded.size() % 4 != 0) {
    throw std::invalid_argument("image requires padded base64");
  }
  const std::size_t padding =
      (encoded.back() == '=') + (encoded[encoded.size() - 2] == '=');
  const auto size = encoded.size() / 4 * 3 - padding;
  if (size == 0 || size > max_bytes) {
    throw std::invalid_argument("encoded image exceeds byte limit");
  }
  auto digit = [](unsigned char c) -> int {
    if (c >= 'A' && c <= 'Z')
      return c - 'A';
    if (c >= 'a' && c <= 'z')
      return c - 'a' + 26;
    if (c >= '0' && c <= '9')
      return c - '0' + 52;
    if (c == '+')
      return 62;
    if (c == '/')
      return 63;
    return -1;
  };
  std::vector<std::uint8_t> data;
  data.reserve(size);
  for (std::size_t i = 0; i < encoded.size(); i += 4) {
    const bool last = i + 4 == encoded.size();
    const auto a = digit(encoded[i]);
    const auto b = digit(encoded[i + 1]);
    const auto c = last && padding == 2 ? 0 : digit(encoded[i + 2]);
    const auto d = last && padding != 0 ? 0 : digit(encoded[i + 3]);
    if (a < 0 || b < 0 || c < 0 || d < 0 ||
        (last && padding == 2 && (b & 15) != 0) ||
        (last && padding == 1 && (c & 3) != 0)) {
      throw std::invalid_argument("image requires canonical base64");
    }
    data.push_back(static_cast<std::uint8_t>((a << 2) | (b >> 4)));
    if (!last || padding < 2)
      data.push_back(static_cast<std::uint8_t>((b << 4) | (c >> 2)));
    if (!last || padding == 0)
      data.push_back(static_cast<std::uint8_t>((c << 6) | d));
  }
  constexpr std::uint8_t png[] = {137, 80, 78, 71, 13, 10, 26, 10};
  const bool signature = mime == "image/png" ? data.size() >= sizeof(png) &&
          std::equal(std::begin(png), std::end(png), data.begin())
                                             : data.size() >= 3 &&
          data[0] == 0xff && data[1] == 0xd8 && data[2] == 0xff;
  if (!signature) {
    throw std::invalid_argument("image contents do not match MIME type");
  }
  return {std::move(data), std::move(mime)};
}

serving::PromptInput prompt_input(
    const Json& request,
    const serving::ServingInfo& info) {
  if (request.contains("prompt") == request.contains("prompt_segments")) {
    throw std::invalid_argument(
        "supply exactly one of prompt and prompt_segments");
  }
  serving::PromptInput prompt;
  if (request.contains("prompt")) {
    prompt.segments.emplace_back(request.at("prompt").get<std::string>());
    return prompt;
  }
  const auto& segments = request.at("prompt_segments");
  if (!segments.is_array()) {
    throw std::invalid_argument("prompt_segments must be an array");
  }
  std::size_t images = 0;
  for (const auto& segment : segments) {
    fields(segment, {"text", "ids", "image"});
    if (segment.size() != 1) {
      throw std::invalid_argument(
          "prompt segment requires exactly one of text, ids, and image");
    }
    if (segment.contains("image")) {
      if (++images > info.max_images) {
        throw std::invalid_argument("image capability limit exceeded");
      }
      prompt.segments.emplace_back(
          image_input(segment.at("image"), info.max_image_encoded_bytes));
    } else if (segment.contains("text")) {
      prompt.segments.emplace_back(segment.at("text").get<std::string>());
    } else {
      const auto& ids = segment.at("ids");
      if (!ids.is_array()) {
        throw std::invalid_argument("ids must be an array");
      }
      std::vector<std::uint64_t> tokens(ids.size());
      std::transform(
          ids.begin(), ids.end(), tokens.begin(), [](const Json& id) {
            return unsigned_integer(id);
          });
      prompt.segments.emplace_back(std::move(tokens));
    }
  }
  return prompt;
}

serving::GenerationOptions generation_options(const Json& request) {
  serving::GenerationOptions options;
  if (request.contains("max_new_tokens")) {
    const auto& value = request.at("max_new_tokens");
    const bool in_range = value.is_number_unsigned()
        ? value.get<std::uint64_t>() <=
            static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())
        : value.is_number_integer() && value.get<std::int64_t>() >= -1 &&
            value.get<std::int64_t>() <=
                std::numeric_limits<std::int32_t>::max();
    if (!in_range) {
      throw std::invalid_argument(
          "max_new_tokens must be an integer in [-1, INT32_MAX]");
    }
    const auto count = value.get<std::int32_t>();
    if (count != -1) {
      options.max_new_tokens = count;
    }
  }
  auto number =
      [&](const char* name, double fallback, double low, double high) {
        if (!request.contains(name))
          return fallback;
        const auto& value = request.at(name);
        if (!value.is_number())
          throw std::invalid_argument("expected numeric sampling field");
        const auto result = value.get<double>();
        if (!std::isfinite(result) || result < low || result > high) {
          throw std::invalid_argument("sampling field out of range");
        }
        return result;
      };
  options.sampling.temperature =
      number("temperature", 0, 0, std::numeric_limits<float>::max());
  options.sampling.top_p =
      number("top_p", 1, std::numeric_limits<float>::min(), 1);
  if (request.contains("top_k")) {
    auto value = unsigned_integer(request.at("top_k"));
    if (value > std::numeric_limits<std::int32_t>::max()) {
      throw std::invalid_argument("top_k exceeds INT32_MAX");
    }
    options.sampling.top_k = value;
  }
  if (request.contains("seed")) {
    const auto seed = unsigned_integer(request.at("seed"));
    if (seed != 0)
      options.seed = seed;
  }
  if (request.contains("stop")) {
    const auto& stops = request.at("stop");
    if (!stops.is_array())
      throw std::invalid_argument("stop must be an array");
    for (const auto& stop : stops)
      options.stop_strings.push_back(stop.get<std::string>());
  }
  return options;
}

Json terminal_json(const serving::TerminalEvent& event) {
  if (event.error)
    return failure(error_code(event.error->code), event.error->message);
  if (event.finish_reason == serving::FinishReason::Failed) {
    return failure("internal", "generation failed");
  }
  const auto& s = event.stats;
  Json out = {
      {"done", true},
      {"finish_reason",
       event.finish_reason == serving::FinishReason::Length ? "length"
                                                            : "stop"},
      {"cancelled", event.finish_reason == serving::FinishReason::Cancelled},
      {"prompt_tokens", s.prompt_tokens},
      {"completion_tokens", s.completion_tokens},
      {"reused_prompt_tokens", s.reused_prompt_tokens},
      {"prefilled_prompt_tokens", s.prefilled_prompt_tokens},
      {"prompt_positions", s.prompt_positions},
      {"reused_prompt_positions", s.reused_prompt_positions},
      {"prefilled_prompt_positions", s.prefilled_prompt_positions},
      {"prefill_ms", s.prefill_ms},
      {"decode_ms", s.decode_ms},
      {"total_ms", s.total_ms},
      {"prefill_tok_s",
       s.prefill_ms > 0 ? s.prefilled_prompt_tokens * 1000.0 / s.prefill_ms
                        : 0},
      {"decode_tok_s",
       s.decode_ms > 0 ? s.completion_tokens * 1000.0 / s.decode_ms : 0},
      {"session_reset_reason", s.session_reset_reason}};
  if (s.generated_token_ids)
    out["generated_token_ids"] = *s.generated_token_ids;
  return out;
}

struct Operation {
  Id id;
  bool control = false;
  bool terminal = false;
  bool cancelled = false;
  bool overflow = false;
  std::size_t frames = 0;
  std::size_t bytes = 0;
  serving::RequestHandle handle;
};

struct Frame {
  std::shared_ptr<Operation> operation;
  std::string bytes;
  bool terminal;
};

class Worker {
 public:
  Worker(
      serving::ServingRuntime& runtime,
      int input,
      int output,
      MultiplexedWorkerConfig config,
      testing::WorkerTestHooks hooks)
      : runtime_(runtime),
        input_(input),
        output_(output),
        config_(config),
        hooks_(std::move(hooks)) {}

  int run() {
    const auto deadline = Clock::now() + config_.startup_timeout;
    while (!runtime_.info().ready && Clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if (!runtime_.info().ready) {
      runtime_.shutdown();
      return 1;
    }
    const int flags = fcntl(output_, F_GETFL);
    if (flags < 0 || fcntl(output_, F_SETFL, flags | O_NONBLOCK) < 0) {
      runtime_.shutdown();
      return 1;
    }
    try {
      const auto info = runtime_.info();
      queue_.push_back(
          {nullptr,
           Json({{"ready", true},
                 {"multiplexed", true},
                 {"max_named_sessions", info.max_sessions},
                 {"max_inflight_requests", config_.max_inflight_requests},
                 {"supports_images", info.max_images != 0},
                 {"max_images", info.max_images},
                 {"max_image_bytes", info.max_image_encoded_bytes},
                 {"max_image_dimension", info.max_image_dimension},
                 {"max_image_pixels", info.max_image_pixels}})
                   .dump() +
               "\n",
           false});
      writer_ = std::thread([this] { write_loop(); });
      read_loop();
    } catch (...) {
      failed_ = true;
    }
    // shutdown waits for every admitted callback; none wait for pipe I/O.
    runtime_.shutdown();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      drained_ = true;
    }
    changed_.notify_all();
    if (writer_.joinable())
      writer_.join();
    fcntl(output_, F_SETFL, flags);
    return failed_ ? 1 : 0;
  }

 private:
  void checkpoint(Checkpoint point, Id id) {
    if (hooks_.checkpoint)
      hooks_.checkpoint(point, id);
  }

  void complete(const std::shared_ptr<Operation>& op, Json message) {
    // mutex_ held; one terminal slot of max_frame_bytes reserved per operation.
    if (op->terminal)
      return;
    op->terminal = true;
    if (op->overflow)
      message = failure("slow_consumer", "worker output backlog exceeded");
    message["request_id"] = op->id;
    auto bytes =
        message.dump(-1, ' ', false, Json::error_handler_t::replace) + "\n";
    if (bytes.size() > config_.max_frame_bytes) {
      bytes = Json({{"request_id", op->id},
                    {"error", "terminal frame exceeds limit"},
                    {"code", "frame_too_large"}})
                  .dump() +
          "\n";
    }
    queue_.push_back({op, std::move(bytes), true});
    changed_.notify_all();
  }

  void event(
      const std::shared_ptr<Operation>& op,
      const serving::GenerationEvent& event) {
    serving::RequestHandle cancel;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (failed_ || op->terminal)
        return;
      if (const auto* terminal = std::get_if<serving::TerminalEvent>(&event)) {
        complete(op, terminal_json(*terminal));
        return;
      }
      if (op->overflow)
        return;
      auto bytes = Json({{"request_id", op->id},
                         {"token", std::get<serving::TextEvent>(event).text}})
                       .dump(-1, ' ', false, Json::error_handler_t::replace) +
          "\n";
      if (bytes.size() > config_.max_frame_bytes ||
          op->frames >= config_.token_frames_per_request ||
          bytes.size() > config_.token_bytes_per_request - op->bytes) {
        op->overflow = op->cancelled = true;
        cancel = op->handle;
        // Discard unsent text only with an explicit failure terminal.
        for (auto it = queue_.begin(); it != queue_.end();) {
          if (it->operation == op && !it->terminal)
            it = queue_.erase(it);
          else
            ++it;
        }
        op->bytes = op->frames = 0;
      } else {
        ++op->frames;
        op->bytes += bytes.size();
        queue_.push_back({op, std::move(bytes), false});
        changed_.notify_all();
        return;
      }
    }
    checkpoint(Checkpoint::OverflowLatched, op->id);
    cancel.cancel();
  }

  void request(const std::string& line) {
    std::vector<std::unordered_set<std::string>> object_keys;
    bool duplicate = false;
    bool ambiguous_id = false;
    auto message =
        Json::parse(line, [&](int, Json::parse_event_t event, Json& value) {
          if (event == Json::parse_event_t::object_start) {
            object_keys.emplace_back();
          } else if (event == Json::parse_event_t::object_end) {
            object_keys.pop_back();
          } else if (event == Json::parse_event_t::key) {
            const auto& key = value.get_ref<const std::string&>();
            if (!object_keys.back().insert(key).second) {
              duplicate = true;
              ambiguous_id |= object_keys.size() == 1 && key == "request_id";
            }
          }
          return true;
        });
    if (ambiguous_id) {
      throw std::invalid_argument("ambiguous request_id");
    }
    const Id id = unsigned_integer(message.at("request_id"), true);
    auto op = std::make_shared<Operation>();
    op->id = id;
    std::string name;
    try {
      name = message.at("op").get<std::string>();
    } catch (...) {
    }
    op->control = name == "cancel";
    {
      std::lock_guard<std::mutex> lock(mutex_);
      // Duplicate identity cannot safely receive a second terminal on this
      // wire.
      if (operations_.count(id))
        throw std::invalid_argument("duplicate in-flight request_id");
      if (requests_ >= config_.max_inflight_requests && !op->control) {
        op->control = true;
        name = "overloaded";
      }
      if (op->control && controls_ >= config_.max_inflight_requests) {
        throw std::invalid_argument("control response budget exhausted");
      }
      operations_.emplace(id, op);
      ++(op->control ? controls_ : requests_);
    }
    checkpoint(Checkpoint::Admitted, id);
    try {
      if (duplicate) {
        throw std::invalid_argument("duplicate JSON field");
      }
      if (name == "overloaded") {
        std::lock_guard<std::mutex> lock(mutex_);
        complete(
            op,
            failure(
                "capacity_exhausted", "worker operation capacity exhausted"));
      } else if (name == "generate") {
        fields(
            message,
            {"op",
             "request_id",
             "prompt",
             "prompt_segments",
             "session_id",
             "max_new_tokens",
             "temperature",
             "top_p",
             "top_k",
             "seed",
             "stop"});
        std::optional<std::string> key;
        if (message.contains("session_id"))
          key = session_key(message);
        auto prompt = prompt_input(message, runtime_.info());
        auto options = generation_options(message);
        // Register first. This serial reader retains op through handle binding,
        // even if a callback publishes its terminal and the writer retires it.
        auto result = runtime_.generate(
            std::move(key),
            std::move(prompt),
            std::move(options),
            [this, weak = std::weak_ptr<Operation>(op)](
                const serving::GenerationEvent& update) {
              try {
                if (auto active = weak.lock()) {
                  event(active, update);
                  if (std::holds_alternative<serving::TerminalEvent>(update))
                    checkpoint(Checkpoint::TerminalEnqueued, active->id);
                }
              } catch (...) {
                failed_ = true;
                changed_.notify_all();
              }
            });
        checkpoint(Checkpoint::BeforeBind, id);
        serving::RequestHandle cancel;
        {
          std::lock_guard<std::mutex> lock(mutex_);
          if (const auto* error = std::get_if<serving::ServingError>(&result)) {
            complete(op, failure(error_code(error->code), error->message));
          } else {
            op->handle = std::get<serving::RequestHandle>(std::move(result));
            if (op->cancelled)
              cancel = op->handle;
          }
        }
        if (hooks_.handle_bound && op->handle.id() != 0)
          hooks_.handle_bound(id, op->handle);
        checkpoint(Checkpoint::Bound, id);
        cancel.cancel();
      } else if (name == "cancel") {
        fields(message, {"op", "request_id", "target_request_id"});
        const auto target =
            unsigned_integer(message.at("target_request_id"), true);
        serving::RequestHandle cancel;
        {
          std::lock_guard<std::mutex> lock(mutex_);
          const auto it = operations_.find(target);
          if (it != operations_.end() && !it->second->terminal) {
            it->second->cancelled = true;
            cancel = it->second->handle;
          }
          complete(op, {{"cancelled", true}});
        }
        cancel.cancel();
      } else if (name == "open" || name == "close" || name == "reset") {
        fields(message, {"op", "request_id", "session_id"});
        auto key = session_key(message);
        const char* ack = name == "open" ? "opened"
            : name == "close"            ? "closed"
                                         : "reset";
        auto completion = [this, weak = std::weak_ptr<Operation>(op), ack](
                              serving::LifecycleResult error) {
          try {
            if (auto active = weak.lock()) {
              {
                std::lock_guard<std::mutex> lock(mutex_);
                if (failed_)
                  return;
                complete(
                    active,
                    error ? failure(error_code(error->code), error->message)
                          : Json{{ack, true}});
              }
              checkpoint(Checkpoint::TerminalEnqueued, active->id);
            }
          } catch (...) {
            failed_ = true;
            changed_.notify_all();
          }
        };
        if (name == "open")
          runtime_.open_session_async(std::move(key), std::move(completion));
        else if (name == "close")
          runtime_.close_session_async(std::move(key), std::move(completion));
        else
          runtime_.reset_session_async(std::move(key), std::move(completion));
      } else {
        throw std::invalid_argument("unsupported or missing op");
      }
    } catch (const std::exception& error) {
      std::lock_guard<std::mutex> lock(mutex_);
      complete(op, failure("invalid_argument", error.what()));
    }
  }

  void read_loop() {
    std::string line;
    char buffer[4096];
    while (!failed_) {
      pollfd descriptor{input_, POLLIN, 0};
      const auto ready = poll(&descriptor, 1, 20);
      if (ready < 0 && errno == EINTR)
        continue;
      if (ready < 0)
        throw std::runtime_error("input poll failed");
      if (ready == 0)
        continue;
      const auto count = read(input_, buffer, sizeof(buffer));
      if (count < 0 && (errno == EINTR || errno == EAGAIN))
        continue;
      if (count < 0)
        throw std::runtime_error("input read failed");
      if (count == 0) {
        if (!line.empty())
          throw std::runtime_error("incomplete input frame");
        return;
      }
      for (ssize_t i = 0; i < count && !failed_; ++i) {
        if (line.size() + 1 > config_.max_frame_bytes)
          throw std::runtime_error("input frame exceeds limit");
        if (buffer[i] == '\n') {
          request(line);
          line.clear();
        } else
          line += buffer[i];
      }
    }
  }

  void write_loop() {
    try {
      while (!failed_) {
        Frame frame;
        {
          std::unique_lock<std::mutex> lock(mutex_);
          changed_.wait(
              lock, [this] { return failed_ || drained_ || !queue_.empty(); });
          if (failed_ || queue_.empty())
            return;
          frame = std::move(queue_.front());
          queue_.pop_front();
          if (frame.operation && !frame.terminal) {
            --frame.operation->frames;
            frame.operation->bytes -= frame.bytes.size();
          }
        }
        const auto deadline = Clock::now() + config_.write_timeout;
        std::size_t offset = 0;
        const auto payload_size = frame.bytes.size() - (frame.terminal ? 1 : 0);
        while (offset < frame.bytes.size() && !failed_) {
          if (Clock::now() >= deadline)
            throw std::runtime_error("output deadline exceeded");
          pollfd descriptor{output_, POLLOUT, 0};
          auto ready = poll(&descriptor, 1, 20);
          if (ready < 0 && errno == EINTR)
            continue;
          if (ready < 0)
            throw std::runtime_error("output poll failed");
          if (ready == 0)
            continue;
          ssize_t count;
          if (frame.terminal && offset == payload_size) {
            // Publish the JSONL delimiter and release admission atomically
            // with respect to the reader. Only this nonblocking byte write
            // runs locked; polling and payload writes always run unlocked.
            const auto error = hooks_.terminal_write_error
                ? hooks_.terminal_write_error(frame.operation->id)
                : 0;
            if (error) {
              errno = error;
              count = -1;
            } else {
              std::lock_guard<std::mutex> lock(mutex_);
              count = write(output_, frame.bytes.data() + offset, 1);
              if (count == 1) {
                const auto& op = frame.operation;
                operations_.erase(op->id);
                --(op->control ? controls_ : requests_);
              }
            }
          } else {
            count = write(
                output_, frame.bytes.data() + offset, payload_size - offset);
          }
          if (count < 0 && (errno == EINTR || errno == EAGAIN))
            continue;
          if (count <= 0)
            throw std::runtime_error("output write failed");
          offset += count;
        }
        if (offset == frame.bytes.size() && frame.operation && frame.terminal) {
          checkpoint(Checkpoint::TerminalPublished, frame.operation->id);
        }
      }
    } catch (...) {
      failed_ = true;
      changed_.notify_all();
    }
  }

  serving::ServingRuntime& runtime_;
  int input_, output_;
  MultiplexedWorkerConfig config_;
  testing::WorkerTestHooks hooks_;
  std::mutex mutex_;
  std::condition_variable changed_;
  std::unordered_map<Id, std::shared_ptr<Operation>> operations_;
  std::deque<Frame> queue_;
  std::size_t requests_ = 0, controls_ = 0;
  std::atomic<bool> failed_{false};
  bool drained_ = false;
  std::thread writer_;
};
} // namespace

int run_multiplexed_worker(
    serving::ServingRuntime& runtime,
    int input_fd,
    int output_fd,
    MultiplexedWorkerConfig config) {
  return testing::run_multiplexed_worker(
      runtime, input_fd, output_fd, config, {});
}

int testing::run_multiplexed_worker(
    serving::ServingRuntime& runtime,
    int input_fd,
    int output_fd,
    MultiplexedWorkerConfig config,
    WorkerTestHooks hooks) {
  if (config.max_inflight_requests == 0 || config.max_frame_bytes < 1024 ||
      config.token_frames_per_request == 0 ||
      config.token_bytes_per_request == 0 ||
      config.write_timeout.count() <= 0 ||
      config.startup_timeout.count() <= 0) {
    runtime.shutdown();
    return 1;
  }
  return Worker(runtime, input_fd, output_fd, config, std::move(hooks)).run();
}
} // namespace executorch::examples::llm_server
