/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Generic pybind wrappers for the Engine/Session serving API, factored out so a
// model's example-local module can expose the SAME LLMEngine / LLMSession
// Python surface without writing its own per-model wrapper classes. The generic
// `_llm_runner` module and an example module both call
// bind_engine_session_api(m) to register the classes (module_local), and an
// example module constructs its own engine and hands it to
// PyLLMEngine(std::unique_ptr<LLMEngine>).

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/llm_session.h>

#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace executorch::extension::llm::pybind_wrappers {

namespace py = pybind11;

inline void throw_if_error(
    ::executorch::runtime::Error error,
    std::string msg) {
  if (error != ::executorch::runtime::Error::Ok) {
    throw std::runtime_error(std::move(msg));
  }
}

// A session handle (LLMSession), the model-agnostic per-conversation API.
// Backend calls (prefill_tokens/decode_one) take the engine-owned lock so
// concurrent sessions of one engine serialize (Module::execute isn't assumed
// thread-safe); cheap state ops (seek/reset/position/stop) don't.
class PyLLMSession {
 public:
  PyLLMSession(
      std::unique_ptr<LLMSession> session,
      std::shared_ptr<std::mutex> exec_mutex)
      : session_(std::move(session)), exec_mutex_(std::move(exec_mutex)) {}

  void prefill_tokens(std::vector<uint64_t> tokens, float temperature = -1.0f) {
    py::gil_scoped_release release;
    auto exec_lock = lock_exec();
    // Pass the first-token sampling so backends that sample during prefill
    // (in-graph sampling) use the request's temperature, not a stale default.
    SamplingConfig sampling;
    sampling.temperature = temperature;
    throw_if_error(
        session_->prefill_tokens(std::move(tokens), &sampling),
        "prefill_tokens failed");
  }

  py::dict decode_one(float temperature = -1.0f) {
    uint64_t token_id;
    std::string text;
    bool is_eos;
    {
      py::gil_scoped_release release;
      auto exec_lock = lock_exec();
      SamplingConfig sampling;
      sampling.temperature = temperature;
      auto res = session_->decode_one(sampling);
      throw_if_error(res.error(), "decode_one failed");
      const auto& r = res.get();
      token_id = r.token_id;
      text = r.text_piece;
      is_eos = r.is_eos;
    }
    py::dict d;
    d["token_id"] = token_id;
    d["text"] = py::bytes(text);
    d["is_eos"] = is_eos;
    return d;
  }

  void seek(int64_t pos) {
    throw_if_error(session_->seek(pos), "seek failed");
  }
  int64_t position() const {
    return session_->position();
  }
  void reset() {
    throw_if_error(session_->reset(), "reset failed");
  }
  void stop() {
    session_->stop();
  }

 private:
  std::unique_lock<std::mutex> lock_exec() {
    return exec_mutex_ ? std::unique_lock<std::mutex>(*exec_mutex_)
                       : std::unique_lock<std::mutex>();
  }
  std::unique_ptr<LLMSession> session_;
  std::shared_ptr<std::mutex> exec_mutex_;
};

// Engine over one loaded Program: loads it once; create_session() returns an
// LLMSession that reuses it but owns its own KV state. Wraps any LLMEngine —
// the built-in "text" engine (via the convenience constructor) or a model
// adapter's engine handed in directly. Physical weight sharing across sessions
// is backend-dependent (serving_capacity() is authoritative).
class PyLLMEngine {
 public:
  // Wrap an already-constructed engine (used by model adapter modules).
  explicit PyLLMEngine(std::unique_ptr<LLMEngine> engine)
      : engine_(std::move(engine)) {}

  // Convenience constructor: the built-in TextLLMEngine.
  PyLLMEngine(
      const std::string& model_path,
      const std::string& tokenizer_path,
      std::optional<const std::string> data_path = std::nullopt,
      const std::string& method_name = "forward",
      float temperature = -1.0f) {
    if (data_path.has_value()) {
      throw std::runtime_error(
          "LLMEngine: shared sessions with external data (.ptd / data_path) are "
          "not yet supported for the text engine; use a self-contained .pte.");
    }
    auto engine = TextLLMEngine::create(
        model_path, tokenizer_path, std::nullopt, temperature, method_name);
    if (!engine) {
      throw std::runtime_error(
          "Failed to create LLMEngine with model: " + model_path);
    }
    engine_ = std::move(engine);
  }

  std::unique_ptr<PyLLMSession> create_session() {
    auto res = engine_->create_session();
    throw_if_error(res.error(), "Failed to create session from LLMEngine");
    // Hand the session the engine-owned lock so backend execution across all
    // sessions of this engine is serialized.
    return std::make_unique<PyLLMSession>(std::move(res.get()), exec_mutex_);
  }

  py::dict serving_capacity() const {
    const auto c = engine_->serving_capacity();
    py::dict d;
    d["max_physical_sessions_without_weight_duplication"] =
        c.max_physical_sessions_without_weight_duplication;
    d["estimated_bytes_per_session"] = c.estimated_bytes_per_session;
    return d;
  }

  py::dict metadata() const {
    py::dict d;
    for (const auto& [key, value] : engine_->metadata()) {
      d[py::str(key)] = value;
    }
    return d;
  }

 private:
  std::unique_ptr<LLMEngine> engine_;
  std::shared_ptr<std::mutex> exec_mutex_ = std::make_shared<std::mutex>();
};

// Bind LLMSession and LLMEngine into `m`. module_local: these wrapper types are
// compiled into BOTH the generic _llm_runner module and any example model
// module, which may be imported in the same process; without module_local
// pybind11's process-global type registry would reject the second registration.
inline void bind_engine_session_api(py::module_& m) {
  py::class_<PyLLMSession>(m, "LLMSession", py::module_local())
      .def(
          "prefill_tokens",
          &PyLLMSession::prefill_tokens,
          py::arg("token_ids"),
          py::arg("temperature") = -1.0f,
          "Prefill pre-tokenized input at the current cache position. "
          "`temperature` is the first-token sampling for backends that sample "
          "during prefill (ignored by decode-time samplers).")
      .def(
          "decode_one",
          &PyLLMSession::decode_one,
          py::arg("temperature") = -1.0f,
          "Decode one token; returns {token_id:int, text:bytes, is_eos:bool}.")
      .def("seek", &PyLLMSession::seek, py::arg("pos"), "Rewind KV to `pos`.")
      .def("position", &PyLLMSession::position, "Resident KV token count.")
      .def("reset", &PyLLMSession::reset, "Clear KV / position.")
      .def("stop", &PyLLMSession::stop, "Signal an in-flight decode to stop.")
      .def("__repr__", [](const PyLLMSession&) { return "<LLMSession>"; });

  py::class_<PyLLMEngine>(m, "LLMEngine", py::module_local())
      .def(
          py::init<
              const std::string&,
              const std::string&,
              std::optional<const std::string>,
              const std::string&,
              float>(),
          py::arg("model_path"),
          py::arg("tokenizer_path"),
          py::arg("data_path") = py::none(),
          py::arg("method_name") = "forward",
          py::arg("temperature") = -1.0f,
          "Load the built-in text model's program once for multi-session serving.")
      .def(
          "create_session",
          &PyLLMEngine::create_session,
          "Create an LLMSession that reuses the engine's program/resources "
          "(weight sharing is backend-dependent — see serving_capacity()) but "
          "owns its own KV cache. Backend execution across sessions is "
          "serialized by an engine-owned lock.")
      .def(
          "serving_capacity",
          &PyLLMEngine::serving_capacity,
          "Serving-capacity dict; the server clamps physical sessions to "
          "max_physical_sessions_without_weight_duplication (1 = single-slot).")
      .def("metadata", &PyLLMEngine::metadata, "Model metadata from the .pte.")
      .def("__repr__", [](const PyLLMEngine&) { return "<LLMEngine>"; });
}

} // namespace executorch::extension::llm::pybind_wrappers
