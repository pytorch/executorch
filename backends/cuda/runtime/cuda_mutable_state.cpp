/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/runtime/cuda_mutable_state.h>

#include <executorch/backends/aoti/slim/c10/core/Device.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/backends/aoti/slim/factory/from_blob.h>
#include <executorch/backends/cuda/runtime/cuda_delegate_handle.h>
#include <executorch/runtime/platform/log.h>

#include <cuda_runtime.h>

#include <iterator>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace executorch {
namespace backends {
namespace cuda {

namespace slimc10 = ::executorch::backends::aoti::slim::c10;
using ::executorch::backends::aoti::slim::from_blob;
using ::executorch::backends::aoti::slim::SlimTensor;
using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

namespace {

// Per-handle descriptor of one mutable constant (AOTI internal name differs per
// compiled method, so this is keyed per delegate handle within a context).
struct Desc {
  std::string internal_name;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  slimc10::ScalarType dtype{slimc10::ScalarType::Float};
  slimc10::Device device{slimc10::DeviceType::CUDA, 0};
  size_t nbytes{0};
};

// Cached user-managed pairs for a (handle, session): SlimTensors wrapping the
// session's GPU buffers (kept alive here) and the flat pairs array AOTI
// rebinds.
struct Bound {
  std::vector<std::unique_ptr<SlimTensor>> tensors;
  std::vector<aoti::AOTInductorConstantMapEntry> pairs;
};

// All per-engine/model mutable state. Keyed by context id in Manager.
struct Context {
  std::vector<std::string> fqns;
  std::unordered_set<std::string> fqn_set;

  bool symbols_checked{false};
  bool symbols_available{false};

  // FQN -> device template (the model's initial mutable contents) + sizes.
  std::unordered_map<std::string, void*> template_ptr;
  std::unordered_map<std::string, size_t> template_nbytes;
  int64_t total_bytes{0};

  // Per-handle descriptor table + the union of discovered FQNs (for coverage).
  std::unordered_map<CudaDelegateHandle*, std::unordered_map<std::string, Desc>>
      desc;
  std::unordered_set<std::string> discovered_fqns;
  Error build_error{Error::Ok};

  std::unordered_set<int> sessions;
  int next_token{0};
  // token -> (fqn -> device buffer) shared across the session's handles.
  std::unordered_map<int, std::unordered_map<std::string, void*>> session_buf;
  // (handle, token) -> cached wrappers + pairs.
  std::unordered_map<CudaDelegateHandle*, std::unordered_map<int, Bound>> bound;
};

struct Manager {
  std::mutex mu;
  std::unordered_map<MutableStateContext, Context> contexts;
  std::unordered_map<CudaDelegateHandle*, MutableStateContext> handle_ctx;
  MutableStateContext next_ctx{1};
};

Manager& mgr() {
  static Manager m;
  return m;
}

// The context whose model is currently being loaded on this thread (so
// note_handle, called from CudaBackend::init, can associate handles). And the
// active (context, session) selected before execute on this thread.
thread_local MutableStateContext tl_loading_ctx = kInvalidMutableContext;
thread_local MutableStateContext tl_active_ctx = kInvalidMutableContext;
thread_local int tl_active_token = kNoMutableSession;

bool handle_has_symbols(CudaDelegateHandle* h) {
  return h->get_num_constants && h->get_constant_name &&
      h->get_constant_original_fqn && h->extract_constants_map &&
      h->update_user_managed_constant_buffer_pairs;
}

// Build the descriptor table for a handle and capture per-FQN initial
// templates. Caller holds mgr().mu. Runs before any session has rebound this
// container, so the constants still hold the model's initial mutable state.
Error build_descriptors(Context& c, CudaDelegateHandle* h) {
  auto container = h->container_handle;

  size_t n = 0;
  ET_CHECK_OK_OR_RETURN_ERROR(
      h->get_num_constants(container, &n),
      "mutable_state: get_num_constants failed");
  std::unordered_map<std::string, std::string> fqn_to_internal;
  for (size_t i = 0; i < n; ++i) {
    const char* internal = nullptr;
    const char* fqn = nullptr;
    ET_CHECK_OK_OR_RETURN_ERROR(
        h->get_constant_name(container, i, &internal),
        "mutable_state: get_constant_name failed");
    ET_CHECK_OK_OR_RETURN_ERROR(
        h->get_constant_original_fqn(container, i, &fqn),
        "mutable_state: get_constant_original_fqn failed");
    // A successful call may still report an unusable (null/empty) name --
    // that's a method-scoped constant, not an error: skip it (another container
    // owns it). A non-OK return code above is a real failure and falls closed.
    if (internal && fqn && fqn[0] != '\0') {
      fqn_to_internal[fqn] = internal;
    }
  }

  std::unordered_map<std::string, aoti::AtenTensorHandle> extracted;
  ET_CHECK_OK_OR_RETURN_ERROR(
      h->extract_constants_map(
          container,
          reinterpret_cast<aoti::AOTInductorConstantMapHandle>(&extracted),
          /*use_inactive=*/false),
      "mutable_state: extract_constants_map failed");

  auto& table = c.desc[h];
  for (const auto& fqn : c.fqns) {
    auto it_name = fqn_to_internal.find(fqn);
    auto it_t = extracted.find(fqn);
    // A mutable FQN not present in this container = a method that does not use
    // it (method-scoped). Skip; another container will own it.
    if (it_name == fqn_to_internal.end() || it_t == extracted.end()) {
      continue;
    }
    auto* t = reinterpret_cast<SlimTensor*>(it_t->second);
    Desc d;
    d.internal_name = it_name->second;
    d.sizes.assign(t->sizes().begin(), t->sizes().end());
    d.strides.assign(t->strides().begin(), t->strides().end());
    d.dtype = t->dtype();
    d.device = t->device();
    d.nbytes = t->nbytes();
    table.emplace(fqn, std::move(d));
    c.discovered_fqns.insert(fqn);

    if (c.template_ptr.find(fqn) == c.template_ptr.end()) {
      void* tpl = nullptr;
      if (cudaMalloc(&tpl, t->nbytes()) != cudaSuccess) {
        ET_LOG(Error, "mutable_state: cudaMalloc template '%s'", fqn.c_str());
        return Error::Internal;
      }
      if (cudaMemcpy(
              tpl, t->data_ptr(), t->nbytes(), cudaMemcpyDeviceToDevice) !=
          cudaSuccess) {
        ET_LOG(Error, "mutable_state: cudaMemcpy template '%s'", fqn.c_str());
        cudaFree(tpl);
        return Error::Internal;
      }
      c.template_ptr[fqn] = tpl;
      c.template_nbytes[fqn] = t->nbytes();
      c.total_bytes += static_cast<int64_t>(t->nbytes());
    }
  }
  return Error::Ok;
}

// Allocate a session's GPU buffers, cloned from the initial templates. Caller
// holds mgr().mu. Allocates PER FQN so a buffer is created for any template
// discovered after the session's first allocation.
Error ensure_session_buffers(Context& c, int token) {
  auto& buf = c.session_buf[token];
  for (const auto& kv : c.template_ptr) {
    const std::string& fqn = kv.first;
    if (buf.find(fqn) != buf.end()) {
      continue; // already allocated for this session
    }
    void* tpl = kv.second;
    size_t nbytes = c.template_nbytes[fqn];
    void* p = nullptr;
    if (cudaMalloc(&p, nbytes) != cudaSuccess) {
      ET_LOG(
          Error, "mutable_state: cudaMalloc session buffer '%s'", fqn.c_str());
      return Error::Internal;
    }
    if (cudaMemcpy(p, tpl, nbytes, cudaMemcpyDeviceToDevice) != cudaSuccess) {
      ET_LOG(
          Error, "mutable_state: cudaMemcpy session buffer '%s'", fqn.c_str());
      cudaFree(p);
      return Error::Internal;
    }
    buf[fqn] = p;
  }
  return Error::Ok;
}

// Build the cached wrappers + pairs for (handle, token). Caller holds mgr().mu.
Error ensure_bound(Context& c, CudaDelegateHandle* h, int token) {
  if (c.bound[h].find(token) != c.bound[h].end()) {
    return Error::Ok;
  }
  Bound b;
  auto& buf = c.session_buf[token];
  for (const auto& fd : c.desc[h]) {
    const std::string& fqn = fd.first;
    const Desc& d = fd.second;
    auto buf_it = buf.find(fqn);
    if (buf_it == buf.end() || buf_it->second == nullptr) {
      // Every descriptor for this handle must have a backing session buffer;
      // a null bind would silently corrupt state.
      ET_LOG(Error, "mutable_state: no session buffer for '%s'", fqn.c_str());
      return Error::Internal;
    }
    void* ptr = buf_it->second;
    auto st = std::make_unique<SlimTensor>(from_blob(
        ptr,
        ::executorch::runtime::makeArrayRef(d.sizes.data(), d.sizes.size()),
        ::executorch::runtime::makeArrayRef(d.strides.data(), d.strides.size()),
        d.dtype,
        d.device));
    aoti::AOTInductorConstantMapEntry entry;
    entry.name = d.internal_name.c_str();
    entry.handle = reinterpret_cast<aoti::AtenTensorHandle>(st.get());
    b.pairs.push_back(entry);
    b.tensors.push_back(std::move(st));
  }
  c.bound[h].emplace(token, std::move(b));
  return Error::Ok;
}

void free_session_buffers(Context& c, int token) {
  auto it = c.session_buf.find(token);
  if (it != c.session_buf.end()) {
    for (auto& kv : it->second) {
      if (kv.second) {
        cudaFree(kv.second);
      }
    }
    c.session_buf.erase(it);
  }
  for (auto& hb : c.bound) {
    hb.second.erase(token);
  }
  c.sessions.erase(token);
}

} // namespace

MutableStateContext mutable_state_create_context() {
  auto& m = mgr();
  std::lock_guard<std::mutex> g(m.mu);
  MutableStateContext id = m.next_ctx++;
  m.contexts[id]; // default-construct
  return id;
}

void mutable_state_destroy_context(MutableStateContext ctx) {
  auto& m = mgr();
  std::lock_guard<std::mutex> g(m.mu);
  auto it = m.contexts.find(ctx);
  if (it == m.contexts.end()) {
    return;
  }
  Context& c = it->second;
  for (auto& kv : c.template_ptr) {
    if (kv.second) {
      cudaFree(kv.second);
    }
  }
  for (auto& sb : c.session_buf) {
    for (auto& kv : sb.second) {
      if (kv.second) {
        cudaFree(kv.second);
      }
    }
  }
  // Drop handle->ctx associations for this context.
  for (auto hit = m.handle_ctx.begin(); hit != m.handle_ctx.end();) {
    hit = (hit->second == ctx) ? m.handle_ctx.erase(hit) : std::next(hit);
  }
  m.contexts.erase(it);
}

void mutable_state_begin_load(MutableStateContext ctx) {
  tl_loading_ctx = ctx;
}

void mutable_state_end_load() {
  tl_loading_ctx = kInvalidMutableContext;
}

void mutable_state_register_fqns(
    MutableStateContext ctx,
    const std::vector<std::string>& fqns) {
  auto& m = mgr();
  std::lock_guard<std::mutex> g(m.mu);
  auto it = m.contexts.find(ctx);
  if (it == m.contexts.end()) {
    return;
  }
  it->second.fqns = fqns;
  it->second.fqn_set.clear();
  it->second.fqn_set.insert(fqns.begin(), fqns.end());
}

bool mutable_state_available(MutableStateContext ctx) {
  auto& m = mgr();
  std::lock_guard<std::mutex> g(m.mu);
  auto it = m.contexts.find(ctx);
  return it != m.contexts.end() && it->second.symbols_available;
}

int64_t mutable_state_bytes_per_session(MutableStateContext ctx) {
  auto& m = mgr();
  std::lock_guard<std::mutex> g(m.mu);
  auto it = m.contexts.find(ctx);
  return it == m.contexts.end() ? 0 : it->second.total_bytes;
}

Error mutable_state_validate_coverage(MutableStateContext ctx) {
  auto& m = mgr();
  std::lock_guard<std::mutex> g(m.mu);
  auto it = m.contexts.find(ctx);
  if (it == m.contexts.end()) {
    return Error::InvalidArgument;
  }
  Context& c = it->second;
  if (!c.symbols_available) {
    return Error::NotSupported;
  }
  if (c.build_error != Error::Ok) {
    return c.build_error;
  }
  bool ok = true;
  for (const auto& fqn : c.fqns) {
    if (c.discovered_fqns.find(fqn) == c.discovered_fqns.end()) {
      ET_LOG(
          Error,
          "mutable_state: declared mutable buffer '%s' not found in any loaded "
          "method's constants (FQN mismatch?)",
          fqn.c_str());
      ok = false;
    }
  }
  return ok ? Error::Ok : Error::InvalidProgram;
}

Result<int> mutable_state_create_session(MutableStateContext ctx) {
  auto& m = mgr();
  std::lock_guard<std::mutex> g(m.mu);
  auto it = m.contexts.find(ctx);
  if (it == m.contexts.end()) {
    return Error::InvalidArgument;
  }
  if (!it->second.symbols_available) {
    ET_LOG(
        Error, "mutable_state: rebinding unavailable; cannot create session");
    return Error::NotSupported;
  }
  int token = it->second.next_token++;
  it->second.sessions.insert(token);
  return token;
}

void mutable_state_destroy_session(MutableStateContext ctx, int token) {
  auto& m = mgr();
  std::lock_guard<std::mutex> g(m.mu);
  auto it = m.contexts.find(ctx);
  if (it == m.contexts.end()) {
    return; // context already torn down; nothing to free
  }
  free_session_buffers(it->second, token);
}

void mutable_state_set_active(MutableStateContext ctx, int token) {
  tl_active_ctx = ctx;
  tl_active_token = token;
}

void mutable_state_note_handle(CudaDelegateHandle* handle) {
  MutableStateContext ctx = tl_loading_ctx;
  if (ctx == kInvalidMutableContext) {
    return; // not loading within a managed context (single-session path)
  }
  auto& m = mgr();
  std::lock_guard<std::mutex> g(m.mu);
  auto it = m.contexts.find(ctx);
  if (it == m.contexts.end()) {
    return;
  }
  Context& c = it->second;
  m.handle_ctx[handle] = ctx;
  bool ok = handle_has_symbols(handle);
  c.symbols_available = c.symbols_checked ? (c.symbols_available && ok) : ok;
  c.symbols_checked = true;
  // Build this method's descriptor table + capture initial templates now, while
  // the container still holds the model's initial mutable state and before any
  // session rebinds. Requires FQNs registered before load_method.
  if (ok && !c.fqns.empty() && c.desc.find(handle) == c.desc.end()) {
    Error e = build_descriptors(c, handle);
    if (e != Error::Ok) {
      c.build_error = e;
    }
  }
}

Error mutable_state_rebind_for_execute(CudaDelegateHandle* handle) {
  if (tl_active_token == kNoMutableSession) {
    return Error::Ok; // single-session / legacy: nothing to rebind
  }
  auto& m = mgr();
  std::lock_guard<std::mutex> g(m.mu);

  auto hit = m.handle_ctx.find(handle);
  if (hit == m.handle_ctx.end()) {
    ET_LOG(
        Error,
        "mutable_state: active session set but handle has no context (load "
        "scope missed?)");
    return Error::Internal;
  }
  MutableStateContext ctx = hit->second;
  if (ctx != tl_active_ctx) {
    ET_LOG(
        Error,
        "mutable_state: active context mismatch (caller set a different context "
        "active than the one executing)");
    return Error::Internal;
  }
  auto cit = m.contexts.find(ctx);
  if (cit == m.contexts.end()) {
    return Error::Internal;
  }
  Context& c = cit->second;
  if (!c.symbols_available) {
    ET_LOG(
        Error, "mutable_state: active session set but rebinding unavailable");
    return Error::NotSupported;
  }
  if (c.desc.find(handle) == c.desc.end()) {
    ET_LOG(
        Error,
        "mutable_state: no descriptors for handle (note_handle missed?)");
    return Error::Internal;
  }
  const int token = tl_active_token;
  ET_CHECK_OK_OR_RETURN_ERROR(ensure_session_buffers(c, token));
  ET_CHECK_OK_OR_RETURN_ERROR(ensure_bound(c, handle, token));

  const Bound& b = c.bound[handle][token];
  if (b.pairs.empty()) {
    return Error::Ok;
  }
  ET_CHECK_OK_OR_RETURN_ERROR(
      handle->update_user_managed_constant_buffer_pairs(
          handle->container_handle,
          b.pairs.data(),
          b.pairs.size(),
          /*use_inactive=*/false,
          /*validate_full_update=*/false),
      "mutable_state: update_user_managed_constant_buffer_pairs failed");
  return Error::Ok;
}

} // namespace cuda
} // namespace backends
} // namespace executorch
