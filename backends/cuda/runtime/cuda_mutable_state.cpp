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

namespace aoti = ::executorch::backends::aoti;
namespace slimc10 = ::executorch::backends::aoti::slim::c10;
using ::executorch::backends::aoti::slim::from_blob;
using ::executorch::backends::aoti::slim::SlimTensor;
using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

namespace {

// AOTI internal constant names are per-handle; the exported FQN is the stable
// identity across methods.
struct Desc {
  std::string internal_name;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  slimc10::ScalarType dtype{slimc10::ScalarType::Float};
  slimc10::Device device{slimc10::DeviceType::CUDA, 0};
  size_t nbytes{0};
};

// Cached user-managed pairs for a (handle, session).
struct Bound {
  std::vector<std::unique_ptr<SlimTensor>> tensors;
  std::vector<aoti::AOTInductorConstantMapEntry> pairs;
};

struct Context {
  std::vector<std::string> fqns;
  std::unordered_set<std::string> fqn_set;

  bool symbols_checked{false};
  bool symbols_available{false};
  bool handles_associated{false};

  std::unordered_map<std::string, void*> template_ptr;
  std::unordered_map<std::string, size_t> template_nbytes;
  std::unordered_map<std::string, int> template_device;
  int64_t total_bytes{0};

  std::unordered_map<CudaDelegateHandle*, std::unordered_map<std::string, Desc>>
      desc;
  std::unordered_set<std::string> discovered_fqns;
  Error build_error{Error::Ok};

  std::unordered_set<int> sessions;
  int next_token{0};
  std::unordered_map<int, std::unordered_map<std::string, void*>> session_buf;
  std::unordered_map<CudaDelegateHandle*, std::unordered_map<int, Bound>> bound;
  // A managed handle must not execute without an active session after sessions
  // exist or after a previous rebind left session pointers installed.
  std::unordered_set<CudaDelegateHandle*> rebound_handles;
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

// Load scopes associate handles with a context; active scopes select a session
// for execute on the current thread.
thread_local MutableStateContext tl_loading_ctx = kInvalidMutableContext;
thread_local MutableStateContext tl_active_ctx = kInvalidMutableContext;
thread_local int tl_active_token = kNoMutableSession;

bool handle_has_symbols(CudaDelegateHandle* h) {
  return h->get_num_constants && h->get_constant_name &&
      h->get_constant_original_fqn && h->extract_constants_map &&
      h->update_user_managed_constant_buffer_pairs;
}

struct CudaDeviceGuard {
  int prev_device{0};
  bool restore{false};

  Error set(int device) {
    if (device < 0) {
      return Error::Ok;
    }
    cudaError_t err = cudaGetDevice(&prev_device);
    if (err != cudaSuccess) {
      ET_LOG(Error, "mutable_state: cudaGetDevice failed");
      return Error::Internal;
    }
    if (prev_device == device) {
      return Error::Ok;
    }
    err = cudaSetDevice(device);
    if (err != cudaSuccess) {
      ET_LOG(
          Error,
          "mutable_state: cudaSetDevice(%d) failed: %s",
          device,
          cudaGetErrorString(err));
      return Error::Internal;
    }
    restore = true;
    return Error::Ok;
  }

  ~CudaDeviceGuard() {
    if (restore) {
      cudaSetDevice(prev_device);
    }
  }
};

Result<int> tensor_cuda_device_index(const SlimTensor& t) {
  const slimc10::Device device = t.device();
  ET_CHECK_OR_RETURN_ERROR(
      device.is_cuda(),
      InvalidArgument,
      "mutable_state: mutable buffer template must be on CUDA, got %s",
      device.str().c_str());
  if (device.index() >= 0) {
    return static_cast<int>(device.index());
  }
  cudaPointerAttributes attr{};
  const cudaError_t err = cudaPointerGetAttributes(&attr, t.data_ptr());
  if (err != cudaSuccess) {
    cudaGetLastError();
    ET_LOG(
        Error,
        "mutable_state: cudaPointerGetAttributes failed for template pointer");
    return Error::Internal;
  }
  return attr.device;
}

void cuda_free_on_pointer_device(void* ptr, bool synchronize) {
  if (!ptr) {
    return;
  }
  int device = -1;
  cudaPointerAttributes attr{};
  const cudaError_t attr_err = cudaPointerGetAttributes(&attr, ptr);
  if (attr_err == cudaSuccess) {
    device = attr.device;
  } else {
    cudaGetLastError();
  }

  CudaDeviceGuard guard;
  if (device >= 0 && guard.set(device) != Error::Ok) {
    ET_LOG(
        Error,
        "mutable_state: freeing pointer %p without switching to device %d",
        ptr,
        device);
  }
  if (synchronize) {
    const cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
      ET_LOG(
          Error,
          "mutable_state: cudaDeviceSynchronize before free failed: %s",
          cudaGetErrorString(sync_err));
    }
  }
  const cudaError_t free_err = cudaFree(ptr);
  if (free_err != cudaSuccess) {
    ET_LOG(
        Error,
        "mutable_state: cudaFree(%p) failed: %s",
        ptr,
        cudaGetErrorString(free_err));
  }
}

bool validate_descriptors(const Context& c) {
  bool ok = true;
  std::unordered_map<std::string, const Desc*> first_desc;
  for (const auto& handle_descs : c.desc) {
    for (const auto& fd : handle_descs.second) {
      const std::string& fqn = fd.first;
      const Desc& d = fd.second;
      auto template_it = c.template_nbytes.find(fqn);
      if (template_it == c.template_nbytes.end()) {
        ET_LOG(
            Error,
            "mutable_state: descriptor '%s' has no captured template",
            fqn.c_str());
        ok = false;
        continue;
      }
      if (d.nbytes > template_it->second) {
        ET_LOG(
            Error,
            "mutable_state: descriptor '%s' (%zu B) exceeds shared template "
            "buffer (%zu B)",
            fqn.c_str(),
            d.nbytes,
            template_it->second);
        ok = false;
      }

      auto inserted = first_desc.emplace(fqn, &d);
      if (!inserted.second) {
        const Desc& base = *inserted.first->second;
        if (d.dtype != base.dtype || d.device != base.device) {
          ET_LOG(
              Error,
              "mutable_state: descriptor '%s' has incompatible dtype/device "
              "across loaded methods",
              fqn.c_str());
          ok = false;
        }
      }
    }
  }
  return ok;
}

Error validate_coverage_locked(Context& c) {
  if (c.build_error != Error::Ok) {
    return c.build_error;
  }
  if (!c.symbols_available) {
    return Error::NotSupported;
  }
  if (c.fqns.empty()) {
    ET_LOG(Error, "mutable_state: no mutable-buffer FQNs registered");
    c.build_error = Error::InvalidState;
    return Error::InvalidState;
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
  ok = validate_descriptors(c) && ok;
  if (!ok) {
    c.build_error = Error::InvalidProgram;
    return Error::InvalidProgram;
  }
  return Error::Ok;
}

// Captures descriptors and initial templates while the container still owns its
// default mutable buffers.
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
    // Empty names are method-scoped constants; skip them.
    if (internal && internal[0] != '\0' && fqn && fqn[0] != '\0') {
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
    if (it_name == fqn_to_internal.end() || it_t == extracted.end()) {
      continue;
    }
    auto* t = reinterpret_cast<SlimTensor*>(it_t->second);
    auto device_res = tensor_cuda_device_index(*t);
    ET_CHECK_OK_OR_RETURN_ERROR(device_res.error());
    const int device = device_res.get();

    Desc d;
    d.internal_name = it_name->second;
    d.sizes.assign(t->sizes().begin(), t->sizes().end());
    d.strides.assign(t->strides().begin(), t->strides().end());
    d.dtype = t->dtype();
    d.device = slimc10::Device(slimc10::DeviceType::CUDA, device);
    d.nbytes = t->nbytes();
    table.emplace(fqn, std::move(d));
    c.discovered_fqns.insert(fqn);

    if (c.template_ptr.find(fqn) == c.template_ptr.end()) {
      CudaDeviceGuard guard;
      ET_CHECK_OK_OR_RETURN_ERROR(guard.set(device));

      void* tpl = nullptr;
      if (cudaMalloc(&tpl, t->nbytes()) != cudaSuccess) {
        ET_LOG(Error, "mutable_state: cudaMalloc template '%s'", fqn.c_str());
        return Error::Internal;
      }
      if (cudaMemcpy(
              tpl, t->data_ptr(), t->nbytes(), cudaMemcpyDeviceToDevice) !=
          cudaSuccess) {
        ET_LOG(Error, "mutable_state: cudaMemcpy template '%s'", fqn.c_str());
        cuda_free_on_pointer_device(tpl, /*synchronize=*/false);
        return Error::Internal;
      }
      c.template_ptr[fqn] = tpl;
      c.template_nbytes[fqn] = t->nbytes();
      c.template_device[fqn] = device;
      c.total_bytes += static_cast<int64_t>(t->nbytes());
    }
  }
  return Error::Ok;
}

// Allocates any missing per-FQN session buffers from the captured templates.
Error ensure_session_buffers(Context& c, int token) {
  auto& buf = c.session_buf[token];
  for (const auto& kv : c.template_ptr) {
    const std::string& fqn = kv.first;
    if (buf.find(fqn) != buf.end()) {
      continue;
    }
    void* tpl = kv.second;
    size_t nbytes = c.template_nbytes[fqn];
    auto device_it = c.template_device.find(fqn);
    if (device_it == c.template_device.end()) {
      ET_LOG(Error, "mutable_state: no template device for '%s'", fqn.c_str());
      return Error::Internal;
    }
    CudaDeviceGuard guard;
    ET_CHECK_OK_OR_RETURN_ERROR(guard.set(device_it->second));

    void* p = nullptr;
    if (cudaMalloc(&p, nbytes) != cudaSuccess) {
      ET_LOG(
          Error, "mutable_state: cudaMalloc session buffer '%s'", fqn.c_str());
      return Error::Internal;
    }
    if (cudaMemcpy(p, tpl, nbytes, cudaMemcpyDeviceToDevice) != cudaSuccess) {
      ET_LOG(
          Error, "mutable_state: cudaMemcpy session buffer '%s'", fqn.c_str());
      cuda_free_on_pointer_device(p, /*synchronize=*/false);
      return Error::Internal;
    }
    buf[fqn] = p;
  }
  return Error::Ok;
}

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
      ET_LOG(Error, "mutable_state: no session buffer for '%s'", fqn.c_str());
      return Error::Internal;
    }
    auto template_it = c.template_nbytes.find(fqn);
    if (template_it == c.template_nbytes.end() ||
        d.nbytes > template_it->second) {
      ET_LOG(
          Error,
          "mutable_state: descriptor '%s' (%zu B) exceeds shared template "
          "buffer (%zu B)",
          fqn.c_str(),
          d.nbytes,
          template_it == c.template_nbytes.end() ? 0 : template_it->second);
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
        cuda_free_on_pointer_device(kv.second, /*synchronize=*/true);
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

namespace detail {

MutableStateContext mutable_state_create_context() {
  auto& m = mgr();
  std::lock_guard<std::mutex> g(m.mu);
  MutableStateContext id = m.next_ctx++;
  m.contexts.emplace(id, Context{});
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
      cuda_free_on_pointer_device(kv.second, /*synchronize=*/true);
    }
  }
  for (auto& sb : c.session_buf) {
    for (auto& kv : sb.second) {
      if (kv.second) {
        cuda_free_on_pointer_device(kv.second, /*synchronize=*/true);
      }
    }
  }
  for (auto hit = m.handle_ctx.begin(); hit != m.handle_ctx.end();) {
    hit = (hit->second == ctx) ? m.handle_ctx.erase(hit) : std::next(hit);
  }
  m.contexts.erase(it);
}

void mutable_state_begin_load(MutableStateContext ctx) {
  if (tl_loading_ctx != kInvalidMutableContext) {
    auto& m = mgr();
    std::lock_guard<std::mutex> g(m.mu);
    auto active = m.contexts.find(tl_loading_ctx);
    if (active != m.contexts.end()) {
      active->second.build_error = Error::InvalidState;
    }
    auto nested = m.contexts.find(ctx);
    if (nested != m.contexts.end()) {
      nested->second.build_error = Error::InvalidState;
    }
    ET_LOG(Error, "mutable_state: nested load scopes are not supported");
    tl_loading_ctx = kInvalidMutableContext;
    return;
  }
  tl_loading_ctx = ctx;
}

void mutable_state_end_load() {
  tl_loading_ctx = kInvalidMutableContext;
}

void mutable_state_set_active(MutableStateContext ctx, int token) {
  tl_active_ctx = ctx;
  tl_active_token = token;
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
  Context& c = it->second;
  if (c.handles_associated || !c.sessions.empty()) {
    ET_LOG(
        Error,
        "mutable_state: mutable-buffer FQNs must be registered before load");
    c.build_error = Error::InvalidState;
    return;
  }
  c.fqns = fqns;
  c.fqn_set.clear();
  c.fqn_set.insert(fqns.begin(), fqns.end());
}

bool mutable_state_available(MutableStateContext ctx) {
  auto& m = mgr();
  std::lock_guard<std::mutex> g(m.mu);
  auto it = m.contexts.find(ctx);
  return it != m.contexts.end() && it->second.build_error == Error::Ok &&
      it->second.symbols_available;
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
  return validate_coverage_locked(it->second);
}

Result<int> mutable_state_create_session(MutableStateContext ctx) {
  auto& m = mgr();
  std::lock_guard<std::mutex> g(m.mu);
  auto it = m.contexts.find(ctx);
  if (it == m.contexts.end()) {
    return Error::InvalidArgument;
  }
  Context& c = it->second;
  ET_CHECK_OK_OR_RETURN_ERROR(validate_coverage_locked(c));
  int token = c.next_token++;
  c.sessions.insert(token);
  return token;
}

void mutable_state_destroy_session(MutableStateContext ctx, int token) {
  auto& m = mgr();
  std::lock_guard<std::mutex> g(m.mu);
  auto it = m.contexts.find(ctx);
  if (it == m.contexts.end()) {
    return;
  }
  free_session_buffers(it->second, token);
}

} // namespace detail

void mutable_state_note_handle(CudaDelegateHandle* handle) {
  MutableStateContext ctx = tl_loading_ctx;
  if (ctx == kInvalidMutableContext) {
    return;
  }
  auto& m = mgr();
  std::lock_guard<std::mutex> g(m.mu);
  auto it = m.contexts.find(ctx);
  if (it == m.contexts.end()) {
    return;
  }
  Context& c = it->second;
  c.handles_associated = true;
  m.handle_ctx[handle] = ctx;
  bool ok = handle_has_symbols(handle);
  c.symbols_available = c.symbols_checked ? (c.symbols_available && ok) : ok;
  c.symbols_checked = true;
  if (ok && !c.fqns.empty() && c.desc.find(handle) == c.desc.end()) {
    Error e = build_descriptors(c, handle);
    if (e != Error::Ok) {
      c.build_error = e;
    }
  }
}

void mutable_state_forget_handle(CudaDelegateHandle* handle) {
  auto& m = mgr();
  std::lock_guard<std::mutex> g(m.mu);
  auto hit = m.handle_ctx.find(handle);
  if (hit == m.handle_ctx.end()) {
    return;
  }
  auto cit = m.contexts.find(hit->second);
  if (cit != m.contexts.end()) {
    cit->second.desc.erase(handle);
    cit->second.bound.erase(handle);
    cit->second.rebound_handles.erase(handle);
  }
  m.handle_ctx.erase(hit);
}

Error mutable_state_rebind_for_execute(CudaDelegateHandle* handle) {
  auto& m = mgr();
  std::lock_guard<std::mutex> g(m.mu);

  auto hit = m.handle_ctx.find(handle);
  if (tl_active_token == kNoMutableSession) {
    if (hit == m.handle_ctx.end()) {
      return Error::Ok;
    }
    auto cit = m.contexts.find(hit->second);
    if (cit != m.contexts.end() &&
        (!cit->second.sessions.empty() ||
         cit->second.rebound_handles.find(handle) !=
             cit->second.rebound_handles.end())) {
      ET_LOG(
          Error, "mutable_state: active session is required for this handle");
      return Error::InvalidState;
    }
    return Error::Ok;
  }
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
  if (c.build_error != Error::Ok) {
    return c.build_error;
  }
  if (!c.symbols_available) {
    ET_LOG(
        Error, "mutable_state: active session set but rebinding unavailable");
    return Error::NotSupported;
  }
  const int token = tl_active_token;
  if (c.sessions.find(token) == c.sessions.end()) {
    ET_LOG(Error, "mutable_state: active session token was not created");
    return Error::InvalidArgument;
  }
  if (handle->cuda_graph_state.phase != CudaGraphPhase::Disabled) {
    ET_LOG(
        Error,
        "mutable_state: per-session rebinding is not supported with CUDA graph");
    return Error::NotSupported;
  }
  if (c.desc.find(handle) == c.desc.end()) {
    ET_LOG(
        Error,
        "mutable_state: no descriptors for handle (note_handle missed?)");
    return Error::Internal;
  }
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
  c.rebound_handles.insert(handle);
  return Error::Ok;
}

} // namespace cuda
} // namespace backends
} // namespace executorch
