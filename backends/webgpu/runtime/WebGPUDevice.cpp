/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUCompat.h>
#include <executorch/backends/webgpu/runtime/WebGPUDevice.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>

namespace executorch {
namespace backends {
namespace webgpu {

namespace {

struct AdapterResult {
  WGPUAdapter adapter = nullptr;
};

struct DeviceResult {
  WGPUDevice device = nullptr;
};

void on_adapter_request(
    WGPURequestAdapterStatus status,
    WGPUAdapter adapter,
    WGPUStringView message,
    void* userdata1,
    void* /*userdata2*/) {
  auto* result = static_cast<AdapterResult*>(userdata1);
  if (status == WGPURequestAdapterStatus_Success) {
    result->adapter = adapter;
  } else {
    fprintf(
        stderr,
        "WebGPU adapter request failed (status %d): %.*s\n",
        static_cast<int>(status),
        static_cast<int>(message.length),
        message.data);
  }
}

void on_device_request(
    WGPURequestDeviceStatus status,
    WGPUDevice device,
    WGPUStringView message,
    void* userdata1,
    void* /*userdata2*/) {
  auto* result = static_cast<DeviceResult*>(userdata1);
  if (status == WGPURequestDeviceStatus_Success) {
    result->device = device;
  } else {
    fprintf(
        stderr,
        "WebGPU device request failed (status %d): %.*s\n",
        static_cast<int>(status),
        static_cast<int>(message.length),
        message.data);
  }
}

void on_device_error(
    WGPUDevice const* /*device*/,
    WGPUErrorType type,
    WGPUStringView message,
    void* /*userdata1*/,
    void* /*userdata2*/) {
  fprintf(
      stderr,
      "WebGPU device error (type %d): %.*s\n",
      static_cast<int>(type),
      static_cast<int>(message.length),
      message.data);
}

} // namespace

WebGPUContext create_webgpu_context() {
  WebGPUContext ctx;

  // TimedWaitAny lets webgpu_wait() block on futures via wgpuInstanceWaitAny.
  WGPUInstanceDescriptor instance_desc = {};
#if defined(__EMSCRIPTEN__)
  instance_desc.capabilities.timedWaitAnyEnable = true;
  instance_desc.capabilities.timedWaitAnyMaxCount = 1;
#else
  WGPUInstanceFeatureName features[1] = {WGPUInstanceFeatureName_TimedWaitAny};
  instance_desc.requiredFeatureCount = 1;
  instance_desc.requiredFeatures = features;
#endif
  ctx.instance = wgpuCreateInstance(&instance_desc);
  if (!ctx.instance) {
    throw std::runtime_error("Failed to create WebGPU instance");
  }

  AdapterResult adapter_result;
  WGPURequestAdapterCallbackInfo adapter_cb = {};
  adapter_cb.mode = WGPUCallbackMode_WaitAnyOnly;
  adapter_cb.callback = on_adapter_request;
  adapter_cb.userdata1 = &adapter_result;

  // No backend pin or forced fallback; Dawn auto-selects the adapter.
  WGPURequestAdapterOptions adapter_opts = {};
  adapter_opts.powerPreference = WGPUPowerPreference_HighPerformance;
  adapter_opts.forceFallbackAdapter = false;
  WGPUWaitStatus adapter_wait = webgpu_wait(
      ctx.instance,
      wgpuInstanceRequestAdapter(ctx.instance, &adapter_opts, adapter_cb));

  if (adapter_wait != WGPUWaitStatus_Success || !adapter_result.adapter) {
    wgpuInstanceRelease(ctx.instance);
    ctx.instance = nullptr;
    throw std::runtime_error(
        "Failed to get WebGPU adapter. "
        "Ensure a GPU with Vulkan (Linux) or Metal (macOS) is available.");
  }
  ctx.adapter = adapter_result.adapter;

  // Request device
  DeviceResult device_result;
  WGPURequestDeviceCallbackInfo device_cb = {};
  device_cb.mode = WGPUCallbackMode_WaitAnyOnly;
  device_cb.callback = on_device_request;
  device_cb.userdata1 = &device_result;

  // Request the adapter's full limits; software adapters default many to 0.
  WGPULimits supported_limits = {};
  WGPUDeviceDescriptor device_desc = {};
  if (wgpuAdapterGetLimits(ctx.adapter, &supported_limits) ==
      WGPUStatus_Success) {
    device_desc.requiredLimits = &supported_limits;
  }
  device_desc.uncapturedErrorCallbackInfo.callback = on_device_error;

  WGPUWaitStatus device_wait = webgpu_wait(
      ctx.instance,
      wgpuAdapterRequestDevice(ctx.adapter, &device_desc, device_cb));

  if (device_wait != WGPUWaitStatus_Success || !device_result.device) {
    wgpuAdapterRelease(ctx.adapter);
    wgpuInstanceRelease(ctx.instance);
    ctx.adapter = nullptr;
    ctx.instance = nullptr;
    throw std::runtime_error("Failed to get WebGPU device");
  }
  ctx.device = device_result.device;
  ctx.queue = wgpuDeviceGetQueue(ctx.device);

  return ctx;
}

namespace {
WebGPUContext* g_default_context = nullptr;
} // namespace

void set_default_webgpu_context(WebGPUContext* ctx) {
  g_default_context = ctx;
}

WebGPUContext* get_default_webgpu_context() {
  if (g_default_context) {
    return g_default_context;
  }
#if !defined(__EMSCRIPTEN__)
  // Native-only lazy process-wide context, mirroring Vulkan api::context().
  static const std::unique_ptr<WebGPUContext, void (*)(WebGPUContext*)>
  lazy_context(
      []() -> WebGPUContext* {
        try {
          return new WebGPUContext(create_webgpu_context());
        } catch (...) {
          return nullptr;
        }
      }(),
      [](WebGPUContext* c) {
        if (c) {
          destroy_webgpu_context(*c);
          delete c;
        }
      });
  return lazy_context.get();
#else
  return nullptr;
#endif
}

void destroy_webgpu_context(WebGPUContext& ctx) {
  if (ctx.queue) {
    wgpuQueueRelease(ctx.queue);
    ctx.queue = nullptr;
  }
  if (ctx.device) {
    wgpuDeviceRelease(ctx.device);
    ctx.device = nullptr;
  }
  if (ctx.adapter) {
    wgpuAdapterRelease(ctx.adapter);
    ctx.adapter = nullptr;
  }
  if (ctx.instance) {
    wgpuInstanceRelease(ctx.instance);
    ctx.instance = nullptr;
  }
}

} // namespace webgpu
} // namespace backends
} // namespace executorch
