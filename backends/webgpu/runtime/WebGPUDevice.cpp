/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUDevice.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace executorch {
namespace backends {
namespace webgpu {

namespace {

struct AdapterResult {
  WGPUAdapter adapter = nullptr;
  bool done = false;
};

struct DeviceResult {
  WGPUDevice device = nullptr;
  bool done = false;
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
  result->done = true;
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
  result->done = true;
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

  ctx.instance = wgpuCreateInstance(nullptr);
  if (!ctx.instance) {
    throw std::runtime_error("Failed to create WebGPU instance");
  }

  // Request adapter using AllowSpontaneous mode (fires during
  // wgpuInstanceProcessEvents or any other API call).
  AdapterResult adapter_result;
  WGPURequestAdapterCallbackInfo adapter_cb = {};
  adapter_cb.mode = WGPUCallbackMode_AllowSpontaneous;
  adapter_cb.callback = on_adapter_request;
  adapter_cb.userdata1 = &adapter_result;

  wgpuInstanceRequestAdapter(ctx.instance, nullptr, adapter_cb);
  while (!adapter_result.done) {
    wgpuInstanceProcessEvents(ctx.instance);
  }

  if (!adapter_result.adapter) {
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
  device_cb.mode = WGPUCallbackMode_AllowSpontaneous;
  device_cb.callback = on_device_request;
  device_cb.userdata1 = &device_result;

  WGPUDeviceDescriptor device_desc = {};
  device_desc.uncapturedErrorCallbackInfo.callback = on_device_error;

  wgpuAdapterRequestDevice(ctx.adapter, &device_desc, device_cb);
  while (!device_result.done) {
    wgpuInstanceProcessEvents(ctx.instance);
  }

  if (!device_result.device) {
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
  return g_default_context;
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
