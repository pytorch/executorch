/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/aoti/slim/c10/core/Device.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/backends/aoti/slim/factory/from_blob.h>
#include <executorch/backends/cuda/runtime/cuda_delegate_handle.h>
#include <executorch/backends/cuda/runtime/cuda_mutable_state.h>
#include <executorch/runtime/core/error.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace cu = ::executorch::backends::cuda;
namespace aoti = ::executorch::backends::aoti;
namespace slim = ::executorch::backends::aoti::slim;
namespace slimc10 = ::executorch::backends::aoti::slim::c10;
using ::executorch::runtime::Error;

namespace {

Error fake_get_num_constants(
    aoti::AOTInductorModelContainerHandle,
    size_t* num_constants) {
  *num_constants = 0;
  return Error::Ok;
}

Error fake_get_constant_name(
    aoti::AOTInductorModelContainerHandle,
    size_t,
    const char**) {
  return Error::Ok;
}

Error fake_get_constant_original_fqn(
    aoti::AOTInductorModelContainerHandle,
    size_t,
    const char**) {
  return Error::Ok;
}

Error fake_extract_constants_map(
    aoti::AOTInductorModelContainerHandle,
    aoti::AOTInductorConstantMapHandle,
    bool) {
  return Error::Ok;
}

Error fake_update_user_managed_pairs(
    aoti::AOTInductorModelContainerHandle,
    const aoti::AOTInductorConstantMapEntry*,
    size_t,
    bool,
    bool) {
  return Error::Ok;
}

struct FakeContainer {
  std::vector<std::string> internal_names;
  std::vector<std::string> fqns;
  std::unordered_map<std::string, aoti::AtenTensorHandle> extracted;
  size_t update_calls = 0;
  size_t last_num_pairs = 0;
  std::string last_name;
  void* last_bound_data = nullptr;
  size_t last_bound_nbytes = 0;
  int last_bound_device_index = -1;
  std::unordered_map<std::string, void*> bound_data_by_name;
  std::unordered_map<std::string, int> bound_device_by_name;
};

Error fake_container_get_num_constants(
    aoti::AOTInductorModelContainerHandle container,
    size_t* num_constants) {
  auto* c = reinterpret_cast<FakeContainer*>(container);
  *num_constants = c->internal_names.size();
  return Error::Ok;
}

Error fake_container_get_constant_name(
    aoti::AOTInductorModelContainerHandle container,
    size_t idx,
    const char** name) {
  auto* c = reinterpret_cast<FakeContainer*>(container);
  *name =
      idx < c->internal_names.size() ? c->internal_names[idx].c_str() : nullptr;
  return Error::Ok;
}

Error fake_container_get_constant_original_fqn(
    aoti::AOTInductorModelContainerHandle container,
    size_t idx,
    const char** fqn) {
  auto* c = reinterpret_cast<FakeContainer*>(container);
  *fqn = idx < c->fqns.size() ? c->fqns[idx].c_str() : nullptr;
  return Error::Ok;
}

Error fake_container_extract_constants_map(
    aoti::AOTInductorModelContainerHandle container,
    aoti::AOTInductorConstantMapHandle map_handle,
    bool) {
  auto* c = reinterpret_cast<FakeContainer*>(container);
  auto* out = reinterpret_cast<
      std::unordered_map<std::string, aoti::AtenTensorHandle>*>(map_handle);
  *out = c->extracted;
  return Error::Ok;
}

Error fake_container_update_user_managed_pairs(
    aoti::AOTInductorModelContainerHandle container,
    const aoti::AOTInductorConstantMapEntry* pairs,
    size_t num_pairs,
    bool,
    bool) {
  auto* c = reinterpret_cast<FakeContainer*>(container);
  c->update_calls++;
  c->last_num_pairs = num_pairs;
  if (num_pairs > 0) {
    c->last_name = pairs[0].name;
    auto* t = reinterpret_cast<slim::SlimTensor*>(pairs[0].handle);
    c->last_bound_data = t->data_ptr();
    c->last_bound_nbytes = t->nbytes();
    c->last_bound_device_index = t->device().index();
  }
  for (size_t i = 0; i < num_pairs; ++i) {
    auto* t = reinterpret_cast<slim::SlimTensor*>(pairs[i].handle);
    c->bound_data_by_name[pairs[i].name] = t->data_ptr();
    c->bound_device_by_name[pairs[i].name] = t->device().index();
  }
  return Error::Ok;
}

cu::CudaDelegateHandle fake_symbol_handle() {
  cu::CudaDelegateHandle handle{};
  handle.get_num_constants = fake_get_num_constants;
  handle.get_constant_name = fake_get_constant_name;
  handle.get_constant_original_fqn = fake_get_constant_original_fqn;
  handle.extract_constants_map = fake_extract_constants_map;
  handle.update_user_managed_constant_buffer_pairs =
      fake_update_user_managed_pairs;
  return handle;
}

cu::CudaDelegateHandle fake_container_handle(FakeContainer* container) {
  cu::CudaDelegateHandle handle{};
  handle.container_handle =
      reinterpret_cast<aoti::AOTInductorModelContainerHandle>(container);
  handle.get_num_constants = fake_container_get_num_constants;
  handle.get_constant_name = fake_container_get_constant_name;
  handle.get_constant_original_fqn = fake_container_get_constant_original_fqn;
  handle.extract_constants_map = fake_container_extract_constants_map;
  handle.update_user_managed_constant_buffer_pairs =
      fake_container_update_user_managed_pairs;
  return handle;
}

bool cuda_device_available() {
  int device_count = 0;
  const cudaError_t err = cudaGetDeviceCount(&device_count);
  return err == cudaSuccess && device_count > 0;
}

std::unique_ptr<slim::SlimTensor> make_device_tensor(
    const std::vector<float>& values,
    void** device_ptr,
    int device_index = 0) {
  *device_ptr = nullptr;
  cudaError_t err = cudaMalloc(device_ptr, values.size() * sizeof(float));
  if (err != cudaSuccess) {
    ADD_FAILURE() << "cudaMalloc failed: " << cudaGetErrorString(err);
    return nullptr;
  }
  err = cudaMemcpy(
      *device_ptr,
      values.data(),
      values.size() * sizeof(float),
      cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    ADD_FAILURE() << "cudaMemcpy failed: " << cudaGetErrorString(err);
    cudaFree(*device_ptr);
    *device_ptr = nullptr;
    return nullptr;
  }
  return std::make_unique<slim::SlimTensor>(slim::from_blob(
      *device_ptr,
      {static_cast<int64_t>(values.size())},
      slimc10::ScalarType::Float,
      slimc10::Device(slimc10::DeviceType::CUDA, device_index)));
}

std::unique_ptr<slim::SlimTensor> make_cpu_tensor(std::vector<float>& values) {
  return std::make_unique<slim::SlimTensor>(slim::from_blob(
      values.data(),
      {static_cast<int64_t>(values.size())},
      slimc10::ScalarType::Float,
      slimc10::Device(slimc10::DeviceType::CPU, 0)));
}

} // namespace

TEST(CudaMutableStateTest, FallClosedDefaults) {
  const cu::MutableStateContext bad = 999999;
  cu::MutableStateContextOwner c1;
  cu::MutableStateContextOwner c2;

  EXPECT_GT(c2.get(), c1.get());
  EXPECT_TRUE(c1);
  EXPECT_FALSE(c1.available());
  EXPECT_EQ(c1.bytes_per_session(), 0);
  EXPECT_EQ(cu::detail::mutable_state_bytes_per_session(bad), 0);
  EXPECT_EQ(
      cu::detail::mutable_state_validate_coverage(bad), Error::InvalidArgument);
  EXPECT_EQ(c1.validate_coverage(), Error::NotSupported);

  c1.register_fqns({"a.b", "c.d"});
  EXPECT_EQ(c1.validate_coverage(), Error::NotSupported);
  EXPECT_EQ(
      cu::detail::mutable_state_create_session(bad).error(),
      Error::InvalidArgument);
  EXPECT_EQ(c1.create_session().error(), Error::NotSupported);

  cu::detail::mutable_state_destroy_session(bad, 0);
  cu::detail::mutable_state_destroy_context(bad);
}

TEST(CudaMutableStateTest, ForgetHandleDropsAssociation) {
  cu::MutableStateContextOwner c;
  cu::CudaDelegateHandle handle{};

  c.with_load_scope([&] { cu::mutable_state_note_handle(&handle); });

  c.with_active_session(0, [&] {
    EXPECT_EQ(
        cu::mutable_state_rebind_for_execute(&handle), Error::NotSupported);

    cu::mutable_state_forget_handle(&handle);
    EXPECT_EQ(cu::mutable_state_rebind_for_execute(&handle), Error::Internal);
  });
}

TEST(CudaMutableStateTest, CreateSessionRejectsEmptyFqns) {
  cu::MutableStateContextOwner c;
  cu::CudaDelegateHandle handle = fake_symbol_handle();

  c.with_load_scope([&] { cu::mutable_state_note_handle(&handle); });
  ASSERT_TRUE(c.available());
  EXPECT_EQ(c.create_session().error(), Error::InvalidState);
  EXPECT_EQ(c.validate_coverage(), Error::InvalidState);
  EXPECT_FALSE(c.available());
}

TEST(CudaMutableStateTest, CreateSessionValidatesCoverageBeforeIssuingToken) {
  cu::MutableStateContextOwner c;
  cu::CudaDelegateHandle handle = fake_symbol_handle();

  c.register_fqns({"missing.state"});
  c.with_load_scope([&] { cu::mutable_state_note_handle(&handle); });

  ASSERT_TRUE(c.available());
  EXPECT_EQ(c.create_session().error(), Error::InvalidProgram);
  EXPECT_EQ(c.validate_coverage(), Error::InvalidProgram);
  EXPECT_FALSE(c.available());
}

TEST(CudaMutableStateTest, RegisterFqnsAfterLoadFailsClosed) {
  cu::MutableStateContextOwner c;
  cu::CudaDelegateHandle handle = fake_symbol_handle();

  c.with_load_scope([&] { cu::mutable_state_note_handle(&handle); });

  ASSERT_TRUE(c.available());
  c.register_fqns({"late.state"});
  EXPECT_FALSE(c.available());
  EXPECT_EQ(c.validate_coverage(), Error::InvalidState);
  EXPECT_EQ(c.create_session().error(), Error::InvalidState);
}

TEST(CudaMutableStateTest, NestedBeginLoadFailsClosed) {
  cu::MutableStateContextOwner c1;
  cu::MutableStateContextOwner c2;
  cu::CudaDelegateHandle handle = fake_symbol_handle();

  c1.with_load_scope([&] {
    c2.with_load_scope([&] { cu::mutable_state_note_handle(&handle); });
  });

  EXPECT_EQ(c1.validate_coverage(), Error::InvalidState);
  EXPECT_EQ(c2.validate_coverage(), Error::InvalidState);
  EXPECT_FALSE(c1.available());
  EXPECT_FALSE(c2.available());
  EXPECT_EQ(c1.create_session().error(), Error::InvalidState);
  EXPECT_EQ(c2.create_session().error(), Error::InvalidState);
}

TEST(CudaMutableStateTest, OwnerLoadScopeClearsThreadLocalLoadState) {
  cu::MutableStateContextOwner c1;
  cu::MutableStateContextOwner c2;
  cu::CudaDelegateHandle h1 = fake_symbol_handle();
  cu::CudaDelegateHandle h2 = fake_symbol_handle();

  c1.with_load_scope([&] { cu::mutable_state_note_handle(&h1); });
  c2.with_load_scope([&] { cu::mutable_state_note_handle(&h2); });

  EXPECT_TRUE(c1.available());
  EXPECT_TRUE(c2.available());
}

TEST(CudaMutableStateTest, RebindRejectsCudaGraphHandle) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "CUDA device unavailable";
  }

  void* source_ptr = nullptr;
  auto source_tensor = make_device_tensor({1.0f}, &source_ptr);
  ASSERT_NE(source_tensor, nullptr);
  ASSERT_NE(source_ptr, nullptr);

  FakeContainer container;
  container.internal_names = {"internal_state"};
  container.fqns = {"model.state"};
  container.extracted["model.state"] =
      reinterpret_cast<aoti::AtenTensorHandle>(source_tensor.get());
  cu::MutableStateContextOwner c;
  cu::CudaDelegateHandle handle = fake_container_handle(&container);

  c.register_fqns({"model.state"});
  c.with_load_scope([&] { cu::mutable_state_note_handle(&handle); });
  ASSERT_TRUE(c.available());
  ASSERT_EQ(c.validate_coverage(), Error::Ok);

  auto token = c.create_session();
  ASSERT_TRUE(token.ok());

  handle.cuda_graph_state.phase = cu::CudaGraphPhase::Warmup;
  EXPECT_EQ(
      c.with_active_session(
          token.get(),
          [&] { return cu::mutable_state_rebind_for_execute(&handle); }),
      Error::NotSupported);

  c.destroy_session(token.get());
  cudaFree(source_ptr);
}

TEST(CudaMutableStateTest, CapturesClonesAndRebindsDeviceBuffer) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "CUDA device unavailable";
  }

  void* source_ptr = nullptr;
  auto source_tensor =
      make_device_tensor({1.0f, 2.0f, 3.0f, 4.0f}, &source_ptr);
  ASSERT_NE(source_tensor, nullptr);
  ASSERT_NE(source_ptr, nullptr);

  FakeContainer container;
  container.internal_names = {"internal_state"};
  container.fqns = {"model.state"};
  container.extracted["model.state"] =
      reinterpret_cast<aoti::AtenTensorHandle>(source_tensor.get());
  cu::CudaDelegateHandle handle = fake_container_handle(&container);

  cu::MutableStateContextOwner c;
  c.register_fqns({"model.state"});
  c.with_load_scope([&] { cu::mutable_state_note_handle(&handle); });

  ASSERT_TRUE(c.available());
  EXPECT_EQ(c.bytes_per_session(), 4 * sizeof(float));
  ASSERT_EQ(c.validate_coverage(), Error::Ok);
  EXPECT_EQ(
      c.with_active_session(
          123, [&] { return cu::mutable_state_rebind_for_execute(&handle); }),
      Error::InvalidArgument);

  auto token = c.create_session();
  ASSERT_TRUE(token.ok());
  EXPECT_EQ(cu::mutable_state_rebind_for_execute(&handle), Error::InvalidState);

  ASSERT_EQ(
      c.with_active_session(
          token.get(),
          [&] { return cu::mutable_state_rebind_for_execute(&handle); }),
      Error::Ok);

  EXPECT_EQ(container.update_calls, 1u);
  EXPECT_EQ(container.last_num_pairs, 1u);
  EXPECT_EQ(container.last_name, "internal_state");
  ASSERT_NE(container.last_bound_data, nullptr);
  EXPECT_NE(container.last_bound_data, source_ptr);
  EXPECT_EQ(container.last_bound_nbytes, 4 * sizeof(float));

  std::vector<float> cloned(4);
  EXPECT_EQ(
      cudaMemcpy(
          cloned.data(),
          container.last_bound_data,
          cloned.size() * sizeof(float),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  EXPECT_EQ(cloned, (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));

  EXPECT_EQ(cu::mutable_state_rebind_for_execute(&handle), Error::InvalidState);

  EXPECT_EQ(
      c.with_active_session(
          token.get(),
          [&] { return cu::mutable_state_rebind_for_execute(&handle); }),
      Error::Ok);

  c.destroy_session(token.get());
  cudaFree(source_ptr);
}

TEST(CudaMutableStateTest, SharedFqnAcrossHandlesUsesSameSessionBuffer) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "CUDA device unavailable";
  }

  void* prefill_ptr = nullptr;
  void* decode_ptr = nullptr;
  auto prefill_tensor = make_device_tensor({1.0f, 2.0f}, &prefill_ptr);
  auto decode_tensor = make_device_tensor({9.0f, 8.0f}, &decode_ptr);
  ASSERT_NE(prefill_tensor, nullptr);
  ASSERT_NE(decode_tensor, nullptr);
  ASSERT_NE(prefill_ptr, nullptr);
  ASSERT_NE(decode_ptr, nullptr);

  FakeContainer prefill_container;
  prefill_container.internal_names = {"prefill_internal_kv"};
  prefill_container.fqns = {"model.kv"};
  prefill_container.extracted["model.kv"] =
      reinterpret_cast<aoti::AtenTensorHandle>(prefill_tensor.get());
  cu::CudaDelegateHandle prefill_handle =
      fake_container_handle(&prefill_container);

  FakeContainer decode_container;
  decode_container.internal_names = {"decode_internal_kv"};
  decode_container.fqns = {"model.kv"};
  decode_container.extracted["model.kv"] =
      reinterpret_cast<aoti::AtenTensorHandle>(decode_tensor.get());
  cu::CudaDelegateHandle decode_handle =
      fake_container_handle(&decode_container);

  cu::MutableStateContextOwner c;
  c.register_fqns({"model.kv"});
  c.with_load_scope([&] {
    cu::mutable_state_note_handle(&prefill_handle);
    cu::mutable_state_note_handle(&decode_handle);
  });

  ASSERT_TRUE(c.available());
  ASSERT_EQ(c.validate_coverage(), Error::Ok);

  auto token = c.create_session();
  ASSERT_TRUE(token.ok());
  ASSERT_EQ(
      c.with_active_session(
          token.get(),
          [&] {
            Error e = cu::mutable_state_rebind_for_execute(&prefill_handle);
            if (e != Error::Ok) {
              return e;
            }
            return cu::mutable_state_rebind_for_execute(&decode_handle);
          }),
      Error::Ok);

  ASSERT_NE(prefill_container.last_bound_data, nullptr);
  ASSERT_NE(decode_container.last_bound_data, nullptr);
  EXPECT_EQ(prefill_container.last_name, "prefill_internal_kv");
  EXPECT_EQ(decode_container.last_name, "decode_internal_kv");
  EXPECT_EQ(
      prefill_container.last_bound_data, decode_container.last_bound_data);
  EXPECT_NE(prefill_container.last_bound_data, prefill_ptr);
  EXPECT_NE(decode_container.last_bound_data, decode_ptr);

  c.destroy_session(token.get());
  cudaFree(prefill_ptr);
  cudaFree(decode_ptr);
}

TEST(CudaMutableStateTest, SessionsStayIsolatedForSameHandleAndFqn) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "CUDA device unavailable";
  }

  void* source_ptr = nullptr;
  auto source_tensor = make_device_tensor({0.0f, 0.0f}, &source_ptr);
  ASSERT_NE(source_tensor, nullptr);
  ASSERT_NE(source_ptr, nullptr);

  FakeContainer container;
  container.internal_names = {"internal_kv"};
  container.fqns = {"model.kv"};
  container.extracted["model.kv"] =
      reinterpret_cast<aoti::AtenTensorHandle>(source_tensor.get());
  cu::CudaDelegateHandle handle = fake_container_handle(&container);

  cu::MutableStateContextOwner c;
  c.register_fqns({"model.kv"});
  c.with_load_scope([&] { cu::mutable_state_note_handle(&handle); });

  ASSERT_TRUE(c.available());
  ASSERT_EQ(c.validate_coverage(), Error::Ok);

  auto session_a = c.create_session();
  auto session_b = c.create_session();
  ASSERT_TRUE(session_a.ok());
  ASSERT_TRUE(session_b.ok());

  void* a_ptr = nullptr;
  const std::vector<float> a_values = {1.0f, 2.0f};
  ASSERT_EQ(
      c.with_active_session(
          session_a.get(),
          [&] { return cu::mutable_state_rebind_for_execute(&handle); }),
      Error::Ok);
  a_ptr = container.bound_data_by_name["internal_kv"];
  ASSERT_NE(a_ptr, nullptr);
  ASSERT_EQ(
      cudaMemcpy(
          a_ptr,
          a_values.data(),
          a_values.size() * sizeof(float),
          cudaMemcpyHostToDevice),
      cudaSuccess);

  void* b_ptr = nullptr;
  const std::vector<float> b_values = {9.0f, 8.0f};
  ASSERT_EQ(
      c.with_active_session(
          session_b.get(),
          [&] { return cu::mutable_state_rebind_for_execute(&handle); }),
      Error::Ok);
  b_ptr = container.bound_data_by_name["internal_kv"];
  ASSERT_NE(b_ptr, nullptr);
  EXPECT_NE(a_ptr, b_ptr);
  ASSERT_EQ(
      cudaMemcpy(
          b_ptr,
          b_values.data(),
          b_values.size() * sizeof(float),
          cudaMemcpyHostToDevice),
      cudaSuccess);

  ASSERT_EQ(
      c.with_active_session(
          session_a.get(),
          [&] { return cu::mutable_state_rebind_for_execute(&handle); }),
      Error::Ok);
  EXPECT_EQ(container.bound_data_by_name["internal_kv"], a_ptr);

  std::vector<float> read_a(2);
  ASSERT_EQ(
      cudaMemcpy(
          read_a.data(),
          a_ptr,
          read_a.size() * sizeof(float),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  EXPECT_EQ(read_a, a_values);

  ASSERT_EQ(
      c.with_active_session(
          session_b.get(),
          [&] { return cu::mutable_state_rebind_for_execute(&handle); }),
      Error::Ok);
  EXPECT_EQ(container.bound_data_by_name["internal_kv"], b_ptr);

  std::vector<float> read_b(2);
  ASSERT_EQ(
      cudaMemcpy(
          read_b.data(),
          b_ptr,
          read_b.size() * sizeof(float),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  EXPECT_EQ(read_b, b_values);

  c.destroy_session(session_a.get());
  c.destroy_session(session_b.get());
  cudaFree(source_ptr);
}

TEST(CudaMutableStateTest, EmptyInternalNameIsSkipped) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "CUDA device unavailable";
  }

  void* skipped_ptr = nullptr;
  void* valid_ptr = nullptr;
  auto skipped_tensor = make_device_tensor({9.0f, 8.0f}, &skipped_ptr);
  auto valid_tensor = make_device_tensor({1.0f, 2.0f}, &valid_ptr);
  ASSERT_NE(skipped_tensor, nullptr);
  ASSERT_NE(valid_tensor, nullptr);
  ASSERT_NE(skipped_ptr, nullptr);
  ASSERT_NE(valid_ptr, nullptr);

  FakeContainer skipped_container;
  skipped_container.internal_names = {""};
  skipped_container.fqns = {"model.kv"};
  skipped_container.extracted["model.kv"] =
      reinterpret_cast<aoti::AtenTensorHandle>(skipped_tensor.get());
  cu::CudaDelegateHandle skipped_handle =
      fake_container_handle(&skipped_container);

  FakeContainer valid_container;
  valid_container.internal_names = {"valid_internal_kv"};
  valid_container.fqns = {"model.kv"};
  valid_container.extracted["model.kv"] =
      reinterpret_cast<aoti::AtenTensorHandle>(valid_tensor.get());
  cu::CudaDelegateHandle valid_handle = fake_container_handle(&valid_container);

  cu::MutableStateContextOwner c;
  c.register_fqns({"model.kv"});
  c.with_load_scope([&] {
    cu::mutable_state_note_handle(&skipped_handle);
    cu::mutable_state_note_handle(&valid_handle);
  });

  ASSERT_TRUE(c.available());
  ASSERT_EQ(c.validate_coverage(), Error::Ok);

  auto token = c.create_session();
  ASSERT_TRUE(token.ok());
  ASSERT_EQ(
      c.with_active_session(
          token.get(),
          [&] {
            Error e = cu::mutable_state_rebind_for_execute(&skipped_handle);
            if (e != Error::Ok) {
              return e;
            }
            return cu::mutable_state_rebind_for_execute(&valid_handle);
          }),
      Error::Ok);
  EXPECT_EQ(skipped_container.update_calls, 0u);

  EXPECT_EQ(valid_container.update_calls, 1u);
  EXPECT_EQ(valid_container.last_name, "valid_internal_kv");
  ASSERT_NE(valid_container.last_bound_data, nullptr);
  EXPECT_NE(valid_container.last_bound_data, valid_ptr);
  EXPECT_NE(valid_container.last_bound_data, skipped_ptr);

  c.destroy_session(token.get());
  cudaFree(skipped_ptr);
  cudaFree(valid_ptr);
}

TEST(
    CudaMutableStateTest,
    ValidateCoverageRejectsLargerDescriptorForSharedFqn) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "CUDA device unavailable";
  }

  void* small_ptr = nullptr;
  void* large_ptr = nullptr;
  auto small_tensor = make_device_tensor({1.0f}, &small_ptr);
  auto large_tensor = make_device_tensor({1.0f, 2.0f}, &large_ptr);
  ASSERT_NE(small_tensor, nullptr);
  ASSERT_NE(large_tensor, nullptr);
  ASSERT_NE(small_ptr, nullptr);
  ASSERT_NE(large_ptr, nullptr);

  FakeContainer small_container;
  small_container.internal_names = {"small_internal"};
  small_container.fqns = {"model.state"};
  small_container.extracted["model.state"] =
      reinterpret_cast<aoti::AtenTensorHandle>(small_tensor.get());
  cu::CudaDelegateHandle small_handle = fake_container_handle(&small_container);

  FakeContainer large_container;
  large_container.internal_names = {"large_internal"};
  large_container.fqns = {"model.state"};
  large_container.extracted["model.state"] =
      reinterpret_cast<aoti::AtenTensorHandle>(large_tensor.get());
  cu::CudaDelegateHandle large_handle = fake_container_handle(&large_container);

  cu::MutableStateContextOwner c;
  c.register_fqns({"model.state"});
  c.with_load_scope([&] {
    cu::mutable_state_note_handle(&small_handle);
    cu::mutable_state_note_handle(&large_handle);
  });

  ASSERT_TRUE(c.available());
  EXPECT_EQ(c.validate_coverage(), Error::InvalidProgram);
  EXPECT_FALSE(c.available());
  EXPECT_EQ(c.create_session().error(), Error::InvalidProgram);
  EXPECT_EQ(large_container.update_calls, 0u);

  cudaFree(small_ptr);
  cudaFree(large_ptr);
}

TEST(
    CudaMutableStateTest,
    ValidateCoverageNormalizesUnspecifiedCudaDeviceForSharedFqn) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "CUDA device unavailable";
  }

  int current_device = 0;
  ASSERT_EQ(cudaGetDevice(&current_device), cudaSuccess);

  void* unspecified_ptr = nullptr;
  void* explicit_ptr = nullptr;
  auto unspecified_tensor =
      make_device_tensor({1.0f}, &unspecified_ptr, /*device_index=*/-1);
  auto explicit_tensor =
      make_device_tensor({2.0f}, &explicit_ptr, current_device);
  ASSERT_NE(unspecified_tensor, nullptr);
  ASSERT_NE(explicit_tensor, nullptr);
  ASSERT_NE(unspecified_ptr, nullptr);
  ASSERT_NE(explicit_ptr, nullptr);

  FakeContainer unspecified_container;
  unspecified_container.internal_names = {"unspecified_internal"};
  unspecified_container.fqns = {"model.state"};
  unspecified_container.extracted["model.state"] =
      reinterpret_cast<aoti::AtenTensorHandle>(unspecified_tensor.get());
  cu::CudaDelegateHandle unspecified_handle =
      fake_container_handle(&unspecified_container);

  FakeContainer explicit_container;
  explicit_container.internal_names = {"explicit_internal"};
  explicit_container.fqns = {"model.state"};
  explicit_container.extracted["model.state"] =
      reinterpret_cast<aoti::AtenTensorHandle>(explicit_tensor.get());
  cu::CudaDelegateHandle explicit_handle =
      fake_container_handle(&explicit_container);

  cu::MutableStateContextOwner c;
  c.register_fqns({"model.state"});
  c.with_load_scope([&] {
    cu::mutable_state_note_handle(&unspecified_handle);
    cu::mutable_state_note_handle(&explicit_handle);
  });

  ASSERT_TRUE(c.available());
  ASSERT_EQ(c.validate_coverage(), Error::Ok);

  auto token = c.create_session();
  ASSERT_TRUE(token.ok());
  ASSERT_EQ(
      c.with_active_session(
          token.get(),
          [&] {
            Error e = cu::mutable_state_rebind_for_execute(&unspecified_handle);
            if (e != Error::Ok) {
              return e;
            }
            return cu::mutable_state_rebind_for_execute(&explicit_handle);
          }),
      Error::Ok);

  EXPECT_EQ(unspecified_container.last_bound_device_index, current_device);
  EXPECT_EQ(explicit_container.last_bound_device_index, current_device);
  EXPECT_EQ(
      unspecified_container.bound_device_by_name["unspecified_internal"],
      current_device);
  EXPECT_EQ(
      explicit_container.bound_device_by_name["explicit_internal"],
      current_device);

  c.destroy_session(token.get());
  cudaFree(unspecified_ptr);
  cudaFree(explicit_ptr);
}

TEST(CudaMutableStateTest, BuildRejectsNonCudaDescriptorForSharedFqn) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "CUDA device unavailable";
  }

  void* cuda_ptr = nullptr;
  auto cuda_tensor = make_device_tensor({1.0f}, &cuda_ptr);
  ASSERT_NE(cuda_tensor, nullptr);
  ASSERT_NE(cuda_ptr, nullptr);

  std::vector<float> cpu_values = {1.0f};
  auto cpu_tensor = make_cpu_tensor(cpu_values);
  ASSERT_NE(cpu_tensor, nullptr);

  FakeContainer cuda_container;
  cuda_container.internal_names = {"cuda_internal"};
  cuda_container.fqns = {"model.state"};
  cuda_container.extracted["model.state"] =
      reinterpret_cast<aoti::AtenTensorHandle>(cuda_tensor.get());
  cu::CudaDelegateHandle cuda_handle = fake_container_handle(&cuda_container);

  FakeContainer cpu_container;
  cpu_container.internal_names = {"cpu_internal"};
  cpu_container.fqns = {"model.state"};
  cpu_container.extracted["model.state"] =
      reinterpret_cast<aoti::AtenTensorHandle>(cpu_tensor.get());
  cu::CudaDelegateHandle cpu_handle = fake_container_handle(&cpu_container);

  cu::MutableStateContextOwner c;
  c.register_fqns({"model.state"});
  c.with_load_scope([&] {
    cu::mutable_state_note_handle(&cuda_handle);
    cu::mutable_state_note_handle(&cpu_handle);
  });

  EXPECT_FALSE(c.available());
  EXPECT_EQ(c.validate_coverage(), Error::InvalidArgument);
  EXPECT_FALSE(c.available());
  EXPECT_EQ(c.create_session().error(), Error::InvalidArgument);
  EXPECT_EQ(cpu_container.update_calls, 0u);

  cudaFree(cuda_ptr);
}
