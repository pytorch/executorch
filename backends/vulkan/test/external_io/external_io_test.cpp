// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// End-to-end tests of external I/O through the Vulkan delegate. They load the
// programs written by export_models.py from the directory named by
// --models_dir.

#include <executorch/backends/vulkan/runtime/ExternalIO.h>
#include <executorch/backends/vulkan/runtime/api/api.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/backend/backend_options_map.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/platform/runtime.h>

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#ifdef __ANDROID__
#include <android/hardware_buffer.h>
#include <dlfcn.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

DEFINE_string(
    models_dir,
    ".",
    "Directory of the programs written by export_models.py.");

namespace etvk = executorch::backends::vulkan;
namespace api = vkcompute::api;
namespace vkapi = vkcompute::vkapi;
namespace utils = vkcompute::utils;

using etvk::ExternalIOTarget;
using etvk::SlotType;
using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::runtime::BackendOptions;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::LoadBackendOptionsMap;

namespace {

// Every input and output of the exported programs has these sizes.
const std::vector<int64_t> kSizes = {1, 8, 16, 24};
constexpr size_t kNumel = 1 * 8 * 16 * 24;

using Data = std::vector<std::vector<float>>;

std::string program_path(const std::string& name) {
  return FLAGS_models_dir + "/" + name + ".pte";
}

// Names the external I/O sets of a test after it, so that a set a failed test
// leaves registered does not affect the others.
std::string set_name(const std::string& suffix = "") {
  return std::string(
             ::testing::UnitTest::GetInstance()->current_test_info()->name()) +
      suffix;
}

Data make_inputs(const size_t count, const float scale = 1.0f) {
  Data inputs(count, std::vector<float>(kNumel));
  for (size_t i = 0; i < count; ++i) {
    for (size_t j = 0; j < kNumel; ++j) {
      inputs[i][j] = scale * std::sin(0.37f * j + 1.3f * i);
    }
  }
  return inputs;
}

Data zeros(const size_t count) {
  return Data(count, std::vector<float>(kNumel, 0.0f));
}

// Loads forward(), bound to external I/O set `set` unless it is null.
Error load(Module& module, const char* set) {
  if (set == nullptr) {
    return module.load_method("forward");
  }
  BackendOptions<1> options;
  options.set_option(etvk::kExternalIOOption, set);
  LoadBackendOptionsMap map;
  map.set_options("VulkanBackend", options.view());
  return module.load_method("forward", nullptr, nullptr, &map);
}

// Runs forward() and returns the data of its output tensors.
Data forward(Module& module, const Data& inputs) {
  const std::vector<executorch::aten::SizesType> sizes(
      kSizes.begin(), kSizes.end());
  std::vector<TensorPtr> tensors;
  std::vector<EValue> args;
  for (const std::vector<float>& input : inputs) {
    tensors.push_back(from_blob(
        const_cast<float*>(input.data()),
        sizes,
        executorch::aten::ScalarType::Float));
    args.emplace_back(tensors.back());
  }
  const auto outputs = module.forward(args);
  if (!outputs.ok()) {
    ADD_FAILURE() << "forward() failed with error "
                  << static_cast<int>(outputs.error());
    return {};
  }
  Data result;
  for (const EValue& output : *outputs) {
    const auto& tensor = output.toTensor();
    const float* const data = tensor.const_data_ptr<float>();
    result.emplace_back(data, data + tensor.numel());
  }
  return result;
}

// Output 0 of `program` through the default path, which copies through host
// memory.
std::vector<float> reference(const std::string& program, const Data& inputs) {
  Module module(program_path(program));
  Data outputs = forward(module, inputs);
  return outputs.empty() ? std::vector<float>() : std::move(outputs[0]);
}

// Elements whose bits differ; external I/O must not change the results.
size_t num_mismatches(
    const std::vector<float>& actual,
    const std::vector<float>& expected) {
  if (actual.size() != expected.size()) {
    return std::max(actual.size(), expected.size());
  }
  size_t count = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    count += std::memcmp(&actual[i], &expected[i], sizeof(float)) != 0;
  }
  return count;
}

//
// Resources of the app. They live on the delegate's VkDevice: every graph
// uses the runtime's adapter, as does the global context used here.
//

class AppBuffer {
 public:
  virtual ~AppBuffer() = default;
  virtual etvk::ExternalIOResource resource() = 0;
  virtual void write(const std::vector<float>& data) = 0;
  virtual std::vector<float> read() = 0;
};

class VkAppBuffer final : public AppBuffer {
  api::StagingBuffer buffer_;

 public:
  explicit VkAppBuffer(const size_t numel = kNumel)
      : buffer_(
            api::context(),
            vkapi::kFloat,
            numel,
            vkapi::CopyDirection::DEVICE_TO_HOST) {}

  etvk::ExternalIOResource resource() override {
    return etvk::ExternalVkBuffer{buffer_.buffer().handle(), buffer_.nbytes()};
  }

  void write(const std::vector<float>& data) override {
    buffer_.copy_from(data.data(), data.size() * sizeof(float));
  }

  std::vector<float> read() override {
    std::vector<float> data(buffer_.numel());
    buffer_.copy_to(data.data(), data.size() * sizeof(float));
    return data;
  }
};

#ifdef __ANDROID__
// Looked up at runtime because they need API 26 at compile time.
struct AHardwareBufferFns {
  int (*allocate)(const AHardwareBuffer_Desc*, AHardwareBuffer**);
  void (*release)(AHardwareBuffer*);
  int (*lock)(AHardwareBuffer*, uint64_t, int32_t, const ARect*, void**);
  int (*unlock)(AHardwareBuffer*, int32_t*);
};

const AHardwareBufferFns& ahardwarebuffer_fns() {
  static const AHardwareBufferFns fns = [] {
    void* const lib = dlopen("libandroid.so", RTLD_NOW);
    return AHardwareBufferFns{
        reinterpret_cast<decltype(AHardwareBufferFns::allocate)>(
            dlsym(lib, "AHardwareBuffer_allocate")),
        reinterpret_cast<decltype(AHardwareBufferFns::release)>(
            dlsym(lib, "AHardwareBuffer_release")),
        reinterpret_cast<decltype(AHardwareBufferFns::lock)>(
            dlsym(lib, "AHardwareBuffer_lock")),
        reinterpret_cast<decltype(AHardwareBufferFns::unlock)>(
            dlsym(lib, "AHardwareBuffer_unlock")),
    };
  }();
  return fns;
}

class AHardwareBufferAppBuffer final : public AppBuffer {
  AHardwareBuffer* ahb_ = nullptr;

 public:
  AHardwareBufferAppBuffer() {
    AHardwareBuffer_Desc desc{};
    desc.width = static_cast<uint32_t>(kNumel * sizeof(float));
    desc.height = 1;
    desc.layers = 1;
    desc.format = AHARDWAREBUFFER_FORMAT_BLOB;
    desc.usage = AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER |
        AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN |
        AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN;
    EXPECT_EQ(ahardwarebuffer_fns().allocate(&desc, &ahb_), 0);
  }

  ~AHardwareBufferAppBuffer() override {
    if (ahb_ != nullptr) {
      ahardwarebuffer_fns().release(ahb_);
    }
  }

  etvk::ExternalIOResource resource() override {
    return etvk::ExternalAHardwareBuffer{ahb_};
  }

  void write(const std::vector<float>& data) override {
    void* mapped = nullptr;
    ahardwarebuffer_fns().lock(
        ahb_, AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN, -1, nullptr, &mapped);
    std::memcpy(mapped, data.data(), data.size() * sizeof(float));
    ahardwarebuffer_fns().unlock(ahb_, nullptr);
  }

  std::vector<float> read() override {
    std::vector<float> data(kNumel);
    void* mapped = nullptr;
    ahardwarebuffer_fns().lock(
        ahb_, AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN, -1, nullptr, &mapped);
    std::memcpy(data.data(), mapped, data.size() * sizeof(float));
    ahardwarebuffer_fns().unlock(ahb_, nullptr);
    return data;
  }
};
#endif // __ANDROID__

using MakeBuffer = std::unique_ptr<AppBuffer> (*)();

std::unique_ptr<AppBuffer> make_vkbuffer() {
  return std::make_unique<VkAppBuffer>();
}

#ifdef __ANDROID__
std::unique_ptr<AppBuffer> make_ahardwarebuffer() {
  return std::make_unique<AHardwareBufferAppBuffer>();
}
#endif // __ANDROID__

// A texture3d image packed as the delegate packs a tensor of `sizes`, in the
// VK_IMAGE_LAYOUT_GENERAL layout that external images must be in.
std::unique_ptr<api::vTensor> make_image(const std::vector<int64_t>& sizes) {
  api::Context* const context = api::context();
  auto image = std::make_unique<api::vTensor>(
      context, sizes, vkapi::kFloat, utils::kTexture3D, utils::kChannelsPacked);
  // A compute shader write puts an image in VK_IMAGE_LAYOUT_GENERAL.
  vkapi::PipelineBarrier barrier{};
  image->image(
      barrier, vkapi::PipelineStage::COMPUTE, vkapi::MemoryAccessType::WRITE);
  context->set_cmd();
  context->register_barrier(barrier);
  vkapi::VulkanFence fence = context->fences().get_fence();
  context->submit_cmd_to_gpu(fence.get_submit_handle());
  fence.wait();
  context->fences().return_fence(fence);
  return image;
}

etvk::ExternalIOResource image_resource(api::vTensor& tensor) {
  const vkapi::VulkanImage& image = tensor.image();
  return etvk::ExternalVkImage{
      image.handle(), image.image_view(), image.format(), image.extents()};
}

class VulkanExternalIOTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    executorch::runtime::runtime_init();
  }
};

// Both inputs and the output of conv_add are staged in app buffers.
void check_staging(const MakeBuffer make) {
  const std::unique_ptr<AppBuffer> x = make();
  const std::unique_ptr<AppBuffer> y = make();
  const std::unique_ptr<AppBuffer> out = make();
  const std::string set = set_name();
  ASSERT_EQ(
      etvk::register_external_io(
          set.c_str(),
          {{SlotType::Input, 0, ExternalIOTarget::Staging, x->resource()},
           {SlotType::Input, 1, ExternalIOTarget::Staging, y->resource()},
           {SlotType::Output, 0, ExternalIOTarget::Staging, out->resource()}}),
      Error::Ok);
  {
    Module module(program_path("conv_add"));
    ASSERT_EQ(load(module, set.c_str()), Error::Ok);
    for (const float scale : {1.0f, 0.5f, -2.0f}) {
      SCOPED_TRACE(scale);
      const Data inputs = make_inputs(2, scale);
      x->write(inputs[0]);
      y->write(inputs[1]);
      out->write(std::vector<float>(kNumel, NAN));
      // The delegate reads the app buffers, not these inputs.
      forward(module, zeros(2));
      EXPECT_EQ(num_mismatches(out->read(), reference("conv_add", inputs)), 0u);
    }
  }
  EXPECT_EQ(etvk::unregister_external_io(set.c_str()), Error::Ok);
}

// The state of a recurrent cell lives in one buffer, as input 1 and output 0.
void check_shared_state(const MakeBuffer make) {
  const std::unique_ptr<AppBuffer> x = make();
  const std::unique_ptr<AppBuffer> state = make();
  const std::string set = set_name();
  ASSERT_EQ(
      etvk::register_external_io(
          set.c_str(),
          {{SlotType::Input, 0, ExternalIOTarget::Staging, x->resource()},
           {SlotType::Input, 1, ExternalIOTarget::Staging, state->resource()},
           {SlotType::Output,
            0,
            ExternalIOTarget::Staging,
            state->resource()}}),
      Error::Ok);
  const Data inputs = make_inputs(2);
  x->write(inputs[0]);
  state->write(inputs[1]);
  std::vector<float> expected = inputs[1];
  {
    Module module(program_path("recurrent"));
    ASSERT_EQ(load(module, set.c_str()), Error::Ok);
    for (int i = 0; i < 20; ++i) {
      forward(module, zeros(2));
      expected = reference("recurrent", {inputs[0], expected});
    }
  }
  EXPECT_EQ(num_mismatches(state->read(), expected), 0u);
  EXPECT_EQ(etvk::unregister_external_io(set.c_str()), Error::Ok);
}

// The output tensor of `producer` and the input tensor of `consumer` are both
// backed by `resource`.
void check_tensor_chain(
    const std::string& producer,
    const std::string& consumer,
    const etvk::ExternalIOResource& resource) {
  const std::string producer_set = set_name("_producer");
  const std::string consumer_set = set_name("_consumer");
  ASSERT_EQ(
      etvk::register_external_io(
          producer_set.c_str(),
          {{SlotType::Output, 0, ExternalIOTarget::Tensor, resource}}),
      Error::Ok);
  ASSERT_EQ(
      etvk::register_external_io(
          consumer_set.c_str(),
          {{SlotType::Input, 0, ExternalIOTarget::Tensor, resource}}),
      Error::Ok);
  {
    Module a(program_path(producer));
    Module b(program_path(consumer));
    ASSERT_EQ(load(a, producer_set.c_str()), Error::Ok);
    ASSERT_EQ(load(b, consumer_set.c_str()), Error::Ok);
    for (const float scale : {1.0f, 0.25f}) {
      SCOPED_TRACE(scale);
      const Data x = make_inputs(1, scale);
      forward(a, x);
      const Data z = forward(b, zeros(1));
      ASSERT_EQ(z.size(), 1u);
      EXPECT_EQ(
          num_mismatches(z[0], reference(consumer, {reference(producer, x)})),
          0u);
    }
  }
  EXPECT_EQ(etvk::unregister_external_io(producer_set.c_str()), Error::Ok);
  EXPECT_EQ(etvk::unregister_external_io(consumer_set.c_str()), Error::Ok);
}

TEST_F(VulkanExternalIOTest, stages_inputs_and_output_in_vkbuffers) {
  check_staging(make_vkbuffer);
}

TEST_F(VulkanExternalIOTest, shares_recurrent_state_in_one_vkbuffer) {
  check_shared_state(make_vkbuffer);
}

TEST_F(VulkanExternalIOTest, chains_buffer_tensors_through_one_vkbuffer) {
  VkAppBuffer buffer;
  check_tensor_chain("affine_buffer", "chain_buffer", buffer.resource());
}

TEST_F(VulkanExternalIOTest, chains_texture_tensors_through_one_vkimage) {
  const std::unique_ptr<api::vTensor> image = make_image(kSizes);
  check_tensor_chain("affine_texture", "chain_texture", image_resource(*image));
}

#ifdef __ANDROID__
TEST_F(VulkanExternalIOTest, stages_inputs_and_output_in_ahardwarebuffers) {
  check_staging(make_ahardwarebuffer);
}

TEST_F(VulkanExternalIOTest, shares_recurrent_state_in_one_ahardwarebuffer) {
  check_shared_state(make_ahardwarebuffer);
}

TEST_F(
    VulkanExternalIOTest,
    chains_buffer_tensors_through_one_ahardwarebuffer) {
  AHardwareBufferAppBuffer buffer;
  check_tensor_chain("affine_buffer", "chain_buffer", buffer.resource());
}
#endif // __ANDROID__

TEST_F(VulkanExternalIOTest, rejects_mismatched_bindings) {
  VkAppBuffer buffer;
  VkAppBuffer small(kNumel / 2);
  const std::unique_ptr<api::vTensor> image = make_image(kSizes);
  std::vector<int64_t> narrower = kSizes;
  narrower.back() -= 4;
  const std::unique_ptr<api::vTensor> narrow_image = make_image(narrower);

  struct Case {
    const char* name;
    const char* program;
    etvk::ExternalIOBinding binding;
    Error expected;
  };
  const Case cases[] = {
      {"two_vulkan_delegates",
       "two_delegates",
       {SlotType::Input, 0, ExternalIOTarget::Staging, buffer.resource()},
       Error::InvalidState},
      {"index_out_of_range",
       "chain_texture",
       {SlotType::Input, 3, ExternalIOTarget::Staging, buffer.resource()},
       Error::InvalidArgument},
      {"staging_too_small",
       "chain_texture",
       {SlotType::Input, 0, ExternalIOTarget::Staging, small.resource()},
       Error::InvalidArgument},
      {"image_with_wrong_extents",
       "chain_texture",
       {SlotType::Input,
        0,
        ExternalIOTarget::Tensor,
        image_resource(*narrow_image)},
       Error::InvalidArgument},
      {"image_for_buffer_tensor",
       "chain_buffer",
       {SlotType::Input, 0, ExternalIOTarget::Tensor, image_resource(*image)},
       Error::InvalidArgument},
      {"buffer_for_texture_tensor",
       "chain_texture",
       {SlotType::Input, 0, ExternalIOTarget::Tensor, buffer.resource()},
       Error::InvalidArgument},
      {"buffer_tensor_too_small",
       "chain_buffer",
       {SlotType::Input, 0, ExternalIOTarget::Tensor, small.resource()},
       Error::InvalidArgument},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.name);
    const std::string set = set_name(std::string("_") + c.name);
    ASSERT_EQ(etvk::register_external_io(set.c_str(), {c.binding}), Error::Ok);
    {
      Module module(program_path(c.program));
      EXPECT_EQ(load(module, set.c_str()), c.expected);
    }
    // A failed load leaves the set unbound.
    EXPECT_EQ(etvk::unregister_external_io(set.c_str()), Error::Ok);
  }
}

TEST_F(VulkanExternalIOTest, rejects_unregistered_set) {
  Module module(program_path("chain_texture"));
  EXPECT_EQ(load(module, set_name().c_str()), Error::NotFound);
}

TEST_F(VulkanExternalIOTest, binds_set_to_one_delegate_at_a_time) {
  VkAppBuffer buffer;
  const std::string set = set_name();
  ASSERT_EQ(
      etvk::register_external_io(
          set.c_str(),
          {{SlotType::Input, 0, ExternalIOTarget::Staging, buffer.resource()}}),
      Error::Ok);
  {
    Module first(program_path("chain_texture"));
    Module second(program_path("chain_texture"));
    EXPECT_EQ(load(first, set.c_str()), Error::Ok);
    EXPECT_EQ(load(second, set.c_str()), Error::InvalidState);
    EXPECT_EQ(etvk::unregister_external_io(set.c_str()), Error::InvalidState);
  }
  {
    Module third(program_path("chain_texture"));
    EXPECT_EQ(load(third, set.c_str()), Error::Ok);
  }
  EXPECT_EQ(etvk::unregister_external_io(set.c_str()), Error::Ok);
}

} // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}
