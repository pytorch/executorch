/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/executor/tensor_parser.h>

#include <array>
#include <cstring>
#include <vector>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/core/tensor_layout.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/executor/test/managed_memory_manager.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/schema/program_generated.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::TensorLayout;
using executorch::runtime::deserialization::parseListOptionalType;
using executorch::runtime::deserialization::parseTensor;
using executorch::runtime::deserialization::parseTensorList;
using executorch::runtime::deserialization::validateTensorLayout;
using executorch::runtime::testing::ManagedMemoryManager;
using torch::executor::util::FileDataLoader;

constexpr size_t kDefaultNonConstMemBytes = 32 * 1024U;
constexpr size_t kDefaultRuntimeMemBytes = 32 * 1024U;

class TensorParserTest : public ::testing::Test {
 protected:
  void SetUp() override {
    executorch::runtime::runtime_init();

    // Load the serialized ModuleAdd data.
    const char* path = std::getenv("ET_MODULE_ADD_PATH");
    Result<FileDataLoader> float_loader = FileDataLoader::from(path);
    ASSERT_EQ(float_loader.error(), Error::Ok);
    float_loader_ =
        std::make_unique<FileDataLoader>(std::move(float_loader.get()));

    // Load the serialized ModuleAddHalf data.
    const char* half_path = std::getenv("ET_MODULE_ADD_HALF_PATH");
    Result<FileDataLoader> half_loader = FileDataLoader::from(half_path);
    ASSERT_EQ(half_loader.error(), Error::Ok);
    half_loader_ =
        std::make_unique<FileDataLoader>(std::move(half_loader.get()));
  }

  std::unique_ptr<FileDataLoader> float_loader_;
  std::unique_ptr<FileDataLoader> half_loader_;
};

namespace executorch {
namespace runtime {
namespace testing {
// Provides access to private Program methods.
class ProgramTestFriend final {
 public:
  const static executorch_flatbuffer::Program* GetInternalProgram(
      const Program* program) {
    return program->internal_program_;
  }
};
} // namespace testing
} // namespace runtime
} // namespace executorch

using executorch::runtime::testing::ProgramTestFriend;

void test_module_add(
    std::unique_ptr<FileDataLoader>& loader,
    ScalarType scalar_type,
    int type_size) {
  Result<Program> program =
      Program::load(loader.get(), Program::Verification::Minimal);
  EXPECT_EQ(program.error(), Error::Ok);

  const Program* program_ = &program.get();
  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);

  const executorch_flatbuffer::Program* internal_program =
      ProgramTestFriend::GetInternalProgram(program_);
  executorch_flatbuffer::ExecutionPlan* execution_plan =
      internal_program->execution_plan()->GetMutableObject(0);
  auto flatbuffer_values = execution_plan->values();

  int tensor_count = 0;
  int double_count = 0;
  for (size_t i = 0; i < flatbuffer_values->size(); ++i) {
    auto serialization_value = flatbuffer_values->Get(i);
    if (serialization_value->val_type() ==
        executorch_flatbuffer::KernelTypes::Tensor) {
      tensor_count++;
      Result<Tensor> tensor = parseTensor(
          program_, &mmm.get(), serialization_value->val_as_Tensor());
      Tensor t = tensor.get();
      ASSERT_EQ(scalar_type, t.scalar_type());
      ASSERT_EQ(2, t.dim()); // [2, 2]
      ASSERT_EQ(4, t.numel());
      ASSERT_EQ(type_size * t.numel(), t.nbytes());
    } else if (
        serialization_value->val_type() ==
        executorch_flatbuffer::KernelTypes::Double) {
      double_count++;
      ASSERT_EQ(1.0, serialization_value->val_as_Double()->double_val());
    }
  }
  ASSERT_EQ(3, tensor_count); // input x2, output
  ASSERT_EQ(2, double_count); // alpha x2
}

TEST_F(TensorParserTest, TestModuleAddFloat) {
  test_module_add(float_loader_, ScalarType::Float, sizeof(float));
}

TEST_F(TensorParserTest, TestModuleAddHalf) {
  test_module_add(
      half_loader_, ScalarType::Half, sizeof(executorch::aten::Half));
}

TEST_F(TensorParserTest, TestMutableState) {
  // Load the serialized ModuleSimpleTrain data.
  const char* path = std::getenv("ET_MODULE_SIMPLE_TRAIN_PATH");
  Result<FileDataLoader> train_loader = FileDataLoader::from(path);
  ASSERT_EQ(train_loader.error(), Error::Ok);

  Result<Program> program =
      Program::load(&train_loader.get(), Program::Verification::Minimal);
  EXPECT_EQ(program.error(), Error::Ok);

  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  ManagedMemoryManager mmm_copy(
      kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);

  const executorch_flatbuffer::Program* internal_program =
      ProgramTestFriend::GetInternalProgram(&program.get());
  executorch_flatbuffer::ExecutionPlan* execution_plan =
      internal_program->execution_plan()->GetMutableObject(0);
  auto flatbuffer_values = execution_plan->values();

  size_t num_mutable_tensors = 0;
  for (size_t i = 0; i < flatbuffer_values->size(); ++i) {
    auto serialization_value = flatbuffer_values->Get(i);
    if (serialization_value->val_type() ==
            executorch_flatbuffer::KernelTypes::Tensor &&
        serialization_value->val_as_Tensor()->allocation_info() != nullptr &&
        serialization_value->val_as_Tensor()->data_buffer_idx() > 0) {
      num_mutable_tensors++;
      Result<torch::executor::Tensor> tensor = parseTensor(
          &program.get(), &mmm.get(), serialization_value->val_as_Tensor());
      torch::executor::Tensor t = tensor.get();
      float loaded_value = t.const_data_ptr<float>()[0];
      ASSERT_NE(nullptr, t.const_data_ptr());
      ASSERT_NE(t.mutable_data_ptr<float>()[0], 0.5);
      t.mutable_data_ptr<float>()[0] = 0.5;
      ASSERT_EQ(
          t.mutable_data_ptr<float>()[0],
          0.5); // 0.5 can be represented perfectly by float so EQ and NE work
                // fine here. Any power of 2 rational can be perfectly
                // represented. See dyadic rationals for more info.

      // Load the same tensor using the same mem manager and show the value is
      // updated again.
      Result<torch::executor::Tensor> tensor1_alias = parseTensor(
          &program.get(), &mmm.get(), serialization_value->val_as_Tensor());
      torch::executor::Tensor t2 = tensor.get();
      ASSERT_NE(t2.mutable_data_ptr<float>()[0], 0.5);

      // Show the tensors are equivalent
      ASSERT_EQ(t.const_data_ptr(), t2.const_data_ptr());
      // Set mutable tensor value back to 0.5 since it got overwritten by second
      // parse.
      t.mutable_data_ptr<float>()[0] = 0.5;

      // Load the same tensor using a different mem manager and show the value
      // is not the same as t.
      Result<torch::executor::Tensor> tensor_new = parseTensor(
          &program.get(),
          &mmm_copy.get(),
          serialization_value->val_as_Tensor());
      torch::executor::Tensor t3 = tensor_new.get();
      ASSERT_NE(t3.mutable_data_ptr<float>()[0], 0.5);
      ASSERT_NE(t3.const_data_ptr(), t.const_data_ptr());
      ASSERT_EQ(loaded_value, t3.const_data_ptr<float>()[0]);
    }
  }
  ASSERT_EQ(num_mutable_tensors, 2);
}

// Tests that validateTensorLayout rejects tensors where dim_order is shorter
// than sizes, preventing out-of-bounds reads.
TEST(ValidateTensorLayoutTest, DimOrderSizeMismatchIsRejected) {
  flatbuffers::FlatBufferBuilder builder;

  std::vector<int32_t> sizes = {2, 3, 4};
  std::vector<uint8_t> dim_order_short = {0};

  auto tensor_offset = executorch_flatbuffer::CreateTensor(
      builder,
      executorch_flatbuffer::ScalarType::FLOAT,
      0,
      builder.CreateVector(sizes),
      builder.CreateVector(dim_order_short));
  builder.Finish(tensor_offset);

  const auto* s_tensor = flatbuffers::GetRoot<executorch_flatbuffer::Tensor>(
      builder.GetBufferPointer());

  std::vector<int32_t> expected_sizes = {2, 3, 4};
  std::vector<uint8_t> expected_dim_order = {0, 1, 2};
  auto layout = TensorLayout::create(
      Span<const int32_t>(expected_sizes.data(), expected_sizes.size()),
      Span<const uint8_t>(expected_dim_order.data(), expected_dim_order.size()),
      ScalarType::Float);
  ASSERT_TRUE(layout.ok());

  EXPECT_EQ(
      validateTensorLayout(s_tensor, layout.get()), Error::InvalidExternalData);
}

// Tests that validateTensorLayout rejects tensors with a null sizes field
// instead of dereferencing it, which would SIGSEGV under the default
// Verification::Minimal load mode.
TEST(ValidateTensorLayoutTest, NullSizesIsRejected) {
  flatbuffers::FlatBufferBuilder builder;

  std::vector<uint8_t> dim_order = {0, 1, 2};

  // Pass 0 for the sizes offset to serialize a null sizes field.
  auto tensor_offset = executorch_flatbuffer::CreateTensor(
      builder,
      executorch_flatbuffer::ScalarType::FLOAT,
      /*storage_offset=*/0,
      /*sizes=*/0,
      builder.CreateVector(dim_order));
  builder.Finish(tensor_offset);

  const auto* s_tensor = flatbuffers::GetRoot<executorch_flatbuffer::Tensor>(
      builder.GetBufferPointer());
  ASSERT_EQ(s_tensor->sizes(), nullptr);

  std::vector<int32_t> expected_sizes = {2, 3, 4};
  std::vector<uint8_t> expected_dim_order = {0, 1, 2};
  auto layout = TensorLayout::create(
      Span<const int32_t>(expected_sizes.data(), expected_sizes.size()),
      Span<const uint8_t>(expected_dim_order.data(), expected_dim_order.size()),
      ScalarType::Float);
  ASSERT_TRUE(layout.ok());

  EXPECT_EQ(
      validateTensorLayout(s_tensor, layout.get()), Error::InvalidExternalData);
}

// Tests that validateTensorLayout rejects tensors with a null dim_order field
// instead of dereferencing it, which would SIGSEGV under the default
// Verification::Minimal load mode.
TEST(ValidateTensorLayoutTest, NullDimOrderIsRejected) {
  flatbuffers::FlatBufferBuilder builder;

  std::vector<int32_t> sizes = {2, 3, 4};

  // Pass 0 for the dim_order offset to serialize a null dim_order field.
  auto tensor_offset = executorch_flatbuffer::CreateTensor(
      builder,
      executorch_flatbuffer::ScalarType::FLOAT,
      /*storage_offset=*/0,
      builder.CreateVector(sizes),
      /*dim_order=*/0);
  builder.Finish(tensor_offset);

  const auto* s_tensor = flatbuffers::GetRoot<executorch_flatbuffer::Tensor>(
      builder.GetBufferPointer());
  ASSERT_EQ(s_tensor->dim_order(), nullptr);

  std::vector<int32_t> expected_sizes = {2, 3, 4};
  std::vector<uint8_t> expected_dim_order = {0, 1, 2};
  auto layout = TensorLayout::create(
      Span<const int32_t>(expected_sizes.data(), expected_sizes.size()),
      Span<const uint8_t>(expected_dim_order.data(), expected_dim_order.size()),
      ScalarType::Float);
  ASSERT_TRUE(layout.ok());

  EXPECT_EQ(
      validateTensorLayout(s_tensor, layout.get()), Error::InvalidExternalData);
}

// Helper to construct a flatbuffers::Vector<int32_t> from raw data.
// FlatBuffer vectors are stored as [uint32_t length][T elements...].
namespace {
struct FlatVectorInt32 {
  static const flatbuffers::Vector<int32_t>* create(
      std::vector<uint8_t>& buf,
      const std::vector<int32_t>& elements) {
    buf.resize(sizeof(uint32_t) + elements.size() * sizeof(int32_t));
    uint32_t len = static_cast<uint32_t>(elements.size());
    memcpy(buf.data(), &len, sizeof(len));
    if (!elements.empty()) {
      memcpy(
          buf.data() + sizeof(uint32_t),
          elements.data(),
          elements.size() * sizeof(int32_t));
    }
    return reinterpret_cast<const flatbuffers::Vector<int32_t>*>(buf.data());
  }
};
} // namespace

// parseTensorList should return an error when the EValue at the given index
// is not a Tensor, instead of aborting.
TEST_F(TensorParserTest, ParseTensorListRejectsNonTensorEValue) {
  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);

  // Create an EValue array with a non-Tensor value at index 0.
  EValue values[2];
  values[0] = EValue(static_cast<int64_t>(42)); // Int, not Tensor
  values[1] = EValue(static_cast<int64_t>(7));

  // Create a vector with index 0 (pointing to the Int EValue).
  std::vector<uint8_t> vec_buf;
  auto* indices = FlatVectorInt32::create(vec_buf, {0});

  auto result = parseTensorList(indices, values, 2, &mmm.get());
  EXPECT_EQ(result.error(), Error::InvalidType);
}

// parseListOptionalType should return an error when the EValue at the given
// index is neither None nor the expected type.
TEST_F(TensorParserTest, ParseListOptionalTypeRejectsWrongType) {
  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);

  // Create an EValue array with a non-Tensor, non-None value at index 0.
  EValue values[2];
  values[0] = EValue(static_cast<int64_t>(42)); // Int, not Tensor or None
  values[1] = EValue(static_cast<int64_t>(7));

  // Create a vector with index 0 (pointing to the Int EValue).
  std::vector<uint8_t> vec_buf;
  auto* indices = FlatVectorInt32::create(vec_buf, {0});

  auto result = parseListOptionalType<Tensor>(indices, values, 2, &mmm.get());
  EXPECT_EQ(result.error(), Error::InvalidType);
}

namespace {

class SpanDataLoader final : public executorch::runtime::DataLoader {
 public:
  SpanDataLoader(const void* data, size_t size)
      : data_(static_cast<const uint8_t*>(data)), size_(size) {}

  ET_NODISCARD Result<executorch::runtime::FreeableBuffer> load(
      size_t offset,
      size_t size,
      ET_UNUSED const SegmentInfo& segment_info) const override {
    ET_CHECK_OR_RETURN_ERROR(
        offset <= size_ && size <= size_ - offset,
        InvalidArgument,
        "SpanDataLoader read out of range");
    return executorch::runtime::FreeableBuffer(
        data_ + offset, size, /*free_fn=*/nullptr);
  }

  ET_NODISCARD Result<size_t> size() const override {
    return size_;
  }

 private:
  const uint8_t* data_;
  size_t size_;
};

struct AlignedProgramBuffer {
  alignas(16) std::array<uint8_t, 4096> storage{};
  size_t len = 0;

  void assign(const flatbuffers::FlatBufferBuilder& builder) {
    ET_CHECK_MSG(
        builder.GetSize() <= storage.size(), "Serialized program too large");
    len = builder.GetSize();
    std::memcpy(storage.data(), builder.GetBufferPointer(), len);
  }

  SpanDataLoader loader() const {
    return SpanDataLoader(storage.data(), len);
  }
};

flatbuffers::Offset<executorch_flatbuffer::EValue> make_tensor_evalue(
    flatbuffers::FlatBufferBuilder& builder,
    executorch_flatbuffer::ScalarType dtype,
    const std::vector<int32_t>& sizes,
    const std::vector<uint8_t>& dim_order,
    bool memory_planned,
    uint32_t memory_offset = 0) {
  auto sizes_vec = builder.CreateVector(sizes);
  auto dim_order_vec = builder.CreateVector(dim_order);
  flatbuffers::Offset<executorch_flatbuffer::AllocationDetails> alloc = 0;
  if (memory_planned) {
    alloc = executorch_flatbuffer::CreateAllocationDetails(
        builder,
        /*memory_id=*/1,
        /*memory_offset_low=*/memory_offset,
        /*memory_offset_high=*/0);
  }
  auto tensor = executorch_flatbuffer::CreateTensor(
      builder,
      dtype,
      /*storage_offset=*/0,
      sizes_vec,
      dim_order_vec,
      /*requires_grad=*/false,
      /*data_buffer_idx=*/0,
      alloc);
  return executorch_flatbuffer::CreateEValue(
      builder, executorch_flatbuffer::KernelTypes::Tensor, tensor.Union());
}

void finish_single_plan_program(
    flatbuffers::FlatBufferBuilder& builder,
    const std::vector<flatbuffers::Offset<executorch_flatbuffer::EValue>>&
        values,
    const std::vector<int32_t>& inputs,
    const std::vector<int32_t>& outputs,
    const std::vector<flatbuffers::Offset<executorch_flatbuffer::Instruction>>&
        instructions,
    const std::vector<flatbuffers::Offset<executorch_flatbuffer::Operator>>&
        operators,
    AlignedProgramBuffer& out) {
  auto chain_inputs = builder.CreateVector(std::vector<int32_t>{});
  auto chain_outputs = builder.CreateVector(std::vector<int32_t>{});
  auto chain_instrs = builder.CreateVector(instructions);
  auto chain = executorch_flatbuffer::CreateChain(
      builder, chain_inputs, chain_outputs, chain_instrs);

  auto plan_name = builder.CreateString("forward");
  auto values_vec = builder.CreateVector(values);
  auto inputs_vec = builder.CreateVector(inputs);
  auto outputs_vec = builder.CreateVector(outputs);
  auto chains_vec = builder.CreateVector(
      std::vector<flatbuffers::Offset<executorch_flatbuffer::Chain>>{chain});
  auto ops_vec = builder.CreateVector(operators);
  auto delegates_vec = builder.CreateVector(
      std::vector<
          flatbuffers::Offset<executorch_flatbuffer::BackendDelegate>>{});
  auto non_const_buffers = builder.CreateVector(std::vector<int64_t>{0, 1024});

  auto plan = executorch_flatbuffer::CreateExecutionPlan(
      builder,
      plan_name,
      /*container_meta_type=*/0,
      values_vec,
      inputs_vec,
      outputs_vec,
      chains_vec,
      ops_vec,
      delegates_vec,
      non_const_buffers);

  auto plans_vec = builder.CreateVector(
      std::vector<flatbuffers::Offset<executorch_flatbuffer::ExecutionPlan>>{
          plan});
  auto const_buf_vec = builder.CreateVector(
      std::vector<flatbuffers::Offset<executorch_flatbuffer::Buffer>>{});
  auto backend_data_vec = builder.CreateVector(
      std::vector<flatbuffers::Offset<
          executorch_flatbuffer::BackendDelegateInlineData>>{});
  auto segments_vec = builder.CreateVector(
      std::vector<flatbuffers::Offset<executorch_flatbuffer::DataSegment>>{});
  auto subsegment_offsets = builder.CreateVector(std::vector<uint64_t>{0});
  auto constant_segment = executorch_flatbuffer::CreateSubsegmentOffsets(
      builder, /*segment_index=*/0, subsegment_offsets);

  auto program = executorch_flatbuffer::CreateProgram(
      builder,
      /*version=*/0,
      plans_vec,
      const_buf_vec,
      backend_data_vec,
      segments_vec,
      constant_segment);
  builder.Finish(program, executorch_flatbuffer::ProgramIdentifier());
  out.assign(builder);
}

void fallback_et_view(
    executorch::runtime::KernelRuntimeContext& context,
    Span<EValue*> stack) {
  if (stack.size() != 3) {
    context.fail(Error::InvalidProgram);
    return;
  }
  auto self = (*stack[0]).toTensor();
  auto out = (*stack[2]).toTensor();
  if (executorch::runtime::internal::set_tensor_data(
          out, self.mutable_data_ptr(), out.nbytes()) != Error::Ok) {
    context.fail(Error::Internal);
  }
}

bool g_custom_et_view_invoked = false;

void custom_et_view_noop(
    ET_UNUSED executorch::runtime::KernelRuntimeContext& context,
    ET_UNUSED Span<EValue*> stack) {
  g_custom_et_view_invoked = true;
}

} // namespace

TEST_F(TensorParserTest, ExecuteValidZeroSizedNullDataTensorSucceeds) {
  flatbuffers::FlatBufferBuilder builder(2048);
  auto v0 = make_tensor_evalue(
      builder,
      executorch_flatbuffer::ScalarType::FLOAT,
      {0, 2},
      {0, 1},
      /*memory_planned=*/false);
  auto v1 = make_tensor_evalue(
      builder,
      executorch_flatbuffer::ScalarType::FLOAT,
      {0, 2},
      {0, 1},
      /*memory_planned=*/false);
  auto alpha = executorch_flatbuffer::CreateDouble(builder, 1.0);
  auto v2 = executorch_flatbuffer::CreateEValue(
      builder, executorch_flatbuffer::KernelTypes::Double, alpha.Union());
  auto v3 = make_tensor_evalue(
      builder,
      executorch_flatbuffer::ScalarType::FLOAT,
      {0, 2},
      {0, 1},
      /*memory_planned=*/false);

  auto op_name = builder.CreateString("aten::add");
  auto op_overload = builder.CreateString("out");
  auto op =
      executorch_flatbuffer::CreateOperator(builder, op_name, op_overload);

  auto call_args = builder.CreateVector(std::vector<int32_t>{0, 1, 2, 3, 3});
  auto call = executorch_flatbuffer::CreateKernelCall(builder, 0, call_args);
  auto instr = executorch_flatbuffer::CreateInstruction(
      builder,
      executorch_flatbuffer::InstructionArguments::KernelCall,
      call.Union());

  AlignedProgramBuffer prog_buf;
  finish_single_plan_program(
      builder, {v0, v1, v2, v3}, {0}, {3}, {instr}, {op}, prog_buf);

  auto loader = prog_buf.loader();
  Result<Program> program =
      Program::load(&loader, Program::Verification::InternalConsistency);
  ASSERT_EQ(program.error(), Error::Ok);

  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  Result<executorch::runtime::Method> method =
      program->load_method("forward", &mmm.get());
  ASSERT_EQ(method.error(), Error::Ok);

  int32_t sizes[2] = {0, 2};
  executorch::aten::TensorImpl impl(
      ScalarType::Float, 2, sizes, /*data=*/nullptr);
  ASSERT_EQ(method->set_input(EValue(Tensor(&impl)), 0), Error::Ok);
  EXPECT_EQ(method->execute(), Error::Ok);
  EXPECT_EQ(method->get_output(0).toTensor().nbytes(), 0U);
}

TEST_F(TensorParserTest, ExecuteEtViewPopulatesNullOutputTensorSucceeds) {
  executorch::runtime::Kernel view_kernel(
      "executorch_prim::et_view.default", fallback_et_view);
  Span<const executorch::runtime::Kernel> extra_kernels;
  if (!executorch::runtime::registry_has_op_function(
          "executorch_prim::et_view.default")) {
    extra_kernels = Span<const executorch::runtime::Kernel>(&view_kernel, 1);
  }

  {
    flatbuffers::FlatBufferBuilder builder(2048);
    auto v0 = make_tensor_evalue(
        builder,
        executorch_flatbuffer::ScalarType::FLOAT,
        {2, 2},
        {0, 1},
        /*memory_planned=*/true,
        /*memory_offset=*/0);
    auto dim0 = executorch_flatbuffer::CreateInt(builder, 4);
    auto v1 = executorch_flatbuffer::CreateEValue(
        builder, executorch_flatbuffer::KernelTypes::Int, dim0.Union());
    auto shape_items = builder.CreateVector(std::vector<int64_t>{1});
    auto shape_list =
        executorch_flatbuffer::CreateIntList(builder, shape_items);
    auto v2 = executorch_flatbuffer::CreateEValue(
        builder,
        executorch_flatbuffer::KernelTypes::IntList,
        shape_list.Union());
    auto v3 = make_tensor_evalue(
        builder,
        executorch_flatbuffer::ScalarType::FLOAT,
        {4},
        {0},
        /*memory_planned=*/false);

    auto op_name = builder.CreateString("executorch_prim::et_view");
    auto op_overload = builder.CreateString("default");
    auto op =
        executorch_flatbuffer::CreateOperator(builder, op_name, op_overload);

    auto call_args = builder.CreateVector(std::vector<int32_t>{0, 2, 3});
    auto call = executorch_flatbuffer::CreateKernelCall(builder, 0, call_args);
    auto instr = executorch_flatbuffer::CreateInstruction(
        builder,
        executorch_flatbuffer::InstructionArguments::KernelCall,
        call.Union());

    AlignedProgramBuffer prog_buf;
    finish_single_plan_program(
        builder, {v0, v1, v2, v3}, {0}, {3}, {instr}, {op}, prog_buf);

    auto loader = prog_buf.loader();
    Result<Program> program =
        Program::load(&loader, Program::Verification::InternalConsistency);
    ASSERT_EQ(program.error(), Error::Ok);

    ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
    Result<executorch::runtime::Method> method = program->load_method(
        "forward", &mmm.get(), nullptr, nullptr, nullptr, extra_kernels);
    ASSERT_EQ(method.error(), Error::Ok);

    float in_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    int32_t sizes[2] = {2, 2};
    executorch::aten::TensorImpl impl(ScalarType::Float, 2, sizes, in_data);
    ASSERT_EQ(method->set_input(EValue(Tensor(&impl)), 0), Error::Ok);
    ASSERT_EQ(method->execute(), Error::Ok);

    const Tensor& out = method->get_output(0).toTensor();
    ASSERT_NE(out.const_data_ptr(), nullptr);
    for (int i = 0; i < 4; ++i) {
      EXPECT_FLOAT_EQ(out.const_data_ptr<float>()[i], in_data[i]);
    }
  }
}

TEST_F(TensorParserTest, ExecuteRejectsMalformedInternalNullDataTensor) {
  flatbuffers::FlatBufferBuilder builder(2048);
  auto v0 = make_tensor_evalue(
      builder,
      executorch_flatbuffer::ScalarType::FLOAT,
      {2, 2},
      {0, 1},
      /*memory_planned=*/true,
      /*memory_offset=*/0);
  // Internal non-I/O tensor with data_buffer_idx == 0 and allocation_info ==
  // nullptr: parseTensor initializes its data pointer to nullptr.
  auto v1 = make_tensor_evalue(
      builder,
      executorch_flatbuffer::ScalarType::FLOAT,
      {2, 2},
      {0, 1},
      /*memory_planned=*/false);
  auto alpha = executorch_flatbuffer::CreateDouble(builder, 1.0);
  auto v2 = executorch_flatbuffer::CreateEValue(
      builder, executorch_flatbuffer::KernelTypes::Double, alpha.Union());
  auto v3 = make_tensor_evalue(
      builder,
      executorch_flatbuffer::ScalarType::FLOAT,
      {2, 2},
      {0, 1},
      /*memory_planned=*/true,
      /*memory_offset=*/64);

  auto op_name = builder.CreateString("aten::add");
  auto op_overload = builder.CreateString("out");
  auto op =
      executorch_flatbuffer::CreateOperator(builder, op_name, op_overload);

  auto call_args = builder.CreateVector(std::vector<int32_t>{0, 1, 2, 3, 3});
  auto call = executorch_flatbuffer::CreateKernelCall(builder, 0, call_args);
  auto instr = executorch_flatbuffer::CreateInstruction(
      builder,
      executorch_flatbuffer::InstructionArguments::KernelCall,
      call.Union());

  AlignedProgramBuffer prog_buf;
  finish_single_plan_program(
      builder, {v0, v1, v2, v3}, {0}, {3}, {instr}, {op}, prog_buf);

  auto loader = prog_buf.loader();
  Result<Program> program =
      Program::load(&loader, Program::Verification::InternalConsistency);
  ASSERT_EQ(program.error(), Error::Ok);

  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  Result<executorch::runtime::Method> method =
      program->load_method("forward", &mmm.get());
  ASSERT_EQ(method.error(), Error::Ok);

  float in_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  int32_t sizes[2] = {2, 2};
  executorch::aten::TensorImpl impl(ScalarType::Float, 2, sizes, in_data);
  ASSERT_EQ(method->set_input(EValue(Tensor(&impl)), 0), Error::Ok);

  EXPECT_EQ(method->execute(), Error::InvalidState);
  EXPECT_FALSE(method->in_progress());
}

TEST_F(TensorParserTest, ExecuteEtViewRejectsNullInputSelfTensor) {
  executorch::runtime::Kernel view_kernel(
      "executorch_prim::et_view.default", fallback_et_view);
  Span<const executorch::runtime::Kernel> extra_kernels;
  if (!executorch::runtime::registry_has_op_function(
          "executorch_prim::et_view.default")) {
    extra_kernels = Span<const executorch::runtime::Kernel>(&view_kernel, 1);
  }

  flatbuffers::FlatBufferBuilder builder(2048);
  auto v0 = make_tensor_evalue(
      builder,
      executorch_flatbuffer::ScalarType::FLOAT,
      {2, 2},
      {0, 1},
      /*memory_planned=*/false);
  auto dim0 = executorch_flatbuffer::CreateInt(builder, 4);
  auto v1 = executorch_flatbuffer::CreateEValue(
      builder, executorch_flatbuffer::KernelTypes::Int, dim0.Union());
  auto shape_items = builder.CreateVector(std::vector<int64_t>{1});
  auto shape_list = executorch_flatbuffer::CreateIntList(builder, shape_items);
  auto v2 = executorch_flatbuffer::CreateEValue(
      builder, executorch_flatbuffer::KernelTypes::IntList, shape_list.Union());
  auto v3 = make_tensor_evalue(
      builder,
      executorch_flatbuffer::ScalarType::FLOAT,
      {4},
      {0},
      /*memory_planned=*/false);

  auto op_name = builder.CreateString("executorch_prim::et_view");
  auto op_overload = builder.CreateString("default");
  auto op =
      executorch_flatbuffer::CreateOperator(builder, op_name, op_overload);

  auto call_args = builder.CreateVector(std::vector<int32_t>{0, 2, 3});
  auto call = executorch_flatbuffer::CreateKernelCall(builder, 0, call_args);
  auto instr = executorch_flatbuffer::CreateInstruction(
      builder,
      executorch_flatbuffer::InstructionArguments::KernelCall,
      call.Union());

  AlignedProgramBuffer prog_buf;
  finish_single_plan_program(
      builder, {v0, v1, v2, v3}, {}, {3}, {instr}, {op}, prog_buf);

  auto loader = prog_buf.loader();
  Result<Program> program =
      Program::load(&loader, Program::Verification::InternalConsistency);
  ASSERT_EQ(program.error(), Error::Ok);

  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  Result<executorch::runtime::Method> method = program->load_method(
      "forward", &mmm.get(), nullptr, nullptr, nullptr, extra_kernels);
  ASSERT_EQ(method.error(), Error::Ok);
  EXPECT_EQ(method->execute(), Error::InvalidState);
}

TEST_F(
    TensorParserTest,
    ExecuteEtViewNonDefaultOverloadRejectsNullOutputTensor) {
  ASSERT_FALSE(
      executorch::runtime::registry_has_op_function(
          "executorch_prim::et_view.custom"));
  executorch::runtime::Kernel custom_view_kernel(
      "executorch_prim::et_view.custom", custom_et_view_noop);
  Span<const executorch::runtime::Kernel> extra_kernels(&custom_view_kernel, 1);

  flatbuffers::FlatBufferBuilder builder(2048);
  auto v0 = make_tensor_evalue(
      builder,
      executorch_flatbuffer::ScalarType::FLOAT,
      {2, 2},
      {0, 1},
      /*memory_planned=*/true,
      /*memory_offset=*/0);
  auto dim0 = executorch_flatbuffer::CreateInt(builder, 4);
  auto v1 = executorch_flatbuffer::CreateEValue(
      builder, executorch_flatbuffer::KernelTypes::Int, dim0.Union());
  auto shape_items = builder.CreateVector(std::vector<int64_t>{1});
  auto shape_list = executorch_flatbuffer::CreateIntList(builder, shape_items);
  auto v2 = executorch_flatbuffer::CreateEValue(
      builder, executorch_flatbuffer::KernelTypes::IntList, shape_list.Union());
  auto v3 = make_tensor_evalue(
      builder,
      executorch_flatbuffer::ScalarType::FLOAT,
      {4},
      {0},
      /*memory_planned=*/false);

  auto op_name = builder.CreateString("executorch_prim::et_view");
  auto op_overload = builder.CreateString("custom");
  auto op =
      executorch_flatbuffer::CreateOperator(builder, op_name, op_overload);

  auto call_args = builder.CreateVector(std::vector<int32_t>{0, 2, 3});
  auto call = executorch_flatbuffer::CreateKernelCall(builder, 0, call_args);
  auto instr = executorch_flatbuffer::CreateInstruction(
      builder,
      executorch_flatbuffer::InstructionArguments::KernelCall,
      call.Union());

  AlignedProgramBuffer prog_buf;
  finish_single_plan_program(
      builder, {v0, v1, v2, v3}, {0}, {0}, {instr}, {op}, prog_buf);

  auto loader = prog_buf.loader();
  Result<Program> program =
      Program::load(&loader, Program::Verification::InternalConsistency);
  ASSERT_EQ(program.error(), Error::Ok);

  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  Result<executorch::runtime::Method> method = program->load_method(
      "forward", &mmm.get(), nullptr, nullptr, nullptr, extra_kernels);
  ASSERT_EQ(method.error(), Error::Ok);

  float in_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  int32_t sizes[2] = {2, 2};
  executorch::aten::TensorImpl impl(ScalarType::Float, 2, sizes, in_data);
  ASSERT_EQ(method->set_input(EValue(Tensor(&impl)), 0), Error::Ok);

  g_custom_et_view_invoked = false;
  EXPECT_EQ(method->execute(), Error::InvalidState);
  EXPECT_FALSE(g_custom_et_view_invoked);
  EXPECT_FALSE(method->in_progress());
}

TEST_F(
    TensorParserTest,
    ExecuteRejectsNullDataElementInTensorListAndOptionalTensorList) {
  // TensorList argument containing a null-data tensor on aten::cat.out.
  {
    flatbuffers::FlatBufferBuilder builder(2048);
    auto v0 = make_tensor_evalue(
        builder,
        executorch_flatbuffer::ScalarType::FLOAT,
        {2, 2},
        {0, 1},
        /*memory_planned=*/true,
        /*memory_offset=*/0);
    auto v1 = make_tensor_evalue(
        builder,
        executorch_flatbuffer::ScalarType::FLOAT,
        {2, 2},
        {0, 1},
        /*memory_planned=*/false);
    auto list_items = builder.CreateVector(std::vector<int32_t>{0, 1});
    auto tensor_list =
        executorch_flatbuffer::CreateTensorList(builder, list_items);
    auto v2 = executorch_flatbuffer::CreateEValue(
        builder,
        executorch_flatbuffer::KernelTypes::TensorList,
        tensor_list.Union());
    auto dim_val = executorch_flatbuffer::CreateInt(builder, 0);
    auto v3 = executorch_flatbuffer::CreateEValue(
        builder, executorch_flatbuffer::KernelTypes::Int, dim_val.Union());
    auto v4 = make_tensor_evalue(
        builder,
        executorch_flatbuffer::ScalarType::FLOAT,
        {4, 2},
        {0, 1},
        /*memory_planned=*/true,
        /*memory_offset=*/64);

    auto op_name = builder.CreateString("aten::cat");
    auto op_overload = builder.CreateString("out");
    auto op =
        executorch_flatbuffer::CreateOperator(builder, op_name, op_overload);

    auto call_args = builder.CreateVector(std::vector<int32_t>{2, 3, 4, 4});
    auto call = executorch_flatbuffer::CreateKernelCall(builder, 0, call_args);
    auto instr = executorch_flatbuffer::CreateInstruction(
        builder,
        executorch_flatbuffer::InstructionArguments::KernelCall,
        call.Union());

    AlignedProgramBuffer prog_buf;
    finish_single_plan_program(
        builder, {v0, v1, v2, v3, v4}, {0}, {4}, {instr}, {op}, prog_buf);

    auto loader = prog_buf.loader();
    Result<Program> program =
        Program::load(&loader, Program::Verification::InternalConsistency);
    ASSERT_EQ(program.error(), Error::Ok);

    ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
    Result<executorch::runtime::Method> method =
        program->load_method("forward", &mmm.get());
    ASSERT_EQ(method.error(), Error::Ok);

    float in_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    int32_t sizes[2] = {2, 2};
    executorch::aten::TensorImpl impl(ScalarType::Float, 2, sizes, in_data);
    ASSERT_EQ(method->set_input(EValue(Tensor(&impl)), 0), Error::Ok);
    EXPECT_EQ(method->execute(), Error::InvalidState);
  }

  // OptionalTensorList argument containing a null-data tensor on
  // aten::index.Tensor_out.
  {
    flatbuffers::FlatBufferBuilder builder(2048);
    auto v0 = make_tensor_evalue(
        builder,
        executorch_flatbuffer::ScalarType::FLOAT,
        {4, 2},
        {0, 1},
        /*memory_planned=*/true,
        /*memory_offset=*/0);
    auto v1 = make_tensor_evalue(
        builder,
        executorch_flatbuffer::ScalarType::LONG,
        {2},
        {0},
        /*memory_planned=*/false);
    auto opt_items = builder.CreateVector(std::vector<int32_t>{1});
    auto opt_list =
        executorch_flatbuffer::CreateOptionalTensorList(builder, opt_items);
    auto v2 = executorch_flatbuffer::CreateEValue(
        builder,
        executorch_flatbuffer::KernelTypes::OptionalTensorList,
        opt_list.Union());
    auto v3 = make_tensor_evalue(
        builder,
        executorch_flatbuffer::ScalarType::FLOAT,
        {2, 2},
        {0, 1},
        /*memory_planned=*/true,
        /*memory_offset=*/64);

    auto op_name = builder.CreateString("aten::index");
    auto op_overload = builder.CreateString("Tensor_out");
    auto op =
        executorch_flatbuffer::CreateOperator(builder, op_name, op_overload);

    auto call_args = builder.CreateVector(std::vector<int32_t>{0, 2, 3, 3});
    auto call = executorch_flatbuffer::CreateKernelCall(builder, 0, call_args);
    auto instr = executorch_flatbuffer::CreateInstruction(
        builder,
        executorch_flatbuffer::InstructionArguments::KernelCall,
        call.Union());

    AlignedProgramBuffer prog_buf;
    finish_single_plan_program(
        builder, {v0, v1, v2, v3}, {0}, {3}, {instr}, {op}, prog_buf);

    auto loader = prog_buf.loader();
    Result<Program> program =
        Program::load(&loader, Program::Verification::InternalConsistency);
    ASSERT_EQ(program.error(), Error::Ok);

    ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
    Result<executorch::runtime::Method> method =
        program->load_method("forward", &mmm.get());
    ASSERT_EQ(method.error(), Error::Ok);

    float in_data[8] = {};
    int32_t sizes[2] = {4, 2};
    executorch::aten::TensorImpl impl(ScalarType::Float, 2, sizes, in_data);
    ASSERT_EQ(method->set_input(EValue(Tensor(&impl)), 0), Error::Ok);
    EXPECT_EQ(method->execute(), Error::InvalidState);
  }
}

TEST_F(
    TensorParserTest,
    ExecuteRejectsUseAfterFreeCallAndEmptyChainNullOutput) {
  // FreeCall clears a tensor's data pointer; a subsequent KernelCall reading it
  // must fail with InvalidState instead of dereferencing nullptr.
  {
    flatbuffers::FlatBufferBuilder builder(2048);
    auto v0 = make_tensor_evalue(
        builder,
        executorch_flatbuffer::ScalarType::FLOAT,
        {2, 2},
        {0, 1},
        /*memory_planned=*/true,
        /*memory_offset=*/0);
    auto v1 = make_tensor_evalue(
        builder,
        executorch_flatbuffer::ScalarType::FLOAT,
        {2, 2},
        {0, 1},
        /*memory_planned=*/true,
        /*memory_offset=*/64);
    auto alpha = executorch_flatbuffer::CreateDouble(builder, 1.0);
    auto v2 = executorch_flatbuffer::CreateEValue(
        builder, executorch_flatbuffer::KernelTypes::Double, alpha.Union());
    auto v3 = make_tensor_evalue(
        builder,
        executorch_flatbuffer::ScalarType::FLOAT,
        {2, 2},
        {0, 1},
        /*memory_planned=*/true,
        /*memory_offset=*/128);

    auto op_name = builder.CreateString("aten::add");
    auto op_overload = builder.CreateString("out");
    auto op =
        executorch_flatbuffer::CreateOperator(builder, op_name, op_overload);

    auto free_call = executorch_flatbuffer::CreateFreeCall(builder, 0);
    auto free_instr = executorch_flatbuffer::CreateInstruction(
        builder,
        executorch_flatbuffer::InstructionArguments::FreeCall,
        free_call.Union());

    auto call_args = builder.CreateVector(std::vector<int32_t>{0, 1, 2, 3, 3});
    auto call = executorch_flatbuffer::CreateKernelCall(builder, 0, call_args);
    auto add_instr = executorch_flatbuffer::CreateInstruction(
        builder,
        executorch_flatbuffer::InstructionArguments::KernelCall,
        call.Union());

    AlignedProgramBuffer prog_buf;
    finish_single_plan_program(
        builder,
        {v0, v1, v2, v3},
        {},
        {3},
        {free_instr, add_instr},
        {op},
        prog_buf);

    auto loader = prog_buf.loader();
    Result<Program> program =
        Program::load(&loader, Program::Verification::InternalConsistency);
    ASSERT_EQ(program.error(), Error::Ok);

    ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
    Result<executorch::runtime::Method> method =
        program->load_method("forward", &mmm.get());
    ASSERT_EQ(method.error(), Error::Ok);
    EXPECT_EQ(method->execute(), Error::InvalidState);
  }

  // Empty chain returning an unbound non-empty output tensor must fail in both
  // execute() and step(), then succeed once output storage is bound.
  {
    flatbuffers::FlatBufferBuilder builder(2048);
    auto v0 = make_tensor_evalue(
        builder,
        executorch_flatbuffer::ScalarType::FLOAT,
        {2, 2},
        {0, 1},
        /*memory_planned=*/false);

    AlignedProgramBuffer prog_buf;
    finish_single_plan_program(builder, {v0}, {}, {0}, {}, {}, prog_buf);

    auto loader = prog_buf.loader();
    Result<Program> program =
        Program::load(&loader, Program::Verification::InternalConsistency);
    ASSERT_EQ(program.error(), Error::Ok);

    ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
    Result<executorch::runtime::Method> method =
        program->load_method("forward", &mmm.get());
    ASSERT_EQ(method.error(), Error::Ok);

    EXPECT_EQ(method->execute(), Error::InvalidState);
    EXPECT_FALSE(method->in_progress());

    EXPECT_EQ(method->step(), Error::InvalidState);
    EXPECT_FALSE(method->in_progress());
    ASSERT_EQ(method->reset_execution(), Error::Ok);

    float out_buf[4] = {};
    ASSERT_EQ(
        method->set_output_data_ptr(out_buf, sizeof(out_buf), 0), Error::Ok);
    EXPECT_EQ(method->execute(), Error::Ok);
    EXPECT_EQ(method->step(), Error::Ok);
    EXPECT_EQ(method->step(), Error::EndOfMethod);
  }
}
