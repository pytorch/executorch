/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/backend_data/initialize_and_save.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>

#include <executorch/extension/backend_data/buffer_data_writer.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/schema/extended_header.h>
#include <executorch/schema/program_generated.h>

namespace {

using executorch::extension::BufferDataWriter;
using executorch::extension::initialize_and_save_backend_data;
using executorch::runtime::ArrayRef;
using executorch::runtime::BackendData;
using executorch::runtime::BackendDataInitContext;
using executorch::runtime::BackendDataInput;
using executorch::runtime::BackendDataWriter;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::BackendInterface;
using executorch::runtime::CompileSpec;
using executorch::runtime::DataLoader;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::NamedBackendData;
using executorch::runtime::Result;
using executorch::runtime::Span;

constexpr char kBackendName[] = "InitializeAndSaveTestBackend";
constexpr size_t kPteHeaderSize = 32;

enum class DelegateConflict { None, Backend, CompileSpec };

uint32_t read_u32(const uint8_t* data) {
  uint32_t value = 0;
  for (size_t i = 0; i < sizeof(value); ++i) {
    value |= static_cast<uint32_t>(data[i]) << (8 * i);
  }
  return value;
}

uint64_t read_u64(const uint8_t* data) {
  uint64_t value = 0;
  for (size_t i = 0; i < sizeof(value); ++i) {
    value |= static_cast<uint64_t>(data[i]) << (8 * i);
  }
  return value;
}

void write_u32(std::vector<uint8_t>& data, size_t offset, uint32_t value) {
  for (size_t i = 0; i < sizeof(value); ++i) {
    data[offset + i] = static_cast<uint8_t>(value >> (8 * i));
  }
}

void write_u64(std::vector<uint8_t>& data, size_t offset, uint64_t value) {
  for (size_t i = 0; i < sizeof(value); ++i) {
    data[offset + i] = static_cast<uint8_t>(value >> (8 * i));
  }
}

size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

class VectorDataLoader final : public DataLoader {
 public:
  explicit VectorDataLoader(std::shared_ptr<std::vector<uint8_t>> data)
      : data_(std::move(data)) {}

  Result<FreeableBuffer> load(size_t offset, size_t size, const SegmentInfo&)
      const override {
    if (offset > data_->size() || size > data_->size() - offset) {
      return Error::InvalidArgument;
    }
    return FreeableBuffer(data_->data() + offset, size, nullptr);
  }

  Error load_into(size_t offset, size_t size, const SegmentInfo&, void* buffer)
      const override {
    if (offset > data_->size() || size > data_->size() - offset) {
      return Error::InvalidArgument;
    }
    std::memcpy(buffer, data_->data() + offset, size);
    return Error::Ok;
  }

  Result<size_t> size() const override {
    return data_->size();
  }

 private:
  std::shared_ptr<std::vector<uint8_t>> data_;
};

class TestPreparingBackend final : public BackendInterface {
 public:
  bool is_available() const override {
    return true;
  }

  Result<DelegateHandle*> init(
      BackendInitContext&,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec>) const override {
    if (processed == nullptr || processed->size() != 1 ||
        static_cast<const uint8_t*>(processed->data())[0] != 1) {
      return Error::DelegateInvalidCompatibility;
    }
    ++init_calls;
    return nullptr;
  }

  Error initialize_backend_data(
      BackendDataInitContext& context,
      BackendDataWriter& output) const override {
    if (!supported) {
      return Error::NotSupported;
    }
    while (true) {
      auto next = context.next_backend_data();
      if (!next.ok()) {
        return next.error();
      }
      std::optional<BackendDataInput> input = std::move(*next);
      if (!input.has_value()) {
        break;
      }
      methods.emplace_back(input->method_name);
    }

    return write_named_data(context, output);
  }

  Error execute(BackendExecutionContext&, DelegateHandle*, Span<EValue*>)
      const override {
    return Error::Ok;
  }

  mutable std::vector<std::string> methods;
  mutable bool supported{true};
  mutable size_t init_calls{0};
  mutable bool use_external_output{false};
  mutable std::optional<size_t> requested_alignment;

  void reset() const {
    methods.clear();
    supported = true;
    init_calls = 0;
    use_external_output = false;
    requested_alignment.reset();
  }

 private:
  Error write_named_data(
      BackendDataInitContext& context,
      BackendDataWriter& output) const {
    const auto* named_data = context.get_named_data_map();
    if (named_data == nullptr) {
      return Error::Ok;
    }
    auto weight = named_data->get_data("weight");
    auto bias = named_data->get_data("bias");
    if (!weight.ok() || !bias.ok()) {
      return Error::Ok;
    }
    auto* packed = static_cast<uint8_t*>(context.allocate(4));
    auto* values = static_cast<NamedBackendData*>(context.allocate(
        2 * sizeof(NamedBackendData), alignof(NamedBackendData)));
    auto* weight_key = static_cast<char*>(context.allocate(7));
    auto* bias_key = static_cast<char*>(context.allocate(5));
    if (packed == nullptr || values == nullptr || weight_key == nullptr ||
        bias_key == nullptr) {
      return Error::MemoryAllocationFailed;
    }
    packed[0] =
        weight->size() > 0 ? static_cast<const uint8_t*>(weight->data())[0] : 3;
    packed[1] =
        weight->size() > 1 ? static_cast<const uint8_t*>(weight->data())[1] : 4;
    packed[2] = static_cast<const uint8_t*>(bias->data())[0];
    packed[3] = static_cast<const uint8_t*>(bias->data())[0];
    std::memcpy(weight_key, "weight", 7);
    std::memcpy(bias_key, "bias", 5);
    new (&values[0]) NamedBackendData{
        bias_key,
        BackendData{Span<const uint8_t>(packed + 3, 1), std::nullopt}};
    static const std::array<uint8_t, 3> external_bytes{{3, 4, 5}};
    const Span<const uint8_t> weight_bytes = use_external_output
        ? Span<const uint8_t>(external_bytes.data(), external_bytes.size())
        : Span<const uint8_t>(packed, 3);
    new (&values[1]) NamedBackendData{
        weight_key, BackendData{weight_bytes, requested_alignment}};
    return output.write_named_data(Span<const NamedBackendData>(values, 2));
  }
};

TestPreparingBackend& test_backend() {
  static TestPreparingBackend backend;
  return backend;
}

Error ensure_test_backend_registered() {
  static const Error registration = executorch::runtime::register_backend(
      executorch::runtime::Backend{kBackendName, &test_backend()});
  return registration;
}

std::shared_ptr<std::vector<uint8_t>> add_pte_header_and_segments(
    flatbuffers::FlatBufferBuilder& builder,
    const std::array<uint8_t, 193>& segments) {
  const size_t metadata_size = builder.GetSize() + kPteHeaderSize;
  const size_t segment_base = align_up(metadata_size, 128);
  auto data =
      std::make_shared<std::vector<uint8_t>>(segment_base + segments.size(), 0);
  std::copy_n(builder.GetBufferPointer(), 8, data->data());
  write_u32(*data, 0, read_u32(builder.GetBufferPointer()) + kPteHeaderSize);
  std::memcpy(data->data() + 8, executorch::runtime::ExtendedHeader::kMagic, 4);
  write_u32(*data, 12, kPteHeaderSize);
  write_u64(*data, 16, metadata_size);
  write_u64(*data, 24, segment_base);
  write_u64(*data, 32, segments.size());
  std::copy(
      builder.GetBufferPointer() + 8,
      builder.GetBufferPointer() + builder.GetSize(),
      data->begin() + 8 + kPteHeaderSize);
  std::copy(segments.begin(), segments.end(), data->begin() + segment_base);
  return data;
}

std::shared_ptr<std::vector<uint8_t>> make_test_pte(
    bool include_named_data,
    bool omit_weight_size = false,
    bool named_constant_alias = false,
    DelegateConflict conflict = DelegateConflict::None) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<executorch_flatbuffer::DataSegment>> segments;
  segments.push_back(executorch_flatbuffer::CreateDataSegment(builder, 0, 1));
  segments.push_back(executorch_flatbuffer::CreateDataSegment(
      builder, 64, omit_weight_size ? 0 : 2));
  segments.push_back(executorch_flatbuffer::CreateDataSegment(builder, 128, 1));
  segments.push_back(executorch_flatbuffer::CreateDataSegment(builder, 192, 1));

  const auto backend_name = builder.CreateString(kBackendName);
  const auto processed =
      executorch_flatbuffer::CreateBackendDelegateDataReference(
          builder, executorch_flatbuffer::DataLocation::SEGMENT, 0);
  const auto first_delegate = executorch_flatbuffer::CreateBackendDelegate(
      builder, backend_name, processed);
  flatbuffers::Offset<flatbuffers::Vector<
      flatbuffers::Offset<executorch_flatbuffer::CompileSpec>>>
      second_specs;
  if (conflict == DelegateConflict::CompileSpec) {
    const std::array<uint8_t, 1> value{{1}};
    const auto spec = executorch_flatbuffer::CreateCompileSpec(
        builder,
        builder.CreateString("variant"),
        builder.CreateVector(value.data(), value.size()));
    second_specs = builder.CreateVector(&spec, 1);
  }
  const auto second_backend_name = conflict == DelegateConflict::Backend
      ? builder.CreateString("ConflictingBackend")
      : backend_name;
  const auto second_delegate = executorch_flatbuffer::CreateBackendDelegate(
      builder, second_backend_name, processed, second_specs);
  const auto first_delegates = builder.CreateVector(&first_delegate, 1);
  const auto second_delegates = builder.CreateVector(&second_delegate, 1);
  std::vector<flatbuffers::Offset<executorch_flatbuffer::ExecutionPlan>> plans;
  plans.push_back(executorch_flatbuffer::CreateExecutionPlan(
      builder,
      builder.CreateString("prefill"),
      0,
      0,
      0,
      0,
      0,
      0,
      first_delegates));
  plans.push_back(executorch_flatbuffer::CreateExecutionPlan(
      builder,
      builder.CreateString("decode"),
      0,
      0,
      0,
      0,
      0,
      0,
      second_delegates));

  std::vector<flatbuffers::Offset<executorch_flatbuffer::NamedData>> named;
  if (include_named_data) {
    named.push_back(executorch_flatbuffer::CreateNamedData(
        builder, builder.CreateString("weight"), 1));
    named.push_back(executorch_flatbuffer::CreateNamedData(
        builder, builder.CreateString("bias"), 2));
  }
  const std::array<uint64_t, 1> constant_offsets{{0}};
  const auto constant_segment = executorch_flatbuffer::CreateSubsegmentOffsets(
      builder,
      named_constant_alias ? 1 : 3,
      builder.CreateVector(constant_offsets.data(), constant_offsets.size()));
  const auto program = executorch_flatbuffer::CreateProgram(
      builder,
      0,
      builder.CreateVector(plans),
      0,
      0,
      builder.CreateVector(segments),
      constant_segment,
      0,
      include_named_data ? builder.CreateVector(named) : 0);
  executorch_flatbuffer::FinishProgramBuffer(builder, program);

  std::array<uint8_t, 193> payload{};
  payload[0] = 1;
  payload[64] = 3;
  payload[65] = 4;
  payload[128] = 5;
  payload[192] = 9;
  return add_pte_header_and_segments(builder, payload);
}

std::shared_ptr<std::vector<uint8_t>> make_old_header_pte() {
  auto data = make_test_pte(true);
  const size_t segment_base = read_u64(data->data() + 24);
  const uint32_t old_root_offset = read_u32(data->data());
  const uint64_t old_program_size = read_u64(data->data() + 16);
  data->erase(data->begin() + 32, data->begin() + 40);
  data->insert(data->begin() + segment_base - 8, 8, uint8_t{0});
  write_u32(*data, 0, old_root_offset - 8);
  write_u32(*data, 12, 24);
  write_u64(*data, 16, old_program_size - 8);
  return data;
}

executorch::runtime::ExtendedHeader parse_pte_header(
    const std::vector<uint8_t>& data) {
  auto header =
      executorch::runtime::ExtendedHeader::Parse(data.data(), data.size());
  EXPECT_TRUE(header.ok());
  return header.ok() ? *header : executorch::runtime::ExtendedHeader{};
}

TEST(InitializeAndSaveBackendDataTest, RewritesOnlyNamedDataInPte) {
  executorch::runtime::runtime_init();
  ASSERT_EQ(ensure_test_backend_registered(), Error::Ok);
  auto& backend = test_backend();
  backend.reset();
  backend.requested_alignment = 256;

  auto original = make_test_pte(true);
  const auto original_header = parse_pte_header(*original);
  const auto* original_program =
      executorch_flatbuffer::GetProgram(original->data());
  const size_t original_segment_count = original_program->segments()->size();
  const size_t original_named_count = original_program->named_data()->size();
  std::vector<uint8_t> output;
  std::array<uint8_t, 4096> temp{};
  executorch::runtime::MemoryAllocator allocator(temp.size(), temp.data());

  ASSERT_EQ(
      initialize_and_save_backend_data(
          std::make_unique<VectorDataLoader>(original),
          std::make_unique<BufferDataWriter>(&output),
          &allocator),
      Error::Ok);
  EXPECT_EQ(backend.methods, (std::vector<std::string>{"prefill"}));
  ASSERT_FALSE(output.empty());

  const auto output_header = parse_pte_header(output);
  EXPECT_EQ(output_header.program_size, original_header.program_size);
  VectorDataLoader loader(std::make_shared<std::vector<uint8_t>>(output));
  EXPECT_TRUE(executorch::runtime::Program::load(&loader).ok());

  const auto* program = executorch_flatbuffer::GetProgram(output.data());
  ASSERT_EQ(program->segments()->size(), original_segment_count);
  ASSERT_EQ(program->named_data()->size(), original_named_count);
  EXPECT_STREQ(program->named_data()->Get(0)->key()->c_str(), "weight");
  EXPECT_STREQ(program->named_data()->Get(1)->key()->c_str(), "bias");
  EXPECT_EQ(program->named_data()->Get(0)->segment_index(), 1U);
  EXPECT_EQ(program->named_data()->Get(1)->segment_index(), 2U);
  EXPECT_EQ(program->segments()->Get(0)->offset(), 0U);
  EXPECT_EQ(program->segments()->Get(0)->size(), 1U);
  EXPECT_EQ(program->segments()->Get(1)->size(), 3U);
  EXPECT_EQ(program->segments()->Get(2)->size(), 1U);
  EXPECT_GT(
      program->segments()->Get(1)->offset(),
      program->segments()->Get(2)->offset());
  const size_t processed_offset =
      output_header.segment_base_offset + program->segments()->Get(0)->offset();
  EXPECT_EQ(output[processed_offset], 1U);
  FreeableBuffer processed(output.data() + processed_offset, 1, nullptr);
  BackendInitContext init_context(&allocator, nullptr, "prefill");
  EXPECT_TRUE(
      backend.init(init_context, &processed, ArrayRef<CompileSpec>()).ok());
  EXPECT_EQ(backend.init_calls, 1U);
  const size_t weight_offset =
      output_header.segment_base_offset + program->segments()->Get(1)->offset();
  EXPECT_EQ(weight_offset % 256, 0U);
  EXPECT_EQ(
      std::vector<uint8_t>(
          output.begin() + weight_offset, output.begin() + weight_offset + 3),
      (std::vector<uint8_t>{3, 4, 5}));
}

TEST(InitializeAndSaveBackendDataTest, RejectsOutputOutsideTempAllocator) {
  executorch::runtime::runtime_init();
  ASSERT_EQ(ensure_test_backend_registered(), Error::Ok);
  auto& backend = test_backend();
  backend.reset();
  backend.use_external_output = true;

  auto original = make_test_pte(true);
  std::vector<uint8_t> output;
  std::array<uint8_t, 4096> temp{};
  executorch::runtime::MemoryAllocator allocator(temp.size(), temp.data());

  EXPECT_EQ(
      initialize_and_save_backend_data(
          std::make_unique<VectorDataLoader>(original),
          std::make_unique<BufferDataWriter>(&output),
          &allocator),
      Error::InvalidArgument);
  EXPECT_TRUE(output.empty());
}

TEST(InitializeAndSaveBackendDataTest, RejectsMissingMutableScalarField) {
  executorch::runtime::runtime_init();
  ASSERT_EQ(ensure_test_backend_registered(), Error::Ok);
  test_backend().reset();

  auto original = make_test_pte(true, true);
  std::vector<uint8_t> output;
  std::array<uint8_t, 4096> temp{};
  executorch::runtime::MemoryAllocator allocator(temp.size(), temp.data());

  EXPECT_EQ(
      initialize_and_save_backend_data(
          std::make_unique<VectorDataLoader>(original),
          std::make_unique<BufferDataWriter>(&output),
          &allocator),
      Error::NotSupported);
  EXPECT_TRUE(output.empty());
}

TEST(InitializeAndSaveBackendDataTest, RejectsOlderHeaderWhenSizeChanges) {
  executorch::runtime::runtime_init();
  ASSERT_EQ(ensure_test_backend_registered(), Error::Ok);
  test_backend().reset();

  auto original = make_old_header_pte();
  std::vector<uint8_t> output;
  std::array<uint8_t, 4096> temp{};
  executorch::runtime::MemoryAllocator allocator(temp.size(), temp.data());

  EXPECT_EQ(
      initialize_and_save_backend_data(
          std::make_unique<VectorDataLoader>(original),
          std::make_unique<BufferDataWriter>(&output),
          &allocator),
      Error::NotSupported);
  EXPECT_TRUE(output.empty());
}

TEST(InitializeAndSaveBackendDataTest, RejectsNamedConstantSegmentAlias) {
  executorch::runtime::runtime_init();
  ASSERT_EQ(ensure_test_backend_registered(), Error::Ok);
  test_backend().reset();

  auto original = make_test_pte(true, false, true);
  std::vector<uint8_t> output;
  std::array<uint8_t, 4096> temp{};
  executorch::runtime::MemoryAllocator allocator(temp.size(), temp.data());

  EXPECT_EQ(
      initialize_and_save_backend_data(
          std::make_unique<VectorDataLoader>(original),
          std::make_unique<BufferDataWriter>(&output),
          &allocator),
      Error::InvalidArgument);
  EXPECT_TRUE(output.empty());
}

TEST(InitializeAndSaveBackendDataTest, RejectsConflictingSharedDelegateData) {
  executorch::runtime::runtime_init();
  ASSERT_EQ(ensure_test_backend_registered(), Error::Ok);

  for (const auto conflict :
       {DelegateConflict::Backend, DelegateConflict::CompileSpec}) {
    auto original = make_test_pte(false, false, false, conflict);
    std::vector<uint8_t> output;
    std::array<uint8_t, 4096> temp{};
    executorch::runtime::MemoryAllocator allocator(temp.size(), temp.data());
    EXPECT_EQ(
        initialize_and_save_backend_data(
            std::make_unique<VectorDataLoader>(original),
            std::make_unique<BufferDataWriter>(&output),
            &allocator),
        Error::NotSupported);
    EXPECT_TRUE(output.empty());
  }
}

TEST(InitializeAndSaveBackendDataTest, UnsupportedBackendDoesNotPublish) {
  executorch::runtime::runtime_init();
  ASSERT_EQ(ensure_test_backend_registered(), Error::Ok);
  auto& backend = test_backend();
  backend.reset();
  backend.supported = false;

  auto original = make_test_pte(false);
  std::vector<uint8_t> output{9};
  std::array<uint8_t, 128> temp{};
  executorch::runtime::MemoryAllocator allocator(temp.size(), temp.data());
  const Error error = initialize_and_save_backend_data(
      std::make_unique<VectorDataLoader>(original),
      std::make_unique<BufferDataWriter>(&output),
      &allocator);
  backend.supported = true;

  EXPECT_EQ(error, Error::Ok);
  EXPECT_EQ(output, (std::vector<uint8_t>{9}));
}

} // namespace
