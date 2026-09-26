/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/flat_tensor/flat_tensor_data_map.h>
#include <executorch/extension/flat_tensor/serialize/flat_tensor_generated.h>
#include <executorch/extension/flat_tensor/serialize/flat_tensor_header.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/runtime.h>

#include <array>
#include <limits>

#include <gtest/gtest.h>

using namespace ::testing;
using namespace executorch::extension;
using namespace executorch::runtime;

class FlatTensorDataMapTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();

    // Load data map. The eager addmul model is defined at:
    // //executorch/test/models/export_program.py
    const char* path = std::getenv("ET_MODULE_ADD_MUL_DATA_PATH");
    Result<FileDataLoader> loader = FileDataLoader::from(path);
    ASSERT_EQ(loader.error(), Error::Ok);

    data_map_loader_ =
        std::make_unique<FileDataLoader>(std::move(loader.get()));
  }
  std::unique_ptr<FileDataLoader> data_map_loader_;
};

TEST_F(FlatTensorDataMapTest, LoadFlatTensorDataMap) {
  Result<FlatTensorDataMap> data_map =
      FlatTensorDataMap::load(data_map_loader_.get());
  EXPECT_EQ(data_map.error(), Error::Ok);
}

TEST_F(FlatTensorDataMapTest, GetMetadata) {
  Result<FlatTensorDataMap> data_map =
      FlatTensorDataMap::load(data_map_loader_.get());
  EXPECT_EQ(data_map.error(), Error::Ok);

  // Check tensor layouts are correct.
  // From //executorch/test/models/linear_model.py, we have the tensors
  // self.a = 3 * torch.ones(2, 2, dtype=torch.float)
  // self.b = 2 * torch.ones(2, 2, dtype=torch.float)
  Result<const TensorLayout> const_a_res = data_map->get_tensor_layout("a");
  ASSERT_EQ(Error::Ok, const_a_res.error());

  const TensorLayout const_a = const_a_res.get();
  EXPECT_EQ(const_a.scalar_type(), executorch::aten::ScalarType::Float);
  auto sizes_a = const_a.sizes();
  EXPECT_EQ(sizes_a.size(), 2);
  EXPECT_EQ(sizes_a[0], 2);
  EXPECT_EQ(sizes_a[1], 2);
  auto dim_order_a = const_a.dim_order();
  EXPECT_EQ(dim_order_a.size(), 2);
  EXPECT_EQ(dim_order_a[0], 0);
  EXPECT_EQ(dim_order_a[1], 1);

  Result<const TensorLayout> const_b_res = data_map->get_tensor_layout("b");
  ASSERT_EQ(Error::Ok, const_b_res.error());

  const TensorLayout const_b = const_b_res.get();
  EXPECT_EQ(const_b.scalar_type(), executorch::aten::ScalarType::Float);
  auto sizes_b = const_b.sizes();
  EXPECT_EQ(sizes_b.size(), 2);
  EXPECT_EQ(sizes_b[0], 2);
  EXPECT_EQ(sizes_b[1], 2);
  auto dim_order_b = const_b.dim_order();
  EXPECT_EQ(dim_order_b.size(), 2);
  EXPECT_EQ(dim_order_b[0], 0);
  EXPECT_EQ(dim_order_b[1], 1);

  // Check get_tensor_layout fails when key is not found.
  Result<const TensorLayout> const_c_res = data_map->get_tensor_layout("c");
  EXPECT_EQ(const_c_res.error(), Error::NotFound);
}

TEST_F(FlatTensorDataMapTest, GetData) {
  Result<FlatTensorDataMap> data_map =
      FlatTensorDataMap::load(data_map_loader_.get());
  EXPECT_EQ(data_map.error(), Error::Ok);

  // Check tensor data sizes are correct.
  Result<FreeableBuffer> data_a_res = data_map->get_data("a");
  ASSERT_EQ(Error::Ok, data_a_res.error());
  FreeableBuffer data_a = std::move(data_a_res.get());
  EXPECT_EQ(data_a.size(), 16);

  Result<FreeableBuffer> data_b_res = data_map->get_data("b");
  ASSERT_EQ(Error::Ok, data_b_res.error());
  FreeableBuffer data_b = std::move(data_b_res.get());
  EXPECT_EQ(data_b.size(), 16);

  // Check get_data fails when key is not found.
  Result<FreeableBuffer> data_c_res = data_map->get_data("c");
  EXPECT_EQ(data_c_res.error(), Error::NotFound);
}

TEST_F(FlatTensorDataMapTest, GetKeys) {
  Result<FlatTensorDataMap> data_map =
      FlatTensorDataMap::load(data_map_loader_.get());
  EXPECT_EQ(data_map.error(), Error::Ok);

  // Check num tensors is 2.
  Result<uint32_t> num_tensors_res = data_map->get_num_keys();
  ASSERT_EQ(Error::Ok, num_tensors_res.error());
  EXPECT_EQ(num_tensors_res.get(), 2);

  // Check get_key returns the correct keys.
  Result<const char*> key0_res = data_map->get_key(0);
  ASSERT_EQ(Error::Ok, key0_res.error());
  EXPECT_EQ(strcmp(key0_res.get(), "a"), 0);

  Result<const char*> key1_res = data_map->get_key(1);
  ASSERT_EQ(Error::Ok, key1_res.error());
  EXPECT_EQ(strcmp(key1_res.get(), "b"), 0);

  // Check get_key fails when out of bounds.
  Result<const char*> key2_res = data_map->get_key(2);
  EXPECT_EQ(key2_res.error(), Error::InvalidArgument);
}

TEST_F(FlatTensorDataMapTest, LoadInto) {
  Result<FlatTensorDataMap> data_map =
      FlatTensorDataMap::load(data_map_loader_.get());
  EXPECT_EQ(data_map.error(), Error::Ok);

  // get the metadata
  auto meta_data_res = data_map->get_tensor_layout("a");
  ASSERT_EQ(meta_data_res.error(), Error::Ok);

  // get data blob
  void* data = malloc(meta_data_res->nbytes());
  auto load_into_error =
      data_map->load_data_into("a", data, meta_data_res->nbytes());
  ASSERT_EQ(load_into_error, Error::Ok);

  // Check tensor data is correct.
  float* data_a = static_cast<float*>(data);
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(data_a[i], 3.0);
  }
  free(data);
}

TEST_F(FlatTensorDataMapTest, LoadAndCheckSize) {
  Result<FlatTensorDataMap> data_map =
      FlatTensorDataMap::load(data_map_loader_.get());
  EXPECT_EQ(data_map.error(), Error::Ok);

  // Truncate the file.
  size_t trunc_size = data_map_loader_->size().get() - 8;
  Result<FreeableBuffer> truncated_file = data_map_loader_->load(
      0,
      trunc_size,
      DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Constant));
  ASSERT_EQ(truncated_file.error(), Error::Ok);

  BufferDataLoader truncated_loader =
      BufferDataLoader(truncated_file->data(), trunc_size);
  Result<FlatTensorDataMap> truncated_program =
      FlatTensorDataMap::load(&truncated_loader);
  ASSERT_EQ(truncated_program.error(), Error::InvalidExternalData);
}

namespace {

constexpr size_t kAlignment = 16;
constexpr char kWideTensorKey[] = "wide_tensor";
constexpr size_t kFlatbufferOffsetFieldOffset =
    FlatTensorHeader::kHeaderOffset + FlatTensorHeader::kMagicSize +
    sizeof(uint32_t);
constexpr size_t kFlatbufferSizeFieldOffset =
    kFlatbufferOffsetFieldOffset + sizeof(uint64_t);
constexpr size_t kSegmentBaseOffsetFieldOffset =
    kFlatbufferSizeFieldOffset + sizeof(uint64_t);
constexpr size_t kSegmentDataSizeFieldOffset =
    kSegmentBaseOffsetFieldOffset + sizeof(uint64_t);

size_t aligned_up(size_t size) {
  return (size + kAlignment - 1) & ~(kAlignment - 1);
}

struct SegmentSpec {
  uint64_t offset;
  uint64_t size;
  uint64_t data_size;
};

// Builds the smallest PTD metadata that FlatTensorDataMap::load() accepts.
// The optional segment is logical: its bytes are supplied separately by the
// test loader, so large source offsets do not require a large allocation.
std::vector<uint8_t> CreateDataWithVersion(
    uint32_t version,
    const SegmentSpec* segment = nullptr) {
  flatbuffers::FlatBufferBuilder builder;

  std::vector<flatbuffers::Offset<flat_tensor_flatbuffer::DataSegment>>
      segments;
  std::vector<flatbuffers::Offset<flat_tensor_flatbuffer::NamedData>>
      named_data;
  if (segment != nullptr) {
    const std::vector<int32_t> sizes{1};
    const std::vector<uint8_t> dim_order{0};
    auto tensor_layout = flat_tensor_flatbuffer::CreateTensorLayout(
        builder,
        executorch_flatbuffer::ScalarType::FLOAT,
        builder.CreateVector(sizes),
        builder.CreateVector(dim_order));
    segments.push_back(flat_tensor_flatbuffer::CreateDataSegment(
        builder, segment->offset, segment->size));
    named_data.push_back(flat_tensor_flatbuffer::CreateNamedData(
        builder,
        builder.CreateString(kWideTensorKey),
        /*segment_index=*/0,
        tensor_layout));
  }

  auto flat_tensor = flat_tensor_flatbuffer::CreateFlatTensor(
      builder,
      version,
      builder.CreateVector(segments),
      builder.CreateVector(named_data));
  builder.Finish(flat_tensor, flat_tensor_flatbuffer::FlatTensorIdentifier());

  const uint8_t* flatbuffer = builder.GetBufferPointer();
  const size_t flatbuffer_size = builder.GetSize();
  const size_t header_size =
      aligned_up(FlatTensorHeader::kHeaderExpectedLength);

  std::vector<uint8_t> data;
  auto append = [&data](const void* bytes, size_t size) {
    const uint8_t* begin = static_cast<const uint8_t*>(bytes);
    data.insert(data.end(), begin, begin + size);
  };

  uint32_t root_table_offset = *reinterpret_cast<const uint32_t*>(flatbuffer) +
      static_cast<uint32_t>(header_size);
  append(&root_table_offset, sizeof(root_table_offset));
  append(flatbuffer + sizeof(root_table_offset), 4); // File identifier.

  append(FlatTensorHeader::kMagic, sizeof(FlatTensorHeader::kMagic));
  uint32_t header_length = FlatTensorHeader::kHeaderExpectedLength;
  append(&header_length, sizeof(header_length));
  const std::array<uint64_t, 4> header_fields = {
      header_size, // Offset to the flatbuffer.
      flatbuffer_size,
      header_size + aligned_up(flatbuffer_size), // Offset to the segments.
      segment == nullptr ? 0 : segment->data_size,
  };
  append(header_fields.data(), sizeof(header_fields));
  data.resize(sizeof(root_table_offset) + 4 + header_size, 0);

  // The first eight bytes of the flatbuffer were written above, before the
  // header.
  append(flatbuffer + 8, flatbuffer_size - 8);
  data.resize(header_size + aligned_up(flatbuffer_size), 0);

  return data;
}

class WideOffsetDataLoader final : public DataLoader {
 public:
  WideOffsetDataLoader(
      const uint8_t* metadata,
      size_t metadata_size,
      const uint8_t* segment_data,
      size_t segment_size,
      uint64_t source_size)
      : metadata_(metadata),
        metadata_size_(metadata_size),
        segment_data_(segment_data),
        segment_size_(segment_size),
        source_size_(source_size) {}

  Result<FreeableBuffer> load(
      size_t,
      size_t,
      const SegmentInfo&) const override {
    return Error::NotSupported;
  }

  Result<size_t> size() const override {
    return Error::NotSupported;
  }

  Result<FreeableBuffer> load_at_offset(
      uint64_t offset,
      size_t size,
      const SegmentInfo& segment_info) const override {
    if (segment_info.segment_type == SegmentInfo::Type::Program) {
      if (offset > metadata_size_ || size > metadata_size_ - offset) {
        return Error::InvalidArgument;
      }
      return FreeableBuffer(
          metadata_ + static_cast<size_t>(offset), size, nullptr);
    }
    if (size > segment_size_) {
      return Error::InvalidArgument;
    }
    last_offset_ = offset;
    return FreeableBuffer(segment_data_, size, nullptr);
  }

  Error load_into_at_offset(
      uint64_t offset,
      size_t size,
      const SegmentInfo&,
      void* buffer) const override {
    if (buffer == nullptr || size > segment_size_) {
      return Error::InvalidArgument;
    }
    last_offset_ = offset;
    std::memcpy(buffer, segment_data_, size);
    return Error::Ok;
  }

  Result<uint64_t> source_size() const override {
    return source_size_;
  }

  uint64_t last_offset() const {
    return last_offset_;
  }

 private:
  const uint8_t* metadata_;
  size_t metadata_size_;
  const uint8_t* segment_data_;
  size_t segment_size_;
  uint64_t source_size_;
  mutable uint64_t last_offset_{0};
};

} // namespace

TEST_F(FlatTensorDataMapTest, SupportedSchemaVersionLoads) {
  std::vector<uint8_t> data =
      CreateDataWithVersion(FlatTensorDataMap::kMaxSupportedSchemaVersion);

  alignas(16) uint8_t aligned_buffer[512];
  ASSERT_LE(data.size(), sizeof(aligned_buffer));
  memcpy(aligned_buffer, data.data(), data.size());

  BufferDataLoader loader(aligned_buffer, data.size());
  Result<FlatTensorDataMap> data_map = FlatTensorDataMap::load(&loader);

  EXPECT_EQ(data_map.error(), Error::Ok);
}

TEST_F(FlatTensorDataMapTest, NewerSchemaVersionFailsToLoad) {
  std::vector<uint8_t> data =
      CreateDataWithVersion(FlatTensorDataMap::kMaxSupportedSchemaVersion + 1);

  alignas(16) uint8_t aligned_buffer[512];
  ASSERT_LE(data.size(), sizeof(aligned_buffer));
  memcpy(aligned_buffer, data.data(), data.size());

  BufferDataLoader loader(aligned_buffer, data.size());
  Result<FlatTensorDataMap> data_map = FlatTensorDataMap::load(&loader);

  EXPECT_EQ(data_map.error(), Error::InvalidExternalData);
}

TEST_F(FlatTensorDataMapTest, RejectsOutOfBoundsRootOffset) {
  // A valid file loads.
  Result<FreeableBuffer> valid = data_map_loader_->load(
      0,
      data_map_loader_->size().get(),
      DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Constant));
  ASSERT_EQ(valid.error(), Error::Ok);

  // Copy it into a max-aligned buffer so we can corrupt the root table offset
  // without tripping the earlier alignment check. std::vector<uint8_t> only
  // guarantees 1-byte alignment, so over-allocate and offset to an aligned
  // start. The first 4 bytes of a (non-size-prefixed) flatbuffer are the
  // uoffset_t pointing at the root table; the next 4 are the file identifier.
  constexpr size_t kBufferAlignment = alignof(std::max_align_t);
  const size_t size = valid->size();
  std::unique_ptr<uint8_t[]> storage(new uint8_t[size + kBufferAlignment]);
  const size_t offset =
      (kBufferAlignment -
       (reinterpret_cast<uintptr_t>(storage.get()) % kBufferAlignment)) %
      kBufferAlignment;
  uint8_t* corrupt = storage.get() + offset;
  std::memcpy(corrupt, valid->data(), size);

  // Overwrite only the root offset with a value far past the end of the buffer,
  // leaving the identifier and alignment intact so the corruption is caught by
  // the root-offset bounds check rather than the identifier or size checks.
  const uint32_t bad_offset = static_cast<uint32_t>(size) + 0x1000u;
  std::memcpy(corrupt, &bad_offset, sizeof(bad_offset));

  // Without the bounds check, GetFlatTensor() would return a pointer outside
  // the buffer and the first field read would walk an out-of-bounds vtable
  // (a use that ASan flags). With it, load() returns cleanly.
  BufferDataLoader corrupt_loader(corrupt, size);
  Result<FlatTensorDataMap> corrupt_map =
      FlatTensorDataMap::load(&corrupt_loader);
  ASSERT_EQ(corrupt_map.error(), Error::InvalidExternalData);
}

TEST_F(FlatTensorDataMapTest, PreservesWideOffsetWhenLoadingData) {
  constexpr uint64_t kSegmentOffset = (uint64_t{1} << 32) + 0x1234;
  constexpr SegmentSpec kSegment{
      kSegmentOffset,
      sizeof(float),
      kSegmentOffset + sizeof(float)};
  std::vector<uint8_t> data = CreateDataWithVersion(
      FlatTensorDataMap::kMaxSupportedSchemaVersion, &kSegment);

  alignas(std::max_align_t) std::array<uint8_t, 1024> aligned_buffer{};
  ASSERT_LE(data.size(), aligned_buffer.size());
  std::memcpy(aligned_buffer.data(), data.data(), data.size());

  Result<FlatTensorHeader> header =
      FlatTensorHeader::Parse(aligned_buffer.data(), data.size());
  ASSERT_TRUE(header.ok());
  const uint64_t absolute_offset =
      header->segment_base_offset + kSegmentOffset;
  const float segment_data = 3.0f;
  WideOffsetDataLoader loader(
      aligned_buffer.data(),
      data.size(),
      reinterpret_cast<const uint8_t*>(&segment_data),
      sizeof(segment_data),
      absolute_offset + sizeof(segment_data));

  Result<FlatTensorDataMap> data_map = FlatTensorDataMap::load(&loader);
  ASSERT_TRUE(data_map.ok());
  Result<FreeableBuffer> loaded = data_map->get_data(kWideTensorKey);

  ASSERT_TRUE(loaded.ok());
  EXPECT_EQ(loader.last_offset(), absolute_offset);
  EXPECT_EQ(loaded->size(), sizeof(segment_data));
  EXPECT_EQ(loaded->data(), &segment_data);
}

TEST_F(FlatTensorDataMapTest, PreservesWideOffsetWhenLoadingDataIntoBuffer) {
  constexpr uint64_t kSegmentOffset = (uint64_t{1} << 32) + 0x5678;
  constexpr SegmentSpec kSegment{
      kSegmentOffset,
      sizeof(float),
      kSegmentOffset + sizeof(float)};
  std::vector<uint8_t> data = CreateDataWithVersion(
      FlatTensorDataMap::kMaxSupportedSchemaVersion, &kSegment);

  alignas(std::max_align_t) std::array<uint8_t, 1024> aligned_buffer{};
  ASSERT_LE(data.size(), aligned_buffer.size());
  std::memcpy(aligned_buffer.data(), data.data(), data.size());

  Result<FlatTensorHeader> header =
      FlatTensorHeader::Parse(aligned_buffer.data(), data.size());
  ASSERT_TRUE(header.ok());
  const uint64_t absolute_offset =
      header->segment_base_offset + kSegmentOffset;
  const float segment_data = 3.0f;
  WideOffsetDataLoader loader(
      aligned_buffer.data(),
      data.size(),
      reinterpret_cast<const uint8_t*>(&segment_data),
      sizeof(segment_data),
      absolute_offset + sizeof(segment_data));
  Result<FlatTensorDataMap> data_map = FlatTensorDataMap::load(&loader);
  ASSERT_TRUE(data_map.ok());

  float loaded = 0.0f;
  EXPECT_EQ(
      data_map->load_data_into(kWideTensorKey, &loaded, sizeof(loaded)),
      Error::Ok);
  EXPECT_EQ(loader.last_offset(), absolute_offset);
  EXPECT_EQ(loaded, segment_data);
}

TEST_F(FlatTensorDataMapTest, ValidatesWideSourceSizeWithoutTruncation) {
  constexpr uint64_t kSegmentOffset = (uint64_t{1} << 32) + 0x9abc;
  constexpr SegmentSpec kSegment{
      kSegmentOffset,
      sizeof(float),
      kSegmentOffset + sizeof(float)};
  std::vector<uint8_t> data = CreateDataWithVersion(
      FlatTensorDataMap::kMaxSupportedSchemaVersion, &kSegment);

  alignas(std::max_align_t) std::array<uint8_t, 1024> aligned_buffer{};
  ASSERT_LE(data.size(), aligned_buffer.size());
  std::memcpy(aligned_buffer.data(), data.data(), data.size());

  Result<FlatTensorHeader> header =
      FlatTensorHeader::Parse(aligned_buffer.data(), data.size());
  ASSERT_TRUE(header.ok());
  const uint64_t required_source_size = header->segment_base_offset +
      kSegmentOffset + sizeof(float);
  const float segment_data = 3.0f;
  WideOffsetDataLoader loader(
      aligned_buffer.data(),
      data.size(),
      reinterpret_cast<const uint8_t*>(&segment_data),
      sizeof(segment_data),
      required_source_size - 1);

  Result<FlatTensorDataMap> data_map = FlatTensorDataMap::load(&loader);

  EXPECT_EQ(data_map.error(), Error::InvalidExternalData);
}

TEST_F(FlatTensorDataMapTest, RejectsSegmentBeyondDeclaredDataRegion) {
  constexpr uint64_t kSegmentOffset = 16;
  constexpr SegmentSpec kSegment{
      kSegmentOffset,
      sizeof(float),
      kSegmentOffset + sizeof(float) - 1};
  std::vector<uint8_t> data = CreateDataWithVersion(
      FlatTensorDataMap::kMaxSupportedSchemaVersion, &kSegment);

  alignas(std::max_align_t) std::array<uint8_t, 1024> aligned_buffer{};
  ASSERT_LE(data.size(), aligned_buffer.size());
  std::memcpy(aligned_buffer.data(), data.data(), data.size());

  Result<FlatTensorHeader> header =
      FlatTensorHeader::Parse(aligned_buffer.data(), data.size());
  ASSERT_TRUE(header.ok());
  const float segment_data = 3.0f;
  WideOffsetDataLoader loader(
      aligned_buffer.data(),
      data.size(),
      reinterpret_cast<const uint8_t*>(&segment_data),
      sizeof(segment_data),
      header->segment_base_offset + kSegmentOffset + sizeof(segment_data));

  Result<FlatTensorDataMap> data_map = FlatTensorDataMap::load(&loader);
  ASSERT_TRUE(data_map.ok());
  EXPECT_EQ(
      data_map->get_data(kWideTensorKey).error(), Error::InvalidExternalData);
}

TEST_F(FlatTensorDataMapTest, LoadIntoHonorsRequestedSize) {
  Result<FlatTensorDataMap> data_map =
      FlatTensorDataMap::load(data_map_loader_.get());
  ASSERT_TRUE(data_map.ok());

  std::array<float, 2> destination{};
  EXPECT_EQ(
      data_map->load_data_into(
          "a", destination.data(), sizeof(destination[0])),
      Error::Ok);
  EXPECT_EQ(destination[0], 3.0f);
  EXPECT_EQ(destination[1], 0.0f);
}

TEST_F(FlatTensorDataMapTest, LoadIntoRejectsRequestBeyondSegmentSize) {
  constexpr uint64_t kSegmentOffset = 16;
  constexpr SegmentSpec kSegment{kSegmentOffset, 1, kSegmentOffset + 1};
  std::vector<uint8_t> data = CreateDataWithVersion(
      FlatTensorDataMap::kMaxSupportedSchemaVersion, &kSegment);

  alignas(std::max_align_t) std::array<uint8_t, 1024> aligned_buffer{};
  ASSERT_LE(data.size(), aligned_buffer.size());
  std::memcpy(aligned_buffer.data(), data.data(), data.size());

  Result<FlatTensorHeader> header =
      FlatTensorHeader::Parse(aligned_buffer.data(), data.size());
  ASSERT_TRUE(header.ok());
  const float segment_data = 3.0f;
  WideOffsetDataLoader loader(
      aligned_buffer.data(),
      data.size(),
      reinterpret_cast<const uint8_t*>(&segment_data),
      sizeof(segment_data),
      header->segment_base_offset + kSegmentOffset + 1);
  Result<FlatTensorDataMap> data_map = FlatTensorDataMap::load(&loader);
  ASSERT_TRUE(data_map.ok());

  float destination = 0.0f;
  EXPECT_EQ(
      data_map->load_data_into(
          kWideTensorKey, &destination, sizeof(destination)),
      Error::InvalidExternalData);
}

TEST_F(FlatTensorDataMapTest, RejectsOverflowingSegmentRegionExtent) {
  std::vector<uint8_t> data =
      CreateDataWithVersion(FlatTensorDataMap::kMaxSupportedSchemaVersion);

  const uint64_t segment_base_offset = std::numeric_limits<uint64_t>::max();
  const uint64_t segment_data_size = 1;
  std::memcpy(
      data.data() + kSegmentBaseOffsetFieldOffset,
      &segment_base_offset,
      sizeof(segment_base_offset));
  std::memcpy(
      data.data() + kSegmentDataSizeFieldOffset,
      &segment_data_size,
      sizeof(segment_data_size));

  alignas(std::max_align_t) std::array<uint8_t, 1024> aligned_buffer{};
  ASSERT_LE(data.size(), aligned_buffer.size());
  std::memcpy(aligned_buffer.data(), data.data(), data.size());
  BufferDataLoader loader(aligned_buffer.data(), data.size());

  EXPECT_EQ(
      FlatTensorDataMap::load(&loader).error(), Error::InvalidExternalData);
}

TEST_F(FlatTensorDataMapTest, RejectsOverflowingDataSegmentExtent) {
  constexpr SegmentSpec kSegment{
      std::numeric_limits<uint64_t>::max() - 1,
      2,
      0};
  std::vector<uint8_t> data = CreateDataWithVersion(
      FlatTensorDataMap::kMaxSupportedSchemaVersion, &kSegment);

  Result<FlatTensorHeader> header =
      FlatTensorHeader::Parse(data.data(), data.size());
  ASSERT_TRUE(header.ok());
  const uint64_t segment_data_size =
      std::numeric_limits<uint64_t>::max() - header->segment_base_offset;
  std::memcpy(
      data.data() + kSegmentDataSizeFieldOffset,
      &segment_data_size,
      sizeof(segment_data_size));

  alignas(std::max_align_t) std::array<uint8_t, 1024> aligned_buffer{};
  ASSERT_LE(data.size(), aligned_buffer.size());
  std::memcpy(aligned_buffer.data(), data.data(), data.size());
  const float segment_data = 3.0f;
  WideOffsetDataLoader loader(
      aligned_buffer.data(),
      data.size(),
      reinterpret_cast<const uint8_t*>(&segment_data),
      sizeof(segment_data),
      std::numeric_limits<uint64_t>::max());

  Result<FlatTensorDataMap> data_map = FlatTensorDataMap::load(&loader);
  ASSERT_TRUE(data_map.ok());
  EXPECT_EQ(
      data_map->get_data(kWideTensorKey).error(), Error::InvalidExternalData);
}

TEST_F(FlatTensorDataMapTest, RejectsOverflowingFlatbufferExtent) {
  std::vector<uint8_t> data =
      CreateDataWithVersion(FlatTensorDataMap::kMaxSupportedSchemaVersion);

  const uint64_t flatbuffer_offset =
      std::numeric_limits<uint64_t>::max();
  const uint64_t flatbuffer_size = 1;
  std::memcpy(
      data.data() + kFlatbufferOffsetFieldOffset,
      &flatbuffer_offset,
      sizeof(flatbuffer_offset));
  std::memcpy(
      data.data() + kFlatbufferSizeFieldOffset,
      &flatbuffer_size,
      sizeof(flatbuffer_size));

  alignas(std::max_align_t) std::array<uint8_t, 1024> aligned_buffer{};
  ASSERT_LE(data.size(), aligned_buffer.size());
  std::memcpy(aligned_buffer.data(), data.data(), data.size());
  BufferDataLoader loader(aligned_buffer.data(), data.size());

  EXPECT_EQ(
      FlatTensorDataMap::load(&loader).error(), Error::InvalidExternalData);
}
