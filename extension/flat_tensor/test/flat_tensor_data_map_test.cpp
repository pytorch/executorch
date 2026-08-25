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

size_t aligned_up(size_t size) {
  return (size + kAlignment - 1) & ~(kAlignment - 1);
}

// Builds the smallest PTD file that FlatTensorDataMap::load() accepts, stamped
// with the given schema version and holding no data. Follows the layout written
// by save_ptd(): the header is embedded in the flatbuffer region, so the offset
// to the root table shifts by the size of the header.
std::vector<uint8_t> CreateDataWithVersion(uint32_t version) {
  flatbuffers::FlatBufferBuilder builder;
  auto flat_tensor = flat_tensor_flatbuffer::CreateFlatTensor(
      builder,
      version,
      builder.CreateVector(
          std::vector<
              flatbuffers::Offset<flat_tensor_flatbuffer::DataSegment>>{}),
      builder.CreateVector(
          std::vector<
              flatbuffers::Offset<flat_tensor_flatbuffer::NamedData>>{}));
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
  uint64_t header_fields[] = {
      header_size, // Offset to the flatbuffer.
      flatbuffer_size,
      header_size + aligned_up(flatbuffer_size), // Offset to the segments.
      0, // Segment data size.
  };
  append(header_fields, sizeof(header_fields));
  data.resize(sizeof(root_table_offset) + 4 + header_size, 0);

  // The first eight bytes of the flatbuffer were written above, before the
  // header.
  append(flatbuffer + 8, flatbuffer_size - 8);
  data.resize(header_size + aligned_up(flatbuffer_size), 0);

  return data;
}

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
