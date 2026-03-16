/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/flat_tensor/serialize/serialize.h>

#include <executorch/extension/flat_tensor/serialize/flat_tensor_generated.h>
#include <executorch/extension/flat_tensor/serialize/flat_tensor_header.h>
#include <executorch/extension/flat_tensor/serialize/scalar_type_generated.h>

#include <fstream>
#include <string>

namespace executorch {
namespace extension {
namespace flat_tensor {

namespace {
size_t padding_required(size_t offset, size_t alignment) {
  // Returns the padding required to align `offset` to `alignment`.
  size_t remainder = offset % alignment;
  if (remainder != 0) {
    return alignment - remainder;
  }
  return 0;
}

size_t aligned_size(size_t input_size, size_t alignment) {
  // Returns input_size padded up to the next whole multiple of alignment.
  return input_size + padding_required(input_size, alignment);
}

void write_nulls(std::ostream& out, size_t num_bytes) {
  for (size_t i = 0; i < num_bytes; i++) {
    out.write("\0", 1);
  }
}
} // namespace

runtime::Error save_ptd(
    const std::string& path,
    const std::map<std::string, executorch::aten::Tensor>& tensor_map,
    const size_t tensor_alignment) {
  // Create File
  std::ofstream file;
  file.open(path);
  runtime::Error e = save_ptd(file, tensor_map, tensor_alignment);
  file.close();
  return e;
}

runtime::Error save_ptd(
    std::ostream& out,
    const std::map<std::string, executorch::aten::Tensor>& tensor_map,
    const size_t tensor_alignment) {
  // Assert the system is little endian. Since we are sending the data over
  // the wire, we need to ensure that the data is always in the same format.
  // for now we only support little endian.
  int n = 1;
  if (*(char*)&n != 1) {
    ET_LOG(Error, "Cannot save_ptd on big endian system");
    return runtime::Error::NotSupported;
  }
  // Create flatbuffer
  flatbuffers::FlatBufferBuilder builder;

  std::vector<flatbuffers::Offset<::flat_tensor_flatbuffer::NamedData>>
      named_data;
  std::vector<flatbuffers::Offset<::flat_tensor_flatbuffer::DataSegment>>
      segments;

  // Write the tensors.
  size_t total_segment_size = 0;
  uint32_t i = 0;
  size_t tensor_count = tensor_map.size();
  for (const auto& [name, tensor] : tensor_map) {
    auto key = builder.CreateString(name);
    // Write the tensor layouts.
    auto tensor_layout = ::flat_tensor_flatbuffer::CreateTensorLayout(
        /*_fbb*=*/builder,
        /*scalar_type=*/
        static_cast<executorch_flatbuffer::ScalarType>(tensor.scalar_type()),
        /*sizes=*/
        builder.CreateVector(tensor.sizes().data(), tensor.sizes().size()),
        /*dim_order=*/
        builder.CreateVector(
            tensor.dim_order().data(), tensor.dim_order().size()));

    named_data.push_back(::flat_tensor_flatbuffer::CreateNamedData(
        /*_fbb=*/builder,
        /*key=*/key,
        /*segment_index=*/i,
        /*tensor_layout=*/tensor_layout));

    segments.push_back(::flat_tensor_flatbuffer::CreateDataSegment(
        /*_fbb=*/builder,
        /*offset=*/total_segment_size,
        /*size=*/tensor.nbytes()));

    // Do not pad the last tensor.
    total_segment_size += (i == tensor_count - 1)
        ? tensor.nbytes()
        : aligned_size(tensor.nbytes(), tensor_alignment);
    i++;
  }

  auto flat_tensor = CreateFlatTensor(
      /*_fbb=*/builder,
      /*version=*/kSchemaVersion,
      /*segments=*/builder.CreateVector(segments),
      /*named_data=*/builder.CreateVector(named_data));
  builder.Finish(flat_tensor, ::flat_tensor_flatbuffer::FlatTensorIdentifier());
  // Our flatbuffer is created now.

  // Calculate flatbuffer padding.
  auto padded_flatbufer_size =
      aligned_size(builder.GetSize(), tensor_alignment);
  auto padded_header_size =
      aligned_size(FlatTensorHeader::kHeaderExpectedLength, tensor_alignment);

  // The general structure of the file is:
  // [flatbuffer offset to root table][flatbuffer file indentifier]
  //   [FlatTensorHeader][padding][flatbuffer contents][padding]
  //   [segment data].
  // This means we first serialize the first 8 bytes of the flatbuffer,
  // updating the offset to the root table, then the header, then the
  // flatbuffer. We are embedding the header inside the flatbuffer doing
  // this which allows us to continue using flatbuffer tools directly on the
  // .ptd file.

  // Calculate new offset to root table.
  uint32_t current_offset =
      *reinterpret_cast<uint32_t*>(builder.GetBufferPointer());
  uint32_t new_offset = current_offset + padded_header_size;

  // Write flatbuffer offset to root table.
  out.write(reinterpret_cast<const char*>(&new_offset), sizeof(new_offset));

  // Write flatbuffer magic bytes.
  out.write(
      reinterpret_cast<const char*>(builder.GetBufferPointer()) +
          sizeof(new_offset),
      4); // This is the file identifier from flat_tensor.fbs.

  // Write header
  out.write(FlatTensorHeader::kMagic, sizeof(FlatTensorHeader::kMagic));
  out.write(
      reinterpret_cast<const char*>(&FlatTensorHeader::kHeaderExpectedLength),
      sizeof(FlatTensorHeader::kHeaderExpectedLength));

  FlatTensorHeader header = {
      padded_header_size, // Offset to flatbuffer
      builder.GetSize(), // flatbuffer size
      padded_header_size + padded_flatbufer_size, // offset to segments
      total_segment_size // segment data size
  };

  out.write(
      reinterpret_cast<const char*>(&header.flatbuffer_offset),
      sizeof(header.flatbuffer_offset));
  out.write(
      reinterpret_cast<const char*>(&header.flatbuffer_size),
      sizeof(header.flatbuffer_size));
  out.write(
      reinterpret_cast<const char*>(&header.segment_base_offset),
      sizeof(header.segment_base_offset));
  out.write(
      reinterpret_cast<const char*>(&header.segment_data_size),
      sizeof(header.segment_data_size));

  // Write header padding
  write_nulls(
      out,
      padding_required(
          FlatTensorHeader::kHeaderExpectedLength, tensor_alignment));

  // Write flatbuffer, offset by 8 bytes (4-byte root table offset + 4-byte
  // file identifier) since we wrote those before the FlatTensorHeader.
  out.write(
      reinterpret_cast<const char*>(builder.GetBufferPointer()) + 8,
      builder.GetSize() - 8);

  // Write flatbuffer padding
  write_nulls(out, padding_required(builder.GetSize(), tensor_alignment));

  // Write segment: buffers + tensor padding
  i = tensor_map.size();
  for (const auto& [name, tensor] : tensor_map) {
    out.write(
        reinterpret_cast<const char*>(tensor.data_ptr()), tensor.nbytes());
    // Don't pad last entry.
    if (i != 1) {
      write_nulls(out, padding_required(tensor.nbytes(), tensor_alignment));
    }
    i--;
  }
  return runtime::Error::Ok;
}

} // namespace flat_tensor
} // namespace extension
} // namespace executorch
