// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <executorch/devtools/etdump/data_sink_base.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/span.h>

namespace executorch {
namespace etdump {

class BufferDataSink : public DataSinkBase {
 public:
  explicit BufferDataSink(::executorch::runtime::Span<uint8_t> buffer)
      : debug_buffer_(buffer), offset_(0) {}

  Result<size_t> write(const void* ptr, size_t length) override;
  Result<size_t> write_tensor(const executorch::aten::Tensor& tensor);
  Result<size_t> get_storage_size() const override;
  size_t get_used_bytes() const override;

 private:
  ::executorch::runtime::Span<uint8_t> debug_buffer_;
  // The offset of the next available location in the debug storage
  size_t offset_;
};

} // namespace etdump
} // namespace executorch
