// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <executorch/devtools/etdump/data_sink.h>
#include <executorch/devtools/etdump/utils.h>

namespace executorch {
namespace etdump {

size_t DataSink::write_tensor(const executorch::aten::Tensor& tensor) {
  if (tensor.nbytes() == 0) {
    return static_cast<size_t>(-1);
  }
  uint8_t* offset_ptr =
      internal::alignPointer(debug_buffer_.data() + offset_, 64);
  offset_ = (offset_ptr - debug_buffer_.data()) + tensor.nbytes();
  ET_CHECK_MSG(
      offset_ <= debug_buffer_.size(),
      "Ran out of space to store tensor data.");
  memcpy(offset_ptr, tensor.const_data_ptr(), tensor.nbytes());
  return (size_t)(offset_ptr - debug_buffer_.data());
}

size_t DataSink::get_storage_size() const {
  return debug_buffer_.size();
}

size_t DataSink::get_used_bytes() const {
  return offset_;
}

} // namespace etdump
} // namespace executorch
