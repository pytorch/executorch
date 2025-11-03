#pragma once

// Implementation file for SlimTensor serialization/dump operations
// This file is included at the end of SlimTensor.h

#include <fstream>
#include <iomanip>

namespace executorch::backends::cuda::slim {

inline bool SlimTensor::dump_binary(const std::string &filename) const {
  // Binary format header for SlimTensor dump
  struct SlimTensorDumpHeader {
    uint32_t magic = 0x534C494D; // "SLIM" in ASCII
    uint32_t version = 1;
    uint32_t dtype_code; // ScalarType as uint32
    uint32_t ndim;
    uint64_t numel;
    uint64_t data_size_bytes;
    // Shape follows immediately after header
    // Data follows after shape
  };

  // Move tensor to CPU for data access
  auto cpu_tensor = this->cpu();

  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    return false;
  }

  // Prepare header
  SlimTensorDumpHeader header;
  header.dtype_code = static_cast<uint32_t>(cpu_tensor.dtype());
  header.ndim = static_cast<uint32_t>(cpu_tensor.dim());
  header.numel = cpu_tensor.numel();
  header.data_size_bytes = cpu_tensor.nbytes();

  // Write header
  file.write(reinterpret_cast<const char *>(&header), sizeof(header));

  // Write shape
  auto sizes = cpu_tensor.sizes();
  std::vector<uint64_t> shape(sizes.begin(), sizes.end());
  file.write(reinterpret_cast<const char *>(shape.data()),
             shape.size() * sizeof(uint64_t));

  // Write raw data
  if (cpu_tensor.numel() > 0) {
    const void *data_ptr = cpu_tensor.data_ptr();
    file.write(static_cast<const char *>(data_ptr), header.data_size_bytes);
  }

  file.close();
  return true;
}

inline bool SlimTensor::dump_text(const std::string &filename,
                                  const std::string &label) const {
  auto cpu_tensor = this->cpu();

  std::ofstream file(filename);
  if (!file.is_open()) {
    return false;
  }

  // Write metadata as comments
  file << "# SlimTensor text dump";
  if (!label.empty()) {
    file << " - " << label;
  }
  file << "\n";

  file << "# dtype: " << static_cast<int>(cpu_tensor.dtype()) << "\n";
  file << "# shape: [";
  auto sizes = cpu_tensor.sizes();
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (i > 0)
      file << ", ";
    file << sizes[i];
  }
  file << "]\n";
  file << "# numel: " << cpu_tensor.numel() << "\n";
  file << "# device: " << this->device().str() << "\n";
  file << "# Data (one value per line):\n";

  // Write data based on dtype
  const size_t numel = cpu_tensor.numel();
  const void *data_ptr = cpu_tensor.data_ptr();

  switch (cpu_tensor.dtype()) {
  case executorch::backends::cuda::c10::ScalarType::Float: {
    const float *data = static_cast<const float *>(data_ptr);
    for (size_t i = 0; i < numel; ++i) {
      file << std::scientific << std::setprecision(8) << data[i] << "\n";
    }
    break;
  }
  case executorch::backends::cuda::c10::ScalarType::BFloat16: {
    // Convert BFloat16 to float for text output
    const executorch::backends::cuda::c10::BFloat16 *data =
        static_cast<const executorch::backends::cuda::c10::BFloat16 *>(data_ptr);
    for (size_t i = 0; i < numel; ++i) {
      float val = static_cast<float>(data[i]);
      file << std::scientific << std::setprecision(8) << val << "\n";
    }
    break;
  }
  case executorch::backends::cuda::c10::ScalarType::Half: {
    // Convert Half to float for text output
    const executorch::backends::cuda::c10::Half *data =
        static_cast<const executorch::backends::cuda::c10::Half *>(data_ptr);
    for (size_t i = 0; i < numel; ++i) {
      float val = static_cast<float>(data[i]);
      file << std::scientific << std::setprecision(8) << val << "\n";
    }
    break;
  }
  case executorch::backends::cuda::c10::ScalarType::Long: {
    const int64_t *data = static_cast<const int64_t *>(data_ptr);
    for (size_t i = 0; i < numel; ++i) {
      file << data[i] << "\n";
    }
    break;
  }
  case executorch::backends::cuda::c10::ScalarType::Int: {
    const int32_t *data = static_cast<const int32_t *>(data_ptr);
    for (size_t i = 0; i < numel; ++i) {
      file << data[i] << "\n";
    }
    break;
  }
  case executorch::backends::cuda::c10::ScalarType::Double: {
    const double *data = static_cast<const double *>(data_ptr);
    for (size_t i = 0; i < numel; ++i) {
      file << std::scientific << std::setprecision(16) << data[i] << "\n";
    }
    break;
  }
  case executorch::backends::cuda::c10::ScalarType::Short: {
    const int16_t *data = static_cast<const int16_t *>(data_ptr);
    for (size_t i = 0; i < numel; ++i) {
      file << data[i] << "\n";
    }
    break;
  }
  case executorch::backends::cuda::c10::ScalarType::Char: {
    const int8_t *data = static_cast<const int8_t *>(data_ptr);
    for (size_t i = 0; i < numel; ++i) {
      file << static_cast<int>(data[i]) << "\n";
    }
    break;
  }
  case executorch::backends::cuda::c10::ScalarType::Byte: {
    const uint8_t *data = static_cast<const uint8_t *>(data_ptr);
    for (size_t i = 0; i < numel; ++i) {
      file << static_cast<unsigned int>(data[i]) << "\n";
    }
    break;
  }
  case executorch::backends::cuda::c10::ScalarType::Bool: {
    const bool *data = static_cast<const bool *>(data_ptr);
    for (size_t i = 0; i < numel; ++i) {
      file << (data[i] ? "1" : "0") << "\n";
    }
    break;
  }
  default:
    file << "# Unsupported dtype for text dumping: "
         << static_cast<int>(cpu_tensor.dtype()) << "\n";
    file.close();
    return false;
  }

  file.close();
  return true;
}

} // namespace executorch::backends::cuda::slim
