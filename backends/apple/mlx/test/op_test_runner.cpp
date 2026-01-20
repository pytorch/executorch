/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Generic op test runner for MLX delegate.
 *
 * Loads a .pte file, reads inputs from .bin files, runs the model,
 * and writes outputs to .bin files.
 *
 * Build:
 *   cd cmake-out-mlx && cmake --build . --target op_test_runner
 *
 * Usage:
 *   ./cmake-out-mlx/backends/apple/mlx/test/op_test_runner \
 *       --pte <model.pte> \
 *       --input <input.bin> \
 *       --output <output.bin>
 *
 * Binary file format:
 *   - 4 bytes: number of tensors (uint32_t)
 *   For each tensor:
 *     - 4 bytes: dtype (0=float32, 1=float16, 2=int32, 3=int64)
 *     - 4 bytes: number of dimensions (uint32_t)
 *     - 4 bytes * ndim: shape (int32_t each)
 *     - N bytes: data (size = product of shape * sizeof(dtype))
 */

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace ::executorch::extension;
using namespace ::executorch::runtime;

enum class DType : uint32_t {
  Float32 = 0,
  Float16 = 1,
  Int32 = 2,
  Int64 = 3,
  BFloat16 = 4,
};

size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::Float32:
      return 4;
    case DType::Float16:
      return 2;
    case DType::Int32:
      return 4;
    case DType::Int64:
      return 8;
    case DType::BFloat16:
      return 2;
    default:
      return 4;
  }
}

exec_aten::ScalarType dtype_to_scalar_type(DType dtype) {
  switch (dtype) {
    case DType::Float32:
      return exec_aten::ScalarType::Float;
    case DType::Float16:
      return exec_aten::ScalarType::Half;
    case DType::Int32:
      return exec_aten::ScalarType::Int;
    case DType::Int64:
      return exec_aten::ScalarType::Long;
    case DType::BFloat16:
      return exec_aten::ScalarType::BFloat16;
    default:
      return exec_aten::ScalarType::Float;
  }
}

DType scalar_type_to_dtype(exec_aten::ScalarType stype) {
  switch (stype) {
    case exec_aten::ScalarType::Float:
      return DType::Float32;
    case exec_aten::ScalarType::Half:
      return DType::Float16;
    case exec_aten::ScalarType::Int:
      return DType::Int32;
    case exec_aten::ScalarType::Long:
      return DType::Int64;
    case exec_aten::ScalarType::BFloat16:
      return DType::BFloat16;
    default:
      return DType::Float32;
  }
}

struct TensorData {
  DType dtype;
  std::vector<int32_t> shape;
  std::vector<uint8_t> data;
};

std::vector<TensorData> read_tensors_from_bin(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open input file: " + path);
  }

  uint32_t num_tensors;
  file.read(reinterpret_cast<char*>(&num_tensors), sizeof(num_tensors));

  std::vector<TensorData> tensors;
  tensors.reserve(num_tensors);

  for (uint32_t i = 0; i < num_tensors; ++i) {
    TensorData t;

    uint32_t dtype_val;
    file.read(reinterpret_cast<char*>(&dtype_val), sizeof(dtype_val));
    t.dtype = static_cast<DType>(dtype_val);

    uint32_t ndim;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));

    t.shape.resize(ndim);
    file.read(reinterpret_cast<char*>(t.shape.data()), ndim * sizeof(int32_t));

    size_t numel = 1;
    for (int32_t s : t.shape) {
      numel *= s;
    }
    size_t data_size = numel * dtype_size(t.dtype);

    t.data.resize(data_size);
    file.read(reinterpret_cast<char*>(t.data.data()), data_size);

    tensors.push_back(std::move(t));
  }

  return tensors;
}

void write_tensors_to_bin(
    const std::string& path,
    const std::vector<TensorData>& tensors) {
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open output file: " + path);
  }

  uint32_t num_tensors = static_cast<uint32_t>(tensors.size());
  file.write(reinterpret_cast<const char*>(&num_tensors), sizeof(num_tensors));

  for (const auto& t : tensors) {
    uint32_t dtype_val = static_cast<uint32_t>(t.dtype);
    file.write(reinterpret_cast<const char*>(&dtype_val), sizeof(dtype_val));

    uint32_t ndim = static_cast<uint32_t>(t.shape.size());
    file.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));

    file.write(
        reinterpret_cast<const char*>(t.shape.data()),
        ndim * sizeof(int32_t));

    file.write(
        reinterpret_cast<const char*>(t.data.data()), t.data.size());
  }
}

void print_usage(const char* prog_name) {
  std::cerr << "Usage: " << prog_name << " [options]\n"
            << "Options:\n"
            << "  --pte <path>     Path to .pte model file (required)\n"
            << "  --input <path>   Path to input .bin file (required)\n"
            << "  --output <path>  Path to output .bin file (required)\n"
            << "  --verbose        Print verbose output\n"
            << std::endl;
}

int main(int argc, char* argv[]) {
  std::string pte_path;
  std::string input_path;
  std::string output_path;
  bool verbose = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--pte" && i + 1 < argc) {
      pte_path = argv[++i];
    } else if (arg == "--input" && i + 1 < argc) {
      input_path = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      output_path = argv[++i];
    } else if (arg == "--verbose") {
      verbose = true;
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      print_usage(argv[0]);
      return 1;
    }
  }

  if (pte_path.empty() || input_path.empty() || output_path.empty()) {
    std::cerr << "Error: --pte, --input, and --output are required\n";
    print_usage(argv[0]);
    return 1;
  }

  try {
    if (verbose) {
      std::cout << "Loading model from: " << pte_path << std::endl;
    }

    Module module(pte_path);
    auto load_error = module.load();
    if (load_error != Error::Ok) {
      std::cerr << "Failed to load model: " << static_cast<int>(load_error)
                << std::endl;
      return 1;
    }

    if (verbose) {
      std::cout << "Model loaded successfully" << std::endl;
    }

    auto load_method_error = module.load_method("forward");
    if (load_method_error != Error::Ok) {
      std::cerr << "Failed to load forward method: "
                << static_cast<int>(load_method_error) << std::endl;
      return 1;
    }

    if (verbose) {
      std::cout << "Reading inputs from: " << input_path << std::endl;
    }

    auto input_tensors = read_tensors_from_bin(input_path);

    if (verbose) {
      std::cout << "Read " << input_tensors.size() << " input tensors"
                << std::endl;
      for (size_t i = 0; i < input_tensors.size(); ++i) {
        std::cout << "  Input " << i << ": dtype="
                  << static_cast<int>(input_tensors[i].dtype) << ", shape=[";
        for (size_t j = 0; j < input_tensors[i].shape.size(); ++j) {
          std::cout << input_tensors[i].shape[j];
          if (j < input_tensors[i].shape.size() - 1)
            std::cout << ", ";
        }
        std::cout << "]" << std::endl;
      }
    }

    std::vector<TensorPtr> tensor_ptrs;
    std::vector<EValue> inputs;
    tensor_ptrs.reserve(input_tensors.size());
    inputs.reserve(input_tensors.size());

    for (const auto& t : input_tensors) {
      auto scalar_type = dtype_to_scalar_type(t.dtype);
      std::vector<exec_aten::SizesType> sizes(t.shape.begin(), t.shape.end());

      TensorPtr tensor_ptr;
      if (t.dtype == DType::Float32) {
        std::vector<float> data(t.data.size() / sizeof(float));
        std::memcpy(data.data(), t.data.data(), t.data.size());
        tensor_ptr = make_tensor_ptr(sizes, std::move(data));
      } else if (t.dtype == DType::Float16) {
        std::vector<exec_aten::Half> data(t.data.size() / sizeof(exec_aten::Half));
        std::memcpy(data.data(), t.data.data(), t.data.size());
        tensor_ptr = make_tensor_ptr(sizes, std::move(data));
      } else if (t.dtype == DType::BFloat16) {
        std::vector<exec_aten::BFloat16> data(t.data.size() / sizeof(exec_aten::BFloat16));
        std::memcpy(data.data(), t.data.data(), t.data.size());
        tensor_ptr = make_tensor_ptr(sizes, std::move(data));
      } else if (t.dtype == DType::Int32) {
        std::vector<int32_t> data(t.data.size() / sizeof(int32_t));
        std::memcpy(data.data(), t.data.data(), t.data.size());
        tensor_ptr = make_tensor_ptr(sizes, std::move(data));
      } else if (t.dtype == DType::Int64) {
        std::vector<int64_t> data(t.data.size() / sizeof(int64_t));
        std::memcpy(data.data(), t.data.data(), t.data.size());
        tensor_ptr = make_tensor_ptr(sizes, std::move(data));
      } else {
        std::cerr << "Unsupported dtype: " << static_cast<int>(t.dtype)
                  << std::endl;
        return 1;
      }

      tensor_ptrs.push_back(tensor_ptr);
      inputs.push_back(tensor_ptr);
    }

    if (verbose) {
      std::cout << "Executing forward..." << std::endl;
    }

    auto result = module.forward(inputs);
    if (result.error() != Error::Ok) {
      std::cerr << "Execution failed: " << static_cast<int>(result.error())
                << std::endl;
      return 1;
    }

    if (verbose) {
      std::cout << "Execution succeeded, " << result->size() << " outputs"
                << std::endl;
    }

    std::vector<TensorData> output_tensors;
    output_tensors.reserve(result->size());

    for (size_t i = 0; i < result->size(); ++i) {
      const auto& evalue = result->at(i);
      if (!evalue.isTensor()) {
        std::cerr << "Output " << i << " is not a tensor" << std::endl;
        return 1;
      }

      const auto& tensor = evalue.toTensor();
      TensorData t;
      t.dtype = scalar_type_to_dtype(tensor.scalar_type());

      t.shape.resize(tensor.dim());
      for (int d = 0; d < tensor.dim(); ++d) {
        t.shape[d] = static_cast<int32_t>(tensor.size(d));
      }

      size_t data_size = tensor.nbytes();
      t.data.resize(data_size);
      std::memcpy(t.data.data(), tensor.const_data_ptr(), data_size);

      if (verbose) {
        std::cout << "  Output " << i << ": dtype=" << static_cast<int>(t.dtype)
                  << ", shape=[";
        for (size_t j = 0; j < t.shape.size(); ++j) {
          std::cout << t.shape[j];
          if (j < t.shape.size() - 1)
            std::cout << ", ";
        }
        std::cout << "]" << std::endl;
      }

      output_tensors.push_back(std::move(t));
    }

    if (verbose) {
      std::cout << "Writing outputs to: " << output_path << std::endl;
    }

    write_tensors_to_bin(output_path, output_tensors);

    std::cout << "OK" << std::endl;
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
