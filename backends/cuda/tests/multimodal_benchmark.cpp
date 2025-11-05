#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/portable_type/tensor.h>

namespace {

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::extension::make_tensor_ptr;
using executorch::extension::TensorPtr;
using executorch::extension::module::Module;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::Result;
using Clock = std::chrono::steady_clock;
using executorch::aten::TensorShapeDynamism;
using DurationMs = std::chrono::duration<double, std::milli>;

enum class ModelType { GEMMA3, VOXTRAL, UNKNOWN };

struct ModelConfig {
  std::string name;
  size_t token_seq_len;
  size_t text_embed_dim;
  std::vector<std::string> expected_methods;
};

const std::map<ModelType, ModelConfig> model_configs = {
    {ModelType::GEMMA3,
     {"gemma3",
      128,
      2304,
      {"vision_encoder", "token_embedding", "text_decoder"}}},
    {ModelType::VOXTRAL,
     {"voxtral",
      1138,
      3072,
      {"audio_encoder", "token_embedding", "text_decoder"}}}};

ModelType parse_model_type(const std::string& model_name) {
  std::string lower_name = model_name;
  std::transform(
      lower_name.begin(),
      lower_name.end(),
      lower_name.begin(),
      [](unsigned char c) { return std::tolower(c); });

  if (lower_name.find("gemma3") != std::string::npos) {
    return ModelType::GEMMA3;
  } else if (lower_name.find("voxtral") != std::string::npos) {
    return ModelType::VOXTRAL;
  }
  return ModelType::UNKNOWN;
}

std::vector<executorch::aten::SizesType> to_sizes(
    std::initializer_list<int64_t> dims) {
  return std::vector<executorch::aten::SizesType>(dims.begin(), dims.end());
}

std::string format_shape(const Tensor& tensor) {
  std::ostringstream oss;
  oss << "[";
  const auto& sizes = tensor.sizes();
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << sizes[i];
  }
  oss << "]";
  return oss.str();
}

void print_tensor_summary(const std::string& label, const Tensor& tensor) {
  std::cout << "    " << label
            << ": dtype=" << executorch::runtime::toString(tensor.scalar_type())
            << ", shape=" << format_shape(tensor)
            << ", numel=" << tensor.numel() << std::endl;
}

void dump_tensor_to_file(const std::string& filename, const Tensor& tensor) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open file for writing: " << filename << std::endl;
    return;
  }

  int32_t dtype = static_cast<int32_t>(tensor.scalar_type());
  file.write(reinterpret_cast<const char*>(&dtype), sizeof(int32_t));

  int32_t ndim = static_cast<int32_t>(tensor.sizes().size());
  file.write(reinterpret_cast<const char*>(&ndim), sizeof(int32_t));

  for (size_t i = 0; i < tensor.sizes().size(); ++i) {
    int64_t dim_size = tensor.sizes()[i];
    file.write(reinterpret_cast<const char*>(&dim_size), sizeof(int64_t));
  }

  const void* data_ptr = tensor.const_data_ptr();
  size_t element_size = 0;

  switch (tensor.scalar_type()) {
    case ScalarType::Float:
      element_size = sizeof(float);
      break;
    case ScalarType::BFloat16:
      element_size = 2;
      break;
    case ScalarType::Half:
      element_size = 2;
      break;
    case ScalarType::Long:
      element_size = sizeof(int64_t);
      break;
    case ScalarType::Int:
      element_size = sizeof(int32_t);
      break;
    default:
      std::cerr << "Unsupported dtype for dumping: "
                << executorch::runtime::toString(tensor.scalar_type())
                << std::endl;
      return;
  }

  size_t data_size = tensor.numel() * element_size;
  file.write(reinterpret_cast<const char*>(data_ptr), data_size);
  file.close();

  std::cout << "Dumped tensor to: " << filename << std::endl;
}

TensorPtr create_vision_input() {
  const auto sizes = to_sizes({1, 3, 896, 896});
  const size_t numel = 1ull * 3ull * 896ull * 896ull;
  std::vector<float> data(numel);
  for (size_t i = 0; i < numel; ++i) {
    data[i] = static_cast<float>((i % 255) / 255.0);
  }
  return make_tensor_ptr<float>(
      sizes,
      std::move(data),
      {},
      {},
      ScalarType::BFloat16,
      TensorShapeDynamism::DYNAMIC_UNBOUND);
}

TensorPtr create_audio_input() {
  const auto sizes = to_sizes({3, 128, 3000});
  const size_t numel = 3ull * 128ull * 3000ull;
  std::vector<float> data(numel, 0.5f);
  return make_tensor_ptr<float>(
      sizes, std::move(data), {}, {}, ScalarType::BFloat16);
}

TensorPtr create_token_ids_input(const ModelConfig& config) {
  const auto sizes = to_sizes({1, static_cast<int64_t>(config.token_seq_len)});
  std::vector<int64_t> data(config.token_seq_len);
  for (size_t i = 0; i < config.token_seq_len; ++i) {
    data[i] = static_cast<int64_t>(i + 1);
  }
  return make_tensor_ptr<int64_t>(sizes, std::move(data));
}

TensorPtr create_positions_input(const ModelConfig& config) {
  const auto sizes = to_sizes({static_cast<int64_t>(config.token_seq_len)});
  std::vector<int64_t> data(config.token_seq_len);
  for (size_t i = 0; i < config.token_seq_len; ++i) {
    data[i] = static_cast<int64_t>(i);
  }
  return make_tensor_ptr<int64_t>(sizes, std::move(data));
}

TensorPtr create_fallback_text_embedding(const ModelConfig& config) {
  const auto sizes = to_sizes(
      {1,
       static_cast<int64_t>(config.token_seq_len),
       static_cast<int64_t>(config.text_embed_dim)});
  const size_t numel = 1ull * config.token_seq_len * config.text_embed_dim;
  std::vector<float> data(numel, 0.0f);
  return make_tensor_ptr<float>(
      sizes, std::move(data), {}, {}, ScalarType::BFloat16);
}

struct MethodTiming {
  double load_ms{0.0};
  double run_ms{0.0};
};

enum class MethodCategory { ENCODER, TOKEN_EMBEDDING, TEXT_DECODER, UNKNOWN };

MethodCategory categorize_method(const std::string& method_name) {
  std::string lower_name = method_name;
  std::transform(
      lower_name.begin(),
      lower_name.end(),
      lower_name.begin(),
      [](unsigned char c) { return std::tolower(c); });

  if (lower_name.find("vision") != std::string::npos ||
      lower_name.find("audio") != std::string::npos ||
      lower_name.find("encoder") != std::string::npos) {
    return MethodCategory::ENCODER;
  } else if (
      lower_name.find("token") != std::string::npos &&
      lower_name.find("embedding") != std::string::npos) {
    return MethodCategory::TOKEN_EMBEDDING;
  } else if (
      lower_name.find("text") != std::string::npos &&
      lower_name.find("decoder") != std::string::npos) {
    return MethodCategory::TEXT_DECODER;
  }
  return MethodCategory::UNKNOWN;
}

std::vector<EValue> create_inputs_for_method(
    const std::string& method_name,
    MethodCategory category,
    ModelType model_type,
    const ModelConfig& config,
    const EValue* token_output,
    std::vector<TensorPtr>& owned_inputs) {
  std::vector<EValue> inputs;

  switch (category) {
    case MethodCategory::ENCODER: {
      if (method_name.find("vision") != std::string::npos) {
        auto input = create_vision_input();
        owned_inputs.emplace_back(input);
        inputs.emplace_back(*input);
      } else if (method_name.find("audio") != std::string::npos) {
        auto input = create_audio_input();
        owned_inputs.emplace_back(input);
        inputs.emplace_back(*input);
      }
      break;
    }

    case MethodCategory::TOKEN_EMBEDDING: {
      auto token_ids = create_token_ids_input(config);
      owned_inputs.emplace_back(token_ids);
      inputs.emplace_back(*token_ids);
      break;
    }

    case MethodCategory::TEXT_DECODER: {
      if (token_output && token_output->isTensor()) {
        inputs.emplace_back(*token_output);
      } else {
        auto fallback_embedding = create_fallback_text_embedding(config);
        owned_inputs.emplace_back(fallback_embedding);
        inputs.emplace_back(*fallback_embedding);
      }

      auto positions = create_positions_input(config);
      owned_inputs.emplace_back(positions);
      inputs.emplace_back(*positions);
      break;
    }

    default:
      break;
  }

  return inputs;
}

Error execute_method(
    Module& module,
    const std::string& method_name,
    MethodCategory category,
    ModelType model_type,
    const ModelConfig& config,
    const EValue* token_output,
    MethodTiming& timing,
    EValue* output_storage = nullptr) {
  ET_LOG(Info, "Loading %s...", method_name.c_str());

  const auto load_start = Clock::now();
  const Error load_err = module.load_method(method_name);
  const auto load_end = Clock::now();
  if (load_err != Error::Ok) {
    std::cerr << "Failed to load method " << method_name << ": error code "
              << static_cast<int>(load_err) << std::endl;
    return load_err;
  }
  timing.load_ms = DurationMs(load_end - load_start).count();

  std::vector<TensorPtr> owned_inputs;
  std::vector<EValue> inputs = create_inputs_for_method(
      method_name, category, model_type, config, token_output, owned_inputs);

  const auto run_start = Clock::now();
  ET_LOG(Info, "%s running", method_name.c_str());
  Result<std::vector<EValue>> output_result =
      module.execute(method_name, inputs);
  ET_LOG(Info, "%s done", method_name.c_str());
  const auto run_end = Clock::now();
  timing.run_ms = DurationMs(run_end - run_start).count();

  if (output_result.error() != Error::Ok) {
    std::cerr << method_name << " execution failed: error code "
              << static_cast<int>(output_result.error()) << std::endl;
    return output_result.error();
  }

  const auto& outputs = output_result.get();
  if (!outputs.empty() && outputs[0].isTensor()) {
    print_tensor_summary(method_name + " output", outputs[0].toTensor());

    if (category == MethodCategory::ENCODER ||
        category == MethodCategory::TOKEN_EMBEDDING) {
      dump_tensor_to_file(method_name + "_output.bin", outputs[0].toTensor());
    }

    if (output_storage) {
      *output_storage = outputs[0];
    }
  }

  return Error::Ok;
}

} // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr
        << "Usage: " << argv[0]
        << " <model_name> <path/to/model.pte> <path/to/aoti_cuda_blob.ptd>"
        << std::endl;
    std::cerr << "  model_name: gemma3 or voxtral" << std::endl;
    return 1;
  }

  const std::string model_name = argv[1];
  const std::string program_path = argv[2];
  const std::string data_map_path = argv[3];

  const ModelType model_type = parse_model_type(model_name);
  if (model_type == ModelType::UNKNOWN) {
    std::cerr << "Unknown model type: " << model_name << std::endl;
    std::cerr << "Supported models: gemma3, voxtral" << std::endl;
    return 1;
  }

  const ModelConfig& config = model_configs.at(model_type);
  std::cout << "Running benchmark for model: " << config.name << std::endl;

  try {
    Module module(program_path, data_map_path);

    const auto program_load_start = Clock::now();
    const Error program_load_error = module.load();
    const auto program_load_end = Clock::now();
    if (program_load_error != Error::Ok) {
      std::cerr << "Failed to load ExecuTorch program: error code "
                << static_cast<int>(program_load_error) << std::endl;
      return 1;
    }
    const DurationMs program_load_latency =
        program_load_end - program_load_start;

    auto method_names_result = module.method_names();
    if (method_names_result.error() != Error::Ok) {
      std::cerr << "Failed to get method names: error code "
                << static_cast<int>(method_names_result.error()) << std::endl;
      return 1;
    }

    const auto& available_methods = method_names_result.get();

    std::cout << "Checking for expected methods..." << std::endl;
    std::vector<std::string> missing_methods;
    for (const auto& expected : config.expected_methods) {
      if (available_methods.find(expected) == available_methods.end()) {
        missing_methods.push_back(expected);
      } else {
        std::cout << "  ✓ " << expected << std::endl;
      }
    }

    if (!missing_methods.empty()) {
      std::cerr << "\nError: Missing expected methods:" << std::endl;
      for (const auto& missing : missing_methods) {
        std::cerr << "  ✗ " << missing << std::endl;
      }
      return 1;
    }

    std::map<std::string, MethodTiming> timings;
    EValue token_output;
    bool token_executed = false;

    for (const auto& method_name : config.expected_methods) {
      MethodCategory category = categorize_method(method_name);
      MethodTiming timing;

      const EValue* input_token_ptr =
          (category == MethodCategory::TEXT_DECODER && token_executed)
          ? &token_output
          : nullptr;

      EValue* output_storage = (category == MethodCategory::TOKEN_EMBEDDING)
          ? &token_output
          : nullptr;

      Error err = execute_method(
          module,
          method_name,
          category,
          model_type,
          config,
          input_token_ptr,
          timing,
          output_storage);

      if (err != Error::Ok) {
        return 1;
      }

      if (category == MethodCategory::TOKEN_EMBEDDING) {
        token_executed = true;
      }

      timings[method_name] = timing;
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n=== Benchmark Results ===" << std::endl;
    std::cout << "Program load latency (ms): " << program_load_latency.count()
              << std::endl;

    std::cout << "\nMethod load latency (ms):" << std::endl;
    for (const auto& [name, timing] : timings) {
      std::cout << "  " << name << ": " << timing.load_ms << std::endl;
    }

    std::cout << "\nRun latency (ms):" << std::endl;
    for (const auto& [name, timing] : timings) {
      std::cout << "  " << name << ": " << timing.run_ms << std::endl;
    }

    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Unhandled exception: " << ex.what() << std::endl;
    return 1;
  }
}
