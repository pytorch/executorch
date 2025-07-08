#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/flat_tensor/flat_tensor_data_map.h>
#include <executorch/extension/flat_tensor/serialize/serialize.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/runtime.h>

#include <iostream>
#include <map>
#include <optional>
#include <tuple>
#include <vector>

#include <gflags/gflags.h>

/*
 * This runner is intended to built and run as part of the backend test flow. It takes a
 * set of inputs from a flat_tensor-format file, runs each case, and then serializes the
 * outputs to a file, also in flat_tensor format.
 */
 
DEFINE_string(
    model_path,
    "model.pte",
    "Model serialized in flatbuffer format.");

DEFINE_string(
    input_path,
    "inputs.ptd",
    "Input tensors in flat tensor (ptd) format.");

DEFINE_string(
    output_path,
    "outputs.ptd",
    "Path to write output tensor in flat tensor (ptd) format.");

DEFINE_string(
    method,
    "forward",
    "The model method to run.");

using executorch::aten::Tensor;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::Result;
using executorch::extension::FileDataLoader;
using executorch::extension::FlatTensorDataMap;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::ET_RUNTIME_NAMESPACE::TensorLayout;

// Contains method inputs for a single run.
struct TestCase {
  std::map<int, TensorPtr> inputs;
};

std::map<std::string, TestCase> collect_test_cases(FlatTensorDataMap& input_map);
TensorPtr create_tensor(TensorLayout& layout, std::unique_ptr<char[], decltype(&free)> buffer);
Result<FlatTensorDataMap> load_input_data(FileDataLoader& loader);
std::optional<std::tuple<std::string, int>> parse_key(const std::string& key);
Result<std::vector<EValue>> run_test_case(Module& module, TestCase& test_case);
void store_outputs(std::map<std::string, TensorPtr>& output_map, const std::string& case_name, const std::vector<EValue>& outputs);

const int TensorAlignment = 16;

int main(int argc, char** argv){
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  executorch::runtime::runtime_init();

  // Load the model.
  Module model(FLAGS_model_path.c_str());
  auto load_method_error = model.load_method(FLAGS_method.c_str());
  if (load_method_error != Error::Ok) {
      std::cerr << "Failed to load method \"" << FLAGS_method << "\": " << static_cast<int>(load_method_error) << std::endl;
      return -1;
  }
  
  // Load the input tensor data. Note that the data loader has to live as long as the flat
  // tensor data map does.
  auto input_loader_result = FileDataLoader::from(FLAGS_input_path.c_str());
  if (!input_loader_result.ok()) {
    std::cerr << "Failed to open input file: error " << static_cast<int>(input_loader_result.error()) << std::endl;
  }

  auto load_result = load_input_data(*input_loader_result);
  if (!load_result.ok()) {
    return -1;   
  }
  auto input_map = std::move(load_result.get());

  auto cases = collect_test_cases(input_map);
  std::map<std::string, TensorPtr> output_map;

  // Run each case and store the outputs.
  for (auto& [name, test_case] : cases) {
    auto result = run_test_case(model, test_case);
    if (!result.ok()) {
      std::cerr << "Failed to run test case \"" << name << "\": " << static_cast<int>(result.error()) << std::endl;
      return -1;
    }

    store_outputs(output_map, name, result.get());
  }

  // Create a map of Tensor (unowned), rather than TensorPtr (owned).
  std::map<std::string, Tensor> output_map_tensors;
  for (auto& [key, value] : output_map) {
    output_map_tensors.emplace(key, *value);
  }

  // Write the output data in .ptd format.
  auto save_result = executorch::extension::flat_tensor::save_ptd(
    FLAGS_output_path.c_str(),
    output_map_tensors,
    TensorAlignment
  );

  if (save_result != Error::Ok) {
    std::cerr << "Failed to save outputs: " << static_cast<int>(save_result) << std::endl;
    return -1;
  }

  std::cout << "Successfully wrote output tensors to " << FLAGS_output_path << "." << std::endl; 
}

// Group inputs by test case and build tensors.
std::map<std::string, TestCase> collect_test_cases(FlatTensorDataMap& input_map) {
  std::map<std::string, TestCase> cases;

  for (auto i = 0u; i < input_map.get_num_keys().get(); i++) {
    auto key = input_map.get_key(i).get();

    // Split key into test_case : input index
    auto [test_case_name, input_index] = *parse_key(key);

    // Get or create the test case instance.
    auto& test_case = cases[test_case_name];
    
    // Create a tensor from the layout and data.
    auto tensor_layout = input_map.get_tensor_layout(key).get();
    auto tensor_data = std::unique_ptr<char[], decltype(&free)>((char*) malloc(tensor_layout.nbytes()), free);
    auto load_result = input_map.load_data_into(key, tensor_data.get(), tensor_layout.nbytes());
    if (load_result != Error::Ok) {
      std::cerr << "Load failed: " << static_cast<int>(load_result) << std::endl;
      exit(-1);
    }

    auto input_tensor = create_tensor(tensor_layout, std::move(tensor_data));
    test_case.inputs[input_index] = std::move(input_tensor);
  }

  return cases;
}

// Create a tensor from a layout and data blob.
TensorPtr create_tensor(TensorLayout& layout, std::unique_ptr<char[], decltype(&free)> buffer) {
  // Sizes and dim order are have different types in TensorLayout vs Tensor.
  std::vector<executorch::aten::SizesType> sizes;
  for (auto x : layout.sizes()) {
    sizes.push_back(x);
  }
  std::vector<executorch::aten::DimOrderType> dim_order;
  for (auto x : layout.dim_order()) {
    dim_order.push_back(x);
  }

  auto raw_data = buffer.release();

  return executorch::extension::make_tensor_ptr(
    sizes,
    raw_data,
    dim_order,
    {}, // Strides - infer from sizes + dim order.
    layout.scalar_type(),
    exec_aten::TensorShapeDynamism::STATIC,
    [](void* ptr) {
      free(ptr);
    }
  );
}

// Load the input data (in .ptd file format) from the given path.
Result<FlatTensorDataMap> load_input_data(FileDataLoader& loader) {
  auto input_data_map_load_result = FlatTensorDataMap::load(&loader);
  if (!input_data_map_load_result.ok()) {
    std::cerr << "Failed to open load input data map: error " << static_cast<int>(input_data_map_load_result.error()) << std::endl;
  }

  return input_data_map_load_result;
}

// Parse a string key of the form "test_case:input index". Returns a tuple of the test case name
// and input index.
std::optional<std::tuple<std::string, int>> parse_key(const std::string& key) {
  auto delimiter = key.find(":");
  if (delimiter == std::string::npos) { return std::nullopt; }

  auto test_case = key.substr(0, delimiter);
  auto index_str = key.substr(delimiter + 1);
  auto index = std::stoi(index_str);

  return {{ test_case, index }};
}

// Run a given test case and return the resulting output values.
Result<std::vector<EValue>> run_test_case(Module& module, TestCase& test_case) {
  for (auto& [index, value] : test_case.inputs) {
    auto set_input_error = module.set_input(FLAGS_method, value, index);
    if (set_input_error != Error::Ok) {
      std::cerr << "Failed to set input " << index << ": " << static_cast<int>(set_input_error) << "." << std::endl;
    }
  }

  return module.execute(FLAGS_method.c_str());
}

// Store output tensors into the named data map.
void store_outputs(
    std::map<std::string, TensorPtr>& output_map, 
    const std::string& case_name, 
    const std::vector<EValue>& outputs) {
  // Because the outputs are likely memory planned, we need to clone the tensor
  // here to avoid having the data clobbered by the next run.
  
  for (auto i = 0u; i < outputs.size(); i++) {
    if (!outputs[i].isTensor()) {
      continue;
    }

    auto key_name = case_name + ":" + std::to_string(i);
    auto& tensor = outputs[i].toTensor();
    
    // Copy tensor storage.
    auto tensor_memory = malloc(tensor.nbytes());
    memcpy(tensor_memory, tensor.const_data_ptr(), tensor.nbytes());

    // Copy tensor metadata.
    std::vector<executorch::aten::SizesType> sizes(
      tensor.sizes().begin(),
      tensor.sizes().end()
    );

    std::vector<executorch::aten::DimOrderType> dim_order(
      tensor.dim_order().begin(),
      tensor.dim_order().end()
    );

    output_map.emplace(key_name, executorch::extension::make_tensor_ptr(
      sizes,
      tensor_memory,
      dim_order,
      {}, // Strides - implicit
      tensor.scalar_type(),
      exec_aten::TensorShapeDynamism::STATIC,
      [](void* ptr) {
        free(ptr);
      }
    ));
  }
}
