
#include <torch/csrc/inductor/aoti_include/cuda.h>
// Definition of AOTI runtime interface functions

#include <torch/csrc/inductor/aoti_runtime/interface.h>
#include <torch/csrc/inductor/aoti_runtime/model_container.h>

#include <iostream>
#include <vector>

#define CONVERT_EXCEPTION_TO_ERROR_CODE(...)      \
  try {                                           \
    __VA_ARGS__                                   \
  } catch (const std::exception& e) {             \
    std::cerr << "Error: " << e.what() << '\n';   \
    return AOTI_RUNTIME_FAILURE;                  \
  } catch (...) {                                 \
    std::cerr << "Unknown exception occurred.\n"; \
    return AOTI_RUNTIME_FAILURE;                  \
  }                                               \
  return AOTI_RUNTIME_SUCCESS;

#define AOTI_VECTOR_SIZE_CHECK(actual_size, expected_size, name)  \
  do {                                                            \
    AOTI_RUNTIME_CHECK(                                           \
        actual_size == expected_size,                             \
        "expected " + std::string(name) + " vector size to be " + \
            std::to_string(expected_size) + ", but got " +        \
            std::to_string(actual_size));                         \
  } while (0)

// AOTInductor uses at::addmm_out, which doesn't supports
// arguments that requires gradient. For this reason, we
// enforce no_grad context for run APIs.
//
// A RAII, thread local (!) guard that enables or disables grad mode upon
// construction, and sets it back to the original value upon destruction.
struct AOTINoGradGuard {
  AOTINoGradGuard() {
    aoti_torch_grad_mode_set_enabled(false);
  }
  AOTINoGradGuard(const AOTINoGradGuard&) = delete;
  AOTINoGradGuard(AOTINoGradGuard&&) noexcept = delete;
  ~AOTINoGradGuard() {
    aoti_torch_grad_mode_set_enabled(prev_mode);
  }
  AOTINoGradGuard& operator=(const AOTINoGradGuard&) = delete;
  AOTINoGradGuard& operator=(AOTINoGradGuard&&) noexcept = delete;
  bool prev_mode{aoti_torch_grad_mode_is_enabled()};
};

extern "C" {

AOTIRuntimeError AOTInductorModelContainerCreate(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    bool is_cpu,
    const char* cubin_dir) {
      return AOTInductorModelContainerCreateWithDevice(
        container_handle,
        num_models,
        is_cpu ? "cpu" : "cuda",
        cubin_dir);
}

AOTIRuntimeError AOTInductorModelContainerCreateWithDevice(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    const char* device_str,
    const char* cubin_dir) {
  if (num_models == 0) {
    std::cerr << "Error: num_models must be positive, but got 0\n";
    return AOTI_RUNTIME_FAILURE;
  }
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    std::optional<std::string> cubin_dir_opt;
    if (cubin_dir != nullptr) {
      cubin_dir_opt.emplace(cubin_dir);
    }
    auto* container = new torch::aot_inductor::AOTInductorModelContainer(
        num_models, std::string(device_str), cubin_dir_opt);
    *container_handle =
        reinterpret_cast<AOTInductorModelContainerHandle>(container);
  })
}

AOTIRuntimeError AOTInductorModelContainerDelete(
    AOTInductorModelContainerHandle container_handle) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* container =
        reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
            container_handle);
    delete container;
  });
}

AOTIRuntimeError AOTInductorModelContainerRun(
    AOTInductorModelContainerHandle container_handle,
    AtenTensorHandle* input_handles, // array of input AtenTensorHandle; handles
                                     // are stolen; the array itself is borrowed
    size_t num_inputs,
    AtenTensorHandle*
        output_handles, // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    size_t num_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  AOTI_VECTOR_SIZE_CHECK(num_inputs, container->num_inputs(), "inputs");
  AOTI_VECTOR_SIZE_CHECK(num_outputs, container->num_outputs(), "outputs");

  auto stream =
      reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    container->run(
        input_handles, output_handles, stream, proxy_executor_handle);
  })
}

AOTIRuntimeError AOTInductorModelContainerRunSingleThreaded(
    AOTInductorModelContainerHandle container_handle,
    AtenTensorHandle* input_handles, // array of input AtenTensorHandle; handles
                                     // are stolen; the array itself is borrowed
    size_t num_inputs,
    AtenTensorHandle*
        output_handles, // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    size_t num_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  AOTI_VECTOR_SIZE_CHECK(num_inputs, container->num_inputs(), "inputs");
  AOTI_VECTOR_SIZE_CHECK(num_outputs, container->num_outputs(), "outputs");

  auto stream =
      reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    container->run_single_threaded(
        input_handles, output_handles, stream, proxy_executor_handle);
  })
}

AOTIRuntimeError AOTInductorModelContainerGetNumConstants(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_constants) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *num_constants = container->num_constants(); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantName(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    const char** name) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *name = container->constant_name(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantOriginalFQN(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    const char** original_fqn) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *original_fqn = container->constant_original_fqn(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantFromFolded(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    bool* from_folded) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *from_folded = container->constant_from_folded(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantType(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    int32_t* type) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *type = container->constant_type(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantDtype(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    int32_t* dtype) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *dtype = container->constant_dtype(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantDataSize(
  AOTInductorModelContainerHandle container_handle,
  size_t idx,
  size_t* data_size) {
  auto* container =
    reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
        container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *data_size = container->constant_data_size(idx); })
}

AOTIRuntimeError AOTInductorModelContainerExtractConstantsMap(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto constants_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { const auto ret = container->extract_constants_map(use_inactive);
      for (const auto& pair: ret) {
        constants_map->emplace(pair.first, pair.second);
      }
    })
}

AOTIRuntimeError AOTInductorModelContainerUpdateUserManagedConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive,
    bool validate_full_update) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto input_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->update_constant_buffer(
        *input_map, use_inactive, validate_full_update, /* user_managed = */ true);
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive,
    bool validate_full_update) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto input_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->update_constant_buffer(
        *input_map, use_inactive, validate_full_update);
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateInactiveConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle) {
  return AOTInductorModelContainerUpdateConstantBuffer(container_handle,
          constant_map_handle,
          /*use_inactive*/ true,
          /*validate_full_update*/ true);
}

AOTIRuntimeError AOTInductorModelContainerFreeInactiveConstantBuffer(
    AOTInductorModelContainerHandle container_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->free_inactive_constant_buffer();
  })
}

AOTIRuntimeError AOTInductorModelContainerRunConstantFolding(
    AOTInductorModelContainerHandle container_handle,
    bool use_inactive,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto stream =
      reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    container->run_const_fold(use_inactive, stream, proxy_executor_handle);
  })
}

AOTIRuntimeError AOTInductorModelContainerSwapConstantBuffer(
    AOTInductorModelContainerHandle container_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->swap_constant_buffer();
  })
}

AOTIRuntimeError AOTInductorModelContainerGetNumInputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_inputs) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_num_inputs = container->num_inputs(); })
}

AOTIRuntimeError AOTInductorModelContainerGetInputName(
    AOTInductorModelContainerHandle container_handle,
    size_t input_idx,
    const char** ret_input_names) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_input_names = container->input_name(input_idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetNumOutputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_outputs) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_num_outputs = container->num_outputs(); })
}

AOTIRuntimeError AOTInductorModelContainerGetOutputName(
    AOTInductorModelContainerHandle container_handle,
    size_t output_idx,
    const char** ret_output_names) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_output_names = container->output_name(output_idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetCallSpec(
    AOTInductorModelContainerHandle container_handle,
    const char** in_spec,
    const char** out_spec) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    *in_spec = container->get_in_spec();
    *out_spec = container->get_out_spec();
  })
}

AOTIRuntimeError AOTInductorModelCreate(
    AOTInductorModelHandle* model_handle,
    AOTInductorConstantMapHandle constant_map_handle){
    CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto constant_map = std::make_shared<torch::aot_inductor::ConstantMap>();
      auto constant_array = std::make_shared<std::vector<torch::aot_inductor::ConstantHandle>>();
      auto input_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);

      auto model = new torch::aot_inductor::AOTInductorModel(
          constant_map,
          constant_array,
          "cpu", // device_str is hardcoded, as AOTInductorModelCreate is only use for CPU models
          ""
      );

      if (input_map) {
        for (auto const& kv : *input_map) {
          constant_map->emplace(kv.first, kv.second);
        }
      } else {
        model->load_constants();
      }

      *model_handle = reinterpret_cast<AOTInductorModelHandle>(model);
    })}

AOTIRuntimeError AOTInductorModelRun(
    AOTInductorModelHandle model_handle,
    AtenTensorHandle* input_handles,
    AtenTensorHandle* output_handles) {
  auto model =
      reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    model->run_impl(
        input_handles,
        output_handles,
        (torch::aot_inductor::DeviceStreamType) nullptr,
        nullptr);
  })
}

AOTIRuntimeError AOTInductorModelDelete(AOTInductorModelHandle model_handle){
    CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto model = reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(
          model_handle);
      delete model;
    })}

AOTIRuntimeError AOTInductorModelGetNumOutputs(
    AOTInductorModelHandle model_handle,
    size_t* ret_num_outputs) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto model = reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
      *ret_num_outputs = model->num_outputs();
  })
}

AOTIRuntimeError AOTInductorModelUpdateConstantsMap(
    AOTInductorModelHandle model_handle,
    AOTInductorConstantMapHandle constant_map_handle) {
  auto model =
      reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto constant_map = std::make_shared<torch::aot_inductor::ConstantMap>();
    auto input_map =
        reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(
            constant_map_handle);

    for (auto const& kv : *input_map) {
      constant_map->emplace(kv.first, kv.second);
    }
    model->update_constants_map(std::move(constant_map));
  })
}

} // extern "C"


#define CUDA_DRIVER_CHECK(EXPR)                    \
do {                                               \
    CUresult code = EXPR;                          \
    const char *msg;                               \
    CUresult code_get_error = cuGetErrorString(code, &msg); \
    if (code_get_error != CUDA_SUCCESS) {          \
        throw std::runtime_error(                  \
            std::string("CUDA driver error: ") +   \
            std::string("invalid error code!"));   \
    }                                              \
    if (code != CUDA_SUCCESS) {                    \
        throw std::runtime_error(                  \
            std::string("CUDA driver error: ") +   \
            std::string(msg));                     \
    }                                              \
} while (0);

static inline CUfunction loadKernel(
        std::string filePath,
        const std::string &funcName,
        uint32_t sharedMemBytes,
        const std::optional<std::string> &cubinDir = std::nullopt) {
    if (cubinDir) {
        std::filesystem::path p1{*cubinDir};
        std::filesystem::path p2{filePath};
        filePath = (p1 / p2.filename()).string();
    }

    CUmodule mod;
    CUfunction func;
    CUDA_DRIVER_CHECK(cuModuleLoad(&mod, filePath.c_str()));
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&func, mod, funcName.c_str()));
    if (sharedMemBytes > 0) {
        CUDA_DRIVER_CHECK(cuFuncSetAttribute(
            func,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            sharedMemBytes
        ))
    }
    return func;
}

static inline CUfunction loadKernel(const void* start, const std::string &funcName, uint32_t sharedMemBytes) {
    CUmodule mod;
    CUfunction func;
    CUDA_DRIVER_CHECK(cuModuleLoadData(&mod, start));
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&func, mod, funcName.c_str()));
    if (sharedMemBytes > 0) {
        CUDA_DRIVER_CHECK(cuFuncSetAttribute(
            func,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            sharedMemBytes
        ))
    }
    return func;
}

static inline void launchKernel(
        CUfunction func,
        uint32_t gridX,
        uint32_t gridY,
        uint32_t gridZ,
        uint32_t numWarps,
        uint32_t sharedMemBytes,
        void* args[],
        cudaStream_t stream) {
    CUDA_DRIVER_CHECK(cuLaunchKernel(
        func, gridX, gridY, gridZ, 32*numWarps, 1, 1, sharedMemBytes, stream, args, nullptr
    ));
}
CACHE_TORCH_DTYPE(float32);
CACHE_TORCH_DEVICE(cuda);
CACHE_TORCH_LAYOUT(strided);
namespace torch::aot_inductor {
namespace {
class AOTInductorModelKernels : public AOTInductorModelKernelsBase {
  public:
    CUfunction triton_per_fused__native_batch_norm_legit_no_training_hardtanh_mean_25{nullptr};
    CUfunction triton_poi_fused__native_batch_norm_legit_no_training_10{nullptr};
    CUfunction triton_poi_fused__native_batch_norm_legit_no_training_14{nullptr};
    CUfunction triton_poi_fused__native_batch_norm_legit_no_training_17{nullptr};
    CUfunction triton_poi_fused__native_batch_norm_legit_no_training_21{nullptr};
    CUfunction triton_poi_fused__native_batch_norm_legit_no_training_24{nullptr};
    CUfunction triton_poi_fused__native_batch_norm_legit_no_training_3{nullptr};
    CUfunction triton_poi_fused__native_batch_norm_legit_no_training_6{nullptr};
    CUfunction triton_poi_fused__native_batch_norm_legit_no_training_add_12{nullptr};
    CUfunction triton_poi_fused__native_batch_norm_legit_no_training_add_16{nullptr};
    CUfunction triton_poi_fused__native_batch_norm_legit_no_training_add_19{nullptr};
    CUfunction triton_poi_fused__native_batch_norm_legit_no_training_add_23{nullptr};
    CUfunction triton_poi_fused__native_batch_norm_legit_no_training_add_8{nullptr};
    CUfunction triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11{nullptr};
    CUfunction triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13{nullptr};
    CUfunction triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15{nullptr};
    CUfunction triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_18{nullptr};
    CUfunction triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_2{nullptr};
    CUfunction triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_20{nullptr};
    CUfunction triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22{nullptr};
    CUfunction triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_4{nullptr};
    CUfunction triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_5{nullptr};
    CUfunction triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_7{nullptr};
    CUfunction triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9{nullptr};
    CUfunction triton_poi_fused_convolution_0{nullptr};
    CUfunction triton_poi_fused_convolution_1{nullptr};
    CUfunction triton_poi_fused_permute_copy_26{nullptr};
};
}  // namespace



AOTInductorModel::AOTInductorModel(std::shared_ptr<ConstantMap> constants_map,
                                   std::shared_ptr<std::vector<ConstantHandle>> constants_array,
                                   const std::string& device_str,
                                   std::optional<std::string> cubin_dir)
    : AOTInductorModelBase(1,
                           1,
                           262,
                           device_str,
                           std::move(cubin_dir),
                           true) {
    inputs_info_[0].name = "arg262_1";
    constants_info_[0].name = "mv2_features_0_0_weight";
    constants_info_[0].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[0].offset = 0;
    constants_info_[0].data_size = 3456;
    constants_info_[0].from_folded = false;
    constants_info_[0].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[0].shape = {32, 3, 3, 3};
    constants_info_[0].stride = {27, 9, 3, 1};
    constants_info_[0].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[0].original_fqn = "mv2.features.0.0.weight";
    constants_info_[1].name = "mv2_features_0_1_weight";
    constants_info_[1].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[1].offset = 0;
    constants_info_[1].data_size = 128;
    constants_info_[1].from_folded = false;
    constants_info_[1].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[1].shape = {32};
    constants_info_[1].stride = {1};
    constants_info_[1].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[1].original_fqn = "mv2.features.0.1.weight";
    constants_info_[2].name = "mv2_features_0_1_bias";
    constants_info_[2].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[2].offset = 0;
    constants_info_[2].data_size = 128;
    constants_info_[2].from_folded = false;
    constants_info_[2].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[2].shape = {32};
    constants_info_[2].stride = {1};
    constants_info_[2].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[2].original_fqn = "mv2.features.0.1.bias";
    constants_info_[3].name = "mv2_features_1_conv_0_0_weight";
    constants_info_[3].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[3].offset = 0;
    constants_info_[3].data_size = 1152;
    constants_info_[3].from_folded = false;
    constants_info_[3].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[3].shape = {32, 1, 3, 3};
    constants_info_[3].stride = {9, 9, 3, 1};
    constants_info_[3].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[3].original_fqn = "mv2.features.1.conv.0.0.weight";
    constants_info_[4].name = "mv2_features_1_conv_0_1_weight";
    constants_info_[4].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[4].offset = 0;
    constants_info_[4].data_size = 128;
    constants_info_[4].from_folded = false;
    constants_info_[4].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[4].shape = {32};
    constants_info_[4].stride = {1};
    constants_info_[4].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[4].original_fqn = "mv2.features.1.conv.0.1.weight";
    constants_info_[5].name = "mv2_features_1_conv_0_1_bias";
    constants_info_[5].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[5].offset = 0;
    constants_info_[5].data_size = 128;
    constants_info_[5].from_folded = false;
    constants_info_[5].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[5].shape = {32};
    constants_info_[5].stride = {1};
    constants_info_[5].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[5].original_fqn = "mv2.features.1.conv.0.1.bias";
    constants_info_[6].name = "mv2_features_1_conv_1_weight";
    constants_info_[6].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[6].offset = 0;
    constants_info_[6].data_size = 2048;
    constants_info_[6].from_folded = false;
    constants_info_[6].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[6].shape = {16, 32, 1, 1};
    constants_info_[6].stride = {32, 1, 1, 1};
    constants_info_[6].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[6].original_fqn = "mv2.features.1.conv.1.weight";
    constants_info_[7].name = "mv2_features_1_conv_2_weight";
    constants_info_[7].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[7].offset = 0;
    constants_info_[7].data_size = 64;
    constants_info_[7].from_folded = false;
    constants_info_[7].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[7].shape = {16};
    constants_info_[7].stride = {1};
    constants_info_[7].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[7].original_fqn = "mv2.features.1.conv.2.weight";
    constants_info_[8].name = "mv2_features_1_conv_2_bias";
    constants_info_[8].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[8].offset = 0;
    constants_info_[8].data_size = 64;
    constants_info_[8].from_folded = false;
    constants_info_[8].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[8].shape = {16};
    constants_info_[8].stride = {1};
    constants_info_[8].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[8].original_fqn = "mv2.features.1.conv.2.bias";
    constants_info_[9].name = "mv2_features_2_conv_0_0_weight";
    constants_info_[9].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[9].offset = 0;
    constants_info_[9].data_size = 6144;
    constants_info_[9].from_folded = false;
    constants_info_[9].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[9].shape = {96, 16, 1, 1};
    constants_info_[9].stride = {16, 1, 1, 1};
    constants_info_[9].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[9].original_fqn = "mv2.features.2.conv.0.0.weight";
    constants_info_[10].name = "mv2_features_2_conv_0_1_weight";
    constants_info_[10].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[10].offset = 0;
    constants_info_[10].data_size = 384;
    constants_info_[10].from_folded = false;
    constants_info_[10].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[10].shape = {96};
    constants_info_[10].stride = {1};
    constants_info_[10].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[10].original_fqn = "mv2.features.2.conv.0.1.weight";
    constants_info_[11].name = "mv2_features_2_conv_0_1_bias";
    constants_info_[11].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[11].offset = 0;
    constants_info_[11].data_size = 384;
    constants_info_[11].from_folded = false;
    constants_info_[11].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[11].shape = {96};
    constants_info_[11].stride = {1};
    constants_info_[11].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[11].original_fqn = "mv2.features.2.conv.0.1.bias";
    constants_info_[12].name = "mv2_features_2_conv_1_0_weight";
    constants_info_[12].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[12].offset = 0;
    constants_info_[12].data_size = 3456;
    constants_info_[12].from_folded = false;
    constants_info_[12].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[12].shape = {96, 1, 3, 3};
    constants_info_[12].stride = {9, 9, 3, 1};
    constants_info_[12].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[12].original_fqn = "mv2.features.2.conv.1.0.weight";
    constants_info_[13].name = "mv2_features_2_conv_1_1_weight";
    constants_info_[13].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[13].offset = 0;
    constants_info_[13].data_size = 384;
    constants_info_[13].from_folded = false;
    constants_info_[13].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[13].shape = {96};
    constants_info_[13].stride = {1};
    constants_info_[13].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[13].original_fqn = "mv2.features.2.conv.1.1.weight";
    constants_info_[14].name = "mv2_features_2_conv_1_1_bias";
    constants_info_[14].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[14].offset = 0;
    constants_info_[14].data_size = 384;
    constants_info_[14].from_folded = false;
    constants_info_[14].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[14].shape = {96};
    constants_info_[14].stride = {1};
    constants_info_[14].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[14].original_fqn = "mv2.features.2.conv.1.1.bias";
    constants_info_[15].name = "mv2_features_2_conv_2_weight";
    constants_info_[15].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[15].offset = 0;
    constants_info_[15].data_size = 9216;
    constants_info_[15].from_folded = false;
    constants_info_[15].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[15].shape = {24, 96, 1, 1};
    constants_info_[15].stride = {96, 1, 1, 1};
    constants_info_[15].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[15].original_fqn = "mv2.features.2.conv.2.weight";
    constants_info_[16].name = "mv2_features_2_conv_3_weight";
    constants_info_[16].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[16].offset = 0;
    constants_info_[16].data_size = 96;
    constants_info_[16].from_folded = false;
    constants_info_[16].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[16].shape = {24};
    constants_info_[16].stride = {1};
    constants_info_[16].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[16].original_fqn = "mv2.features.2.conv.3.weight";
    constants_info_[17].name = "mv2_features_2_conv_3_bias";
    constants_info_[17].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[17].offset = 0;
    constants_info_[17].data_size = 96;
    constants_info_[17].from_folded = false;
    constants_info_[17].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[17].shape = {24};
    constants_info_[17].stride = {1};
    constants_info_[17].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[17].original_fqn = "mv2.features.2.conv.3.bias";
    constants_info_[18].name = "mv2_features_3_conv_0_0_weight";
    constants_info_[18].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[18].offset = 0;
    constants_info_[18].data_size = 13824;
    constants_info_[18].from_folded = false;
    constants_info_[18].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[18].shape = {144, 24, 1, 1};
    constants_info_[18].stride = {24, 1, 1, 1};
    constants_info_[18].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[18].original_fqn = "mv2.features.3.conv.0.0.weight";
    constants_info_[19].name = "mv2_features_3_conv_0_1_weight";
    constants_info_[19].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[19].offset = 0;
    constants_info_[19].data_size = 576;
    constants_info_[19].from_folded = false;
    constants_info_[19].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[19].shape = {144};
    constants_info_[19].stride = {1};
    constants_info_[19].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[19].original_fqn = "mv2.features.3.conv.0.1.weight";
    constants_info_[20].name = "mv2_features_3_conv_0_1_bias";
    constants_info_[20].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[20].offset = 0;
    constants_info_[20].data_size = 576;
    constants_info_[20].from_folded = false;
    constants_info_[20].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[20].shape = {144};
    constants_info_[20].stride = {1};
    constants_info_[20].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[20].original_fqn = "mv2.features.3.conv.0.1.bias";
    constants_info_[21].name = "mv2_features_3_conv_1_0_weight";
    constants_info_[21].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[21].offset = 0;
    constants_info_[21].data_size = 5184;
    constants_info_[21].from_folded = false;
    constants_info_[21].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[21].shape = {144, 1, 3, 3};
    constants_info_[21].stride = {9, 9, 3, 1};
    constants_info_[21].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[21].original_fqn = "mv2.features.3.conv.1.0.weight";
    constants_info_[22].name = "mv2_features_3_conv_1_1_weight";
    constants_info_[22].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[22].offset = 0;
    constants_info_[22].data_size = 576;
    constants_info_[22].from_folded = false;
    constants_info_[22].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[22].shape = {144};
    constants_info_[22].stride = {1};
    constants_info_[22].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[22].original_fqn = "mv2.features.3.conv.1.1.weight";
    constants_info_[23].name = "mv2_features_3_conv_1_1_bias";
    constants_info_[23].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[23].offset = 0;
    constants_info_[23].data_size = 576;
    constants_info_[23].from_folded = false;
    constants_info_[23].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[23].shape = {144};
    constants_info_[23].stride = {1};
    constants_info_[23].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[23].original_fqn = "mv2.features.3.conv.1.1.bias";
    constants_info_[24].name = "mv2_features_3_conv_2_weight";
    constants_info_[24].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[24].offset = 0;
    constants_info_[24].data_size = 13824;
    constants_info_[24].from_folded = false;
    constants_info_[24].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[24].shape = {24, 144, 1, 1};
    constants_info_[24].stride = {144, 1, 1, 1};
    constants_info_[24].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[24].original_fqn = "mv2.features.3.conv.2.weight";
    constants_info_[25].name = "mv2_features_3_conv_3_weight";
    constants_info_[25].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[25].offset = 0;
    constants_info_[25].data_size = 96;
    constants_info_[25].from_folded = false;
    constants_info_[25].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[25].shape = {24};
    constants_info_[25].stride = {1};
    constants_info_[25].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[25].original_fqn = "mv2.features.3.conv.3.weight";
    constants_info_[26].name = "mv2_features_3_conv_3_bias";
    constants_info_[26].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[26].offset = 0;
    constants_info_[26].data_size = 96;
    constants_info_[26].from_folded = false;
    constants_info_[26].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[26].shape = {24};
    constants_info_[26].stride = {1};
    constants_info_[26].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[26].original_fqn = "mv2.features.3.conv.3.bias";
    constants_info_[27].name = "mv2_features_4_conv_0_0_weight";
    constants_info_[27].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[27].offset = 0;
    constants_info_[27].data_size = 13824;
    constants_info_[27].from_folded = false;
    constants_info_[27].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[27].shape = {144, 24, 1, 1};
    constants_info_[27].stride = {24, 1, 1, 1};
    constants_info_[27].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[27].original_fqn = "mv2.features.4.conv.0.0.weight";
    constants_info_[28].name = "mv2_features_4_conv_0_1_weight";
    constants_info_[28].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[28].offset = 0;
    constants_info_[28].data_size = 576;
    constants_info_[28].from_folded = false;
    constants_info_[28].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[28].shape = {144};
    constants_info_[28].stride = {1};
    constants_info_[28].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[28].original_fqn = "mv2.features.4.conv.0.1.weight";
    constants_info_[29].name = "mv2_features_4_conv_0_1_bias";
    constants_info_[29].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[29].offset = 0;
    constants_info_[29].data_size = 576;
    constants_info_[29].from_folded = false;
    constants_info_[29].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[29].shape = {144};
    constants_info_[29].stride = {1};
    constants_info_[29].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[29].original_fqn = "mv2.features.4.conv.0.1.bias";
    constants_info_[30].name = "mv2_features_4_conv_1_0_weight";
    constants_info_[30].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[30].offset = 0;
    constants_info_[30].data_size = 5184;
    constants_info_[30].from_folded = false;
    constants_info_[30].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[30].shape = {144, 1, 3, 3};
    constants_info_[30].stride = {9, 9, 3, 1};
    constants_info_[30].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[30].original_fqn = "mv2.features.4.conv.1.0.weight";
    constants_info_[31].name = "mv2_features_4_conv_1_1_weight";
    constants_info_[31].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[31].offset = 0;
    constants_info_[31].data_size = 576;
    constants_info_[31].from_folded = false;
    constants_info_[31].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[31].shape = {144};
    constants_info_[31].stride = {1};
    constants_info_[31].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[31].original_fqn = "mv2.features.4.conv.1.1.weight";
    constants_info_[32].name = "mv2_features_4_conv_1_1_bias";
    constants_info_[32].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[32].offset = 0;
    constants_info_[32].data_size = 576;
    constants_info_[32].from_folded = false;
    constants_info_[32].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[32].shape = {144};
    constants_info_[32].stride = {1};
    constants_info_[32].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[32].original_fqn = "mv2.features.4.conv.1.1.bias";
    constants_info_[33].name = "mv2_features_4_conv_2_weight";
    constants_info_[33].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[33].offset = 0;
    constants_info_[33].data_size = 18432;
    constants_info_[33].from_folded = false;
    constants_info_[33].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[33].shape = {32, 144, 1, 1};
    constants_info_[33].stride = {144, 1, 1, 1};
    constants_info_[33].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[33].original_fqn = "mv2.features.4.conv.2.weight";
    constants_info_[34].name = "mv2_features_4_conv_3_weight";
    constants_info_[34].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[34].offset = 0;
    constants_info_[34].data_size = 128;
    constants_info_[34].from_folded = false;
    constants_info_[34].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[34].shape = {32};
    constants_info_[34].stride = {1};
    constants_info_[34].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[34].original_fqn = "mv2.features.4.conv.3.weight";
    constants_info_[35].name = "mv2_features_4_conv_3_bias";
    constants_info_[35].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[35].offset = 0;
    constants_info_[35].data_size = 128;
    constants_info_[35].from_folded = false;
    constants_info_[35].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[35].shape = {32};
    constants_info_[35].stride = {1};
    constants_info_[35].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[35].original_fqn = "mv2.features.4.conv.3.bias";
    constants_info_[36].name = "mv2_features_5_conv_0_0_weight";
    constants_info_[36].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[36].offset = 0;
    constants_info_[36].data_size = 24576;
    constants_info_[36].from_folded = false;
    constants_info_[36].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[36].shape = {192, 32, 1, 1};
    constants_info_[36].stride = {32, 1, 1, 1};
    constants_info_[36].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[36].original_fqn = "mv2.features.5.conv.0.0.weight";
    constants_info_[37].name = "mv2_features_5_conv_0_1_weight";
    constants_info_[37].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[37].offset = 0;
    constants_info_[37].data_size = 768;
    constants_info_[37].from_folded = false;
    constants_info_[37].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[37].shape = {192};
    constants_info_[37].stride = {1};
    constants_info_[37].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[37].original_fqn = "mv2.features.5.conv.0.1.weight";
    constants_info_[38].name = "mv2_features_5_conv_0_1_bias";
    constants_info_[38].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[38].offset = 0;
    constants_info_[38].data_size = 768;
    constants_info_[38].from_folded = false;
    constants_info_[38].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[38].shape = {192};
    constants_info_[38].stride = {1};
    constants_info_[38].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[38].original_fqn = "mv2.features.5.conv.0.1.bias";
    constants_info_[39].name = "mv2_features_5_conv_1_0_weight";
    constants_info_[39].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[39].offset = 0;
    constants_info_[39].data_size = 6912;
    constants_info_[39].from_folded = false;
    constants_info_[39].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[39].shape = {192, 1, 3, 3};
    constants_info_[39].stride = {9, 9, 3, 1};
    constants_info_[39].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[39].original_fqn = "mv2.features.5.conv.1.0.weight";
    constants_info_[40].name = "mv2_features_5_conv_1_1_weight";
    constants_info_[40].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[40].offset = 0;
    constants_info_[40].data_size = 768;
    constants_info_[40].from_folded = false;
    constants_info_[40].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[40].shape = {192};
    constants_info_[40].stride = {1};
    constants_info_[40].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[40].original_fqn = "mv2.features.5.conv.1.1.weight";
    constants_info_[41].name = "mv2_features_5_conv_1_1_bias";
    constants_info_[41].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[41].offset = 0;
    constants_info_[41].data_size = 768;
    constants_info_[41].from_folded = false;
    constants_info_[41].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[41].shape = {192};
    constants_info_[41].stride = {1};
    constants_info_[41].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[41].original_fqn = "mv2.features.5.conv.1.1.bias";
    constants_info_[42].name = "mv2_features_5_conv_2_weight";
    constants_info_[42].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[42].offset = 0;
    constants_info_[42].data_size = 24576;
    constants_info_[42].from_folded = false;
    constants_info_[42].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[42].shape = {32, 192, 1, 1};
    constants_info_[42].stride = {192, 1, 1, 1};
    constants_info_[42].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[42].original_fqn = "mv2.features.5.conv.2.weight";
    constants_info_[43].name = "mv2_features_5_conv_3_weight";
    constants_info_[43].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[43].offset = 0;
    constants_info_[43].data_size = 128;
    constants_info_[43].from_folded = false;
    constants_info_[43].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[43].shape = {32};
    constants_info_[43].stride = {1};
    constants_info_[43].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[43].original_fqn = "mv2.features.5.conv.3.weight";
    constants_info_[44].name = "mv2_features_5_conv_3_bias";
    constants_info_[44].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[44].offset = 0;
    constants_info_[44].data_size = 128;
    constants_info_[44].from_folded = false;
    constants_info_[44].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[44].shape = {32};
    constants_info_[44].stride = {1};
    constants_info_[44].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[44].original_fqn = "mv2.features.5.conv.3.bias";
    constants_info_[45].name = "mv2_features_6_conv_0_0_weight";
    constants_info_[45].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[45].offset = 0;
    constants_info_[45].data_size = 24576;
    constants_info_[45].from_folded = false;
    constants_info_[45].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[45].shape = {192, 32, 1, 1};
    constants_info_[45].stride = {32, 1, 1, 1};
    constants_info_[45].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[45].original_fqn = "mv2.features.6.conv.0.0.weight";
    constants_info_[46].name = "mv2_features_6_conv_0_1_weight";
    constants_info_[46].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[46].offset = 0;
    constants_info_[46].data_size = 768;
    constants_info_[46].from_folded = false;
    constants_info_[46].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[46].shape = {192};
    constants_info_[46].stride = {1};
    constants_info_[46].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[46].original_fqn = "mv2.features.6.conv.0.1.weight";
    constants_info_[47].name = "mv2_features_6_conv_0_1_bias";
    constants_info_[47].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[47].offset = 0;
    constants_info_[47].data_size = 768;
    constants_info_[47].from_folded = false;
    constants_info_[47].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[47].shape = {192};
    constants_info_[47].stride = {1};
    constants_info_[47].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[47].original_fqn = "mv2.features.6.conv.0.1.bias";
    constants_info_[48].name = "mv2_features_6_conv_1_0_weight";
    constants_info_[48].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[48].offset = 0;
    constants_info_[48].data_size = 6912;
    constants_info_[48].from_folded = false;
    constants_info_[48].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[48].shape = {192, 1, 3, 3};
    constants_info_[48].stride = {9, 9, 3, 1};
    constants_info_[48].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[48].original_fqn = "mv2.features.6.conv.1.0.weight";
    constants_info_[49].name = "mv2_features_6_conv_1_1_weight";
    constants_info_[49].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[49].offset = 0;
    constants_info_[49].data_size = 768;
    constants_info_[49].from_folded = false;
    constants_info_[49].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[49].shape = {192};
    constants_info_[49].stride = {1};
    constants_info_[49].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[49].original_fqn = "mv2.features.6.conv.1.1.weight";
    constants_info_[50].name = "mv2_features_6_conv_1_1_bias";
    constants_info_[50].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[50].offset = 0;
    constants_info_[50].data_size = 768;
    constants_info_[50].from_folded = false;
    constants_info_[50].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[50].shape = {192};
    constants_info_[50].stride = {1};
    constants_info_[50].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[50].original_fqn = "mv2.features.6.conv.1.1.bias";
    constants_info_[51].name = "mv2_features_6_conv_2_weight";
    constants_info_[51].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[51].offset = 0;
    constants_info_[51].data_size = 24576;
    constants_info_[51].from_folded = false;
    constants_info_[51].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[51].shape = {32, 192, 1, 1};
    constants_info_[51].stride = {192, 1, 1, 1};
    constants_info_[51].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[51].original_fqn = "mv2.features.6.conv.2.weight";
    constants_info_[52].name = "mv2_features_6_conv_3_weight";
    constants_info_[52].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[52].offset = 0;
    constants_info_[52].data_size = 128;
    constants_info_[52].from_folded = false;
    constants_info_[52].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[52].shape = {32};
    constants_info_[52].stride = {1};
    constants_info_[52].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[52].original_fqn = "mv2.features.6.conv.3.weight";
    constants_info_[53].name = "mv2_features_6_conv_3_bias";
    constants_info_[53].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[53].offset = 0;
    constants_info_[53].data_size = 128;
    constants_info_[53].from_folded = false;
    constants_info_[53].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[53].shape = {32};
    constants_info_[53].stride = {1};
    constants_info_[53].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[53].original_fqn = "mv2.features.6.conv.3.bias";
    constants_info_[54].name = "mv2_features_7_conv_0_0_weight";
    constants_info_[54].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[54].offset = 0;
    constants_info_[54].data_size = 24576;
    constants_info_[54].from_folded = false;
    constants_info_[54].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[54].shape = {192, 32, 1, 1};
    constants_info_[54].stride = {32, 1, 1, 1};
    constants_info_[54].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[54].original_fqn = "mv2.features.7.conv.0.0.weight";
    constants_info_[55].name = "mv2_features_7_conv_0_1_weight";
    constants_info_[55].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[55].offset = 0;
    constants_info_[55].data_size = 768;
    constants_info_[55].from_folded = false;
    constants_info_[55].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[55].shape = {192};
    constants_info_[55].stride = {1};
    constants_info_[55].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[55].original_fqn = "mv2.features.7.conv.0.1.weight";
    constants_info_[56].name = "mv2_features_7_conv_0_1_bias";
    constants_info_[56].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[56].offset = 0;
    constants_info_[56].data_size = 768;
    constants_info_[56].from_folded = false;
    constants_info_[56].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[56].shape = {192};
    constants_info_[56].stride = {1};
    constants_info_[56].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[56].original_fqn = "mv2.features.7.conv.0.1.bias";
    constants_info_[57].name = "mv2_features_7_conv_1_0_weight";
    constants_info_[57].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[57].offset = 0;
    constants_info_[57].data_size = 6912;
    constants_info_[57].from_folded = false;
    constants_info_[57].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[57].shape = {192, 1, 3, 3};
    constants_info_[57].stride = {9, 9, 3, 1};
    constants_info_[57].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[57].original_fqn = "mv2.features.7.conv.1.0.weight";
    constants_info_[58].name = "mv2_features_7_conv_1_1_weight";
    constants_info_[58].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[58].offset = 0;
    constants_info_[58].data_size = 768;
    constants_info_[58].from_folded = false;
    constants_info_[58].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[58].shape = {192};
    constants_info_[58].stride = {1};
    constants_info_[58].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[58].original_fqn = "mv2.features.7.conv.1.1.weight";
    constants_info_[59].name = "mv2_features_7_conv_1_1_bias";
    constants_info_[59].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[59].offset = 0;
    constants_info_[59].data_size = 768;
    constants_info_[59].from_folded = false;
    constants_info_[59].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[59].shape = {192};
    constants_info_[59].stride = {1};
    constants_info_[59].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[59].original_fqn = "mv2.features.7.conv.1.1.bias";
    constants_info_[60].name = "mv2_features_7_conv_2_weight";
    constants_info_[60].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[60].offset = 0;
    constants_info_[60].data_size = 49152;
    constants_info_[60].from_folded = false;
    constants_info_[60].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[60].shape = {64, 192, 1, 1};
    constants_info_[60].stride = {192, 1, 1, 1};
    constants_info_[60].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[60].original_fqn = "mv2.features.7.conv.2.weight";
    constants_info_[61].name = "mv2_features_7_conv_3_weight";
    constants_info_[61].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[61].offset = 0;
    constants_info_[61].data_size = 256;
    constants_info_[61].from_folded = false;
    constants_info_[61].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[61].shape = {64};
    constants_info_[61].stride = {1};
    constants_info_[61].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[61].original_fqn = "mv2.features.7.conv.3.weight";
    constants_info_[62].name = "mv2_features_7_conv_3_bias";
    constants_info_[62].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[62].offset = 0;
    constants_info_[62].data_size = 256;
    constants_info_[62].from_folded = false;
    constants_info_[62].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[62].shape = {64};
    constants_info_[62].stride = {1};
    constants_info_[62].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[62].original_fqn = "mv2.features.7.conv.3.bias";
    constants_info_[63].name = "mv2_features_8_conv_0_0_weight";
    constants_info_[63].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[63].offset = 0;
    constants_info_[63].data_size = 98304;
    constants_info_[63].from_folded = false;
    constants_info_[63].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[63].shape = {384, 64, 1, 1};
    constants_info_[63].stride = {64, 1, 1, 1};
    constants_info_[63].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[63].original_fqn = "mv2.features.8.conv.0.0.weight";
    constants_info_[64].name = "mv2_features_8_conv_0_1_weight";
    constants_info_[64].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[64].offset = 0;
    constants_info_[64].data_size = 1536;
    constants_info_[64].from_folded = false;
    constants_info_[64].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[64].shape = {384};
    constants_info_[64].stride = {1};
    constants_info_[64].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[64].original_fqn = "mv2.features.8.conv.0.1.weight";
    constants_info_[65].name = "mv2_features_8_conv_0_1_bias";
    constants_info_[65].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[65].offset = 0;
    constants_info_[65].data_size = 1536;
    constants_info_[65].from_folded = false;
    constants_info_[65].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[65].shape = {384};
    constants_info_[65].stride = {1};
    constants_info_[65].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[65].original_fqn = "mv2.features.8.conv.0.1.bias";
    constants_info_[66].name = "mv2_features_8_conv_1_0_weight";
    constants_info_[66].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[66].offset = 0;
    constants_info_[66].data_size = 13824;
    constants_info_[66].from_folded = false;
    constants_info_[66].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[66].shape = {384, 1, 3, 3};
    constants_info_[66].stride = {9, 9, 3, 1};
    constants_info_[66].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[66].original_fqn = "mv2.features.8.conv.1.0.weight";
    constants_info_[67].name = "mv2_features_8_conv_1_1_weight";
    constants_info_[67].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[67].offset = 0;
    constants_info_[67].data_size = 1536;
    constants_info_[67].from_folded = false;
    constants_info_[67].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[67].shape = {384};
    constants_info_[67].stride = {1};
    constants_info_[67].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[67].original_fqn = "mv2.features.8.conv.1.1.weight";
    constants_info_[68].name = "mv2_features_8_conv_1_1_bias";
    constants_info_[68].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[68].offset = 0;
    constants_info_[68].data_size = 1536;
    constants_info_[68].from_folded = false;
    constants_info_[68].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[68].shape = {384};
    constants_info_[68].stride = {1};
    constants_info_[68].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[68].original_fqn = "mv2.features.8.conv.1.1.bias";
    constants_info_[69].name = "mv2_features_8_conv_2_weight";
    constants_info_[69].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[69].offset = 0;
    constants_info_[69].data_size = 98304;
    constants_info_[69].from_folded = false;
    constants_info_[69].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[69].shape = {64, 384, 1, 1};
    constants_info_[69].stride = {384, 1, 1, 1};
    constants_info_[69].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[69].original_fqn = "mv2.features.8.conv.2.weight";
    constants_info_[70].name = "mv2_features_8_conv_3_weight";
    constants_info_[70].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[70].offset = 0;
    constants_info_[70].data_size = 256;
    constants_info_[70].from_folded = false;
    constants_info_[70].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[70].shape = {64};
    constants_info_[70].stride = {1};
    constants_info_[70].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[70].original_fqn = "mv2.features.8.conv.3.weight";
    constants_info_[71].name = "mv2_features_8_conv_3_bias";
    constants_info_[71].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[71].offset = 0;
    constants_info_[71].data_size = 256;
    constants_info_[71].from_folded = false;
    constants_info_[71].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[71].shape = {64};
    constants_info_[71].stride = {1};
    constants_info_[71].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[71].original_fqn = "mv2.features.8.conv.3.bias";
    constants_info_[72].name = "mv2_features_9_conv_0_0_weight";
    constants_info_[72].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[72].offset = 0;
    constants_info_[72].data_size = 98304;
    constants_info_[72].from_folded = false;
    constants_info_[72].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[72].shape = {384, 64, 1, 1};
    constants_info_[72].stride = {64, 1, 1, 1};
    constants_info_[72].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[72].original_fqn = "mv2.features.9.conv.0.0.weight";
    constants_info_[73].name = "mv2_features_9_conv_0_1_weight";
    constants_info_[73].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[73].offset = 0;
    constants_info_[73].data_size = 1536;
    constants_info_[73].from_folded = false;
    constants_info_[73].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[73].shape = {384};
    constants_info_[73].stride = {1};
    constants_info_[73].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[73].original_fqn = "mv2.features.9.conv.0.1.weight";
    constants_info_[74].name = "mv2_features_9_conv_0_1_bias";
    constants_info_[74].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[74].offset = 0;
    constants_info_[74].data_size = 1536;
    constants_info_[74].from_folded = false;
    constants_info_[74].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[74].shape = {384};
    constants_info_[74].stride = {1};
    constants_info_[74].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[74].original_fqn = "mv2.features.9.conv.0.1.bias";
    constants_info_[75].name = "mv2_features_9_conv_1_0_weight";
    constants_info_[75].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[75].offset = 0;
    constants_info_[75].data_size = 13824;
    constants_info_[75].from_folded = false;
    constants_info_[75].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[75].shape = {384, 1, 3, 3};
    constants_info_[75].stride = {9, 9, 3, 1};
    constants_info_[75].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[75].original_fqn = "mv2.features.9.conv.1.0.weight";
    constants_info_[76].name = "mv2_features_9_conv_1_1_weight";
    constants_info_[76].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[76].offset = 0;
    constants_info_[76].data_size = 1536;
    constants_info_[76].from_folded = false;
    constants_info_[76].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[76].shape = {384};
    constants_info_[76].stride = {1};
    constants_info_[76].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[76].original_fqn = "mv2.features.9.conv.1.1.weight";
    constants_info_[77].name = "mv2_features_9_conv_1_1_bias";
    constants_info_[77].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[77].offset = 0;
    constants_info_[77].data_size = 1536;
    constants_info_[77].from_folded = false;
    constants_info_[77].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[77].shape = {384};
    constants_info_[77].stride = {1};
    constants_info_[77].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[77].original_fqn = "mv2.features.9.conv.1.1.bias";
    constants_info_[78].name = "mv2_features_9_conv_2_weight";
    constants_info_[78].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[78].offset = 0;
    constants_info_[78].data_size = 98304;
    constants_info_[78].from_folded = false;
    constants_info_[78].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[78].shape = {64, 384, 1, 1};
    constants_info_[78].stride = {384, 1, 1, 1};
    constants_info_[78].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[78].original_fqn = "mv2.features.9.conv.2.weight";
    constants_info_[79].name = "mv2_features_9_conv_3_weight";
    constants_info_[79].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[79].offset = 0;
    constants_info_[79].data_size = 256;
    constants_info_[79].from_folded = false;
    constants_info_[79].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[79].shape = {64};
    constants_info_[79].stride = {1};
    constants_info_[79].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[79].original_fqn = "mv2.features.9.conv.3.weight";
    constants_info_[80].name = "mv2_features_9_conv_3_bias";
    constants_info_[80].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[80].offset = 0;
    constants_info_[80].data_size = 256;
    constants_info_[80].from_folded = false;
    constants_info_[80].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[80].shape = {64};
    constants_info_[80].stride = {1};
    constants_info_[80].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[80].original_fqn = "mv2.features.9.conv.3.bias";
    constants_info_[81].name = "mv2_features_10_conv_0_0_weight";
    constants_info_[81].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[81].offset = 0;
    constants_info_[81].data_size = 98304;
    constants_info_[81].from_folded = false;
    constants_info_[81].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[81].shape = {384, 64, 1, 1};
    constants_info_[81].stride = {64, 1, 1, 1};
    constants_info_[81].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[81].original_fqn = "mv2.features.10.conv.0.0.weight";
    constants_info_[82].name = "mv2_features_10_conv_0_1_weight";
    constants_info_[82].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[82].offset = 0;
    constants_info_[82].data_size = 1536;
    constants_info_[82].from_folded = false;
    constants_info_[82].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[82].shape = {384};
    constants_info_[82].stride = {1};
    constants_info_[82].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[82].original_fqn = "mv2.features.10.conv.0.1.weight";
    constants_info_[83].name = "mv2_features_10_conv_0_1_bias";
    constants_info_[83].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[83].offset = 0;
    constants_info_[83].data_size = 1536;
    constants_info_[83].from_folded = false;
    constants_info_[83].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[83].shape = {384};
    constants_info_[83].stride = {1};
    constants_info_[83].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[83].original_fqn = "mv2.features.10.conv.0.1.bias";
    constants_info_[84].name = "mv2_features_10_conv_1_0_weight";
    constants_info_[84].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[84].offset = 0;
    constants_info_[84].data_size = 13824;
    constants_info_[84].from_folded = false;
    constants_info_[84].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[84].shape = {384, 1, 3, 3};
    constants_info_[84].stride = {9, 9, 3, 1};
    constants_info_[84].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[84].original_fqn = "mv2.features.10.conv.1.0.weight";
    constants_info_[85].name = "mv2_features_10_conv_1_1_weight";
    constants_info_[85].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[85].offset = 0;
    constants_info_[85].data_size = 1536;
    constants_info_[85].from_folded = false;
    constants_info_[85].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[85].shape = {384};
    constants_info_[85].stride = {1};
    constants_info_[85].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[85].original_fqn = "mv2.features.10.conv.1.1.weight";
    constants_info_[86].name = "mv2_features_10_conv_1_1_bias";
    constants_info_[86].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[86].offset = 0;
    constants_info_[86].data_size = 1536;
    constants_info_[86].from_folded = false;
    constants_info_[86].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[86].shape = {384};
    constants_info_[86].stride = {1};
    constants_info_[86].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[86].original_fqn = "mv2.features.10.conv.1.1.bias";
    constants_info_[87].name = "mv2_features_10_conv_2_weight";
    constants_info_[87].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[87].offset = 0;
    constants_info_[87].data_size = 98304;
    constants_info_[87].from_folded = false;
    constants_info_[87].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[87].shape = {64, 384, 1, 1};
    constants_info_[87].stride = {384, 1, 1, 1};
    constants_info_[87].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[87].original_fqn = "mv2.features.10.conv.2.weight";
    constants_info_[88].name = "mv2_features_10_conv_3_weight";
    constants_info_[88].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[88].offset = 0;
    constants_info_[88].data_size = 256;
    constants_info_[88].from_folded = false;
    constants_info_[88].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[88].shape = {64};
    constants_info_[88].stride = {1};
    constants_info_[88].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[88].original_fqn = "mv2.features.10.conv.3.weight";
    constants_info_[89].name = "mv2_features_10_conv_3_bias";
    constants_info_[89].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[89].offset = 0;
    constants_info_[89].data_size = 256;
    constants_info_[89].from_folded = false;
    constants_info_[89].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[89].shape = {64};
    constants_info_[89].stride = {1};
    constants_info_[89].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[89].original_fqn = "mv2.features.10.conv.3.bias";
    constants_info_[90].name = "mv2_features_11_conv_0_0_weight";
    constants_info_[90].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[90].offset = 0;
    constants_info_[90].data_size = 98304;
    constants_info_[90].from_folded = false;
    constants_info_[90].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[90].shape = {384, 64, 1, 1};
    constants_info_[90].stride = {64, 1, 1, 1};
    constants_info_[90].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[90].original_fqn = "mv2.features.11.conv.0.0.weight";
    constants_info_[91].name = "mv2_features_11_conv_0_1_weight";
    constants_info_[91].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[91].offset = 0;
    constants_info_[91].data_size = 1536;
    constants_info_[91].from_folded = false;
    constants_info_[91].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[91].shape = {384};
    constants_info_[91].stride = {1};
    constants_info_[91].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[91].original_fqn = "mv2.features.11.conv.0.1.weight";
    constants_info_[92].name = "mv2_features_11_conv_0_1_bias";
    constants_info_[92].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[92].offset = 0;
    constants_info_[92].data_size = 1536;
    constants_info_[92].from_folded = false;
    constants_info_[92].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[92].shape = {384};
    constants_info_[92].stride = {1};
    constants_info_[92].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[92].original_fqn = "mv2.features.11.conv.0.1.bias";
    constants_info_[93].name = "mv2_features_11_conv_1_0_weight";
    constants_info_[93].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[93].offset = 0;
    constants_info_[93].data_size = 13824;
    constants_info_[93].from_folded = false;
    constants_info_[93].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[93].shape = {384, 1, 3, 3};
    constants_info_[93].stride = {9, 9, 3, 1};
    constants_info_[93].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[93].original_fqn = "mv2.features.11.conv.1.0.weight";
    constants_info_[94].name = "mv2_features_11_conv_1_1_weight";
    constants_info_[94].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[94].offset = 0;
    constants_info_[94].data_size = 1536;
    constants_info_[94].from_folded = false;
    constants_info_[94].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[94].shape = {384};
    constants_info_[94].stride = {1};
    constants_info_[94].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[94].original_fqn = "mv2.features.11.conv.1.1.weight";
    constants_info_[95].name = "mv2_features_11_conv_1_1_bias";
    constants_info_[95].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[95].offset = 0;
    constants_info_[95].data_size = 1536;
    constants_info_[95].from_folded = false;
    constants_info_[95].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[95].shape = {384};
    constants_info_[95].stride = {1};
    constants_info_[95].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[95].original_fqn = "mv2.features.11.conv.1.1.bias";
    constants_info_[96].name = "mv2_features_11_conv_2_weight";
    constants_info_[96].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[96].offset = 0;
    constants_info_[96].data_size = 147456;
    constants_info_[96].from_folded = false;
    constants_info_[96].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[96].shape = {96, 384, 1, 1};
    constants_info_[96].stride = {384, 1, 1, 1};
    constants_info_[96].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[96].original_fqn = "mv2.features.11.conv.2.weight";
    constants_info_[97].name = "mv2_features_11_conv_3_weight";
    constants_info_[97].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[97].offset = 0;
    constants_info_[97].data_size = 384;
    constants_info_[97].from_folded = false;
    constants_info_[97].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[97].shape = {96};
    constants_info_[97].stride = {1};
    constants_info_[97].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[97].original_fqn = "mv2.features.11.conv.3.weight";
    constants_info_[98].name = "mv2_features_11_conv_3_bias";
    constants_info_[98].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[98].offset = 0;
    constants_info_[98].data_size = 384;
    constants_info_[98].from_folded = false;
    constants_info_[98].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[98].shape = {96};
    constants_info_[98].stride = {1};
    constants_info_[98].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[98].original_fqn = "mv2.features.11.conv.3.bias";
    constants_info_[99].name = "mv2_features_12_conv_0_0_weight";
    constants_info_[99].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[99].offset = 0;
    constants_info_[99].data_size = 221184;
    constants_info_[99].from_folded = false;
    constants_info_[99].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[99].shape = {576, 96, 1, 1};
    constants_info_[99].stride = {96, 1, 1, 1};
    constants_info_[99].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[99].original_fqn = "mv2.features.12.conv.0.0.weight";
    constants_info_[100].name = "mv2_features_12_conv_0_1_weight";
    constants_info_[100].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[100].offset = 0;
    constants_info_[100].data_size = 2304;
    constants_info_[100].from_folded = false;
    constants_info_[100].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[100].shape = {576};
    constants_info_[100].stride = {1};
    constants_info_[100].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[100].original_fqn = "mv2.features.12.conv.0.1.weight";
    constants_info_[101].name = "mv2_features_12_conv_0_1_bias";
    constants_info_[101].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[101].offset = 0;
    constants_info_[101].data_size = 2304;
    constants_info_[101].from_folded = false;
    constants_info_[101].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[101].shape = {576};
    constants_info_[101].stride = {1};
    constants_info_[101].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[101].original_fqn = "mv2.features.12.conv.0.1.bias";
    constants_info_[102].name = "mv2_features_12_conv_1_0_weight";
    constants_info_[102].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[102].offset = 0;
    constants_info_[102].data_size = 20736;
    constants_info_[102].from_folded = false;
    constants_info_[102].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[102].shape = {576, 1, 3, 3};
    constants_info_[102].stride = {9, 9, 3, 1};
    constants_info_[102].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[102].original_fqn = "mv2.features.12.conv.1.0.weight";
    constants_info_[103].name = "mv2_features_12_conv_1_1_weight";
    constants_info_[103].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[103].offset = 0;
    constants_info_[103].data_size = 2304;
    constants_info_[103].from_folded = false;
    constants_info_[103].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[103].shape = {576};
    constants_info_[103].stride = {1};
    constants_info_[103].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[103].original_fqn = "mv2.features.12.conv.1.1.weight";
    constants_info_[104].name = "mv2_features_12_conv_1_1_bias";
    constants_info_[104].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[104].offset = 0;
    constants_info_[104].data_size = 2304;
    constants_info_[104].from_folded = false;
    constants_info_[104].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[104].shape = {576};
    constants_info_[104].stride = {1};
    constants_info_[104].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[104].original_fqn = "mv2.features.12.conv.1.1.bias";
    constants_info_[105].name = "mv2_features_12_conv_2_weight";
    constants_info_[105].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[105].offset = 0;
    constants_info_[105].data_size = 221184;
    constants_info_[105].from_folded = false;
    constants_info_[105].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[105].shape = {96, 576, 1, 1};
    constants_info_[105].stride = {576, 1, 1, 1};
    constants_info_[105].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[105].original_fqn = "mv2.features.12.conv.2.weight";
    constants_info_[106].name = "mv2_features_12_conv_3_weight";
    constants_info_[106].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[106].offset = 0;
    constants_info_[106].data_size = 384;
    constants_info_[106].from_folded = false;
    constants_info_[106].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[106].shape = {96};
    constants_info_[106].stride = {1};
    constants_info_[106].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[106].original_fqn = "mv2.features.12.conv.3.weight";
    constants_info_[107].name = "mv2_features_12_conv_3_bias";
    constants_info_[107].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[107].offset = 0;
    constants_info_[107].data_size = 384;
    constants_info_[107].from_folded = false;
    constants_info_[107].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[107].shape = {96};
    constants_info_[107].stride = {1};
    constants_info_[107].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[107].original_fqn = "mv2.features.12.conv.3.bias";
    constants_info_[108].name = "mv2_features_13_conv_0_0_weight";
    constants_info_[108].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[108].offset = 0;
    constants_info_[108].data_size = 221184;
    constants_info_[108].from_folded = false;
    constants_info_[108].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[108].shape = {576, 96, 1, 1};
    constants_info_[108].stride = {96, 1, 1, 1};
    constants_info_[108].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[108].original_fqn = "mv2.features.13.conv.0.0.weight";
    constants_info_[109].name = "mv2_features_13_conv_0_1_weight";
    constants_info_[109].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[109].offset = 0;
    constants_info_[109].data_size = 2304;
    constants_info_[109].from_folded = false;
    constants_info_[109].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[109].shape = {576};
    constants_info_[109].stride = {1};
    constants_info_[109].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[109].original_fqn = "mv2.features.13.conv.0.1.weight";
    constants_info_[110].name = "mv2_features_13_conv_0_1_bias";
    constants_info_[110].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[110].offset = 0;
    constants_info_[110].data_size = 2304;
    constants_info_[110].from_folded = false;
    constants_info_[110].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[110].shape = {576};
    constants_info_[110].stride = {1};
    constants_info_[110].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[110].original_fqn = "mv2.features.13.conv.0.1.bias";
    constants_info_[111].name = "mv2_features_13_conv_1_0_weight";
    constants_info_[111].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[111].offset = 0;
    constants_info_[111].data_size = 20736;
    constants_info_[111].from_folded = false;
    constants_info_[111].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[111].shape = {576, 1, 3, 3};
    constants_info_[111].stride = {9, 9, 3, 1};
    constants_info_[111].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[111].original_fqn = "mv2.features.13.conv.1.0.weight";
    constants_info_[112].name = "mv2_features_13_conv_1_1_weight";
    constants_info_[112].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[112].offset = 0;
    constants_info_[112].data_size = 2304;
    constants_info_[112].from_folded = false;
    constants_info_[112].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[112].shape = {576};
    constants_info_[112].stride = {1};
    constants_info_[112].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[112].original_fqn = "mv2.features.13.conv.1.1.weight";
    constants_info_[113].name = "mv2_features_13_conv_1_1_bias";
    constants_info_[113].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[113].offset = 0;
    constants_info_[113].data_size = 2304;
    constants_info_[113].from_folded = false;
    constants_info_[113].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[113].shape = {576};
    constants_info_[113].stride = {1};
    constants_info_[113].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[113].original_fqn = "mv2.features.13.conv.1.1.bias";
    constants_info_[114].name = "mv2_features_13_conv_2_weight";
    constants_info_[114].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[114].offset = 0;
    constants_info_[114].data_size = 221184;
    constants_info_[114].from_folded = false;
    constants_info_[114].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[114].shape = {96, 576, 1, 1};
    constants_info_[114].stride = {576, 1, 1, 1};
    constants_info_[114].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[114].original_fqn = "mv2.features.13.conv.2.weight";
    constants_info_[115].name = "mv2_features_13_conv_3_weight";
    constants_info_[115].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[115].offset = 0;
    constants_info_[115].data_size = 384;
    constants_info_[115].from_folded = false;
    constants_info_[115].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[115].shape = {96};
    constants_info_[115].stride = {1};
    constants_info_[115].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[115].original_fqn = "mv2.features.13.conv.3.weight";
    constants_info_[116].name = "mv2_features_13_conv_3_bias";
    constants_info_[116].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[116].offset = 0;
    constants_info_[116].data_size = 384;
    constants_info_[116].from_folded = false;
    constants_info_[116].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[116].shape = {96};
    constants_info_[116].stride = {1};
    constants_info_[116].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[116].original_fqn = "mv2.features.13.conv.3.bias";
    constants_info_[117].name = "mv2_features_14_conv_0_0_weight";
    constants_info_[117].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[117].offset = 0;
    constants_info_[117].data_size = 221184;
    constants_info_[117].from_folded = false;
    constants_info_[117].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[117].shape = {576, 96, 1, 1};
    constants_info_[117].stride = {96, 1, 1, 1};
    constants_info_[117].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[117].original_fqn = "mv2.features.14.conv.0.0.weight";
    constants_info_[118].name = "mv2_features_14_conv_0_1_weight";
    constants_info_[118].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[118].offset = 0;
    constants_info_[118].data_size = 2304;
    constants_info_[118].from_folded = false;
    constants_info_[118].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[118].shape = {576};
    constants_info_[118].stride = {1};
    constants_info_[118].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[118].original_fqn = "mv2.features.14.conv.0.1.weight";
    constants_info_[119].name = "mv2_features_14_conv_0_1_bias";
    constants_info_[119].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[119].offset = 0;
    constants_info_[119].data_size = 2304;
    constants_info_[119].from_folded = false;
    constants_info_[119].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[119].shape = {576};
    constants_info_[119].stride = {1};
    constants_info_[119].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[119].original_fqn = "mv2.features.14.conv.0.1.bias";
    constants_info_[120].name = "mv2_features_14_conv_1_0_weight";
    constants_info_[120].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[120].offset = 0;
    constants_info_[120].data_size = 20736;
    constants_info_[120].from_folded = false;
    constants_info_[120].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[120].shape = {576, 1, 3, 3};
    constants_info_[120].stride = {9, 9, 3, 1};
    constants_info_[120].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[120].original_fqn = "mv2.features.14.conv.1.0.weight";
    constants_info_[121].name = "mv2_features_14_conv_1_1_weight";
    constants_info_[121].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[121].offset = 0;
    constants_info_[121].data_size = 2304;
    constants_info_[121].from_folded = false;
    constants_info_[121].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[121].shape = {576};
    constants_info_[121].stride = {1};
    constants_info_[121].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[121].original_fqn = "mv2.features.14.conv.1.1.weight";
    constants_info_[122].name = "mv2_features_14_conv_1_1_bias";
    constants_info_[122].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[122].offset = 0;
    constants_info_[122].data_size = 2304;
    constants_info_[122].from_folded = false;
    constants_info_[122].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[122].shape = {576};
    constants_info_[122].stride = {1};
    constants_info_[122].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[122].original_fqn = "mv2.features.14.conv.1.1.bias";
    constants_info_[123].name = "mv2_features_14_conv_2_weight";
    constants_info_[123].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[123].offset = 0;
    constants_info_[123].data_size = 368640;
    constants_info_[123].from_folded = false;
    constants_info_[123].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[123].shape = {160, 576, 1, 1};
    constants_info_[123].stride = {576, 1, 1, 1};
    constants_info_[123].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[123].original_fqn = "mv2.features.14.conv.2.weight";
    constants_info_[124].name = "mv2_features_14_conv_3_weight";
    constants_info_[124].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[124].offset = 0;
    constants_info_[124].data_size = 640;
    constants_info_[124].from_folded = false;
    constants_info_[124].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[124].shape = {160};
    constants_info_[124].stride = {1};
    constants_info_[124].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[124].original_fqn = "mv2.features.14.conv.3.weight";
    constants_info_[125].name = "mv2_features_14_conv_3_bias";
    constants_info_[125].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[125].offset = 0;
    constants_info_[125].data_size = 640;
    constants_info_[125].from_folded = false;
    constants_info_[125].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[125].shape = {160};
    constants_info_[125].stride = {1};
    constants_info_[125].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[125].original_fqn = "mv2.features.14.conv.3.bias";
    constants_info_[126].name = "mv2_features_15_conv_0_0_weight";
    constants_info_[126].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[126].offset = 0;
    constants_info_[126].data_size = 614400;
    constants_info_[126].from_folded = false;
    constants_info_[126].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[126].shape = {960, 160, 1, 1};
    constants_info_[126].stride = {160, 1, 1, 1};
    constants_info_[126].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[126].original_fqn = "mv2.features.15.conv.0.0.weight";
    constants_info_[127].name = "mv2_features_15_conv_0_1_weight";
    constants_info_[127].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[127].offset = 0;
    constants_info_[127].data_size = 3840;
    constants_info_[127].from_folded = false;
    constants_info_[127].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[127].shape = {960};
    constants_info_[127].stride = {1};
    constants_info_[127].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[127].original_fqn = "mv2.features.15.conv.0.1.weight";
    constants_info_[128].name = "mv2_features_15_conv_0_1_bias";
    constants_info_[128].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[128].offset = 0;
    constants_info_[128].data_size = 3840;
    constants_info_[128].from_folded = false;
    constants_info_[128].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[128].shape = {960};
    constants_info_[128].stride = {1};
    constants_info_[128].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[128].original_fqn = "mv2.features.15.conv.0.1.bias";
    constants_info_[129].name = "mv2_features_15_conv_1_0_weight";
    constants_info_[129].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[129].offset = 0;
    constants_info_[129].data_size = 34560;
    constants_info_[129].from_folded = false;
    constants_info_[129].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[129].shape = {960, 1, 3, 3};
    constants_info_[129].stride = {9, 9, 3, 1};
    constants_info_[129].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[129].original_fqn = "mv2.features.15.conv.1.0.weight";
    constants_info_[130].name = "mv2_features_15_conv_1_1_weight";
    constants_info_[130].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[130].offset = 0;
    constants_info_[130].data_size = 3840;
    constants_info_[130].from_folded = false;
    constants_info_[130].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[130].shape = {960};
    constants_info_[130].stride = {1};
    constants_info_[130].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[130].original_fqn = "mv2.features.15.conv.1.1.weight";
    constants_info_[131].name = "mv2_features_15_conv_1_1_bias";
    constants_info_[131].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[131].offset = 0;
    constants_info_[131].data_size = 3840;
    constants_info_[131].from_folded = false;
    constants_info_[131].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[131].shape = {960};
    constants_info_[131].stride = {1};
    constants_info_[131].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[131].original_fqn = "mv2.features.15.conv.1.1.bias";
    constants_info_[132].name = "mv2_features_15_conv_2_weight";
    constants_info_[132].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[132].offset = 0;
    constants_info_[132].data_size = 614400;
    constants_info_[132].from_folded = false;
    constants_info_[132].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[132].shape = {160, 960, 1, 1};
    constants_info_[132].stride = {960, 1, 1, 1};
    constants_info_[132].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[132].original_fqn = "mv2.features.15.conv.2.weight";
    constants_info_[133].name = "mv2_features_15_conv_3_weight";
    constants_info_[133].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[133].offset = 0;
    constants_info_[133].data_size = 640;
    constants_info_[133].from_folded = false;
    constants_info_[133].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[133].shape = {160};
    constants_info_[133].stride = {1};
    constants_info_[133].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[133].original_fqn = "mv2.features.15.conv.3.weight";
    constants_info_[134].name = "mv2_features_15_conv_3_bias";
    constants_info_[134].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[134].offset = 0;
    constants_info_[134].data_size = 640;
    constants_info_[134].from_folded = false;
    constants_info_[134].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[134].shape = {160};
    constants_info_[134].stride = {1};
    constants_info_[134].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[134].original_fqn = "mv2.features.15.conv.3.bias";
    constants_info_[135].name = "mv2_features_16_conv_0_0_weight";
    constants_info_[135].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[135].offset = 0;
    constants_info_[135].data_size = 614400;
    constants_info_[135].from_folded = false;
    constants_info_[135].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[135].shape = {960, 160, 1, 1};
    constants_info_[135].stride = {160, 1, 1, 1};
    constants_info_[135].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[135].original_fqn = "mv2.features.16.conv.0.0.weight";
    constants_info_[136].name = "mv2_features_16_conv_0_1_weight";
    constants_info_[136].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[136].offset = 0;
    constants_info_[136].data_size = 3840;
    constants_info_[136].from_folded = false;
    constants_info_[136].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[136].shape = {960};
    constants_info_[136].stride = {1};
    constants_info_[136].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[136].original_fqn = "mv2.features.16.conv.0.1.weight";
    constants_info_[137].name = "mv2_features_16_conv_0_1_bias";
    constants_info_[137].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[137].offset = 0;
    constants_info_[137].data_size = 3840;
    constants_info_[137].from_folded = false;
    constants_info_[137].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[137].shape = {960};
    constants_info_[137].stride = {1};
    constants_info_[137].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[137].original_fqn = "mv2.features.16.conv.0.1.bias";
    constants_info_[138].name = "mv2_features_16_conv_1_0_weight";
    constants_info_[138].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[138].offset = 0;
    constants_info_[138].data_size = 34560;
    constants_info_[138].from_folded = false;
    constants_info_[138].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[138].shape = {960, 1, 3, 3};
    constants_info_[138].stride = {9, 9, 3, 1};
    constants_info_[138].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[138].original_fqn = "mv2.features.16.conv.1.0.weight";
    constants_info_[139].name = "mv2_features_16_conv_1_1_weight";
    constants_info_[139].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[139].offset = 0;
    constants_info_[139].data_size = 3840;
    constants_info_[139].from_folded = false;
    constants_info_[139].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[139].shape = {960};
    constants_info_[139].stride = {1};
    constants_info_[139].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[139].original_fqn = "mv2.features.16.conv.1.1.weight";
    constants_info_[140].name = "mv2_features_16_conv_1_1_bias";
    constants_info_[140].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[140].offset = 0;
    constants_info_[140].data_size = 3840;
    constants_info_[140].from_folded = false;
    constants_info_[140].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[140].shape = {960};
    constants_info_[140].stride = {1};
    constants_info_[140].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[140].original_fqn = "mv2.features.16.conv.1.1.bias";
    constants_info_[141].name = "mv2_features_16_conv_2_weight";
    constants_info_[141].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[141].offset = 0;
    constants_info_[141].data_size = 614400;
    constants_info_[141].from_folded = false;
    constants_info_[141].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[141].shape = {160, 960, 1, 1};
    constants_info_[141].stride = {960, 1, 1, 1};
    constants_info_[141].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[141].original_fqn = "mv2.features.16.conv.2.weight";
    constants_info_[142].name = "mv2_features_16_conv_3_weight";
    constants_info_[142].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[142].offset = 0;
    constants_info_[142].data_size = 640;
    constants_info_[142].from_folded = false;
    constants_info_[142].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[142].shape = {160};
    constants_info_[142].stride = {1};
    constants_info_[142].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[142].original_fqn = "mv2.features.16.conv.3.weight";
    constants_info_[143].name = "mv2_features_16_conv_3_bias";
    constants_info_[143].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[143].offset = 0;
    constants_info_[143].data_size = 640;
    constants_info_[143].from_folded = false;
    constants_info_[143].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[143].shape = {160};
    constants_info_[143].stride = {1};
    constants_info_[143].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[143].original_fqn = "mv2.features.16.conv.3.bias";
    constants_info_[144].name = "mv2_features_17_conv_0_0_weight";
    constants_info_[144].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[144].offset = 0;
    constants_info_[144].data_size = 614400;
    constants_info_[144].from_folded = false;
    constants_info_[144].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[144].shape = {960, 160, 1, 1};
    constants_info_[144].stride = {160, 1, 1, 1};
    constants_info_[144].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[144].original_fqn = "mv2.features.17.conv.0.0.weight";
    constants_info_[145].name = "mv2_features_17_conv_0_1_weight";
    constants_info_[145].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[145].offset = 0;
    constants_info_[145].data_size = 3840;
    constants_info_[145].from_folded = false;
    constants_info_[145].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[145].shape = {960};
    constants_info_[145].stride = {1};
    constants_info_[145].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[145].original_fqn = "mv2.features.17.conv.0.1.weight";
    constants_info_[146].name = "mv2_features_17_conv_0_1_bias";
    constants_info_[146].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[146].offset = 0;
    constants_info_[146].data_size = 3840;
    constants_info_[146].from_folded = false;
    constants_info_[146].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[146].shape = {960};
    constants_info_[146].stride = {1};
    constants_info_[146].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[146].original_fqn = "mv2.features.17.conv.0.1.bias";
    constants_info_[147].name = "mv2_features_17_conv_1_0_weight";
    constants_info_[147].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[147].offset = 0;
    constants_info_[147].data_size = 34560;
    constants_info_[147].from_folded = false;
    constants_info_[147].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[147].shape = {960, 1, 3, 3};
    constants_info_[147].stride = {9, 9, 3, 1};
    constants_info_[147].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[147].original_fqn = "mv2.features.17.conv.1.0.weight";
    constants_info_[148].name = "mv2_features_17_conv_1_1_weight";
    constants_info_[148].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[148].offset = 0;
    constants_info_[148].data_size = 3840;
    constants_info_[148].from_folded = false;
    constants_info_[148].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[148].shape = {960};
    constants_info_[148].stride = {1};
    constants_info_[148].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[148].original_fqn = "mv2.features.17.conv.1.1.weight";
    constants_info_[149].name = "mv2_features_17_conv_1_1_bias";
    constants_info_[149].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[149].offset = 0;
    constants_info_[149].data_size = 3840;
    constants_info_[149].from_folded = false;
    constants_info_[149].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[149].shape = {960};
    constants_info_[149].stride = {1};
    constants_info_[149].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[149].original_fqn = "mv2.features.17.conv.1.1.bias";
    constants_info_[150].name = "mv2_features_17_conv_2_weight";
    constants_info_[150].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[150].offset = 0;
    constants_info_[150].data_size = 1228800;
    constants_info_[150].from_folded = false;
    constants_info_[150].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[150].shape = {320, 960, 1, 1};
    constants_info_[150].stride = {960, 1, 1, 1};
    constants_info_[150].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[150].original_fqn = "mv2.features.17.conv.2.weight";
    constants_info_[151].name = "mv2_features_17_conv_3_weight";
    constants_info_[151].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[151].offset = 0;
    constants_info_[151].data_size = 1280;
    constants_info_[151].from_folded = false;
    constants_info_[151].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[151].shape = {320};
    constants_info_[151].stride = {1};
    constants_info_[151].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[151].original_fqn = "mv2.features.17.conv.3.weight";
    constants_info_[152].name = "mv2_features_17_conv_3_bias";
    constants_info_[152].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[152].offset = 0;
    constants_info_[152].data_size = 1280;
    constants_info_[152].from_folded = false;
    constants_info_[152].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[152].shape = {320};
    constants_info_[152].stride = {1};
    constants_info_[152].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[152].original_fqn = "mv2.features.17.conv.3.bias";
    constants_info_[153].name = "mv2_features_18_0_weight";
    constants_info_[153].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[153].offset = 0;
    constants_info_[153].data_size = 1638400;
    constants_info_[153].from_folded = false;
    constants_info_[153].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[153].shape = {1280, 320, 1, 1};
    constants_info_[153].stride = {320, 1, 1, 1};
    constants_info_[153].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[153].original_fqn = "mv2.features.18.0.weight";
    constants_info_[154].name = "mv2_features_18_1_weight";
    constants_info_[154].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[154].offset = 0;
    constants_info_[154].data_size = 5120;
    constants_info_[154].from_folded = false;
    constants_info_[154].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[154].shape = {1280};
    constants_info_[154].stride = {1};
    constants_info_[154].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[154].original_fqn = "mv2.features.18.1.weight";
    constants_info_[155].name = "mv2_features_18_1_bias";
    constants_info_[155].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[155].offset = 0;
    constants_info_[155].data_size = 5120;
    constants_info_[155].from_folded = false;
    constants_info_[155].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[155].shape = {1280};
    constants_info_[155].stride = {1};
    constants_info_[155].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[155].original_fqn = "mv2.features.18.1.bias";
    constants_info_[156].name = "mv2_classifier_1_weight";
    constants_info_[156].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[156].offset = 0;
    constants_info_[156].data_size = 5120000;
    constants_info_[156].from_folded = false;
    constants_info_[156].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[156].shape = {1000, 1280};
    constants_info_[156].stride = {1280, 1};
    constants_info_[156].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[156].original_fqn = "mv2.classifier.1.weight";
    constants_info_[157].name = "mv2_classifier_1_bias";
    constants_info_[157].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[157].offset = 0;
    constants_info_[157].data_size = 4000;
    constants_info_[157].from_folded = false;
    constants_info_[157].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[157].shape = {1000};
    constants_info_[157].stride = {1};
    constants_info_[157].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[157].original_fqn = "mv2.classifier.1.bias";
    constants_info_[158].name = "mv2_features_0_1_running_mean";
    constants_info_[158].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[158].offset = 0;
    constants_info_[158].data_size = 128;
    constants_info_[158].from_folded = false;
    constants_info_[158].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[158].shape = {32};
    constants_info_[158].stride = {1};
    constants_info_[158].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[158].original_fqn = "mv2.features.0.1.running_mean";
    constants_info_[159].name = "mv2_features_0_1_running_var";
    constants_info_[159].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[159].offset = 0;
    constants_info_[159].data_size = 128;
    constants_info_[159].from_folded = false;
    constants_info_[159].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[159].shape = {32};
    constants_info_[159].stride = {1};
    constants_info_[159].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[159].original_fqn = "mv2.features.0.1.running_var";
    constants_info_[160].name = "mv2_features_1_conv_0_1_running_mean";
    constants_info_[160].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[160].offset = 0;
    constants_info_[160].data_size = 128;
    constants_info_[160].from_folded = false;
    constants_info_[160].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[160].shape = {32};
    constants_info_[160].stride = {1};
    constants_info_[160].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[160].original_fqn = "mv2.features.1.conv.0.1.running_mean";
    constants_info_[161].name = "mv2_features_1_conv_0_1_running_var";
    constants_info_[161].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[161].offset = 0;
    constants_info_[161].data_size = 128;
    constants_info_[161].from_folded = false;
    constants_info_[161].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[161].shape = {32};
    constants_info_[161].stride = {1};
    constants_info_[161].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[161].original_fqn = "mv2.features.1.conv.0.1.running_var";
    constants_info_[162].name = "mv2_features_1_conv_2_running_mean";
    constants_info_[162].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[162].offset = 0;
    constants_info_[162].data_size = 64;
    constants_info_[162].from_folded = false;
    constants_info_[162].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[162].shape = {16};
    constants_info_[162].stride = {1};
    constants_info_[162].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[162].original_fqn = "mv2.features.1.conv.2.running_mean";
    constants_info_[163].name = "mv2_features_1_conv_2_running_var";
    constants_info_[163].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[163].offset = 0;
    constants_info_[163].data_size = 64;
    constants_info_[163].from_folded = false;
    constants_info_[163].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[163].shape = {16};
    constants_info_[163].stride = {1};
    constants_info_[163].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[163].original_fqn = "mv2.features.1.conv.2.running_var";
    constants_info_[164].name = "mv2_features_2_conv_0_1_running_mean";
    constants_info_[164].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[164].offset = 0;
    constants_info_[164].data_size = 384;
    constants_info_[164].from_folded = false;
    constants_info_[164].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[164].shape = {96};
    constants_info_[164].stride = {1};
    constants_info_[164].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[164].original_fqn = "mv2.features.2.conv.0.1.running_mean";
    constants_info_[165].name = "mv2_features_2_conv_0_1_running_var";
    constants_info_[165].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[165].offset = 0;
    constants_info_[165].data_size = 384;
    constants_info_[165].from_folded = false;
    constants_info_[165].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[165].shape = {96};
    constants_info_[165].stride = {1};
    constants_info_[165].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[165].original_fqn = "mv2.features.2.conv.0.1.running_var";
    constants_info_[166].name = "mv2_features_2_conv_1_1_running_mean";
    constants_info_[166].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[166].offset = 0;
    constants_info_[166].data_size = 384;
    constants_info_[166].from_folded = false;
    constants_info_[166].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[166].shape = {96};
    constants_info_[166].stride = {1};
    constants_info_[166].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[166].original_fqn = "mv2.features.2.conv.1.1.running_mean";
    constants_info_[167].name = "mv2_features_2_conv_1_1_running_var";
    constants_info_[167].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[167].offset = 0;
    constants_info_[167].data_size = 384;
    constants_info_[167].from_folded = false;
    constants_info_[167].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[167].shape = {96};
    constants_info_[167].stride = {1};
    constants_info_[167].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[167].original_fqn = "mv2.features.2.conv.1.1.running_var";
    constants_info_[168].name = "mv2_features_2_conv_3_running_mean";
    constants_info_[168].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[168].offset = 0;
    constants_info_[168].data_size = 96;
    constants_info_[168].from_folded = false;
    constants_info_[168].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[168].shape = {24};
    constants_info_[168].stride = {1};
    constants_info_[168].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[168].original_fqn = "mv2.features.2.conv.3.running_mean";
    constants_info_[169].name = "mv2_features_2_conv_3_running_var";
    constants_info_[169].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[169].offset = 0;
    constants_info_[169].data_size = 96;
    constants_info_[169].from_folded = false;
    constants_info_[169].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[169].shape = {24};
    constants_info_[169].stride = {1};
    constants_info_[169].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[169].original_fqn = "mv2.features.2.conv.3.running_var";
    constants_info_[170].name = "mv2_features_3_conv_0_1_running_mean";
    constants_info_[170].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[170].offset = 0;
    constants_info_[170].data_size = 576;
    constants_info_[170].from_folded = false;
    constants_info_[170].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[170].shape = {144};
    constants_info_[170].stride = {1};
    constants_info_[170].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[170].original_fqn = "mv2.features.3.conv.0.1.running_mean";
    constants_info_[171].name = "mv2_features_3_conv_0_1_running_var";
    constants_info_[171].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[171].offset = 0;
    constants_info_[171].data_size = 576;
    constants_info_[171].from_folded = false;
    constants_info_[171].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[171].shape = {144};
    constants_info_[171].stride = {1};
    constants_info_[171].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[171].original_fqn = "mv2.features.3.conv.0.1.running_var";
    constants_info_[172].name = "mv2_features_3_conv_1_1_running_mean";
    constants_info_[172].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[172].offset = 0;
    constants_info_[172].data_size = 576;
    constants_info_[172].from_folded = false;
    constants_info_[172].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[172].shape = {144};
    constants_info_[172].stride = {1};
    constants_info_[172].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[172].original_fqn = "mv2.features.3.conv.1.1.running_mean";
    constants_info_[173].name = "mv2_features_3_conv_1_1_running_var";
    constants_info_[173].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[173].offset = 0;
    constants_info_[173].data_size = 576;
    constants_info_[173].from_folded = false;
    constants_info_[173].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[173].shape = {144};
    constants_info_[173].stride = {1};
    constants_info_[173].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[173].original_fqn = "mv2.features.3.conv.1.1.running_var";
    constants_info_[174].name = "mv2_features_3_conv_3_running_mean";
    constants_info_[174].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[174].offset = 0;
    constants_info_[174].data_size = 96;
    constants_info_[174].from_folded = false;
    constants_info_[174].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[174].shape = {24};
    constants_info_[174].stride = {1};
    constants_info_[174].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[174].original_fqn = "mv2.features.3.conv.3.running_mean";
    constants_info_[175].name = "mv2_features_3_conv_3_running_var";
    constants_info_[175].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[175].offset = 0;
    constants_info_[175].data_size = 96;
    constants_info_[175].from_folded = false;
    constants_info_[175].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[175].shape = {24};
    constants_info_[175].stride = {1};
    constants_info_[175].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[175].original_fqn = "mv2.features.3.conv.3.running_var";
    constants_info_[176].name = "mv2_features_4_conv_0_1_running_mean";
    constants_info_[176].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[176].offset = 0;
    constants_info_[176].data_size = 576;
    constants_info_[176].from_folded = false;
    constants_info_[176].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[176].shape = {144};
    constants_info_[176].stride = {1};
    constants_info_[176].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[176].original_fqn = "mv2.features.4.conv.0.1.running_mean";
    constants_info_[177].name = "mv2_features_4_conv_0_1_running_var";
    constants_info_[177].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[177].offset = 0;
    constants_info_[177].data_size = 576;
    constants_info_[177].from_folded = false;
    constants_info_[177].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[177].shape = {144};
    constants_info_[177].stride = {1};
    constants_info_[177].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[177].original_fqn = "mv2.features.4.conv.0.1.running_var";
    constants_info_[178].name = "mv2_features_4_conv_1_1_running_mean";
    constants_info_[178].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[178].offset = 0;
    constants_info_[178].data_size = 576;
    constants_info_[178].from_folded = false;
    constants_info_[178].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[178].shape = {144};
    constants_info_[178].stride = {1};
    constants_info_[178].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[178].original_fqn = "mv2.features.4.conv.1.1.running_mean";
    constants_info_[179].name = "mv2_features_4_conv_1_1_running_var";
    constants_info_[179].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[179].offset = 0;
    constants_info_[179].data_size = 576;
    constants_info_[179].from_folded = false;
    constants_info_[179].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[179].shape = {144};
    constants_info_[179].stride = {1};
    constants_info_[179].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[179].original_fqn = "mv2.features.4.conv.1.1.running_var";
    constants_info_[180].name = "mv2_features_4_conv_3_running_mean";
    constants_info_[180].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[180].offset = 0;
    constants_info_[180].data_size = 128;
    constants_info_[180].from_folded = false;
    constants_info_[180].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[180].shape = {32};
    constants_info_[180].stride = {1};
    constants_info_[180].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[180].original_fqn = "mv2.features.4.conv.3.running_mean";
    constants_info_[181].name = "mv2_features_4_conv_3_running_var";
    constants_info_[181].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[181].offset = 0;
    constants_info_[181].data_size = 128;
    constants_info_[181].from_folded = false;
    constants_info_[181].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[181].shape = {32};
    constants_info_[181].stride = {1};
    constants_info_[181].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[181].original_fqn = "mv2.features.4.conv.3.running_var";
    constants_info_[182].name = "mv2_features_5_conv_0_1_running_mean";
    constants_info_[182].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[182].offset = 0;
    constants_info_[182].data_size = 768;
    constants_info_[182].from_folded = false;
    constants_info_[182].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[182].shape = {192};
    constants_info_[182].stride = {1};
    constants_info_[182].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[182].original_fqn = "mv2.features.5.conv.0.1.running_mean";
    constants_info_[183].name = "mv2_features_5_conv_0_1_running_var";
    constants_info_[183].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[183].offset = 0;
    constants_info_[183].data_size = 768;
    constants_info_[183].from_folded = false;
    constants_info_[183].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[183].shape = {192};
    constants_info_[183].stride = {1};
    constants_info_[183].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[183].original_fqn = "mv2.features.5.conv.0.1.running_var";
    constants_info_[184].name = "mv2_features_5_conv_1_1_running_mean";
    constants_info_[184].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[184].offset = 0;
    constants_info_[184].data_size = 768;
    constants_info_[184].from_folded = false;
    constants_info_[184].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[184].shape = {192};
    constants_info_[184].stride = {1};
    constants_info_[184].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[184].original_fqn = "mv2.features.5.conv.1.1.running_mean";
    constants_info_[185].name = "mv2_features_5_conv_1_1_running_var";
    constants_info_[185].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[185].offset = 0;
    constants_info_[185].data_size = 768;
    constants_info_[185].from_folded = false;
    constants_info_[185].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[185].shape = {192};
    constants_info_[185].stride = {1};
    constants_info_[185].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[185].original_fqn = "mv2.features.5.conv.1.1.running_var";
    constants_info_[186].name = "mv2_features_5_conv_3_running_mean";
    constants_info_[186].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[186].offset = 0;
    constants_info_[186].data_size = 128;
    constants_info_[186].from_folded = false;
    constants_info_[186].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[186].shape = {32};
    constants_info_[186].stride = {1};
    constants_info_[186].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[186].original_fqn = "mv2.features.5.conv.3.running_mean";
    constants_info_[187].name = "mv2_features_5_conv_3_running_var";
    constants_info_[187].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[187].offset = 0;
    constants_info_[187].data_size = 128;
    constants_info_[187].from_folded = false;
    constants_info_[187].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[187].shape = {32};
    constants_info_[187].stride = {1};
    constants_info_[187].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[187].original_fqn = "mv2.features.5.conv.3.running_var";
    constants_info_[188].name = "mv2_features_6_conv_0_1_running_mean";
    constants_info_[188].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[188].offset = 0;
    constants_info_[188].data_size = 768;
    constants_info_[188].from_folded = false;
    constants_info_[188].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[188].shape = {192};
    constants_info_[188].stride = {1};
    constants_info_[188].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[188].original_fqn = "mv2.features.6.conv.0.1.running_mean";
    constants_info_[189].name = "mv2_features_6_conv_0_1_running_var";
    constants_info_[189].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[189].offset = 0;
    constants_info_[189].data_size = 768;
    constants_info_[189].from_folded = false;
    constants_info_[189].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[189].shape = {192};
    constants_info_[189].stride = {1};
    constants_info_[189].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[189].original_fqn = "mv2.features.6.conv.0.1.running_var";
    constants_info_[190].name = "mv2_features_6_conv_1_1_running_mean";
    constants_info_[190].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[190].offset = 0;
    constants_info_[190].data_size = 768;
    constants_info_[190].from_folded = false;
    constants_info_[190].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[190].shape = {192};
    constants_info_[190].stride = {1};
    constants_info_[190].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[190].original_fqn = "mv2.features.6.conv.1.1.running_mean";
    constants_info_[191].name = "mv2_features_6_conv_1_1_running_var";
    constants_info_[191].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[191].offset = 0;
    constants_info_[191].data_size = 768;
    constants_info_[191].from_folded = false;
    constants_info_[191].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[191].shape = {192};
    constants_info_[191].stride = {1};
    constants_info_[191].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[191].original_fqn = "mv2.features.6.conv.1.1.running_var";
    constants_info_[192].name = "mv2_features_6_conv_3_running_mean";
    constants_info_[192].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[192].offset = 0;
    constants_info_[192].data_size = 128;
    constants_info_[192].from_folded = false;
    constants_info_[192].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[192].shape = {32};
    constants_info_[192].stride = {1};
    constants_info_[192].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[192].original_fqn = "mv2.features.6.conv.3.running_mean";
    constants_info_[193].name = "mv2_features_6_conv_3_running_var";
    constants_info_[193].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[193].offset = 0;
    constants_info_[193].data_size = 128;
    constants_info_[193].from_folded = false;
    constants_info_[193].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[193].shape = {32};
    constants_info_[193].stride = {1};
    constants_info_[193].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[193].original_fqn = "mv2.features.6.conv.3.running_var";
    constants_info_[194].name = "mv2_features_7_conv_0_1_running_mean";
    constants_info_[194].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[194].offset = 0;
    constants_info_[194].data_size = 768;
    constants_info_[194].from_folded = false;
    constants_info_[194].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[194].shape = {192};
    constants_info_[194].stride = {1};
    constants_info_[194].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[194].original_fqn = "mv2.features.7.conv.0.1.running_mean";
    constants_info_[195].name = "mv2_features_7_conv_0_1_running_var";
    constants_info_[195].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[195].offset = 0;
    constants_info_[195].data_size = 768;
    constants_info_[195].from_folded = false;
    constants_info_[195].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[195].shape = {192};
    constants_info_[195].stride = {1};
    constants_info_[195].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[195].original_fqn = "mv2.features.7.conv.0.1.running_var";
    constants_info_[196].name = "mv2_features_7_conv_1_1_running_mean";
    constants_info_[196].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[196].offset = 0;
    constants_info_[196].data_size = 768;
    constants_info_[196].from_folded = false;
    constants_info_[196].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[196].shape = {192};
    constants_info_[196].stride = {1};
    constants_info_[196].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[196].original_fqn = "mv2.features.7.conv.1.1.running_mean";
    constants_info_[197].name = "mv2_features_7_conv_1_1_running_var";
    constants_info_[197].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[197].offset = 0;
    constants_info_[197].data_size = 768;
    constants_info_[197].from_folded = false;
    constants_info_[197].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[197].shape = {192};
    constants_info_[197].stride = {1};
    constants_info_[197].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[197].original_fqn = "mv2.features.7.conv.1.1.running_var";
    constants_info_[198].name = "mv2_features_7_conv_3_running_mean";
    constants_info_[198].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[198].offset = 0;
    constants_info_[198].data_size = 256;
    constants_info_[198].from_folded = false;
    constants_info_[198].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[198].shape = {64};
    constants_info_[198].stride = {1};
    constants_info_[198].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[198].original_fqn = "mv2.features.7.conv.3.running_mean";
    constants_info_[199].name = "mv2_features_7_conv_3_running_var";
    constants_info_[199].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[199].offset = 0;
    constants_info_[199].data_size = 256;
    constants_info_[199].from_folded = false;
    constants_info_[199].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[199].shape = {64};
    constants_info_[199].stride = {1};
    constants_info_[199].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[199].original_fqn = "mv2.features.7.conv.3.running_var";
    constants_info_[200].name = "mv2_features_8_conv_0_1_running_mean";
    constants_info_[200].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[200].offset = 0;
    constants_info_[200].data_size = 1536;
    constants_info_[200].from_folded = false;
    constants_info_[200].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[200].shape = {384};
    constants_info_[200].stride = {1};
    constants_info_[200].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[200].original_fqn = "mv2.features.8.conv.0.1.running_mean";
    constants_info_[201].name = "mv2_features_8_conv_0_1_running_var";
    constants_info_[201].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[201].offset = 0;
    constants_info_[201].data_size = 1536;
    constants_info_[201].from_folded = false;
    constants_info_[201].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[201].shape = {384};
    constants_info_[201].stride = {1};
    constants_info_[201].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[201].original_fqn = "mv2.features.8.conv.0.1.running_var";
    constants_info_[202].name = "mv2_features_8_conv_1_1_running_mean";
    constants_info_[202].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[202].offset = 0;
    constants_info_[202].data_size = 1536;
    constants_info_[202].from_folded = false;
    constants_info_[202].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[202].shape = {384};
    constants_info_[202].stride = {1};
    constants_info_[202].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[202].original_fqn = "mv2.features.8.conv.1.1.running_mean";
    constants_info_[203].name = "mv2_features_8_conv_1_1_running_var";
    constants_info_[203].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[203].offset = 0;
    constants_info_[203].data_size = 1536;
    constants_info_[203].from_folded = false;
    constants_info_[203].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[203].shape = {384};
    constants_info_[203].stride = {1};
    constants_info_[203].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[203].original_fqn = "mv2.features.8.conv.1.1.running_var";
    constants_info_[204].name = "mv2_features_8_conv_3_running_mean";
    constants_info_[204].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[204].offset = 0;
    constants_info_[204].data_size = 256;
    constants_info_[204].from_folded = false;
    constants_info_[204].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[204].shape = {64};
    constants_info_[204].stride = {1};
    constants_info_[204].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[204].original_fqn = "mv2.features.8.conv.3.running_mean";
    constants_info_[205].name = "mv2_features_8_conv_3_running_var";
    constants_info_[205].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[205].offset = 0;
    constants_info_[205].data_size = 256;
    constants_info_[205].from_folded = false;
    constants_info_[205].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[205].shape = {64};
    constants_info_[205].stride = {1};
    constants_info_[205].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[205].original_fqn = "mv2.features.8.conv.3.running_var";
    constants_info_[206].name = "mv2_features_9_conv_0_1_running_mean";
    constants_info_[206].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[206].offset = 0;
    constants_info_[206].data_size = 1536;
    constants_info_[206].from_folded = false;
    constants_info_[206].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[206].shape = {384};
    constants_info_[206].stride = {1};
    constants_info_[206].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[206].original_fqn = "mv2.features.9.conv.0.1.running_mean";
    constants_info_[207].name = "mv2_features_9_conv_0_1_running_var";
    constants_info_[207].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[207].offset = 0;
    constants_info_[207].data_size = 1536;
    constants_info_[207].from_folded = false;
    constants_info_[207].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[207].shape = {384};
    constants_info_[207].stride = {1};
    constants_info_[207].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[207].original_fqn = "mv2.features.9.conv.0.1.running_var";
    constants_info_[208].name = "mv2_features_9_conv_1_1_running_mean";
    constants_info_[208].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[208].offset = 0;
    constants_info_[208].data_size = 1536;
    constants_info_[208].from_folded = false;
    constants_info_[208].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[208].shape = {384};
    constants_info_[208].stride = {1};
    constants_info_[208].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[208].original_fqn = "mv2.features.9.conv.1.1.running_mean";
    constants_info_[209].name = "mv2_features_9_conv_1_1_running_var";
    constants_info_[209].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[209].offset = 0;
    constants_info_[209].data_size = 1536;
    constants_info_[209].from_folded = false;
    constants_info_[209].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[209].shape = {384};
    constants_info_[209].stride = {1};
    constants_info_[209].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[209].original_fqn = "mv2.features.9.conv.1.1.running_var";
    constants_info_[210].name = "mv2_features_9_conv_3_running_mean";
    constants_info_[210].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[210].offset = 0;
    constants_info_[210].data_size = 256;
    constants_info_[210].from_folded = false;
    constants_info_[210].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[210].shape = {64};
    constants_info_[210].stride = {1};
    constants_info_[210].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[210].original_fqn = "mv2.features.9.conv.3.running_mean";
    constants_info_[211].name = "mv2_features_9_conv_3_running_var";
    constants_info_[211].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[211].offset = 0;
    constants_info_[211].data_size = 256;
    constants_info_[211].from_folded = false;
    constants_info_[211].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[211].shape = {64};
    constants_info_[211].stride = {1};
    constants_info_[211].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[211].original_fqn = "mv2.features.9.conv.3.running_var";
    constants_info_[212].name = "mv2_features_10_conv_0_1_running_mean";
    constants_info_[212].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[212].offset = 0;
    constants_info_[212].data_size = 1536;
    constants_info_[212].from_folded = false;
    constants_info_[212].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[212].shape = {384};
    constants_info_[212].stride = {1};
    constants_info_[212].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[212].original_fqn = "mv2.features.10.conv.0.1.running_mean";
    constants_info_[213].name = "mv2_features_10_conv_0_1_running_var";
    constants_info_[213].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[213].offset = 0;
    constants_info_[213].data_size = 1536;
    constants_info_[213].from_folded = false;
    constants_info_[213].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[213].shape = {384};
    constants_info_[213].stride = {1};
    constants_info_[213].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[213].original_fqn = "mv2.features.10.conv.0.1.running_var";
    constants_info_[214].name = "mv2_features_10_conv_1_1_running_mean";
    constants_info_[214].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[214].offset = 0;
    constants_info_[214].data_size = 1536;
    constants_info_[214].from_folded = false;
    constants_info_[214].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[214].shape = {384};
    constants_info_[214].stride = {1};
    constants_info_[214].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[214].original_fqn = "mv2.features.10.conv.1.1.running_mean";
    constants_info_[215].name = "mv2_features_10_conv_1_1_running_var";
    constants_info_[215].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[215].offset = 0;
    constants_info_[215].data_size = 1536;
    constants_info_[215].from_folded = false;
    constants_info_[215].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[215].shape = {384};
    constants_info_[215].stride = {1};
    constants_info_[215].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[215].original_fqn = "mv2.features.10.conv.1.1.running_var";
    constants_info_[216].name = "mv2_features_10_conv_3_running_mean";
    constants_info_[216].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[216].offset = 0;
    constants_info_[216].data_size = 256;
    constants_info_[216].from_folded = false;
    constants_info_[216].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[216].shape = {64};
    constants_info_[216].stride = {1};
    constants_info_[216].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[216].original_fqn = "mv2.features.10.conv.3.running_mean";
    constants_info_[217].name = "mv2_features_10_conv_3_running_var";
    constants_info_[217].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[217].offset = 0;
    constants_info_[217].data_size = 256;
    constants_info_[217].from_folded = false;
    constants_info_[217].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[217].shape = {64};
    constants_info_[217].stride = {1};
    constants_info_[217].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[217].original_fqn = "mv2.features.10.conv.3.running_var";
    constants_info_[218].name = "mv2_features_11_conv_0_1_running_mean";
    constants_info_[218].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[218].offset = 0;
    constants_info_[218].data_size = 1536;
    constants_info_[218].from_folded = false;
    constants_info_[218].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[218].shape = {384};
    constants_info_[218].stride = {1};
    constants_info_[218].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[218].original_fqn = "mv2.features.11.conv.0.1.running_mean";
    constants_info_[219].name = "mv2_features_11_conv_0_1_running_var";
    constants_info_[219].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[219].offset = 0;
    constants_info_[219].data_size = 1536;
    constants_info_[219].from_folded = false;
    constants_info_[219].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[219].shape = {384};
    constants_info_[219].stride = {1};
    constants_info_[219].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[219].original_fqn = "mv2.features.11.conv.0.1.running_var";
    constants_info_[220].name = "mv2_features_11_conv_1_1_running_mean";
    constants_info_[220].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[220].offset = 0;
    constants_info_[220].data_size = 1536;
    constants_info_[220].from_folded = false;
    constants_info_[220].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[220].shape = {384};
    constants_info_[220].stride = {1};
    constants_info_[220].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[220].original_fqn = "mv2.features.11.conv.1.1.running_mean";
    constants_info_[221].name = "mv2_features_11_conv_1_1_running_var";
    constants_info_[221].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[221].offset = 0;
    constants_info_[221].data_size = 1536;
    constants_info_[221].from_folded = false;
    constants_info_[221].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[221].shape = {384};
    constants_info_[221].stride = {1};
    constants_info_[221].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[221].original_fqn = "mv2.features.11.conv.1.1.running_var";
    constants_info_[222].name = "mv2_features_11_conv_3_running_mean";
    constants_info_[222].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[222].offset = 0;
    constants_info_[222].data_size = 384;
    constants_info_[222].from_folded = false;
    constants_info_[222].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[222].shape = {96};
    constants_info_[222].stride = {1};
    constants_info_[222].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[222].original_fqn = "mv2.features.11.conv.3.running_mean";
    constants_info_[223].name = "mv2_features_11_conv_3_running_var";
    constants_info_[223].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[223].offset = 0;
    constants_info_[223].data_size = 384;
    constants_info_[223].from_folded = false;
    constants_info_[223].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[223].shape = {96};
    constants_info_[223].stride = {1};
    constants_info_[223].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[223].original_fqn = "mv2.features.11.conv.3.running_var";
    constants_info_[224].name = "mv2_features_12_conv_0_1_running_mean";
    constants_info_[224].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[224].offset = 0;
    constants_info_[224].data_size = 2304;
    constants_info_[224].from_folded = false;
    constants_info_[224].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[224].shape = {576};
    constants_info_[224].stride = {1};
    constants_info_[224].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[224].original_fqn = "mv2.features.12.conv.0.1.running_mean";
    constants_info_[225].name = "mv2_features_12_conv_0_1_running_var";
    constants_info_[225].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[225].offset = 0;
    constants_info_[225].data_size = 2304;
    constants_info_[225].from_folded = false;
    constants_info_[225].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[225].shape = {576};
    constants_info_[225].stride = {1};
    constants_info_[225].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[225].original_fqn = "mv2.features.12.conv.0.1.running_var";
    constants_info_[226].name = "mv2_features_12_conv_1_1_running_mean";
    constants_info_[226].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[226].offset = 0;
    constants_info_[226].data_size = 2304;
    constants_info_[226].from_folded = false;
    constants_info_[226].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[226].shape = {576};
    constants_info_[226].stride = {1};
    constants_info_[226].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[226].original_fqn = "mv2.features.12.conv.1.1.running_mean";
    constants_info_[227].name = "mv2_features_12_conv_1_1_running_var";
    constants_info_[227].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[227].offset = 0;
    constants_info_[227].data_size = 2304;
    constants_info_[227].from_folded = false;
    constants_info_[227].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[227].shape = {576};
    constants_info_[227].stride = {1};
    constants_info_[227].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[227].original_fqn = "mv2.features.12.conv.1.1.running_var";
    constants_info_[228].name = "mv2_features_12_conv_3_running_mean";
    constants_info_[228].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[228].offset = 0;
    constants_info_[228].data_size = 384;
    constants_info_[228].from_folded = false;
    constants_info_[228].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[228].shape = {96};
    constants_info_[228].stride = {1};
    constants_info_[228].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[228].original_fqn = "mv2.features.12.conv.3.running_mean";
    constants_info_[229].name = "mv2_features_12_conv_3_running_var";
    constants_info_[229].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[229].offset = 0;
    constants_info_[229].data_size = 384;
    constants_info_[229].from_folded = false;
    constants_info_[229].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[229].shape = {96};
    constants_info_[229].stride = {1};
    constants_info_[229].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[229].original_fqn = "mv2.features.12.conv.3.running_var";
    constants_info_[230].name = "mv2_features_13_conv_0_1_running_mean";
    constants_info_[230].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[230].offset = 0;
    constants_info_[230].data_size = 2304;
    constants_info_[230].from_folded = false;
    constants_info_[230].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[230].shape = {576};
    constants_info_[230].stride = {1};
    constants_info_[230].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[230].original_fqn = "mv2.features.13.conv.0.1.running_mean";
    constants_info_[231].name = "mv2_features_13_conv_0_1_running_var";
    constants_info_[231].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[231].offset = 0;
    constants_info_[231].data_size = 2304;
    constants_info_[231].from_folded = false;
    constants_info_[231].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[231].shape = {576};
    constants_info_[231].stride = {1};
    constants_info_[231].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[231].original_fqn = "mv2.features.13.conv.0.1.running_var";
    constants_info_[232].name = "mv2_features_13_conv_1_1_running_mean";
    constants_info_[232].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[232].offset = 0;
    constants_info_[232].data_size = 2304;
    constants_info_[232].from_folded = false;
    constants_info_[232].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[232].shape = {576};
    constants_info_[232].stride = {1};
    constants_info_[232].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[232].original_fqn = "mv2.features.13.conv.1.1.running_mean";
    constants_info_[233].name = "mv2_features_13_conv_1_1_running_var";
    constants_info_[233].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[233].offset = 0;
    constants_info_[233].data_size = 2304;
    constants_info_[233].from_folded = false;
    constants_info_[233].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[233].shape = {576};
    constants_info_[233].stride = {1};
    constants_info_[233].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[233].original_fqn = "mv2.features.13.conv.1.1.running_var";
    constants_info_[234].name = "mv2_features_13_conv_3_running_mean";
    constants_info_[234].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[234].offset = 0;
    constants_info_[234].data_size = 384;
    constants_info_[234].from_folded = false;
    constants_info_[234].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[234].shape = {96};
    constants_info_[234].stride = {1};
    constants_info_[234].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[234].original_fqn = "mv2.features.13.conv.3.running_mean";
    constants_info_[235].name = "mv2_features_13_conv_3_running_var";
    constants_info_[235].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[235].offset = 0;
    constants_info_[235].data_size = 384;
    constants_info_[235].from_folded = false;
    constants_info_[235].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[235].shape = {96};
    constants_info_[235].stride = {1};
    constants_info_[235].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[235].original_fqn = "mv2.features.13.conv.3.running_var";
    constants_info_[236].name = "mv2_features_14_conv_0_1_running_mean";
    constants_info_[236].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[236].offset = 0;
    constants_info_[236].data_size = 2304;
    constants_info_[236].from_folded = false;
    constants_info_[236].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[236].shape = {576};
    constants_info_[236].stride = {1};
    constants_info_[236].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[236].original_fqn = "mv2.features.14.conv.0.1.running_mean";
    constants_info_[237].name = "mv2_features_14_conv_0_1_running_var";
    constants_info_[237].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[237].offset = 0;
    constants_info_[237].data_size = 2304;
    constants_info_[237].from_folded = false;
    constants_info_[237].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[237].shape = {576};
    constants_info_[237].stride = {1};
    constants_info_[237].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[237].original_fqn = "mv2.features.14.conv.0.1.running_var";
    constants_info_[238].name = "mv2_features_14_conv_1_1_running_mean";
    constants_info_[238].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[238].offset = 0;
    constants_info_[238].data_size = 2304;
    constants_info_[238].from_folded = false;
    constants_info_[238].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[238].shape = {576};
    constants_info_[238].stride = {1};
    constants_info_[238].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[238].original_fqn = "mv2.features.14.conv.1.1.running_mean";
    constants_info_[239].name = "mv2_features_14_conv_1_1_running_var";
    constants_info_[239].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[239].offset = 0;
    constants_info_[239].data_size = 2304;
    constants_info_[239].from_folded = false;
    constants_info_[239].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[239].shape = {576};
    constants_info_[239].stride = {1};
    constants_info_[239].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[239].original_fqn = "mv2.features.14.conv.1.1.running_var";
    constants_info_[240].name = "mv2_features_14_conv_3_running_mean";
    constants_info_[240].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[240].offset = 0;
    constants_info_[240].data_size = 640;
    constants_info_[240].from_folded = false;
    constants_info_[240].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[240].shape = {160};
    constants_info_[240].stride = {1};
    constants_info_[240].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[240].original_fqn = "mv2.features.14.conv.3.running_mean";
    constants_info_[241].name = "mv2_features_14_conv_3_running_var";
    constants_info_[241].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[241].offset = 0;
    constants_info_[241].data_size = 640;
    constants_info_[241].from_folded = false;
    constants_info_[241].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[241].shape = {160};
    constants_info_[241].stride = {1};
    constants_info_[241].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[241].original_fqn = "mv2.features.14.conv.3.running_var";
    constants_info_[242].name = "mv2_features_15_conv_0_1_running_mean";
    constants_info_[242].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[242].offset = 0;
    constants_info_[242].data_size = 3840;
    constants_info_[242].from_folded = false;
    constants_info_[242].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[242].shape = {960};
    constants_info_[242].stride = {1};
    constants_info_[242].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[242].original_fqn = "mv2.features.15.conv.0.1.running_mean";
    constants_info_[243].name = "mv2_features_15_conv_0_1_running_var";
    constants_info_[243].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[243].offset = 0;
    constants_info_[243].data_size = 3840;
    constants_info_[243].from_folded = false;
    constants_info_[243].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[243].shape = {960};
    constants_info_[243].stride = {1};
    constants_info_[243].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[243].original_fqn = "mv2.features.15.conv.0.1.running_var";
    constants_info_[244].name = "mv2_features_15_conv_1_1_running_mean";
    constants_info_[244].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[244].offset = 0;
    constants_info_[244].data_size = 3840;
    constants_info_[244].from_folded = false;
    constants_info_[244].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[244].shape = {960};
    constants_info_[244].stride = {1};
    constants_info_[244].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[244].original_fqn = "mv2.features.15.conv.1.1.running_mean";
    constants_info_[245].name = "mv2_features_15_conv_1_1_running_var";
    constants_info_[245].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[245].offset = 0;
    constants_info_[245].data_size = 3840;
    constants_info_[245].from_folded = false;
    constants_info_[245].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[245].shape = {960};
    constants_info_[245].stride = {1};
    constants_info_[245].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[245].original_fqn = "mv2.features.15.conv.1.1.running_var";
    constants_info_[246].name = "mv2_features_15_conv_3_running_mean";
    constants_info_[246].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[246].offset = 0;
    constants_info_[246].data_size = 640;
    constants_info_[246].from_folded = false;
    constants_info_[246].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[246].shape = {160};
    constants_info_[246].stride = {1};
    constants_info_[246].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[246].original_fqn = "mv2.features.15.conv.3.running_mean";
    constants_info_[247].name = "mv2_features_15_conv_3_running_var";
    constants_info_[247].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[247].offset = 0;
    constants_info_[247].data_size = 640;
    constants_info_[247].from_folded = false;
    constants_info_[247].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[247].shape = {160};
    constants_info_[247].stride = {1};
    constants_info_[247].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[247].original_fqn = "mv2.features.15.conv.3.running_var";
    constants_info_[248].name = "mv2_features_16_conv_0_1_running_mean";
    constants_info_[248].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[248].offset = 0;
    constants_info_[248].data_size = 3840;
    constants_info_[248].from_folded = false;
    constants_info_[248].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[248].shape = {960};
    constants_info_[248].stride = {1};
    constants_info_[248].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[248].original_fqn = "mv2.features.16.conv.0.1.running_mean";
    constants_info_[249].name = "mv2_features_16_conv_0_1_running_var";
    constants_info_[249].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[249].offset = 0;
    constants_info_[249].data_size = 3840;
    constants_info_[249].from_folded = false;
    constants_info_[249].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[249].shape = {960};
    constants_info_[249].stride = {1};
    constants_info_[249].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[249].original_fqn = "mv2.features.16.conv.0.1.running_var";
    constants_info_[250].name = "mv2_features_16_conv_1_1_running_mean";
    constants_info_[250].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[250].offset = 0;
    constants_info_[250].data_size = 3840;
    constants_info_[250].from_folded = false;
    constants_info_[250].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[250].shape = {960};
    constants_info_[250].stride = {1};
    constants_info_[250].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[250].original_fqn = "mv2.features.16.conv.1.1.running_mean";
    constants_info_[251].name = "mv2_features_16_conv_1_1_running_var";
    constants_info_[251].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[251].offset = 0;
    constants_info_[251].data_size = 3840;
    constants_info_[251].from_folded = false;
    constants_info_[251].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[251].shape = {960};
    constants_info_[251].stride = {1};
    constants_info_[251].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[251].original_fqn = "mv2.features.16.conv.1.1.running_var";
    constants_info_[252].name = "mv2_features_16_conv_3_running_mean";
    constants_info_[252].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[252].offset = 0;
    constants_info_[252].data_size = 640;
    constants_info_[252].from_folded = false;
    constants_info_[252].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[252].shape = {160};
    constants_info_[252].stride = {1};
    constants_info_[252].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[252].original_fqn = "mv2.features.16.conv.3.running_mean";
    constants_info_[253].name = "mv2_features_16_conv_3_running_var";
    constants_info_[253].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[253].offset = 0;
    constants_info_[253].data_size = 640;
    constants_info_[253].from_folded = false;
    constants_info_[253].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[253].shape = {160};
    constants_info_[253].stride = {1};
    constants_info_[253].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[253].original_fqn = "mv2.features.16.conv.3.running_var";
    constants_info_[254].name = "mv2_features_17_conv_0_1_running_mean";
    constants_info_[254].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[254].offset = 0;
    constants_info_[254].data_size = 3840;
    constants_info_[254].from_folded = false;
    constants_info_[254].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[254].shape = {960};
    constants_info_[254].stride = {1};
    constants_info_[254].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[254].original_fqn = "mv2.features.17.conv.0.1.running_mean";
    constants_info_[255].name = "mv2_features_17_conv_0_1_running_var";
    constants_info_[255].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[255].offset = 0;
    constants_info_[255].data_size = 3840;
    constants_info_[255].from_folded = false;
    constants_info_[255].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[255].shape = {960};
    constants_info_[255].stride = {1};
    constants_info_[255].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[255].original_fqn = "mv2.features.17.conv.0.1.running_var";
    constants_info_[256].name = "mv2_features_17_conv_1_1_running_mean";
    constants_info_[256].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[256].offset = 0;
    constants_info_[256].data_size = 3840;
    constants_info_[256].from_folded = false;
    constants_info_[256].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[256].shape = {960};
    constants_info_[256].stride = {1};
    constants_info_[256].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[256].original_fqn = "mv2.features.17.conv.1.1.running_mean";
    constants_info_[257].name = "mv2_features_17_conv_1_1_running_var";
    constants_info_[257].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[257].offset = 0;
    constants_info_[257].data_size = 3840;
    constants_info_[257].from_folded = false;
    constants_info_[257].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[257].shape = {960};
    constants_info_[257].stride = {1};
    constants_info_[257].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[257].original_fqn = "mv2.features.17.conv.1.1.running_var";
    constants_info_[258].name = "mv2_features_17_conv_3_running_mean";
    constants_info_[258].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[258].offset = 0;
    constants_info_[258].data_size = 1280;
    constants_info_[258].from_folded = false;
    constants_info_[258].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[258].shape = {320};
    constants_info_[258].stride = {1};
    constants_info_[258].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[258].original_fqn = "mv2.features.17.conv.3.running_mean";
    constants_info_[259].name = "mv2_features_17_conv_3_running_var";
    constants_info_[259].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[259].offset = 0;
    constants_info_[259].data_size = 1280;
    constants_info_[259].from_folded = false;
    constants_info_[259].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[259].shape = {320};
    constants_info_[259].stride = {1};
    constants_info_[259].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[259].original_fqn = "mv2.features.17.conv.3.running_var";
    constants_info_[260].name = "mv2_features_18_1_running_mean";
    constants_info_[260].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[260].offset = 0;
    constants_info_[260].data_size = 5120;
    constants_info_[260].from_folded = false;
    constants_info_[260].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[260].shape = {1280};
    constants_info_[260].stride = {1};
    constants_info_[260].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[260].original_fqn = "mv2.features.18.1.running_mean";
    constants_info_[261].name = "mv2_features_18_1_running_var";
    constants_info_[261].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[261].offset = 0;
    constants_info_[261].data_size = 5120;
    constants_info_[261].from_folded = false;
    constants_info_[261].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[261].shape = {1280};
    constants_info_[261].stride = {1};
    constants_info_[261].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[261].original_fqn = "mv2.features.18.1.running_var";
    update_constants_map(std::move(constants_map));
    update_constants_array(std::move(constants_array));
    in_spec_ = R"([1, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.dict", "context": "[]", "children_spec": []}]}])";
    out_spec_ = R"([1, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}]}])";
    outputs_info_[0].name = "output0";
    this->kernels_ = std::make_unique<AOTInductorModelKernels>();
}

std::unordered_map<std::string, AtenTensorHandle> AOTInductorModel::const_run_impl(
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor,
    bool initialization
) {

    if (!initialization) {
        std::cerr << "[WARNING] Calling constant_folding in model, but compiled with config: "
                  << "aot_inductor.use_runtime_constant_folding=False\n";
    }
    return {};
}
} // namespace torch::aot_inductor
using namespace torch::aot_inductor;

template <typename in_ptr0_type_, typename out_ptr0_type_, typename kernels_type_>
static inline void call_triton_poi_fused_convolution_0(
    const in_ptr0_type_& in_ptr0,
    const out_ptr0_type_& out_ptr0,
    int64_t ynumel,
    int64_t xnumel,
    int32_t device_idx_,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    /*
    async_compile.triton('triton_poi_fused_convolution_0', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.pointwise(
        size_hints={'y': 4, 'x': 65536}, tile_hint=TileHint.SQUARE,
        filename=__file__,
        triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4F87BAC7C78026030CE21ABCD241F4211145E4ACCDC53C53E0CC97717CB6F329', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'tiling_scores': {'y': 451584, 'x': 602112}},
        min_elem_per_thread=0
    )
    @triton.jit
    def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
        ynumel = 3
        xnumel = 50176
        yoffset = tl.program_id(1) * YBLOCK
        yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
        ymask = yindex < ynumel
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
        xmask = xindex < xnumel
        x1 = xindex
        y0 = yindex
        tmp0 = tl.load(in_ptr0 + (x1 + 50176*y0), xmask & ymask, eviction_policy='evict_last')
        tl.store(out_ptr0 + (y0 + 3*x1), tmp0, xmask & ymask)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = ((xnumel + (256 - 1)) / (256));
    uint32_t grid_1 = ((ynumel + (4 - 1)) / (4));
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused_convolution_0 == nullptr) {
        kernels_.triton_poi_fused_convolution_0 = loadKernel("/home/gasoonjia/executorch/cxzopurug2u2kff3zliyvn25jrj6hvbvo6qrp26tzvi5i7zoaq2b.cubin", "triton_poi_fused_convolution_0", 4160, cubin_dir_); 
    }
    CUdeviceptr var_0 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_1 = reinterpret_cast<CUdeviceptr>(out_ptr0.data_ptr());
    int var_2 = ynumel;
    int var_3 = xnumel;
    CUdeviceptr global_scratch_4 = 0;
    void* kernel_args_[] = {&var_0, &var_1, &var_2, &var_3, &global_scratch_4};
    launchKernel(kernels_.triton_poi_fused_convolution_0, grid_0, grid_1, grid_2, 4, 4160, kernel_args_, stream_);
}

template <typename in_ptr0_type_, typename out_ptr0_type_, typename kernels_type_>
static inline void call_triton_poi_fused_convolution_1(
    const in_ptr0_type_& in_ptr0,
    const out_ptr0_type_& out_ptr0,
    int64_t ynumel,
    int64_t xnumel,
    int32_t device_idx_,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    /*
    async_compile.triton('triton_poi_fused_convolution_1', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.pointwise(
        size_hints={'y': 128, 'x': 16}, tile_hint=TileHint.SQUARE,
        filename=__file__,
        triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4F87BAC7C78026030CE21ABCD241F4211145E4ACCDC53C53E0CC97717CB6F329', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'tiling_scores': {'y': 6912, 'x': 3456}},
        min_elem_per_thread=0
    )
    @triton.jit
    def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
        ynumel = 96
        xnumel = 9
        yoffset = tl.program_id(1) * YBLOCK
        yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
        ymask = yindex < ynumel
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
        xmask = xindex < xnumel
        x2 = xindex
        y3 = yindex
        y0 = (yindex % 3)
        y1 = yindex // 3
        tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
        tl.store(out_ptr0 + (y0 + 3*x2 + 27*y1), tmp0, xmask & ymask)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = ((xnumel + (16 - 1)) / (16));
    uint32_t grid_1 = ((ynumel + (64 - 1)) / (64));
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused_convolution_1 == nullptr) {
        kernels_.triton_poi_fused_convolution_1 = loadKernel("/home/gasoonjia/executorch/cwvumepeeo7fjwjgwncwiji54ff6le55tfzp4kzgc4qgueefvrjb.cubin", "triton_poi_fused_convolution_1", 4352, cubin_dir_); 
    }
    CUdeviceptr var_5 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_6 = reinterpret_cast<CUdeviceptr>(out_ptr0.data_ptr());
    int var_7 = ynumel;
    int var_8 = xnumel;
    CUdeviceptr global_scratch_9 = 0;
    void* kernel_args_[] = {&var_5, &var_6, &var_7, &var_8, &global_scratch_9};
    launchKernel(kernels_.triton_poi_fused_convolution_1, grid_0, grid_1, grid_2, 4, 4352, kernel_args_, stream_);
}

template <typename in_out_ptr0_type_, typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename kernels_type_>
static inline void call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_2(
    const in_out_ptr0_type_& in_out_ptr0,
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    int64_t xnumel,
    int32_t device_idx_,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    /*
    async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_2', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.pointwise(
        size_hints={'x': 524288}, 
        filename=__file__,
        triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '4F87BAC7C78026030CE21ABCD241F4211145E4ACCDC53C53E0CC97717CB6F329', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'tiling_scores': {'x': 4817408}},
        min_elem_per_thread=0
    )
    @triton.jit
    def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
        xnumel = 401408
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        x2 = xindex
        x0 = (xindex % 32)
        tmp0 = tl.load(in_out_ptr0 + (x2), None)
        tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
        tmp2 = tmp0 - tmp1
        tmp4 = 1e-05
        tmp5 = tmp3 + tmp4
        tmp6 = libdevice.sqrt(tmp5)
        tmp7 = tl.full([1], 1, tl.int32)
        tmp8 = (tmp7 / tmp6)
        tmp9 = 1.0
        tmp10 = tmp8 * tmp9
        tmp11 = tmp2 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = 0.0
        tmp17 = triton_helpers.maximum(tmp15, tmp16)
        tmp18 = 6.0
        tmp19 = triton_helpers.minimum(tmp17, tmp18)
        tl.store(in_out_ptr0 + (x2), tmp19, None)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = ((xnumel + (512 - 1)) / (512));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_2 == nullptr) {
        kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_2 = loadKernel("/home/gasoonjia/executorch/c74zcdwgzyij2kup6edvwy6x4v2o3kzogatnfm3fd4ttgs3qq26p.cubin", "triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_2", 0, cubin_dir_); 
    }
    CUdeviceptr var_10 = reinterpret_cast<CUdeviceptr>(in_out_ptr0.data_ptr());
    CUdeviceptr var_11 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_12 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_13 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_14 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    int var_15 = xnumel;
    CUdeviceptr global_scratch_16 = 0;
    void* kernel_args_[] = {&var_10, &var_11, &var_12, &var_13, &var_14, &var_15, &global_scratch_16};
    launchKernel(kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_2, grid_0, grid_1, grid_2, 8, 0, kernel_args_, stream_);
}

template <typename in_out_ptr0_type_, typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename kernels_type_>
static inline void call_triton_poi_fused__native_batch_norm_legit_no_training_3(
    const in_out_ptr0_type_& in_out_ptr0,
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    int64_t xnumel,
    int32_t device_idx_,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    /*
    async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_3', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.pointwise(
        size_hints={'x': 262144}, 
        filename=__file__,
        triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '4F87BAC7C78026030CE21ABCD241F4211145E4ACCDC53C53E0CC97717CB6F329', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'tiling_scores': {'x': 2408704}},
        min_elem_per_thread=0
    )
    @triton.jit
    def triton_poi_fused__native_batch_norm_legit_no_training_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
        xnumel = 200704
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        x2 = xindex
        x0 = (xindex % 16)
        tmp0 = tl.load(in_out_ptr0 + (x2), None)
        tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
        tmp2 = tmp0 - tmp1
        tmp4 = 1e-05
        tmp5 = tmp3 + tmp4
        tmp6 = libdevice.sqrt(tmp5)
        tmp7 = tl.full([1], 1, tl.int32)
        tmp8 = (tmp7 / tmp6)
        tmp9 = 1.0
        tmp10 = tmp8 * tmp9
        tmp11 = tmp2 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tl.store(in_out_ptr0 + (x2), tmp15, None)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = ((xnumel + (512 - 1)) / (512));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused__native_batch_norm_legit_no_training_3 == nullptr) {
        kernels_.triton_poi_fused__native_batch_norm_legit_no_training_3 = loadKernel("/home/gasoonjia/executorch/cgpouheql4rpwtcaretoqzvk65fkvmoma6frdyhd3ilsvuggrlzy.cubin", "triton_poi_fused__native_batch_norm_legit_no_training_3", 0, cubin_dir_); 
    }
    CUdeviceptr var_17 = reinterpret_cast<CUdeviceptr>(in_out_ptr0.data_ptr());
    CUdeviceptr var_18 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_19 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_20 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_21 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    int var_22 = xnumel;
    CUdeviceptr global_scratch_23 = 0;
    void* kernel_args_[] = {&var_17, &var_18, &var_19, &var_20, &var_21, &var_22, &global_scratch_23};
    launchKernel(kernels_.triton_poi_fused__native_batch_norm_legit_no_training_3, grid_0, grid_1, grid_2, 8, 0, kernel_args_, stream_);
}

template <typename in_out_ptr0_type_, typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename kernels_type_>
static inline void call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_4(
    const in_out_ptr0_type_& in_out_ptr0,
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    int64_t xnumel,
    int32_t device_idx_,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    /*
    async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_4', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.pointwise(
        size_hints={'x': 2097152}, 
        filename=__file__,
        triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '4F87BAC7C78026030CE21ABCD241F4211145E4ACCDC53C53E0CC97717CB6F329', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'tiling_scores': {'x': 14452224}},
        min_elem_per_thread=0
    )
    @triton.jit
    def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
        xnumel = 1204224
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        x2 = xindex
        x0 = (xindex % 96)
        tmp0 = tl.load(in_out_ptr0 + (x2), None)
        tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
        tmp2 = tmp0 - tmp1
        tmp4 = 1e-05
        tmp5 = tmp3 + tmp4
        tmp6 = libdevice.sqrt(tmp5)
        tmp7 = tl.full([1], 1, tl.int32)
        tmp8 = (tmp7 / tmp6)
        tmp9 = 1.0
        tmp10 = tmp8 * tmp9
        tmp11 = tmp2 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = 0.0
        tmp17 = triton_helpers.maximum(tmp15, tmp16)
        tmp18 = 6.0
        tmp19 = triton_helpers.minimum(tmp17, tmp18)
        tl.store(in_out_ptr0 + (x2), tmp19, None)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = ((xnumel + (512 - 1)) / (512));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_4 == nullptr) {
        kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_4 = loadKernel("/home/gasoonjia/executorch/cd4lomi6yttiqc3qnhhhc675ta5iienuto5t67ybtshlxzp6p4ud.cubin", "triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_4", 0, cubin_dir_); 
    }
    CUdeviceptr var_24 = reinterpret_cast<CUdeviceptr>(in_out_ptr0.data_ptr());
    CUdeviceptr var_25 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_26 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_27 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_28 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    int var_29 = xnumel;
    CUdeviceptr global_scratch_30 = 0;
    void* kernel_args_[] = {&var_24, &var_25, &var_26, &var_27, &var_28, &var_29, &global_scratch_30};
    launchKernel(kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_4, grid_0, grid_1, grid_2, 8, 0, kernel_args_, stream_);
}

template <typename in_out_ptr0_type_, typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename kernels_type_>
static inline void call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_5(
    const in_out_ptr0_type_& in_out_ptr0,
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    int64_t xnumel,
    int32_t device_idx_,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    /*
    async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_5', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.pointwise(
        size_hints={'x': 524288}, 
        filename=__file__,
        triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '4F87BAC7C78026030CE21ABCD241F4211145E4ACCDC53C53E0CC97717CB6F329', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'tiling_scores': {'x': 3614208}},
        min_elem_per_thread=0
    )
    @triton.jit
    def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
        xnumel = 301056
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        x0 = (xindex % 96)
        tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
        tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
        tmp2 = tmp0 - tmp1
        tmp4 = 1e-05
        tmp5 = tmp3 + tmp4
        tmp6 = libdevice.sqrt(tmp5)
        tmp7 = tl.full([1], 1, tl.int32)
        tmp8 = (tmp7 / tmp6)
        tmp9 = 1.0
        tmp10 = tmp8 * tmp9
        tmp11 = tmp2 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = 0.0
        tmp17 = triton_helpers.maximum(tmp15, tmp16)
        tmp18 = 6.0
        tmp19 = triton_helpers.minimum(tmp17, tmp18)
        tl.store(in_out_ptr0 + (x2), tmp19, xmask)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = ((xnumel + (512 - 1)) / (512));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_5 == nullptr) {
        kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_5 = loadKernel("/home/gasoonjia/executorch/c7k3euhriolgsebdxauqyj6p2zdkse6qa6e4ylwbrc7765zcfd3m.cubin", "triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_5", 0, cubin_dir_); 
    }
    CUdeviceptr var_31 = reinterpret_cast<CUdeviceptr>(in_out_ptr0.data_ptr());
    CUdeviceptr var_32 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_33 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_34 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_35 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    int var_36 = xnumel;
    CUdeviceptr global_scratch_37 = 0;
    void* kernel_args_[] = {&var_31, &var_32, &var_33, &var_34, &var_35, &var_36, &global_scratch_37};
    launchKernel(kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_5, grid_0, grid_1, grid_2, 8, 0, kernel_args_, stream_);
}

template <typename in_out_ptr0_type_, typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename kernels_type_>
static inline void call_triton_poi_fused__native_batch_norm_legit_no_training_6(
    const in_out_ptr0_type_& in_out_ptr0,
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    int64_t xnumel,
    int32_t device_idx_,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    /*
    async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_6', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.pointwise(
        size_hints={'x': 131072}, 
        filename=__file__,
        triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '4F87BAC7C78026030CE21ABCD241F4211145E4ACCDC53C53E0CC97717CB6F329', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'tiling_scores': {'x': 903552}},
        min_elem_per_thread=0
    )
    @triton.jit
    def triton_poi_fused__native_batch_norm_legit_no_training_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
        xnumel = 75264
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        x0 = (xindex % 24)
        tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
        tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
        tmp2 = tmp0 - tmp1
        tmp4 = 1e-05
        tmp5 = tmp3 + tmp4
        tmp6 = libdevice.sqrt(tmp5)
        tmp7 = tl.full([1], 1, tl.int32)
        tmp8 = (tmp7 / tmp6)
        tmp9 = 1.0
        tmp10 = tmp8 * tmp9
        tmp11 = tmp2 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tl.store(in_out_ptr0 + (x2), tmp15, xmask)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = ((xnumel + (512 - 1)) / (512));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused__native_batch_norm_legit_no_training_6 == nullptr) {
        kernels_.triton_poi_fused__native_batch_norm_legit_no_training_6 = loadKernel("/home/gasoonjia/executorch/ckneyyhrfy6dkwkb6gaodbhn3l2khublcfvrwlajocypscgzcbft.cubin", "triton_poi_fused__native_batch_norm_legit_no_training_6", 0, cubin_dir_); 
    }
    CUdeviceptr var_38 = reinterpret_cast<CUdeviceptr>(in_out_ptr0.data_ptr());
    CUdeviceptr var_39 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_40 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_41 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_42 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    int var_43 = xnumel;
    CUdeviceptr global_scratch_44 = 0;
    void* kernel_args_[] = {&var_38, &var_39, &var_40, &var_41, &var_42, &var_43, &global_scratch_44};
    launchKernel(kernels_.triton_poi_fused__native_batch_norm_legit_no_training_6, grid_0, grid_1, grid_2, 8, 0, kernel_args_, stream_);
}

template <typename in_out_ptr0_type_, typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename kernels_type_>
static inline void call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_7(
    const in_out_ptr0_type_& in_out_ptr0,
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    int64_t xnumel,
    int32_t device_idx_,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    /*
    async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_7', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.pointwise(
        size_hints={'x': 524288}, 
        filename=__file__,
        triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '4F87BAC7C78026030CE21ABCD241F4211145E4ACCDC53C53E0CC97717CB6F329', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'tiling_scores': {'x': 5421312}},
        min_elem_per_thread=0
    )
    @triton.jit
    def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
        xnumel = 451584
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        x0 = (xindex % 144)
        tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
        tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
        tmp2 = tmp0 - tmp1
        tmp4 = 1e-05
        tmp5 = tmp3 + tmp4
        tmp6 = libdevice.sqrt(tmp5)
        tmp7 = tl.full([1], 1, tl.int32)
        tmp8 = (tmp7 / tmp6)
        tmp9 = 1.0
        tmp10 = tmp8 * tmp9
        tmp11 = tmp2 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = 0.0
        tmp17 = triton_helpers.maximum(tmp15, tmp16)
        tmp18 = 6.0
        tmp19 = triton_helpers.minimum(tmp17, tmp18)
        tl.store(in_out_ptr0 + (x2), tmp19, xmask)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = ((xnumel + (512 - 1)) / (512));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_7 == nullptr) {
        kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_7 = loadKernel("/home/gasoonjia/executorch/c656cklj2pms2iadvspxywzssohwg3dtxcy4dlztkpnbgadleo2n.cubin", "triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_7", 0, cubin_dir_); 
    }
    CUdeviceptr var_45 = reinterpret_cast<CUdeviceptr>(in_out_ptr0.data_ptr());
    CUdeviceptr var_46 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_47 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_48 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_49 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    int var_50 = xnumel;
    CUdeviceptr global_scratch_51 = 0;
    void* kernel_args_[] = {&var_45, &var_46, &var_47, &var_48, &var_49, &var_50, &global_scratch_51};
    launchKernel(kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_7, grid_0, grid_1, grid_2, 8, 0, kernel_args_, stream_);
}

template <typename in_out_ptr0_type_, typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename in_ptr4_type_, typename kernels_type_>
static inline void call_triton_poi_fused__native_batch_norm_legit_no_training_add_8(
    const in_out_ptr0_type_& in_out_ptr0,
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    const in_ptr4_type_& in_ptr4,
    int64_t xnumel,
    int32_t device_idx_,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    /*
    async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_8', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.pointwise(
        size_hints={'x': 131072}, 
        filename=__file__,
        triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '4F87BAC7C78026030CE21ABCD241F4211145E4ACCDC53C53E0CC97717CB6F329', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'tiling_scores': {'x': 1204608}},
        min_elem_per_thread=0
    )
    @triton.jit
    def triton_poi_fused__native_batch_norm_legit_no_training_add_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
        xnumel = 75264
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        x0 = (xindex % 24)
        tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
        tmp1 = tl.load(in_ptr0 + (x2), xmask)
        tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
        tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
        tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
        tmp3 = tmp1 - tmp2
        tmp5 = 1e-05
        tmp6 = tmp4 + tmp5
        tmp7 = libdevice.sqrt(tmp6)
        tmp8 = tl.full([1], 1, tl.int32)
        tmp9 = (tmp8 / tmp7)
        tmp10 = 1.0
        tmp11 = tmp9 * tmp10
        tmp12 = tmp3 * tmp11
        tmp14 = tmp12 * tmp13
        tmp16 = tmp14 + tmp15
        tmp17 = tmp0 + tmp16
        tl.store(in_out_ptr0 + (x2), tmp17, xmask)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = ((xnumel + (512 - 1)) / (512));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused__native_batch_norm_legit_no_training_add_8 == nullptr) {
        kernels_.triton_poi_fused__native_batch_norm_legit_no_training_add_8 = loadKernel("/home/gasoonjia/executorch/cx6i7mlkzaxbh5vk47jvftmw7ls63iczwax45psdovflgeuxo4z5.cubin", "triton_poi_fused__native_batch_norm_legit_no_training_add_8", 0, cubin_dir_); 
    }
    CUdeviceptr var_52 = reinterpret_cast<CUdeviceptr>(in_out_ptr0.data_ptr());
    CUdeviceptr var_53 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_54 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_55 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_56 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    CUdeviceptr var_57 = reinterpret_cast<CUdeviceptr>(in_ptr4.data_ptr());
    int var_58 = xnumel;
    CUdeviceptr global_scratch_59 = 0;
    void* kernel_args_[] = {&var_52, &var_53, &var_54, &var_55, &var_56, &var_57, &var_58, &global_scratch_59};
    launchKernel(kernels_.triton_poi_fused__native_batch_norm_legit_no_training_add_8, grid_0, grid_1, grid_2, 8, 0, kernel_args_, stream_);
}

template <typename in_out_ptr0_type_, typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename kernels_type_>
static inline void call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9(
    const in_out_ptr0_type_& in_out_ptr0,
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    int64_t xnumel,
    int32_t device_idx_,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    /*
    async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.pointwise(
        size_hints={'x': 131072}, 
        filename=__file__,
        triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '4F87BAC7C78026030CE21ABCD241F4211145E4ACCDC53C53E0CC97717CB6F329', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'tiling_scores': {'x': 1357056}},
        min_elem_per_thread=0
    )
    @triton.jit
    def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
        xnumel = 112896
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        x0 = (xindex % 144)
        tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
        tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
        tmp2 = tmp0 - tmp1
        tmp4 = 1e-05
        tmp5 = tmp3 + tmp4
        tmp6 = libdevice.sqrt(tmp5)
        tmp7 = tl.full([1], 1, tl.int32)
        tmp8 = (tmp7 / tmp6)
        tmp9 = 1.0
        tmp10 = tmp8 * tmp9
        tmp11 = tmp2 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = 0.0
        tmp17 = triton_helpers.maximum(tmp15, tmp16)
        tmp18 = 6.0
        tmp19 = triton_helpers.minimum(tmp17, tmp18)
        tl.store(in_out_ptr0 + (x2), tmp19, xmask)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = ((xnumel + (512 - 1)) / (512));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9 == nullptr) {
        kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9 = loadKernel("/home/gasoonjia/executorch/cguqxqtxyno4btxkugwlps3lbm56okihdtohl53vad3fobxqjmuc.cubin", "triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9", 0, cubin_dir_); 
    }
    CUdeviceptr var_60 = reinterpret_cast<CUdeviceptr>(in_out_ptr0.data_ptr());
    CUdeviceptr var_61 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_62 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_63 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_64 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    int var_65 = xnumel;
    CUdeviceptr global_scratch_66 = 0;
    void* kernel_args_[] = {&var_60, &var_61, &var_62, &var_63, &var_64, &var_65, &global_scratch_66};
    launchKernel(kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9, grid_0, grid_1, grid_2, 8, 0, kernel_args_, stream_);
}

template <typename in_out_ptr0_type_, typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename kernels_type_>
static inline void call_triton_poi_fused__native_batch_norm_legit_no_training_10(
    const in_out_ptr0_type_& in_out_ptr0,
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    int64_t xnumel,
    int32_t device_idx_,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    /*
    async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_10', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.pointwise(
        size_hints={'x': 32768}, 
        filename=__file__,
        triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '4F87BAC7C78026030CE21ABCD241F4211145E4ACCDC53C53E0CC97717CB6F329', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'tiling_scores': {'x': 301568}},
        min_elem_per_thread=0
    )
    @triton.jit
    def triton_poi_fused__native_batch_norm_legit_no_training_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
        xnumel = 25088
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        x0 = (xindex % 32)
        tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
        tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
        tmp2 = tmp0 - tmp1
        tmp4 = 1e-05
        tmp5 = tmp3 + tmp4
        tmp6 = libdevice.sqrt(tmp5)
        tmp7 = tl.full([1], 1, tl.int32)
        tmp8 = (tmp7 / tmp6)
        tmp9 = 1.0
        tmp10 = tmp8 * tmp9
        tmp11 = tmp2 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tl.store(in_out_ptr0 + (x2), tmp15, xmask)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = ((xnumel + (128 - 1)) / (128));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused__native_batch_norm_legit_no_training_10 == nullptr) {
        kernels_.triton_poi_fused__native_batch_norm_legit_no_training_10 = loadKernel("/home/gasoonjia/executorch/cxurxwta5vlfbwctjkkticzdokzzr73dnqi2s4asnb4ckdieiii5.cubin", "triton_poi_fused__native_batch_norm_legit_no_training_10", 0, cubin_dir_); 
    }
    CUdeviceptr var_67 = reinterpret_cast<CUdeviceptr>(in_out_ptr0.data_ptr());
    CUdeviceptr var_68 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_69 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_70 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_71 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    int var_72 = xnumel;
    CUdeviceptr global_scratch_73 = 0;
    void* kernel_args_[] = {&var_67, &var_68, &var_69, &var_70, &var_71, &var_72, &global_scratch_73};
    launchKernel(kernels_.triton_poi_fused__native_batch_norm_legit_no_training_10, grid_0, grid_1, grid_2, 4, 0, kernel_args_, stream_);
}

template <typename in_out_ptr0_type_, typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename kernels_type_>
static inline void call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11(
    const in_out_ptr0_type_& in_out_ptr0,
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    int64_t xnumel,
    int32_t device_idx_,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    /*
    async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.pointwise(
        size_hints={'x': 262144}, 
        filename=__file__,
        triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '4F87BAC7C78026030CE21ABCD241F4211145E4ACCDC53C53E0CC97717CB6F329', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'tiling_scores': {'x': 1809408}},
        min_elem_per_thread=0
    )
    @triton.jit
    def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
        xnumel = 150528
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        x0 = (xindex % 192)
        tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
        tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
        tmp2 = tmp0 - tmp1
        tmp4 = 1e-05
        tmp5 = tmp3 + tmp4
        tmp6 = libdevice.sqrt(tmp5)
        tmp7 = tl.full([1], 1, tl.int32)
        tmp8 = (tmp7 / tmp6)
        tmp9 = 1.0
        tmp10 = tmp8 * tmp9
        tmp11 = tmp2 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = 0.0
        tmp17 = triton_helpers.maximum(tmp15, tmp16)
        tmp18 = 6.0
        tmp19 = triton_helpers.minimum(tmp17, tmp18)
        tl.store(in_out_ptr0 + (x2), tmp19, xmask)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = ((xnumel + (512 - 1)) / (512));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11 == nullptr) {
        kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11 = loadKernel("/home/gasoonjia/executorch/cedahkafk34ku7ldx6xjj5g7kdphvxc3vywwrxoqogx6xqos4uft.cubin", "triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11", 0, cubin_dir_); 
    }
    CUdeviceptr var_74 = reinterpret_cast<CUdeviceptr>(in_out_ptr0.data_ptr());
    CUdeviceptr var_75 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_76 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_77 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_78 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    int var_79 = xnumel;
    CUdeviceptr global_scratch_80 = 0;
    void* kernel_args_[] = {&var_74, &var_75, &var_76, &var_77, &var_78, &var_79, &global_scratch_80};
    launchKernel(kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11, grid_0, grid_1, grid_2, 8, 0, kernel_args_, stream_);
}

template <typename in_out_ptr0_type_, typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename in_ptr4_type_, typename kernels_type_>
static inline void call_triton_poi_fused__native_batch_norm_legit_no_training_add_12(
    const in_out_ptr0_type_& in_out_ptr0,
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    const in_ptr4_type_& in_ptr4,
    int64_t xnumel,
    int32_t device_idx_,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    /*
    async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_12', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.pointwise(
        size_hints={'x': 32768}, 
        filename=__file__,
        triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '4F87BAC7C78026030CE21ABCD241F4211145E4ACCDC53C53E0CC97717CB6F329', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'tiling_scores': {'x': 401920}},
        min_elem_per_thread=0
    )
    @triton.jit
    def triton_poi_fused__native_batch_norm_legit_no_training_add_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
        xnumel = 25088
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        x0 = (xindex % 32)
        tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
        tmp1 = tl.load(in_ptr0 + (x2), xmask)
        tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
        tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
        tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
        tmp3 = tmp1 - tmp2
        tmp5 = 1e-05
        tmp6 = tmp4 + tmp5
        tmp7 = libdevice.sqrt(tmp6)
        tmp8 = tl.full([1], 1, tl.int32)
        tmp9 = (tmp8 / tmp7)
        tmp10 = 1.0
        tmp11 = tmp9 * tmp10
        tmp12 = tmp3 * tmp11
        tmp14 = tmp12 * tmp13
        tmp16 = tmp14 + tmp15
        tmp17 = tmp0 + tmp16
        tl.store(in_out_ptr0 + (x2), tmp17, xmask)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = ((xnumel + (128 - 1)) / (128));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused__native_batch_norm_legit_no_training_add_12 == nullptr) {
        kernels_.triton_poi_fused__native_batch_norm_legit_no_training_add_12 = loadKernel("/home/gasoonjia/executorch/c4id4zognxxqwo4qci5zcry3oobj4eoerxfp5yxnlo5pdfcwnqtn.cubin", "triton_poi_fused__native_batch_norm_legit_no_training_add_12", 0, cubin_dir_); 
    }
    CUdeviceptr var_81 = reinterpret_cast<CUdeviceptr>(in_out_ptr0.data_ptr());
    CUdeviceptr var_82 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_83 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_84 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_85 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    CUdeviceptr var_86 = reinterpret_cast<CUdeviceptr>(in_ptr4.data_ptr());
    int var_87 = xnumel;
    CUdeviceptr global_scratch_88 = 0;
    void* kernel_args_[] = {&var_81, &var_82, &var_83, &var_84, &var_85, &var_86, &var_87, &global_scratch_88};
    launchKernel(kernels_.triton_poi_fused__native_batch_norm_legit_no_training_add_12, grid_0, grid_1, grid_2, 4, 0, kernel_args_, stream_);
}

template <typename in_out_ptr0_type_, typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename kernels_type_>
static inline void call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13(
    const in_out_ptr0_type_& in_out_ptr0,
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    int64_t xnumel,
    int32_t device_idx_,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    /*
    async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.pointwise(
        size_hints={'x': 65536}, 
        filename=__file__,
        triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '4F87BAC7C78026030CE21ABCD241F4211145E4ACCDC53C53E0CC97717CB6F329', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'tiling_scores': {'x': 454656}},
        min_elem_per_thread=0
    )
    @triton.jit
    def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
        xnumel = 37632
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        x0 = (xindex % 192)
        tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
        tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
        tmp2 = tmp0 - tmp1
        tmp4 = 1e-05
        tmp5 = tmp3 + tmp4
        tmp6 = libdevice.sqrt(tmp5)
        tmp7 = tl.full([1], 1, tl.int32)
        tmp8 = (tmp7 / tmp6)
        tmp9 = 1.0
        tmp10 = tmp8 * tmp9
        tmp11 = tmp2 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = 0.0
        tmp17 = triton_helpers.maximum(tmp15, tmp16)
        tmp18 = 6.0
        tmp19 = triton_helpers.minimum(tmp17, tmp18)
        tl.store(in_out_ptr0 + (x2), tmp19, xmask)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = ((xnumel + (256 - 1)) / (256));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13 == nullptr) {
        kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13 = loadKernel("/home/gasoonjia/executorch/cxn357cdpjzfyhgfzkziumdqzvax6wmbfva3bo36qlb2w5deusut.cubin", "triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13", 0, cubin_dir_); 
    }
    CUdeviceptr var_89 = reinterpret_cast<CUdeviceptr>(in_out_ptr0.data_ptr());
    CUdeviceptr var_90 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_91 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_92 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_93 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    int var_94 = xnumel;
    CUdeviceptr global_scratch_95 = 0;
    void* kernel_args_[] = {&var_89, &var_90, &var_91, &var_92, &var_93, &var_94, &global_scratch_95};
    launchKernel(kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13, grid_0, grid_1, grid_2, 4, 0, kernel_args_, stream_);
}

template <typename in_out_ptr0_type_, typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename kernels_type_>
static inline void call_triton_poi_fused__native_batch_norm_legit_no_training_14(
    const in_out_ptr0_type_& in_out_ptr0,
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    int64_t xnumel,
    int32_t device_idx_,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    /*
    async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_14', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.pointwise(
        size_hints={'x': 16384}, 
        filename=__file__,
        triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '4F87BAC7C78026030CE21ABCD241F4211145E4ACCDC53C53E0CC97717CB6F329', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'tiling_scores': {'x': 151552}},
        min_elem_per_thread=0
    )
    @triton.jit
    def triton_poi_fused__native_batch_norm_legit_no_training_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
        xnumel = 12544
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        x0 = (xindex % 64)
        tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
        tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
        tmp2 = tmp0 - tmp1
        tmp4 = 1e-05
        tmp5 = tmp3 + tmp4
        tmp6 = libdevice.sqrt(tmp5)
        tmp7 = tl.full([1], 1, tl.int32)
        tmp8 = (tmp7 / tmp6)
        tmp9 = 1.0
        tmp10 = tmp8 * tmp9
        tmp11 = tmp2 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tl.store(in_out_ptr0 + (x2), tmp15, xmask)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = ((xnumel + (128 - 1)) / (128));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused__native_batch_norm_legit_no_training_14 == nullptr) {
        kernels_.triton_poi_fused__native_batch_norm_legit_no_training_14 = loadKernel("/home/gasoonjia/executorch/cmwzm6zpgnuflon4ux22vbg463wrhvpwsjsryjid3yzwslq5jy6j.cubin", "triton_poi_fused__native_batch_norm_legit_no_training_14", 0, cubin_dir_); 
    }
    CUdeviceptr var_96 = reinterpret_cast<CUdeviceptr>(in_out_ptr0.data_ptr());
    CUdeviceptr var_97 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_98 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_99 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_100 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    int var_101 = xnumel;
    CUdeviceptr global_scratch_102 = 0;
    void* kernel_args_[] = {&var_96, &var_97, &var_98, &var_99, &var_100, &var_101, &global_scratch_102};
    launchKernel(kernels_.triton_poi_fused__native_batch_norm_legit_no_training_14, grid_0, grid_1, grid_2, 4, 0, kernel_args_, stream_);
}

template <typename in_out_ptr0_type_, typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename kernels_type_>
static inline void call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15(
    const in_out_ptr0_type_& in_out_ptr0,
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    int64_t xnumel,
    int32_t device_idx_,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    /*
    async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.pointwise(
        size_hints={'x': 131072}, 
        filename=__file__,
        triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '4F87BAC7C78026030CE21ABCD241F4211145E4ACCDC53C53E0CC97717CB6F329', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'tiling_scores': {'x': 909312}},
        min_elem_per_thread=0
    )
    @triton.jit
    def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
        xnumel = 75264
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        x0 = (xindex % 384)
        tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
        tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
        tmp2 = tmp0 - tmp1
        tmp4 = 1e-05
        tmp5 = tmp3 + tmp4
        tmp6 = libdevice.sqrt(tmp5)
        tmp7 = tl.full([1], 1, tl.int32)
        tmp8 = (tmp7 / tmp6)
        tmp9 = 1.0
        tmp10 = tmp8 * tmp9
        tmp11 = tmp2 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = 0.0
        tmp17 = triton_helpers.maximum(tmp15, tmp16)
        tmp18 = 6.0
        tmp19 = triton_helpers.minimum(tmp17, tmp18)
        tl.store(in_out_ptr0 + (x2), tmp19, xmask)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = ((xnumel + (512 - 1)) / (512));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15 == nullptr) {
        kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15 = loadKernel("/home/gasoonjia/executorch/caqye62oxfgou2x7ke4dl35rberxbjhgbjfnpcgtkr4avrno4ixy.cubin", "triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15", 0, cubin_dir_); 
    }
    CUdeviceptr var_103 = reinterpret_cast<CUdeviceptr>(in_out_ptr0.data_ptr());
    CUdeviceptr var_104 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_105 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_106 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_107 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    int var_108 = xnumel;
    CUdeviceptr global_scratch_109 = 0;
    void* kernel_args_[] = {&var_103, &var_104, &var_105, &var_106, &var_107, &var_108, &global_scratch_109};
    launchKernel(kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15, grid_0, grid_1, grid_2, 8, 0, kernel_args_, stream_);
}

template <typename in_out_ptr0_type_, typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename in_ptr4_type_, typename kernels_type_>
static inline void call_triton_poi_fused__native_batch_norm_legit_no_training_add_16(
    const in_out_ptr0_type_& in_out_ptr0,
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    const in_ptr4_type_& in_ptr4,
    int64_t xnumel,
    int32_t device_idx_,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    /*
    async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_16', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.pointwise(
        size_hints={'x': 16384}, 
        filename=__file__,
        triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '4F87BAC7C78026030CE21ABCD241F4211145E4ACCDC53C53E0CC97717CB6F329', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'tiling_scores': {'x': 201728}},
        min_elem_per_thread=0
    )
    @triton.jit
    def triton_poi_fused__native_batch_norm_legit_no_training_add_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
        xnumel = 12544
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        x0 = (xindex % 64)
        tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
        tmp1 = tl.load(in_ptr0 + (x2), xmask)
        tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
        tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
        tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
        tmp3 = tmp1 - tmp2
        tmp5 = 1e-05
        tmp6 = tmp4 + tmp5
        tmp7 = libdevice.sqrt(tmp6)
        tmp8 = tl.full([1], 1, tl.int32)
        tmp9 = (tmp8 / tmp7)
        tmp10 = 1.0
        tmp11 = tmp9 * tmp10
        tmp12 = tmp3 * tmp11
        tmp14 = tmp12 * tmp13
        tmp16 = tmp14 + tmp15
        tmp17 = tmp0 + tmp16
        tl.store(in_out_ptr0 + (x2), tmp17, xmask)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = ((xnumel + (256 - 1)) / (256));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused__native_batch_norm_legit_no_training_add_16 == nullptr) {
        kernels_.triton_poi_fused__native_batch_norm_legit_no_training_add_16 = loadKernel("/home/gasoonjia/executorch/cafig5mi4e5ufzbj47ahikyfz3zcex4yxqvcdqpm27f6d4mtoxbo.cubin", "triton_poi_fused__native_batch_norm_legit_no_training_add_16", 0, cubin_dir_); 
    }
    CUdeviceptr var_110 = reinterpret_cast<CUdeviceptr>(in_out_ptr0.data_ptr());
    CUdeviceptr var_111 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_112 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_113 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_114 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    CUdeviceptr var_115 = reinterpret_cast<CUdeviceptr>(in_ptr4.data_ptr());
    int var_116 = xnumel;
    CUdeviceptr global_scratch_117 = 0;
    void* kernel_args_[] = {&var_110, &var_111, &var_112, &var_113, &var_114, &var_115, &var_116, &global_scratch_117};
    launchKernel(kernels_.triton_poi_fused__native_batch_norm_legit_no_training_add_16, grid_0, grid_1, grid_2, 4, 0, kernel_args_, stream_);
}

template <typename in_out_ptr0_type_, typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename kernels_type_>
static inline void call_triton_poi_fused__native_batch_norm_legit_no_training_17(
    const in_out_ptr0_type_& in_out_ptr0,
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    int64_t xnumel,
    int32_t device_idx_,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    /*
    async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_17', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.pointwise(
        size_hints={'x': 32768}, 
        filename=__file__,
        triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '4F87BAC7C78026030CE21ABCD241F4211145E4ACCDC53C53E0CC97717CB6F329', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'tiling_scores': {'x': 227328}},
        min_elem_per_thread=0
    )
    @triton.jit
    def triton_poi_fused__native_batch_norm_legit_no_training_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
        xnumel = 18816
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        x0 = (xindex % 96)
        tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
        tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
        tmp2 = tmp0 - tmp1
        tmp4 = 1e-05
        tmp5 = tmp3 + tmp4
        tmp6 = libdevice.sqrt(tmp5)
        tmp7 = tl.full([1], 1, tl.int32)
        tmp8 = (tmp7 / tmp6)
        tmp9 = 1.0
        tmp10 = tmp8 * tmp9
        tmp11 = tmp2 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tl.store(in_out_ptr0 + (x2), tmp15, xmask)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = ((xnumel + (256 - 1)) / (256));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused__native_batch_norm_legit_no_training_17 == nullptr) {
        kernels_.triton_poi_fused__native_batch_norm_legit_no_training_17 = loadKernel("/home/gasoonjia/executorch/ctc4njxfwewhkkjkreaoqgsbyrr7s3dbfmgdfcunjbmfgrzqksu4.cubin", "triton_poi_fused__native_batch_norm_legit_no_training_17", 0, cubin_dir_); 
    }
    CUdeviceptr var_118 = reinterpret_cast<CUdeviceptr>(in_out_ptr0.data_ptr());
    CUdeviceptr var_119 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_120 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_121 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_122 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    int var_123 = xnumel;
    CUdeviceptr global_scratch_124 = 0;
    void* kernel_args_[] = {&var_118, &var_119, &var_120, &var_121, &var_122, &var_123, &global_scratch_124};
    launchKernel(kernels_.triton_poi_fused__native_batch_norm_legit_no_training_17, grid_0, grid_1, grid_2, 4, 0, kernel_args_, stream_);
}

template <typename in_out_ptr0_type_, typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename kernels_type_>
static inline void call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_18(
    const in_out_ptr0_type_& in_out_ptr0,
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    int64_t xnumel,
    int32_t device_idx_,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    /*
    async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_18', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.pointwise(
        size_hints={'x': 131072}, 
        filename=__file__,
        triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '4F87BAC7C78026030CE21ABCD241F4211145E4ACCDC53C53E0CC97717CB6F329', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'tiling_scores': {'x': 1363968}},
        min_elem_per_thread=0
    )
    @triton.jit
    def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
        xnumel = 112896
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        x0 = (xindex % 576)
        tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
        tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
        tmp2 = tmp0 - tmp1
        tmp4 = 1e-05
        tmp5 = tmp3 + tmp4
        tmp6 = libdevice.sqrt(tmp5)
        tmp7 = tl.full([1], 1, tl.int32)
        tmp8 = (tmp7 / tmp6)
        tmp9 = 1.0
        tmp10 = tmp8 * tmp9
        tmp11 = tmp2 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = 0.0
        tmp17 = triton_helpers.maximum(tmp15, tmp16)
        tmp18 = 6.0
        tmp19 = triton_helpers.minimum(tmp17, tmp18)
        tl.store(in_out_ptr0 + (x2), tmp19, xmask)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = ((xnumel + (512 - 1)) / (512));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_18 == nullptr) {
        kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_18 = loadKernel("/home/gasoonjia/executorch/cklg2ezqvtkbhlekhvyenxwrgnlwt2msvmc7427nuluwqezzy5lx.cubin", "triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_18", 0, cubin_dir_); 
    }
    CUdeviceptr var_125 = reinterpret_cast<CUdeviceptr>(in_out_ptr0.data_ptr());
    CUdeviceptr var_126 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_127 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_128 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_129 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    int var_130 = xnumel;
    CUdeviceptr global_scratch_131 = 0;
    void* kernel_args_[] = {&var_125, &var_126, &var_127, &var_128, &var_129, &var_130, &global_scratch_131};
    launchKernel(kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_18, grid_0, grid_1, grid_2, 8, 0, kernel_args_, stream_);
}

template <typename in_out_ptr0_type_, typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename in_ptr4_type_, typename kernels_type_>
static inline void call_triton_poi_fused__native_batch_norm_legit_no_training_add_19(
    const in_out_ptr0_type_& in_out_ptr0,
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    const in_ptr4_type_& in_ptr4,
    int64_t xnumel,
    int32_t device_idx_,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    /*
    async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_19', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.pointwise(
        size_hints={'x': 32768}, 
        filename=__file__,
        triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '4F87BAC7C78026030CE21ABCD241F4211145E4ACCDC53C53E0CC97717CB6F329', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'tiling_scores': {'x': 302592}},
        min_elem_per_thread=0
    )
    @triton.jit
    def triton_poi_fused__native_batch_norm_legit_no_training_add_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
        xnumel = 18816
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        x0 = (xindex % 96)
        tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
        tmp1 = tl.load(in_ptr0 + (x2), xmask)
        tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
        tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
        tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
        tmp3 = tmp1 - tmp2
        tmp5 = 1e-05
        tmp6 = tmp4 + tmp5
        tmp7 = libdevice.sqrt(tmp6)
        tmp8 = tl.full([1], 1, tl.int32)
        tmp9 = (tmp8 / tmp7)
        tmp10 = 1.0
        tmp11 = tmp9 * tmp10
        tmp12 = tmp3 * tmp11
        tmp14 = tmp12 * tmp13
        tmp16 = tmp14 + tmp15
        tmp17 = tmp0 + tmp16
        tl.store(in_out_ptr0 + (x2), tmp17, xmask)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = ((xnumel + (256 - 1)) / (256));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused__native_batch_norm_legit_no_training_add_19 == nullptr) {
        kernels_.triton_poi_fused__native_batch_norm_legit_no_training_add_19 = loadKernel("/home/gasoonjia/executorch/c3sj66uvazrx3drgx5zzvxlffnqf3kezaikukfqbiue2bb2vcbdg.cubin", "triton_poi_fused__native_batch_norm_legit_no_training_add_19", 0, cubin_dir_); 
    }
    CUdeviceptr var_132 = reinterpret_cast<CUdeviceptr>(in_out_ptr0.data_ptr());
    CUdeviceptr var_133 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_134 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_135 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_136 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    CUdeviceptr var_137 = reinterpret_cast<CUdeviceptr>(in_ptr4.data_ptr());
    int var_138 = xnumel;
    CUdeviceptr global_scratch_139 = 0;
    void* kernel_args_[] = {&var_132, &var_133, &var_134, &var_135, &var_136, &var_137, &var_138, &global_scratch_139};
    launchKernel(kernels_.triton_poi_fused__native_batch_norm_legit_no_training_add_19, grid_0, grid_1, grid_2, 4, 0, kernel_args_, stream_);
}

template <typename in_out_ptr0_type_, typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename kernels_type_>
static inline void call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_20(
    const in_out_ptr0_type_& in_out_ptr0,
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    int64_t xnumel,
    int32_t device_idx_,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    /*
    async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_20', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.pointwise(
        size_hints={'x': 32768}, 
        filename=__file__,
        triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '4F87BAC7C78026030CE21ABCD241F4211145E4ACCDC53C53E0CC97717CB6F329', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'tiling_scores': {'x': 347904}},
        min_elem_per_thread=0
    )
    @triton.jit
    def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
        xnumel = 28224
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        x0 = (xindex % 576)
        tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
        tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
        tmp2 = tmp0 - tmp1
        tmp4 = 1e-05
        tmp5 = tmp3 + tmp4
        tmp6 = libdevice.sqrt(tmp5)
        tmp7 = tl.full([1], 1, tl.int32)
        tmp8 = (tmp7 / tmp6)
        tmp9 = 1.0
        tmp10 = tmp8 * tmp9
        tmp11 = tmp2 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = 0.0
        tmp17 = triton_helpers.maximum(tmp15, tmp16)
        tmp18 = 6.0
        tmp19 = triton_helpers.minimum(tmp17, tmp18)
        tl.store(in_out_ptr0 + (x2), tmp19, xmask)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = ((xnumel + (256 - 1)) / (256));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_20 == nullptr) {
        kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_20 = loadKernel("/home/gasoonjia/executorch/c2oewcn4k655ga3vky43nudfhqe4py7nuxkauuy7fcrnhwyg4gsl.cubin", "triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_20", 0, cubin_dir_); 
    }
    CUdeviceptr var_140 = reinterpret_cast<CUdeviceptr>(in_out_ptr0.data_ptr());
    CUdeviceptr var_141 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_142 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_143 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_144 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    int var_145 = xnumel;
    CUdeviceptr global_scratch_146 = 0;
    void* kernel_args_[] = {&var_140, &var_141, &var_142, &var_143, &var_144, &var_145, &global_scratch_146};
    launchKernel(kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_20, grid_0, grid_1, grid_2, 4, 0, kernel_args_, stream_);
}

template <typename in_out_ptr0_type_, typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename kernels_type_>
static inline void call_triton_poi_fused__native_batch_norm_legit_no_training_21(
    const in_out_ptr0_type_& in_out_ptr0,
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    int64_t xnumel,
    int32_t device_idx_,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    /*
    async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_21', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.pointwise(
        size_hints={'x': 8192}, 
        filename=__file__,
        triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '4F87BAC7C78026030CE21ABCD241F4211145E4ACCDC53C53E0CC97717CB6F329', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'tiling_scores': {'x': 96640}},
        min_elem_per_thread=0
    )
    @triton.jit
    def triton_poi_fused__native_batch_norm_legit_no_training_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
        xnumel = 7840
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        x0 = (xindex % 160)
        tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
        tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
        tmp2 = tmp0 - tmp1
        tmp4 = 1e-05
        tmp5 = tmp3 + tmp4
        tmp6 = libdevice.sqrt(tmp5)
        tmp7 = tl.full([1], 1, tl.int32)
        tmp8 = (tmp7 / tmp6)
        tmp9 = 1.0
        tmp10 = tmp8 * tmp9
        tmp11 = tmp2 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tl.store(in_out_ptr0 + (x2), tmp15, xmask)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = ((xnumel + (128 - 1)) / (128));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused__native_batch_norm_legit_no_training_21 == nullptr) {
        kernels_.triton_poi_fused__native_batch_norm_legit_no_training_21 = loadKernel("/home/gasoonjia/executorch/crikv76bp356w3xfrsl6v7yjgadifnrrfofduf4qs74u5yah7y3u.cubin", "triton_poi_fused__native_batch_norm_legit_no_training_21", 0, cubin_dir_); 
    }
    CUdeviceptr var_147 = reinterpret_cast<CUdeviceptr>(in_out_ptr0.data_ptr());
    CUdeviceptr var_148 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_149 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_150 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_151 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    int var_152 = xnumel;
    CUdeviceptr global_scratch_153 = 0;
    void* kernel_args_[] = {&var_147, &var_148, &var_149, &var_150, &var_151, &var_152, &global_scratch_153};
    launchKernel(kernels_.triton_poi_fused__native_batch_norm_legit_no_training_21, grid_0, grid_1, grid_2, 4, 0, kernel_args_, stream_);
}

template <typename in_out_ptr0_type_, typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename kernels_type_>
static inline void call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22(
    const in_out_ptr0_type_& in_out_ptr0,
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    int64_t xnumel,
    int32_t device_idx_,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    /*
    async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.pointwise(
        size_hints={'x': 65536}, 
        filename=__file__,
        triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '4F87BAC7C78026030CE21ABCD241F4211145E4ACCDC53C53E0CC97717CB6F329', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'tiling_scores': {'x': 579840}},
        min_elem_per_thread=0
    )
    @triton.jit
    def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
        xnumel = 47040
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        x0 = (xindex % 960)
        tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
        tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
        tmp2 = tmp0 - tmp1
        tmp4 = 1e-05
        tmp5 = tmp3 + tmp4
        tmp6 = libdevice.sqrt(tmp5)
        tmp7 = tl.full([1], 1, tl.int32)
        tmp8 = (tmp7 / tmp6)
        tmp9 = 1.0
        tmp10 = tmp8 * tmp9
        tmp11 = tmp2 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = 0.0
        tmp17 = triton_helpers.maximum(tmp15, tmp16)
        tmp18 = 6.0
        tmp19 = triton_helpers.minimum(tmp17, tmp18)
        tl.store(in_out_ptr0 + (x2), tmp19, xmask)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = ((xnumel + (512 - 1)) / (512));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22 == nullptr) {
        kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22 = loadKernel("/home/gasoonjia/executorch/cluvzszdtr4ykyrpkxlp2moyesdw57fomp6qblpztzjs77ltlqpm.cubin", "triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22", 0, cubin_dir_); 
    }
    CUdeviceptr var_154 = reinterpret_cast<CUdeviceptr>(in_out_ptr0.data_ptr());
    CUdeviceptr var_155 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_156 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_157 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_158 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    int var_159 = xnumel;
    CUdeviceptr global_scratch_160 = 0;
    void* kernel_args_[] = {&var_154, &var_155, &var_156, &var_157, &var_158, &var_159, &global_scratch_160};
    launchKernel(kernels_.triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22, grid_0, grid_1, grid_2, 4, 0, kernel_args_, stream_);
}

template <typename in_out_ptr0_type_, typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename in_ptr4_type_, typename kernels_type_>
static inline void call_triton_poi_fused__native_batch_norm_legit_no_training_add_23(
    const in_out_ptr0_type_& in_out_ptr0,
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    const in_ptr4_type_& in_ptr4,
    int64_t xnumel,
    int32_t device_idx_,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    /*
    async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_23', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.pointwise(
        size_hints={'x': 8192}, 
        filename=__file__,
        triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '4F87BAC7C78026030CE21ABCD241F4211145E4ACCDC53C53E0CC97717CB6F329', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'tiling_scores': {'x': 128000}},
        min_elem_per_thread=0
    )
    @triton.jit
    def triton_poi_fused__native_batch_norm_legit_no_training_add_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
        xnumel = 7840
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        x0 = (xindex % 160)
        tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
        tmp1 = tl.load(in_ptr0 + (x2), xmask)
        tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
        tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
        tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
        tmp3 = tmp1 - tmp2
        tmp5 = 1e-05
        tmp6 = tmp4 + tmp5
        tmp7 = libdevice.sqrt(tmp6)
        tmp8 = tl.full([1], 1, tl.int32)
        tmp9 = (tmp8 / tmp7)
        tmp10 = 1.0
        tmp11 = tmp9 * tmp10
        tmp12 = tmp3 * tmp11
        tmp14 = tmp12 * tmp13
        tmp16 = tmp14 + tmp15
        tmp17 = tmp0 + tmp16
        tl.store(in_out_ptr0 + (x2), tmp17, xmask)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = ((xnumel + (128 - 1)) / (128));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused__native_batch_norm_legit_no_training_add_23 == nullptr) {
        kernels_.triton_poi_fused__native_batch_norm_legit_no_training_add_23 = loadKernel("/home/gasoonjia/executorch/c2yybeoyrkfdeh34rwaadbn7z3xbhkdmautjebwjj3cnspt7codl.cubin", "triton_poi_fused__native_batch_norm_legit_no_training_add_23", 0, cubin_dir_); 
    }
    CUdeviceptr var_161 = reinterpret_cast<CUdeviceptr>(in_out_ptr0.data_ptr());
    CUdeviceptr var_162 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_163 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_164 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_165 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    CUdeviceptr var_166 = reinterpret_cast<CUdeviceptr>(in_ptr4.data_ptr());
    int var_167 = xnumel;
    CUdeviceptr global_scratch_168 = 0;
    void* kernel_args_[] = {&var_161, &var_162, &var_163, &var_164, &var_165, &var_166, &var_167, &global_scratch_168};
    launchKernel(kernels_.triton_poi_fused__native_batch_norm_legit_no_training_add_23, grid_0, grid_1, grid_2, 4, 0, kernel_args_, stream_);
}

template <typename in_out_ptr0_type_, typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename kernels_type_>
static inline void call_triton_poi_fused__native_batch_norm_legit_no_training_24(
    const in_out_ptr0_type_& in_out_ptr0,
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    int64_t xnumel,
    int32_t device_idx_,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    /*
    async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_24', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.pointwise(
        size_hints={'x': 16384}, 
        filename=__file__,
        triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '4F87BAC7C78026030CE21ABCD241F4211145E4ACCDC53C53E0CC97717CB6F329', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'tiling_scores': {'x': 193280}},
        min_elem_per_thread=0
    )
    @triton.jit
    def triton_poi_fused__native_batch_norm_legit_no_training_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
        xnumel = 15680
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        x0 = (xindex % 320)
        tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
        tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
        tmp2 = tmp0 - tmp1
        tmp4 = 1e-05
        tmp5 = tmp3 + tmp4
        tmp6 = libdevice.sqrt(tmp5)
        tmp7 = tl.full([1], 1, tl.int32)
        tmp8 = (tmp7 / tmp6)
        tmp9 = 1.0
        tmp10 = tmp8 * tmp9
        tmp11 = tmp2 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tl.store(in_out_ptr0 + (x2), tmp15, xmask)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = ((xnumel + (256 - 1)) / (256));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused__native_batch_norm_legit_no_training_24 == nullptr) {
        kernels_.triton_poi_fused__native_batch_norm_legit_no_training_24 = loadKernel("/home/gasoonjia/executorch/cwmiqau7t5rssvjroylm2qwtew7tkyixr7l2y5x22afsem5iac72.cubin", "triton_poi_fused__native_batch_norm_legit_no_training_24", 0, cubin_dir_); 
    }
    CUdeviceptr var_169 = reinterpret_cast<CUdeviceptr>(in_out_ptr0.data_ptr());
    CUdeviceptr var_170 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_171 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_172 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_173 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    int var_174 = xnumel;
    CUdeviceptr global_scratch_175 = 0;
    void* kernel_args_[] = {&var_169, &var_170, &var_171, &var_172, &var_173, &var_174, &global_scratch_175};
    launchKernel(kernels_.triton_poi_fused__native_batch_norm_legit_no_training_24, grid_0, grid_1, grid_2, 4, 0, kernel_args_, stream_);
}

template <typename in_out_ptr0_type_, typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename in_ptr4_type_, typename kernels_type_>
static inline void call_triton_per_fused__native_batch_norm_legit_no_training_hardtanh_mean_25(
    const in_out_ptr0_type_& in_out_ptr0,
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    const in_ptr4_type_& in_ptr4,
    int64_t xnumel,
    int64_t r0_numel,
    int32_t device_idx_,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    /*
    async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_hardtanh_mean_25', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.persistent_reduction(
        size_hints={'x': 2048, 'r0_': 64},
        reduction_hint=ReductionHint.OUTER,
        filename=__file__,
        triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_hardtanh_mean_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4F87BAC7C78026030CE21ABCD241F4211145E4ACCDC53C53E0CC97717CB6F329', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'tiling_scores': {'x': 281600, 'r0_': 0}}
    )
    @triton.jit
    def triton_per_fused__native_batch_norm_legit_no_training_hardtanh_mean_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
        xnumel = 1280
        r0_numel = 49
        R0_BLOCK: tl.constexpr = 64
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel
        r0_index = tl.arange(0, R0_BLOCK)[None, :]
        r0_offset = 0
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0 + 1280*r0_1), r0_mask & xmask, other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
        tmp2 = tmp0 - tmp1
        tmp4 = 1e-05
        tmp5 = tmp3 + tmp4
        tmp6 = libdevice.sqrt(tmp5)
        tmp7 = tl.full([1, 1], 1, tl.int32)
        tmp8 = (tmp7 / tmp6)
        tmp9 = 1.0
        tmp10 = tmp8 * tmp9
        tmp11 = tmp2 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = 0.0
        tmp17 = triton_helpers.maximum(tmp15, tmp16)
        tmp18 = 6.0
        tmp19 = triton_helpers.minimum(tmp17, tmp18)
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
        tmp22 = tl.where(r0_mask & xmask, tmp20, 0)
        tmp23 = tl.sum(tmp22, 1)[:, None]
        tmp24 = 49.0
        tmp25 = (tmp23 / tmp24)
        tl.debug_barrier()
        tl.store(in_out_ptr0 + (x0), tmp25, xmask)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = ((xnumel + (32 - 1)) / (32));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_per_fused__native_batch_norm_legit_no_training_hardtanh_mean_25 == nullptr) {
        kernels_.triton_per_fused__native_batch_norm_legit_no_training_hardtanh_mean_25 = loadKernel("/home/gasoonjia/executorch/csitc2tbez7ytfakpudstbhsobm3wlczsly46p5oeax43spr3eab.cubin", "triton_per_fused__native_batch_norm_legit_no_training_hardtanh_mean_25", 1024, cubin_dir_); 
    }
    CUdeviceptr var_176 = reinterpret_cast<CUdeviceptr>(in_out_ptr0.data_ptr());
    CUdeviceptr var_177 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_178 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_179 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_180 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    CUdeviceptr var_181 = reinterpret_cast<CUdeviceptr>(in_ptr4.data_ptr());
    int var_182 = xnumel;
    int var_183 = r0_numel;
    CUdeviceptr global_scratch_184 = 0;
    void* kernel_args_[] = {&var_176, &var_177, &var_178, &var_179, &var_180, &var_181, &var_182, &var_183, &global_scratch_184};
    launchKernel(kernels_.triton_per_fused__native_batch_norm_legit_no_training_hardtanh_mean_25, grid_0, grid_1, grid_2, 8, 1024, kernel_args_, stream_);
}

template <typename in_ptr0_type_, typename out_ptr0_type_, typename kernels_type_>
static inline void call_triton_poi_fused_permute_copy_26(
    const in_ptr0_type_& in_ptr0,
    const out_ptr0_type_& out_ptr0,
    int64_t xnumel,
    int32_t device_idx_,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    /*
    async_compile.triton('triton_poi_fused_permute_copy_26', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.pointwise(
        size_hints={'x': 2097152}, 
        filename=__file__,
        triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_permute_copy_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4F87BAC7C78026030CE21ABCD241F4211145E4ACCDC53C53E0CC97717CB6F329', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'tiling_scores': {'x': 15360000}},
        min_elem_per_thread=0
    )
    @triton.jit
    def triton_poi_fused_permute_copy_26(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
        xnumel = 1280000
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tl.store(out_ptr0 + (x0), tmp0, xmask)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = ((xnumel + (1024 - 1)) / (1024));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused_permute_copy_26 == nullptr) {
        kernels_.triton_poi_fused_permute_copy_26 = loadKernel("/home/gasoonjia/executorch/czj7vvfy745m4rwqvkdetdltbkwsdx6kjaldi7zklwlc3zi37bno.cubin", "triton_poi_fused_permute_copy_26", 0, cubin_dir_); 
    }
    CUdeviceptr var_185 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_186 = reinterpret_cast<CUdeviceptr>(out_ptr0.data_ptr());
    int var_187 = xnumel;
    CUdeviceptr global_scratch_188 = 0;
    void* kernel_args_[] = {&var_185, &var_186, &var_187, &global_scratch_188};
    launchKernel(kernels_.triton_poi_fused_permute_copy_26, grid_0, grid_1, grid_2, 4, 0, kernel_args_, stream_);
}

namespace torch::aot_inductor {

void AOTInductorModel::_const_run_impl(
    std::vector<AtenTensorHandle>& output_handles,
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor
) {}

AOTI_NOINLINE static void check_input_0(
    AtenTensorHandle* input_handles
) {
    ConstantHandle arg262_1 = ConstantHandle(input_handles[0]);
    int32_t arg262_1_dtype;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(arg262_1, &arg262_1_dtype));

    int32_t arg262_1_expected_dtype = aoti_torch_dtype_float32();
    if (arg262_1_expected_dtype != arg262_1_dtype) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched dtype, "
           << "expected: " << arg262_1_expected_dtype << "(at::kFloat), "
           << "but got: " << arg262_1_dtype << "\n";
        throw std::runtime_error(ss.str());
    }
    auto arg262_1_size = arg262_1.sizes();

    if (1 != arg262_1_size[0]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched dim value at 0, "
           << "expected: 1, " << "but got: " << arg262_1_size[0]
           << "\n";
        throw std::runtime_error(ss.str());
    }

    if (3 != arg262_1_size[1]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched dim value at 1, "
           << "expected: 3, " << "but got: " << arg262_1_size[1]
           << "\n";
        throw std::runtime_error(ss.str());
    }

    if (224 != arg262_1_size[2]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched dim value at 2, "
           << "expected: 224, " << "but got: " << arg262_1_size[2]
           << "\n";
        throw std::runtime_error(ss.str());
    }

    if (224 != arg262_1_size[3]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched dim value at 3, "
           << "expected: 224, " << "but got: " << arg262_1_size[3]
           << "\n";
        throw std::runtime_error(ss.str());
    }
    auto arg262_1_stride = arg262_1.strides();

    if (150528 != arg262_1_stride[0]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched stride value at 0, "
           << "expected: 150528, " << "but got: " << arg262_1_stride[0]
           << "\n";
        throw std::runtime_error(ss.str());
    }

    if (50176 != arg262_1_stride[1]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched stride value at 1, "
           << "expected: 50176, " << "but got: " << arg262_1_stride[1]
           << "\n";
        throw std::runtime_error(ss.str());
    }

    if (224 != arg262_1_stride[2]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched stride value at 2, "
           << "expected: 224, " << "but got: " << arg262_1_stride[2]
           << "\n";
        throw std::runtime_error(ss.str());
    }

    if (1 != arg262_1_stride[3]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched stride value at 3, "
           << "expected: 1, " << "but got: " << arg262_1_stride[3]
           << "\n";
        throw std::runtime_error(ss.str());
    }
    int32_t arg262_1_device_type;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type(arg262_1, &arg262_1_device_type));

    int32_t arg262_1_expected_device_type = 1;
    if (arg262_1_expected_device_type != arg262_1_device_type) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched device type, "
        << "expected: " << arg262_1_expected_device_type << "1(cuda), "
        << "but got: " << arg262_1_device_type << "\n";
        throw std::runtime_error(ss.str());
    }
}

static bool _check_aoti_runtime_check_inputs_env() {
    const static char* env_var_value = getenv("AOTI_RUNTIME_CHECK_INPUTS");
    const static bool result = env_var_value != nullptr && env_var_value[0] != '0';
    return result;
}

AOTI_NOINLINE static void __check_inputs_outputs(
    AtenTensorHandle* input_handles,
    AtenTensorHandle* output_handles) {
    if (!_check_aoti_runtime_check_inputs_env()){
        return;
    }
    check_input_0(input_handles);
}

void AOTInductorModel::run_impl(
    AtenTensorHandle*
        input_handles, // array of input AtenTensorHandle; handles
                        // are stolen; the array itself is borrowed
    AtenTensorHandle*
        output_handles, // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor
) {
    __check_inputs_outputs(input_handles, output_handles);

    auto inputs = steal_from_raw_handles_to_raii_handles(input_handles, 1);
    auto arg262_1 = std::move(inputs[0]);
    [[maybe_unused]] auto& mv2_features_0_0_weight = constants_->at(0);
    [[maybe_unused]] auto& mv2_features_0_1_weight = constants_->at(1);
    [[maybe_unused]] auto& mv2_features_0_1_bias = constants_->at(2);
    [[maybe_unused]] auto& mv2_features_1_conv_0_0_weight = constants_->at(3);
    [[maybe_unused]] auto& mv2_features_1_conv_0_1_weight = constants_->at(4);
    [[maybe_unused]] auto& mv2_features_1_conv_0_1_bias = constants_->at(5);
    [[maybe_unused]] auto& mv2_features_1_conv_1_weight = constants_->at(6);
    [[maybe_unused]] auto& mv2_features_1_conv_2_weight = constants_->at(7);
    [[maybe_unused]] auto& mv2_features_1_conv_2_bias = constants_->at(8);
    [[maybe_unused]] auto& mv2_features_2_conv_0_0_weight = constants_->at(9);
    [[maybe_unused]] auto& mv2_features_2_conv_0_1_weight = constants_->at(10);
    [[maybe_unused]] auto& mv2_features_2_conv_0_1_bias = constants_->at(11);
    [[maybe_unused]] auto& mv2_features_2_conv_1_0_weight = constants_->at(12);
    [[maybe_unused]] auto& mv2_features_2_conv_1_1_weight = constants_->at(13);
    [[maybe_unused]] auto& mv2_features_2_conv_1_1_bias = constants_->at(14);
    [[maybe_unused]] auto& mv2_features_2_conv_2_weight = constants_->at(15);
    [[maybe_unused]] auto& mv2_features_2_conv_3_weight = constants_->at(16);
    [[maybe_unused]] auto& mv2_features_2_conv_3_bias = constants_->at(17);
    [[maybe_unused]] auto& mv2_features_3_conv_0_0_weight = constants_->at(18);
    [[maybe_unused]] auto& mv2_features_3_conv_0_1_weight = constants_->at(19);
    [[maybe_unused]] auto& mv2_features_3_conv_0_1_bias = constants_->at(20);
    [[maybe_unused]] auto& mv2_features_3_conv_1_0_weight = constants_->at(21);
    [[maybe_unused]] auto& mv2_features_3_conv_1_1_weight = constants_->at(22);
    [[maybe_unused]] auto& mv2_features_3_conv_1_1_bias = constants_->at(23);
    [[maybe_unused]] auto& mv2_features_3_conv_2_weight = constants_->at(24);
    [[maybe_unused]] auto& mv2_features_3_conv_3_weight = constants_->at(25);
    [[maybe_unused]] auto& mv2_features_3_conv_3_bias = constants_->at(26);
    [[maybe_unused]] auto& mv2_features_4_conv_0_0_weight = constants_->at(27);
    [[maybe_unused]] auto& mv2_features_4_conv_0_1_weight = constants_->at(28);
    [[maybe_unused]] auto& mv2_features_4_conv_0_1_bias = constants_->at(29);
    [[maybe_unused]] auto& mv2_features_4_conv_1_0_weight = constants_->at(30);
    [[maybe_unused]] auto& mv2_features_4_conv_1_1_weight = constants_->at(31);
    [[maybe_unused]] auto& mv2_features_4_conv_1_1_bias = constants_->at(32);
    [[maybe_unused]] auto& mv2_features_4_conv_2_weight = constants_->at(33);
    [[maybe_unused]] auto& mv2_features_4_conv_3_weight = constants_->at(34);
    [[maybe_unused]] auto& mv2_features_4_conv_3_bias = constants_->at(35);
    [[maybe_unused]] auto& mv2_features_5_conv_0_0_weight = constants_->at(36);
    [[maybe_unused]] auto& mv2_features_5_conv_0_1_weight = constants_->at(37);
    [[maybe_unused]] auto& mv2_features_5_conv_0_1_bias = constants_->at(38);
    [[maybe_unused]] auto& mv2_features_5_conv_1_0_weight = constants_->at(39);
    [[maybe_unused]] auto& mv2_features_5_conv_1_1_weight = constants_->at(40);
    [[maybe_unused]] auto& mv2_features_5_conv_1_1_bias = constants_->at(41);
    [[maybe_unused]] auto& mv2_features_5_conv_2_weight = constants_->at(42);
    [[maybe_unused]] auto& mv2_features_5_conv_3_weight = constants_->at(43);
    [[maybe_unused]] auto& mv2_features_5_conv_3_bias = constants_->at(44);
    [[maybe_unused]] auto& mv2_features_6_conv_0_0_weight = constants_->at(45);
    [[maybe_unused]] auto& mv2_features_6_conv_0_1_weight = constants_->at(46);
    [[maybe_unused]] auto& mv2_features_6_conv_0_1_bias = constants_->at(47);
    [[maybe_unused]] auto& mv2_features_6_conv_1_0_weight = constants_->at(48);
    [[maybe_unused]] auto& mv2_features_6_conv_1_1_weight = constants_->at(49);
    [[maybe_unused]] auto& mv2_features_6_conv_1_1_bias = constants_->at(50);
    [[maybe_unused]] auto& mv2_features_6_conv_2_weight = constants_->at(51);
    [[maybe_unused]] auto& mv2_features_6_conv_3_weight = constants_->at(52);
    [[maybe_unused]] auto& mv2_features_6_conv_3_bias = constants_->at(53);
    [[maybe_unused]] auto& mv2_features_7_conv_0_0_weight = constants_->at(54);
    [[maybe_unused]] auto& mv2_features_7_conv_0_1_weight = constants_->at(55);
    [[maybe_unused]] auto& mv2_features_7_conv_0_1_bias = constants_->at(56);
    [[maybe_unused]] auto& mv2_features_7_conv_1_0_weight = constants_->at(57);
    [[maybe_unused]] auto& mv2_features_7_conv_1_1_weight = constants_->at(58);
    [[maybe_unused]] auto& mv2_features_7_conv_1_1_bias = constants_->at(59);
    [[maybe_unused]] auto& mv2_features_7_conv_2_weight = constants_->at(60);
    [[maybe_unused]] auto& mv2_features_7_conv_3_weight = constants_->at(61);
    [[maybe_unused]] auto& mv2_features_7_conv_3_bias = constants_->at(62);
    [[maybe_unused]] auto& mv2_features_8_conv_0_0_weight = constants_->at(63);
    [[maybe_unused]] auto& mv2_features_8_conv_0_1_weight = constants_->at(64);
    [[maybe_unused]] auto& mv2_features_8_conv_0_1_bias = constants_->at(65);
    [[maybe_unused]] auto& mv2_features_8_conv_1_0_weight = constants_->at(66);
    [[maybe_unused]] auto& mv2_features_8_conv_1_1_weight = constants_->at(67);
    [[maybe_unused]] auto& mv2_features_8_conv_1_1_bias = constants_->at(68);
    [[maybe_unused]] auto& mv2_features_8_conv_2_weight = constants_->at(69);
    [[maybe_unused]] auto& mv2_features_8_conv_3_weight = constants_->at(70);
    [[maybe_unused]] auto& mv2_features_8_conv_3_bias = constants_->at(71);
    [[maybe_unused]] auto& mv2_features_9_conv_0_0_weight = constants_->at(72);
    [[maybe_unused]] auto& mv2_features_9_conv_0_1_weight = constants_->at(73);
    [[maybe_unused]] auto& mv2_features_9_conv_0_1_bias = constants_->at(74);
    [[maybe_unused]] auto& mv2_features_9_conv_1_0_weight = constants_->at(75);
    [[maybe_unused]] auto& mv2_features_9_conv_1_1_weight = constants_->at(76);
    [[maybe_unused]] auto& mv2_features_9_conv_1_1_bias = constants_->at(77);
    [[maybe_unused]] auto& mv2_features_9_conv_2_weight = constants_->at(78);
    [[maybe_unused]] auto& mv2_features_9_conv_3_weight = constants_->at(79);
    [[maybe_unused]] auto& mv2_features_9_conv_3_bias = constants_->at(80);
    [[maybe_unused]] auto& mv2_features_10_conv_0_0_weight = constants_->at(81);
    [[maybe_unused]] auto& mv2_features_10_conv_0_1_weight = constants_->at(82);
    [[maybe_unused]] auto& mv2_features_10_conv_0_1_bias = constants_->at(83);
    [[maybe_unused]] auto& mv2_features_10_conv_1_0_weight = constants_->at(84);
    [[maybe_unused]] auto& mv2_features_10_conv_1_1_weight = constants_->at(85);
    [[maybe_unused]] auto& mv2_features_10_conv_1_1_bias = constants_->at(86);
    [[maybe_unused]] auto& mv2_features_10_conv_2_weight = constants_->at(87);
    [[maybe_unused]] auto& mv2_features_10_conv_3_weight = constants_->at(88);
    [[maybe_unused]] auto& mv2_features_10_conv_3_bias = constants_->at(89);
    [[maybe_unused]] auto& mv2_features_11_conv_0_0_weight = constants_->at(90);
    [[maybe_unused]] auto& mv2_features_11_conv_0_1_weight = constants_->at(91);
    [[maybe_unused]] auto& mv2_features_11_conv_0_1_bias = constants_->at(92);
    [[maybe_unused]] auto& mv2_features_11_conv_1_0_weight = constants_->at(93);
    [[maybe_unused]] auto& mv2_features_11_conv_1_1_weight = constants_->at(94);
    [[maybe_unused]] auto& mv2_features_11_conv_1_1_bias = constants_->at(95);
    [[maybe_unused]] auto& mv2_features_11_conv_2_weight = constants_->at(96);
    [[maybe_unused]] auto& mv2_features_11_conv_3_weight = constants_->at(97);
    [[maybe_unused]] auto& mv2_features_11_conv_3_bias = constants_->at(98);
    [[maybe_unused]] auto& mv2_features_12_conv_0_0_weight = constants_->at(99);
    [[maybe_unused]] auto& mv2_features_12_conv_0_1_weight = constants_->at(100);
    [[maybe_unused]] auto& mv2_features_12_conv_0_1_bias = constants_->at(101);
    [[maybe_unused]] auto& mv2_features_12_conv_1_0_weight = constants_->at(102);
    [[maybe_unused]] auto& mv2_features_12_conv_1_1_weight = constants_->at(103);
    [[maybe_unused]] auto& mv2_features_12_conv_1_1_bias = constants_->at(104);
    [[maybe_unused]] auto& mv2_features_12_conv_2_weight = constants_->at(105);
    [[maybe_unused]] auto& mv2_features_12_conv_3_weight = constants_->at(106);
    [[maybe_unused]] auto& mv2_features_12_conv_3_bias = constants_->at(107);
    [[maybe_unused]] auto& mv2_features_13_conv_0_0_weight = constants_->at(108);
    [[maybe_unused]] auto& mv2_features_13_conv_0_1_weight = constants_->at(109);
    [[maybe_unused]] auto& mv2_features_13_conv_0_1_bias = constants_->at(110);
    [[maybe_unused]] auto& mv2_features_13_conv_1_0_weight = constants_->at(111);
    [[maybe_unused]] auto& mv2_features_13_conv_1_1_weight = constants_->at(112);
    [[maybe_unused]] auto& mv2_features_13_conv_1_1_bias = constants_->at(113);
    [[maybe_unused]] auto& mv2_features_13_conv_2_weight = constants_->at(114);
    [[maybe_unused]] auto& mv2_features_13_conv_3_weight = constants_->at(115);
    [[maybe_unused]] auto& mv2_features_13_conv_3_bias = constants_->at(116);
    [[maybe_unused]] auto& mv2_features_14_conv_0_0_weight = constants_->at(117);
    [[maybe_unused]] auto& mv2_features_14_conv_0_1_weight = constants_->at(118);
    [[maybe_unused]] auto& mv2_features_14_conv_0_1_bias = constants_->at(119);
    [[maybe_unused]] auto& mv2_features_14_conv_1_0_weight = constants_->at(120);
    [[maybe_unused]] auto& mv2_features_14_conv_1_1_weight = constants_->at(121);
    [[maybe_unused]] auto& mv2_features_14_conv_1_1_bias = constants_->at(122);
    [[maybe_unused]] auto& mv2_features_14_conv_2_weight = constants_->at(123);
    [[maybe_unused]] auto& mv2_features_14_conv_3_weight = constants_->at(124);
    [[maybe_unused]] auto& mv2_features_14_conv_3_bias = constants_->at(125);
    [[maybe_unused]] auto& mv2_features_15_conv_0_0_weight = constants_->at(126);
    [[maybe_unused]] auto& mv2_features_15_conv_0_1_weight = constants_->at(127);
    [[maybe_unused]] auto& mv2_features_15_conv_0_1_bias = constants_->at(128);
    [[maybe_unused]] auto& mv2_features_15_conv_1_0_weight = constants_->at(129);
    [[maybe_unused]] auto& mv2_features_15_conv_1_1_weight = constants_->at(130);
    [[maybe_unused]] auto& mv2_features_15_conv_1_1_bias = constants_->at(131);
    [[maybe_unused]] auto& mv2_features_15_conv_2_weight = constants_->at(132);
    [[maybe_unused]] auto& mv2_features_15_conv_3_weight = constants_->at(133);
    [[maybe_unused]] auto& mv2_features_15_conv_3_bias = constants_->at(134);
    [[maybe_unused]] auto& mv2_features_16_conv_0_0_weight = constants_->at(135);
    [[maybe_unused]] auto& mv2_features_16_conv_0_1_weight = constants_->at(136);
    [[maybe_unused]] auto& mv2_features_16_conv_0_1_bias = constants_->at(137);
    [[maybe_unused]] auto& mv2_features_16_conv_1_0_weight = constants_->at(138);
    [[maybe_unused]] auto& mv2_features_16_conv_1_1_weight = constants_->at(139);
    [[maybe_unused]] auto& mv2_features_16_conv_1_1_bias = constants_->at(140);
    [[maybe_unused]] auto& mv2_features_16_conv_2_weight = constants_->at(141);
    [[maybe_unused]] auto& mv2_features_16_conv_3_weight = constants_->at(142);
    [[maybe_unused]] auto& mv2_features_16_conv_3_bias = constants_->at(143);
    [[maybe_unused]] auto& mv2_features_17_conv_0_0_weight = constants_->at(144);
    [[maybe_unused]] auto& mv2_features_17_conv_0_1_weight = constants_->at(145);
    [[maybe_unused]] auto& mv2_features_17_conv_0_1_bias = constants_->at(146);
    [[maybe_unused]] auto& mv2_features_17_conv_1_0_weight = constants_->at(147);
    [[maybe_unused]] auto& mv2_features_17_conv_1_1_weight = constants_->at(148);
    [[maybe_unused]] auto& mv2_features_17_conv_1_1_bias = constants_->at(149);
    [[maybe_unused]] auto& mv2_features_17_conv_2_weight = constants_->at(150);
    [[maybe_unused]] auto& mv2_features_17_conv_3_weight = constants_->at(151);
    [[maybe_unused]] auto& mv2_features_17_conv_3_bias = constants_->at(152);
    [[maybe_unused]] auto& mv2_features_18_0_weight = constants_->at(153);
    [[maybe_unused]] auto& mv2_features_18_1_weight = constants_->at(154);
    [[maybe_unused]] auto& mv2_features_18_1_bias = constants_->at(155);
    [[maybe_unused]] auto& mv2_classifier_1_weight = constants_->at(156);
    [[maybe_unused]] auto& mv2_classifier_1_bias = constants_->at(157);
    [[maybe_unused]] auto& mv2_features_0_1_running_mean = constants_->at(158);
    [[maybe_unused]] auto& mv2_features_0_1_running_var = constants_->at(159);
    [[maybe_unused]] auto& mv2_features_1_conv_0_1_running_mean = constants_->at(160);
    [[maybe_unused]] auto& mv2_features_1_conv_0_1_running_var = constants_->at(161);
    [[maybe_unused]] auto& mv2_features_1_conv_2_running_mean = constants_->at(162);
    [[maybe_unused]] auto& mv2_features_1_conv_2_running_var = constants_->at(163);
    [[maybe_unused]] auto& mv2_features_2_conv_0_1_running_mean = constants_->at(164);
    [[maybe_unused]] auto& mv2_features_2_conv_0_1_running_var = constants_->at(165);
    [[maybe_unused]] auto& mv2_features_2_conv_1_1_running_mean = constants_->at(166);
    [[maybe_unused]] auto& mv2_features_2_conv_1_1_running_var = constants_->at(167);
    [[maybe_unused]] auto& mv2_features_2_conv_3_running_mean = constants_->at(168);
    [[maybe_unused]] auto& mv2_features_2_conv_3_running_var = constants_->at(169);
    [[maybe_unused]] auto& mv2_features_3_conv_0_1_running_mean = constants_->at(170);
    [[maybe_unused]] auto& mv2_features_3_conv_0_1_running_var = constants_->at(171);
    [[maybe_unused]] auto& mv2_features_3_conv_1_1_running_mean = constants_->at(172);
    [[maybe_unused]] auto& mv2_features_3_conv_1_1_running_var = constants_->at(173);
    [[maybe_unused]] auto& mv2_features_3_conv_3_running_mean = constants_->at(174);
    [[maybe_unused]] auto& mv2_features_3_conv_3_running_var = constants_->at(175);
    [[maybe_unused]] auto& mv2_features_4_conv_0_1_running_mean = constants_->at(176);
    [[maybe_unused]] auto& mv2_features_4_conv_0_1_running_var = constants_->at(177);
    [[maybe_unused]] auto& mv2_features_4_conv_1_1_running_mean = constants_->at(178);
    [[maybe_unused]] auto& mv2_features_4_conv_1_1_running_var = constants_->at(179);
    [[maybe_unused]] auto& mv2_features_4_conv_3_running_mean = constants_->at(180);
    [[maybe_unused]] auto& mv2_features_4_conv_3_running_var = constants_->at(181);
    [[maybe_unused]] auto& mv2_features_5_conv_0_1_running_mean = constants_->at(182);
    [[maybe_unused]] auto& mv2_features_5_conv_0_1_running_var = constants_->at(183);
    [[maybe_unused]] auto& mv2_features_5_conv_1_1_running_mean = constants_->at(184);
    [[maybe_unused]] auto& mv2_features_5_conv_1_1_running_var = constants_->at(185);
    [[maybe_unused]] auto& mv2_features_5_conv_3_running_mean = constants_->at(186);
    [[maybe_unused]] auto& mv2_features_5_conv_3_running_var = constants_->at(187);
    [[maybe_unused]] auto& mv2_features_6_conv_0_1_running_mean = constants_->at(188);
    [[maybe_unused]] auto& mv2_features_6_conv_0_1_running_var = constants_->at(189);
    [[maybe_unused]] auto& mv2_features_6_conv_1_1_running_mean = constants_->at(190);
    [[maybe_unused]] auto& mv2_features_6_conv_1_1_running_var = constants_->at(191);
    [[maybe_unused]] auto& mv2_features_6_conv_3_running_mean = constants_->at(192);
    [[maybe_unused]] auto& mv2_features_6_conv_3_running_var = constants_->at(193);
    [[maybe_unused]] auto& mv2_features_7_conv_0_1_running_mean = constants_->at(194);
    [[maybe_unused]] auto& mv2_features_7_conv_0_1_running_var = constants_->at(195);
    [[maybe_unused]] auto& mv2_features_7_conv_1_1_running_mean = constants_->at(196);
    [[maybe_unused]] auto& mv2_features_7_conv_1_1_running_var = constants_->at(197);
    [[maybe_unused]] auto& mv2_features_7_conv_3_running_mean = constants_->at(198);
    [[maybe_unused]] auto& mv2_features_7_conv_3_running_var = constants_->at(199);
    [[maybe_unused]] auto& mv2_features_8_conv_0_1_running_mean = constants_->at(200);
    [[maybe_unused]] auto& mv2_features_8_conv_0_1_running_var = constants_->at(201);
    [[maybe_unused]] auto& mv2_features_8_conv_1_1_running_mean = constants_->at(202);
    [[maybe_unused]] auto& mv2_features_8_conv_1_1_running_var = constants_->at(203);
    [[maybe_unused]] auto& mv2_features_8_conv_3_running_mean = constants_->at(204);
    [[maybe_unused]] auto& mv2_features_8_conv_3_running_var = constants_->at(205);
    [[maybe_unused]] auto& mv2_features_9_conv_0_1_running_mean = constants_->at(206);
    [[maybe_unused]] auto& mv2_features_9_conv_0_1_running_var = constants_->at(207);
    [[maybe_unused]] auto& mv2_features_9_conv_1_1_running_mean = constants_->at(208);
    [[maybe_unused]] auto& mv2_features_9_conv_1_1_running_var = constants_->at(209);
    [[maybe_unused]] auto& mv2_features_9_conv_3_running_mean = constants_->at(210);
    [[maybe_unused]] auto& mv2_features_9_conv_3_running_var = constants_->at(211);
    [[maybe_unused]] auto& mv2_features_10_conv_0_1_running_mean = constants_->at(212);
    [[maybe_unused]] auto& mv2_features_10_conv_0_1_running_var = constants_->at(213);
    [[maybe_unused]] auto& mv2_features_10_conv_1_1_running_mean = constants_->at(214);
    [[maybe_unused]] auto& mv2_features_10_conv_1_1_running_var = constants_->at(215);
    [[maybe_unused]] auto& mv2_features_10_conv_3_running_mean = constants_->at(216);
    [[maybe_unused]] auto& mv2_features_10_conv_3_running_var = constants_->at(217);
    [[maybe_unused]] auto& mv2_features_11_conv_0_1_running_mean = constants_->at(218);
    [[maybe_unused]] auto& mv2_features_11_conv_0_1_running_var = constants_->at(219);
    [[maybe_unused]] auto& mv2_features_11_conv_1_1_running_mean = constants_->at(220);
    [[maybe_unused]] auto& mv2_features_11_conv_1_1_running_var = constants_->at(221);
    [[maybe_unused]] auto& mv2_features_11_conv_3_running_mean = constants_->at(222);
    [[maybe_unused]] auto& mv2_features_11_conv_3_running_var = constants_->at(223);
    [[maybe_unused]] auto& mv2_features_12_conv_0_1_running_mean = constants_->at(224);
    [[maybe_unused]] auto& mv2_features_12_conv_0_1_running_var = constants_->at(225);
    [[maybe_unused]] auto& mv2_features_12_conv_1_1_running_mean = constants_->at(226);
    [[maybe_unused]] auto& mv2_features_12_conv_1_1_running_var = constants_->at(227);
    [[maybe_unused]] auto& mv2_features_12_conv_3_running_mean = constants_->at(228);
    [[maybe_unused]] auto& mv2_features_12_conv_3_running_var = constants_->at(229);
    [[maybe_unused]] auto& mv2_features_13_conv_0_1_running_mean = constants_->at(230);
    [[maybe_unused]] auto& mv2_features_13_conv_0_1_running_var = constants_->at(231);
    [[maybe_unused]] auto& mv2_features_13_conv_1_1_running_mean = constants_->at(232);
    [[maybe_unused]] auto& mv2_features_13_conv_1_1_running_var = constants_->at(233);
    [[maybe_unused]] auto& mv2_features_13_conv_3_running_mean = constants_->at(234);
    [[maybe_unused]] auto& mv2_features_13_conv_3_running_var = constants_->at(235);
    [[maybe_unused]] auto& mv2_features_14_conv_0_1_running_mean = constants_->at(236);
    [[maybe_unused]] auto& mv2_features_14_conv_0_1_running_var = constants_->at(237);
    [[maybe_unused]] auto& mv2_features_14_conv_1_1_running_mean = constants_->at(238);
    [[maybe_unused]] auto& mv2_features_14_conv_1_1_running_var = constants_->at(239);
    [[maybe_unused]] auto& mv2_features_14_conv_3_running_mean = constants_->at(240);
    [[maybe_unused]] auto& mv2_features_14_conv_3_running_var = constants_->at(241);
    [[maybe_unused]] auto& mv2_features_15_conv_0_1_running_mean = constants_->at(242);
    [[maybe_unused]] auto& mv2_features_15_conv_0_1_running_var = constants_->at(243);
    [[maybe_unused]] auto& mv2_features_15_conv_1_1_running_mean = constants_->at(244);
    [[maybe_unused]] auto& mv2_features_15_conv_1_1_running_var = constants_->at(245);
    [[maybe_unused]] auto& mv2_features_15_conv_3_running_mean = constants_->at(246);
    [[maybe_unused]] auto& mv2_features_15_conv_3_running_var = constants_->at(247);
    [[maybe_unused]] auto& mv2_features_16_conv_0_1_running_mean = constants_->at(248);
    [[maybe_unused]] auto& mv2_features_16_conv_0_1_running_var = constants_->at(249);
    [[maybe_unused]] auto& mv2_features_16_conv_1_1_running_mean = constants_->at(250);
    [[maybe_unused]] auto& mv2_features_16_conv_1_1_running_var = constants_->at(251);
    [[maybe_unused]] auto& mv2_features_16_conv_3_running_mean = constants_->at(252);
    [[maybe_unused]] auto& mv2_features_16_conv_3_running_var = constants_->at(253);
    [[maybe_unused]] auto& mv2_features_17_conv_0_1_running_mean = constants_->at(254);
    [[maybe_unused]] auto& mv2_features_17_conv_0_1_running_var = constants_->at(255);
    [[maybe_unused]] auto& mv2_features_17_conv_1_1_running_mean = constants_->at(256);
    [[maybe_unused]] auto& mv2_features_17_conv_1_1_running_var = constants_->at(257);
    [[maybe_unused]] auto& mv2_features_17_conv_3_running_mean = constants_->at(258);
    [[maybe_unused]] auto& mv2_features_17_conv_3_running_var = constants_->at(259);
    [[maybe_unused]] auto& mv2_features_18_1_running_mean = constants_->at(260);
    [[maybe_unused]] auto& mv2_features_18_1_running_var = constants_->at(261);

    if ((long(arg262_1.data_ptr()) & (16 -1)) != 0) {
        AOTI_TORCH_WARN("Input 0 was compiled as 16-bytes aligned, but it is not aligned at run time. Copying to an aligned tensor to guarantee correctness, but expect a performance hit.");
        AtenTensorHandle arg262_1_aligned;
        aoti_torch_clone_preserve_strides(arg262_1, &arg262_1_aligned);
        arg262_1 = std::move(RAIIAtenTensorHandle(arg262_1_aligned));
    }
    inputs.clear();
    [[maybe_unused]] auto& kernels = static_cast<AOTInductorModelKernels&>(*this->kernels_.get());

    AOTICudaStreamGuard stream_guard(stream, this->device_idx_);
    static constexpr int64_t int_array_0[] = {1L, 3L, 224L, 224L};
    static constexpr int64_t int_array_1[] = {150528L, 1L, 672L, 3L};
    AtenTensorHandle buf0_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(4, int_array_0, int_array_1, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf0_handle));
    RAIIAtenTensorHandle buf0(buf0_handle);
    // Topologically Sorted Source Nodes: [aten_convolution_default], Original ATen: [aten.convolution]
    call_triton_poi_fused_convolution_0(arg262_1, buf0, 3L, 50176L, this->device_idx_, stream, kernels, this->cubin_dir_);
    arg262_1.reset();
    static constexpr int64_t int_array_2[] = {32L, 3L, 3L, 3L};
    static constexpr int64_t int_array_3[] = {27L, 1L, 9L, 3L};
    AtenTensorHandle buf1_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(4, int_array_2, int_array_3, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf1_handle));
    RAIIAtenTensorHandle buf1(buf1_handle);
    // Topologically Sorted Source Nodes: [aten_convolution_default], Original ATen: [aten.convolution]
    call_triton_poi_fused_convolution_1(mv2_features_0_0_weight, buf1, 96L, 9L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten_convolution_default], Original ATen: [aten.convolution]
    AtenTensorHandle buf2_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf0, buf1, nullptr, std::array<int64_t, 2>{2L, 2L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf2_handle));
    RAIIAtenTensorHandle buf2(buf2_handle);
    buf0.reset();
    buf1.reset();
    auto buf3 = std::move(buf2);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default, aten_hardtanh_default], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_2(buf3, mv2_features_0_1_running_mean, mv2_features_0_1_running_var, mv2_features_0_1_weight, mv2_features_0_1_bias, 401408L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default, aten_hardtanh_default, aten_convolution_default_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf4_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf3, mv2_features_1_conv_0_0_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 32L, &buf4_handle));
    RAIIAtenTensorHandle buf4(buf4_handle);
    buf3.reset();
    auto buf5 = std::move(buf4);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_1, aten_hardtanh_default_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_2(buf5, mv2_features_1_conv_0_1_running_mean, mv2_features_1_conv_0_1_running_var, mv2_features_1_conv_0_1_weight, mv2_features_1_conv_0_1_bias, 401408L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_1, aten_hardtanh_default_1, aten_convolution_default_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf6_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf5, mv2_features_1_conv_1_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf6_handle));
    RAIIAtenTensorHandle buf6(buf6_handle);
    buf5.reset();
    auto buf7 = std::move(buf6);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_2], Original ATen: [aten._native_batch_norm_legit_no_training]
    call_triton_poi_fused__native_batch_norm_legit_no_training_3(buf7, mv2_features_1_conv_2_running_mean, mv2_features_1_conv_2_running_var, mv2_features_1_conv_2_weight, mv2_features_1_conv_2_bias, 200704L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_2, aten_convolution_default_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    AtenTensorHandle buf8_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf7, mv2_features_2_conv_0_0_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf8_handle));
    RAIIAtenTensorHandle buf8(buf8_handle);
    buf7.reset();
    auto buf9 = std::move(buf8);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_3, aten_hardtanh_default_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_4(buf9, mv2_features_2_conv_0_1_running_mean, mv2_features_2_conv_0_1_running_var, mv2_features_2_conv_0_1_weight, mv2_features_2_conv_0_1_bias, 1204224L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_3, aten_hardtanh_default_2, aten_convolution_default_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf10_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf9, mv2_features_2_conv_1_0_weight, nullptr, std::array<int64_t, 2>{2L, 2L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 96L, &buf10_handle));
    RAIIAtenTensorHandle buf10(buf10_handle);
    buf9.reset();
    auto buf11 = std::move(buf10);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_4, aten_hardtanh_default_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_5(buf11, mv2_features_2_conv_1_1_running_mean, mv2_features_2_conv_1_1_running_var, mv2_features_2_conv_1_1_weight, mv2_features_2_conv_1_1_bias, 301056L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_4, aten_hardtanh_default_3, aten_convolution_default_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf12_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf11, mv2_features_2_conv_2_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf12_handle));
    RAIIAtenTensorHandle buf12(buf12_handle);
    buf11.reset();
    auto buf13 = std::move(buf12);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_5], Original ATen: [aten._native_batch_norm_legit_no_training]
    call_triton_poi_fused__native_batch_norm_legit_no_training_6(buf13, mv2_features_2_conv_3_running_mean, mv2_features_2_conv_3_running_var, mv2_features_2_conv_3_weight, mv2_features_2_conv_3_bias, 75264L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten_convolution_default_6], Original ATen: [aten.convolution]
    AtenTensorHandle buf14_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf13, mv2_features_3_conv_0_0_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf14_handle));
    RAIIAtenTensorHandle buf14(buf14_handle);
    auto buf15 = std::move(buf14);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_6, aten_hardtanh_default_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_7(buf15, mv2_features_3_conv_0_1_running_mean, mv2_features_3_conv_0_1_running_var, mv2_features_3_conv_0_1_weight, mv2_features_3_conv_0_1_bias, 451584L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_6, aten_hardtanh_default_4, aten_convolution_default_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf16_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf15, mv2_features_3_conv_1_0_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 144L, &buf16_handle));
    RAIIAtenTensorHandle buf16(buf16_handle);
    buf15.reset();
    auto buf17 = std::move(buf16);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_7, aten_hardtanh_default_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_7(buf17, mv2_features_3_conv_1_1_running_mean, mv2_features_3_conv_1_1_running_var, mv2_features_3_conv_1_1_weight, mv2_features_3_conv_1_1_bias, 451584L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_7, aten_hardtanh_default_5, aten_convolution_default_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf18_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf17, mv2_features_3_conv_2_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf18_handle));
    RAIIAtenTensorHandle buf18(buf18_handle);
    buf17.reset();
    auto buf19 = std::move(buf13);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_8, aten_add_tensor], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
    call_triton_poi_fused__native_batch_norm_legit_no_training_add_8(buf19, buf18, mv2_features_3_conv_3_running_mean, mv2_features_3_conv_3_running_var, mv2_features_3_conv_3_weight, mv2_features_3_conv_3_bias, 75264L, this->device_idx_, stream, kernels, this->cubin_dir_);
    buf18.reset();
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_8, aten_add_tensor, aten_convolution_default_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    AtenTensorHandle buf20_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf19, mv2_features_4_conv_0_0_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf20_handle));
    RAIIAtenTensorHandle buf20(buf20_handle);
    buf19.reset();
    auto buf21 = std::move(buf20);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_9, aten_hardtanh_default_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_7(buf21, mv2_features_4_conv_0_1_running_mean, mv2_features_4_conv_0_1_running_var, mv2_features_4_conv_0_1_weight, mv2_features_4_conv_0_1_bias, 451584L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_9, aten_hardtanh_default_6, aten_convolution_default_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf22_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf21, mv2_features_4_conv_1_0_weight, nullptr, std::array<int64_t, 2>{2L, 2L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 144L, &buf22_handle));
    RAIIAtenTensorHandle buf22(buf22_handle);
    buf21.reset();
    auto buf23 = std::move(buf22);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_10, aten_hardtanh_default_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9(buf23, mv2_features_4_conv_1_1_running_mean, mv2_features_4_conv_1_1_running_var, mv2_features_4_conv_1_1_weight, mv2_features_4_conv_1_1_bias, 112896L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_10, aten_hardtanh_default_7, aten_convolution_default_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf24_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf23, mv2_features_4_conv_2_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf24_handle));
    RAIIAtenTensorHandle buf24(buf24_handle);
    buf23.reset();
    auto buf25 = std::move(buf24);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_11], Original ATen: [aten._native_batch_norm_legit_no_training]
    call_triton_poi_fused__native_batch_norm_legit_no_training_10(buf25, mv2_features_4_conv_3_running_mean, mv2_features_4_conv_3_running_var, mv2_features_4_conv_3_weight, mv2_features_4_conv_3_bias, 25088L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten_convolution_default_12], Original ATen: [aten.convolution]
    AtenTensorHandle buf26_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf25, mv2_features_5_conv_0_0_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf26_handle));
    RAIIAtenTensorHandle buf26(buf26_handle);
    auto buf27 = std::move(buf26);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_12, aten_hardtanh_default_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11(buf27, mv2_features_5_conv_0_1_running_mean, mv2_features_5_conv_0_1_running_var, mv2_features_5_conv_0_1_weight, mv2_features_5_conv_0_1_bias, 150528L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_12, aten_hardtanh_default_8, aten_convolution_default_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf28_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf27, mv2_features_5_conv_1_0_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 192L, &buf28_handle));
    RAIIAtenTensorHandle buf28(buf28_handle);
    buf27.reset();
    auto buf29 = std::move(buf28);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_13, aten_hardtanh_default_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11(buf29, mv2_features_5_conv_1_1_running_mean, mv2_features_5_conv_1_1_running_var, mv2_features_5_conv_1_1_weight, mv2_features_5_conv_1_1_bias, 150528L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_13, aten_hardtanh_default_9, aten_convolution_default_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf30_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf29, mv2_features_5_conv_2_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf30_handle));
    RAIIAtenTensorHandle buf30(buf30_handle);
    buf29.reset();
    auto buf31 = std::move(buf25);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_14, aten_add_tensor_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
    call_triton_poi_fused__native_batch_norm_legit_no_training_add_12(buf31, buf30, mv2_features_5_conv_3_running_mean, mv2_features_5_conv_3_running_var, mv2_features_5_conv_3_weight, mv2_features_5_conv_3_bias, 25088L, this->device_idx_, stream, kernels, this->cubin_dir_);
    buf30.reset();
    // Topologically Sorted Source Nodes: [aten_convolution_default_15], Original ATen: [aten.convolution]
    AtenTensorHandle buf32_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf31, mv2_features_6_conv_0_0_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf32_handle));
    RAIIAtenTensorHandle buf32(buf32_handle);
    auto buf33 = std::move(buf32);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_15, aten_hardtanh_default_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11(buf33, mv2_features_6_conv_0_1_running_mean, mv2_features_6_conv_0_1_running_var, mv2_features_6_conv_0_1_weight, mv2_features_6_conv_0_1_bias, 150528L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_15, aten_hardtanh_default_10, aten_convolution_default_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf34_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf33, mv2_features_6_conv_1_0_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 192L, &buf34_handle));
    RAIIAtenTensorHandle buf34(buf34_handle);
    buf33.reset();
    auto buf35 = std::move(buf34);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_16, aten_hardtanh_default_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11(buf35, mv2_features_6_conv_1_1_running_mean, mv2_features_6_conv_1_1_running_var, mv2_features_6_conv_1_1_weight, mv2_features_6_conv_1_1_bias, 150528L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_16, aten_hardtanh_default_11, aten_convolution_default_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf36_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf35, mv2_features_6_conv_2_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf36_handle));
    RAIIAtenTensorHandle buf36(buf36_handle);
    buf35.reset();
    auto buf37 = std::move(buf31);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_17, aten_add_tensor_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
    call_triton_poi_fused__native_batch_norm_legit_no_training_add_12(buf37, buf36, mv2_features_6_conv_3_running_mean, mv2_features_6_conv_3_running_var, mv2_features_6_conv_3_weight, mv2_features_6_conv_3_bias, 25088L, this->device_idx_, stream, kernels, this->cubin_dir_);
    buf36.reset();
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_17, aten_add_tensor_2, aten_convolution_default_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    AtenTensorHandle buf38_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf37, mv2_features_7_conv_0_0_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf38_handle));
    RAIIAtenTensorHandle buf38(buf38_handle);
    buf37.reset();
    auto buf39 = std::move(buf38);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_18, aten_hardtanh_default_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11(buf39, mv2_features_7_conv_0_1_running_mean, mv2_features_7_conv_0_1_running_var, mv2_features_7_conv_0_1_weight, mv2_features_7_conv_0_1_bias, 150528L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_18, aten_hardtanh_default_12, aten_convolution_default_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf40_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf39, mv2_features_7_conv_1_0_weight, nullptr, std::array<int64_t, 2>{2L, 2L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 192L, &buf40_handle));
    RAIIAtenTensorHandle buf40(buf40_handle);
    buf39.reset();
    auto buf41 = std::move(buf40);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_19, aten_hardtanh_default_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13(buf41, mv2_features_7_conv_1_1_running_mean, mv2_features_7_conv_1_1_running_var, mv2_features_7_conv_1_1_weight, mv2_features_7_conv_1_1_bias, 37632L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_19, aten_hardtanh_default_13, aten_convolution_default_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf42_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf41, mv2_features_7_conv_2_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf42_handle));
    RAIIAtenTensorHandle buf42(buf42_handle);
    buf41.reset();
    auto buf43 = std::move(buf42);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_20], Original ATen: [aten._native_batch_norm_legit_no_training]
    call_triton_poi_fused__native_batch_norm_legit_no_training_14(buf43, mv2_features_7_conv_3_running_mean, mv2_features_7_conv_3_running_var, mv2_features_7_conv_3_weight, mv2_features_7_conv_3_bias, 12544L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten_convolution_default_21], Original ATen: [aten.convolution]
    AtenTensorHandle buf44_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf43, mv2_features_8_conv_0_0_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf44_handle));
    RAIIAtenTensorHandle buf44(buf44_handle);
    auto buf45 = std::move(buf44);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_21, aten_hardtanh_default_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15(buf45, mv2_features_8_conv_0_1_running_mean, mv2_features_8_conv_0_1_running_var, mv2_features_8_conv_0_1_weight, mv2_features_8_conv_0_1_bias, 75264L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_21, aten_hardtanh_default_14, aten_convolution_default_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf46_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf45, mv2_features_8_conv_1_0_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 384L, &buf46_handle));
    RAIIAtenTensorHandle buf46(buf46_handle);
    buf45.reset();
    auto buf47 = std::move(buf46);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_22, aten_hardtanh_default_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15(buf47, mv2_features_8_conv_1_1_running_mean, mv2_features_8_conv_1_1_running_var, mv2_features_8_conv_1_1_weight, mv2_features_8_conv_1_1_bias, 75264L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_22, aten_hardtanh_default_15, aten_convolution_default_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf48_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf47, mv2_features_8_conv_2_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf48_handle));
    RAIIAtenTensorHandle buf48(buf48_handle);
    buf47.reset();
    auto buf49 = std::move(buf43);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_23, aten_add_tensor_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
    call_triton_poi_fused__native_batch_norm_legit_no_training_add_16(buf49, buf48, mv2_features_8_conv_3_running_mean, mv2_features_8_conv_3_running_var, mv2_features_8_conv_3_weight, mv2_features_8_conv_3_bias, 12544L, this->device_idx_, stream, kernels, this->cubin_dir_);
    buf48.reset();
    // Topologically Sorted Source Nodes: [aten_convolution_default_24], Original ATen: [aten.convolution]
    AtenTensorHandle buf50_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf49, mv2_features_9_conv_0_0_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf50_handle));
    RAIIAtenTensorHandle buf50(buf50_handle);
    auto buf51 = std::move(buf50);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_24, aten_hardtanh_default_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15(buf51, mv2_features_9_conv_0_1_running_mean, mv2_features_9_conv_0_1_running_var, mv2_features_9_conv_0_1_weight, mv2_features_9_conv_0_1_bias, 75264L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_24, aten_hardtanh_default_16, aten_convolution_default_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf52_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf51, mv2_features_9_conv_1_0_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 384L, &buf52_handle));
    RAIIAtenTensorHandle buf52(buf52_handle);
    buf51.reset();
    auto buf53 = std::move(buf52);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_25, aten_hardtanh_default_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15(buf53, mv2_features_9_conv_1_1_running_mean, mv2_features_9_conv_1_1_running_var, mv2_features_9_conv_1_1_weight, mv2_features_9_conv_1_1_bias, 75264L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_25, aten_hardtanh_default_17, aten_convolution_default_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf54_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf53, mv2_features_9_conv_2_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf54_handle));
    RAIIAtenTensorHandle buf54(buf54_handle);
    buf53.reset();
    auto buf55 = std::move(buf49);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_26, aten_add_tensor_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
    call_triton_poi_fused__native_batch_norm_legit_no_training_add_16(buf55, buf54, mv2_features_9_conv_3_running_mean, mv2_features_9_conv_3_running_var, mv2_features_9_conv_3_weight, mv2_features_9_conv_3_bias, 12544L, this->device_idx_, stream, kernels, this->cubin_dir_);
    buf54.reset();
    // Topologically Sorted Source Nodes: [aten_convolution_default_27], Original ATen: [aten.convolution]
    AtenTensorHandle buf56_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf55, mv2_features_10_conv_0_0_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf56_handle));
    RAIIAtenTensorHandle buf56(buf56_handle);
    auto buf57 = std::move(buf56);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_27, aten_hardtanh_default_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15(buf57, mv2_features_10_conv_0_1_running_mean, mv2_features_10_conv_0_1_running_var, mv2_features_10_conv_0_1_weight, mv2_features_10_conv_0_1_bias, 75264L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_27, aten_hardtanh_default_18, aten_convolution_default_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf58_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf57, mv2_features_10_conv_1_0_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 384L, &buf58_handle));
    RAIIAtenTensorHandle buf58(buf58_handle);
    buf57.reset();
    auto buf59 = std::move(buf58);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_28, aten_hardtanh_default_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15(buf59, mv2_features_10_conv_1_1_running_mean, mv2_features_10_conv_1_1_running_var, mv2_features_10_conv_1_1_weight, mv2_features_10_conv_1_1_bias, 75264L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_28, aten_hardtanh_default_19, aten_convolution_default_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf60_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf59, mv2_features_10_conv_2_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf60_handle));
    RAIIAtenTensorHandle buf60(buf60_handle);
    buf59.reset();
    auto buf61 = std::move(buf55);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_29, aten_add_tensor_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
    call_triton_poi_fused__native_batch_norm_legit_no_training_add_16(buf61, buf60, mv2_features_10_conv_3_running_mean, mv2_features_10_conv_3_running_var, mv2_features_10_conv_3_weight, mv2_features_10_conv_3_bias, 12544L, this->device_idx_, stream, kernels, this->cubin_dir_);
    buf60.reset();
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_29, aten_add_tensor_5, aten_convolution_default_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    AtenTensorHandle buf62_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf61, mv2_features_11_conv_0_0_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf62_handle));
    RAIIAtenTensorHandle buf62(buf62_handle);
    buf61.reset();
    auto buf63 = std::move(buf62);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_30, aten_hardtanh_default_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15(buf63, mv2_features_11_conv_0_1_running_mean, mv2_features_11_conv_0_1_running_var, mv2_features_11_conv_0_1_weight, mv2_features_11_conv_0_1_bias, 75264L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_30, aten_hardtanh_default_20, aten_convolution_default_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf64_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf63, mv2_features_11_conv_1_0_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 384L, &buf64_handle));
    RAIIAtenTensorHandle buf64(buf64_handle);
    buf63.reset();
    auto buf65 = std::move(buf64);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_31, aten_hardtanh_default_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15(buf65, mv2_features_11_conv_1_1_running_mean, mv2_features_11_conv_1_1_running_var, mv2_features_11_conv_1_1_weight, mv2_features_11_conv_1_1_bias, 75264L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_31, aten_hardtanh_default_21, aten_convolution_default_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf66_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf65, mv2_features_11_conv_2_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf66_handle));
    RAIIAtenTensorHandle buf66(buf66_handle);
    buf65.reset();
    auto buf67 = std::move(buf66);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_32], Original ATen: [aten._native_batch_norm_legit_no_training]
    call_triton_poi_fused__native_batch_norm_legit_no_training_17(buf67, mv2_features_11_conv_3_running_mean, mv2_features_11_conv_3_running_var, mv2_features_11_conv_3_weight, mv2_features_11_conv_3_bias, 18816L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten_convolution_default_33], Original ATen: [aten.convolution]
    AtenTensorHandle buf68_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf67, mv2_features_12_conv_0_0_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf68_handle));
    RAIIAtenTensorHandle buf68(buf68_handle);
    auto buf69 = std::move(buf68);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_33, aten_hardtanh_default_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_18(buf69, mv2_features_12_conv_0_1_running_mean, mv2_features_12_conv_0_1_running_var, mv2_features_12_conv_0_1_weight, mv2_features_12_conv_0_1_bias, 112896L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_33, aten_hardtanh_default_22, aten_convolution_default_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf70_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf69, mv2_features_12_conv_1_0_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 576L, &buf70_handle));
    RAIIAtenTensorHandle buf70(buf70_handle);
    buf69.reset();
    auto buf71 = std::move(buf70);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_34, aten_hardtanh_default_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_18(buf71, mv2_features_12_conv_1_1_running_mean, mv2_features_12_conv_1_1_running_var, mv2_features_12_conv_1_1_weight, mv2_features_12_conv_1_1_bias, 112896L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_34, aten_hardtanh_default_23, aten_convolution_default_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf72_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf71, mv2_features_12_conv_2_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf72_handle));
    RAIIAtenTensorHandle buf72(buf72_handle);
    buf71.reset();
    auto buf73 = std::move(buf67);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_35, aten_add_tensor_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
    call_triton_poi_fused__native_batch_norm_legit_no_training_add_19(buf73, buf72, mv2_features_12_conv_3_running_mean, mv2_features_12_conv_3_running_var, mv2_features_12_conv_3_weight, mv2_features_12_conv_3_bias, 18816L, this->device_idx_, stream, kernels, this->cubin_dir_);
    buf72.reset();
    // Topologically Sorted Source Nodes: [aten_convolution_default_36], Original ATen: [aten.convolution]
    AtenTensorHandle buf74_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf73, mv2_features_13_conv_0_0_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf74_handle));
    RAIIAtenTensorHandle buf74(buf74_handle);
    auto buf75 = std::move(buf74);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_36, aten_hardtanh_default_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_18(buf75, mv2_features_13_conv_0_1_running_mean, mv2_features_13_conv_0_1_running_var, mv2_features_13_conv_0_1_weight, mv2_features_13_conv_0_1_bias, 112896L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_36, aten_hardtanh_default_24, aten_convolution_default_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf76_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf75, mv2_features_13_conv_1_0_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 576L, &buf76_handle));
    RAIIAtenTensorHandle buf76(buf76_handle);
    buf75.reset();
    auto buf77 = std::move(buf76);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_37, aten_hardtanh_default_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_18(buf77, mv2_features_13_conv_1_1_running_mean, mv2_features_13_conv_1_1_running_var, mv2_features_13_conv_1_1_weight, mv2_features_13_conv_1_1_bias, 112896L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_37, aten_hardtanh_default_25, aten_convolution_default_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf78_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf77, mv2_features_13_conv_2_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf78_handle));
    RAIIAtenTensorHandle buf78(buf78_handle);
    buf77.reset();
    auto buf79 = std::move(buf73);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_38, aten_add_tensor_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
    call_triton_poi_fused__native_batch_norm_legit_no_training_add_19(buf79, buf78, mv2_features_13_conv_3_running_mean, mv2_features_13_conv_3_running_var, mv2_features_13_conv_3_weight, mv2_features_13_conv_3_bias, 18816L, this->device_idx_, stream, kernels, this->cubin_dir_);
    buf78.reset();
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_38, aten_add_tensor_7, aten_convolution_default_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    AtenTensorHandle buf80_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf79, mv2_features_14_conv_0_0_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf80_handle));
    RAIIAtenTensorHandle buf80(buf80_handle);
    buf79.reset();
    auto buf81 = std::move(buf80);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_39, aten_hardtanh_default_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_18(buf81, mv2_features_14_conv_0_1_running_mean, mv2_features_14_conv_0_1_running_var, mv2_features_14_conv_0_1_weight, mv2_features_14_conv_0_1_bias, 112896L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_39, aten_hardtanh_default_26, aten_convolution_default_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf82_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf81, mv2_features_14_conv_1_0_weight, nullptr, std::array<int64_t, 2>{2L, 2L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 576L, &buf82_handle));
    RAIIAtenTensorHandle buf82(buf82_handle);
    buf81.reset();
    auto buf83 = std::move(buf82);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_40, aten_hardtanh_default_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_20(buf83, mv2_features_14_conv_1_1_running_mean, mv2_features_14_conv_1_1_running_var, mv2_features_14_conv_1_1_weight, mv2_features_14_conv_1_1_bias, 28224L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_40, aten_hardtanh_default_27, aten_convolution_default_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf84_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf83, mv2_features_14_conv_2_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf84_handle));
    RAIIAtenTensorHandle buf84(buf84_handle);
    buf83.reset();
    auto buf85 = std::move(buf84);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_41], Original ATen: [aten._native_batch_norm_legit_no_training]
    call_triton_poi_fused__native_batch_norm_legit_no_training_21(buf85, mv2_features_14_conv_3_running_mean, mv2_features_14_conv_3_running_var, mv2_features_14_conv_3_weight, mv2_features_14_conv_3_bias, 7840L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten_convolution_default_42], Original ATen: [aten.convolution]
    AtenTensorHandle buf86_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf85, mv2_features_15_conv_0_0_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf86_handle));
    RAIIAtenTensorHandle buf86(buf86_handle);
    auto buf87 = std::move(buf86);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_42, aten_hardtanh_default_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22(buf87, mv2_features_15_conv_0_1_running_mean, mv2_features_15_conv_0_1_running_var, mv2_features_15_conv_0_1_weight, mv2_features_15_conv_0_1_bias, 47040L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_42, aten_hardtanh_default_28, aten_convolution_default_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf88_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf87, mv2_features_15_conv_1_0_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 960L, &buf88_handle));
    RAIIAtenTensorHandle buf88(buf88_handle);
    buf87.reset();
    auto buf89 = std::move(buf88);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_43, aten_hardtanh_default_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22(buf89, mv2_features_15_conv_1_1_running_mean, mv2_features_15_conv_1_1_running_var, mv2_features_15_conv_1_1_weight, mv2_features_15_conv_1_1_bias, 47040L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_43, aten_hardtanh_default_29, aten_convolution_default_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf90_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf89, mv2_features_15_conv_2_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf90_handle));
    RAIIAtenTensorHandle buf90(buf90_handle);
    buf89.reset();
    auto buf91 = std::move(buf85);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_44, aten_add_tensor_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
    call_triton_poi_fused__native_batch_norm_legit_no_training_add_23(buf91, buf90, mv2_features_15_conv_3_running_mean, mv2_features_15_conv_3_running_var, mv2_features_15_conv_3_weight, mv2_features_15_conv_3_bias, 7840L, this->device_idx_, stream, kernels, this->cubin_dir_);
    buf90.reset();
    // Topologically Sorted Source Nodes: [aten_convolution_default_45], Original ATen: [aten.convolution]
    AtenTensorHandle buf92_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf91, mv2_features_16_conv_0_0_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf92_handle));
    RAIIAtenTensorHandle buf92(buf92_handle);
    auto buf93 = std::move(buf92);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_45, aten_hardtanh_default_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22(buf93, mv2_features_16_conv_0_1_running_mean, mv2_features_16_conv_0_1_running_var, mv2_features_16_conv_0_1_weight, mv2_features_16_conv_0_1_bias, 47040L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_45, aten_hardtanh_default_30, aten_convolution_default_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf94_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf93, mv2_features_16_conv_1_0_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 960L, &buf94_handle));
    RAIIAtenTensorHandle buf94(buf94_handle);
    buf93.reset();
    auto buf95 = std::move(buf94);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_46, aten_hardtanh_default_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22(buf95, mv2_features_16_conv_1_1_running_mean, mv2_features_16_conv_1_1_running_var, mv2_features_16_conv_1_1_weight, mv2_features_16_conv_1_1_bias, 47040L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_46, aten_hardtanh_default_31, aten_convolution_default_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf96_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf95, mv2_features_16_conv_2_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf96_handle));
    RAIIAtenTensorHandle buf96(buf96_handle);
    buf95.reset();
    auto buf97 = std::move(buf91);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_47, aten_add_tensor_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
    call_triton_poi_fused__native_batch_norm_legit_no_training_add_23(buf97, buf96, mv2_features_16_conv_3_running_mean, mv2_features_16_conv_3_running_var, mv2_features_16_conv_3_weight, mv2_features_16_conv_3_bias, 7840L, this->device_idx_, stream, kernels, this->cubin_dir_);
    buf96.reset();
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_47, aten_add_tensor_9, aten_convolution_default_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    AtenTensorHandle buf98_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf97, mv2_features_17_conv_0_0_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf98_handle));
    RAIIAtenTensorHandle buf98(buf98_handle);
    buf97.reset();
    auto buf99 = std::move(buf98);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_48, aten_hardtanh_default_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22(buf99, mv2_features_17_conv_0_1_running_mean, mv2_features_17_conv_0_1_running_var, mv2_features_17_conv_0_1_weight, mv2_features_17_conv_0_1_bias, 47040L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_48, aten_hardtanh_default_32, aten_convolution_default_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf100_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf99, mv2_features_17_conv_1_0_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 960L, &buf100_handle));
    RAIIAtenTensorHandle buf100(buf100_handle);
    buf99.reset();
    auto buf101 = std::move(buf100);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_49, aten_hardtanh_default_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
    call_triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22(buf101, mv2_features_17_conv_1_1_running_mean, mv2_features_17_conv_1_1_running_var, mv2_features_17_conv_1_1_weight, mv2_features_17_conv_1_1_bias, 47040L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_49, aten_hardtanh_default_33, aten_convolution_default_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
    AtenTensorHandle buf102_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf101, mv2_features_17_conv_2_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf102_handle));
    RAIIAtenTensorHandle buf102(buf102_handle);
    buf101.reset();
    auto buf103 = std::move(buf102);  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_50], Original ATen: [aten._native_batch_norm_legit_no_training]
    call_triton_poi_fused__native_batch_norm_legit_no_training_24(buf103, mv2_features_17_conv_3_running_mean, mv2_features_17_conv_3_running_var, mv2_features_17_conv_3_weight, mv2_features_17_conv_3_bias, 15680L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_50, aten_convolution_default_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    AtenTensorHandle buf104_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf103, mv2_features_18_0_weight, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf104_handle));
    RAIIAtenTensorHandle buf104(buf104_handle);
    buf103.reset();
    static constexpr int64_t int_array_4[] = {1L, 1280L, 1L, 1L};
    static constexpr int64_t int_array_5[] = {1280L, 1L, 1280L, 1280L};
    AtenTensorHandle buf105_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(4, int_array_4, int_array_5, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf105_handle));
    RAIIAtenTensorHandle buf105(buf105_handle);
    static constexpr int64_t int_array_6[] = {1280L, 1L, 1L, 1L};
    auto buf106 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf105, 4, int_array_4, int_array_6, 0L)); buf105.reset();  // reuse
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_51, aten_hardtanh_default_34, aten_mean_dim], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.mean]
    call_triton_per_fused__native_batch_norm_legit_no_training_hardtanh_mean_25(buf106, buf104, mv2_features_18_1_running_mean, mv2_features_18_1_running_var, mv2_features_18_1_weight, mv2_features_18_1_bias, 1280L, 49L, this->device_idx_, stream, kernels, this->cubin_dir_);
    buf104.reset();
    static constexpr int64_t int_array_7[] = {1280L, 1000L};
    static constexpr int64_t int_array_8[] = {1L, 1280L};
    AtenTensorHandle buf107_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(2, int_array_7, int_array_8, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf107_handle));
    RAIIAtenTensorHandle buf107(buf107_handle);
    // Topologically Sorted Source Nodes: [aten_permute_copy_default], Original ATen: [aten.permute_copy]
    call_triton_poi_fused_permute_copy_26(mv2_classifier_1_weight, buf107, 1280000L, this->device_idx_, stream, kernels, this->cubin_dir_);
    static constexpr int64_t int_array_9[] = {1L, 1000L};
    static constexpr int64_t int_array_10[] = {1000L, 1L};
    AtenTensorHandle buf108_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(2, int_array_9, int_array_10, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf108_handle));
    RAIIAtenTensorHandle buf108(buf108_handle);
    // Topologically Sorted Source Nodes: [aten__native_batch_norm_legit_no_training_default_51, aten_hardtanh_default_34, aten_mean_dim, aten_view_copy_default, aten_permute_copy_default, aten_addmm_default], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.mean, aten.view_copy, aten.permute_copy, aten.addmm]
    static constexpr int64_t int_array_11[] = {0L, 1L};
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_addmm_out(buf108, mv2_classifier_1_bias, wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf106, 2, int_array_8, int_array_11, 0L)), buf107, 1L, 1L));
    buf106.reset();
    buf107.reset();
    output_handles[0] = buf108.release();
} // AOTInductorModel::run_impl
} // namespace torch::aot_inductor




// Compile cmd
// g++ /home/gasoonjia/executorch/ce5v7wqyagkbtsmdw5kshpjd2t6vrjvl6ndtpaca5r3ct3piucq7.wrapper.cpp -D TORCH_INDUCTOR_CPP_WRAPPER -D STANDALONE_TORCH_HEADER -D  C10_USING_CUSTOM_GENERATED_MACROS -D CPU_CAPABILITY_AVX512 -D  USE_CUDA  -fPIC -O1 -DNDEBUG -fno-trapping-math -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros -fno-math-errno -fexcess-precision=fast -fno-finite-math-only -fno-unsafe-math-optimizations -ffp-contract=off -fno-tree-loop-vectorize -march=native -Wall -std=c++17 -Wno-unused-variable -Wno-unknown-pragmas -pedantic -fopenmp  -include /tmp/torchinductor_gasoonjia/precompiled_headers/c4cub4usfsuwqkbp3pfgzit6fkb6qpm3anlkt22y6d2ks3tdluhg.h -I/home/gasoonjia/.conda/envs/aoti/include/python3.10 -I/home/gasoonjia/.conda/envs/aoti/lib/python3.10/site-packages/torch/include -I/home/gasoonjia/.conda/envs/aoti/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/usr/local/cuda-12.6/include   -mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma  -c -o /home/gasoonjia/executorch/ce5v7wqyagkbtsmdw5kshpjd2t6vrjvl6ndtpaca5r3ct3piucq7.wrapper.o
// Link cmd
// g++ /home/gasoonjia/executorch/ce5v7wqyagkbtsmdw5kshpjd2t6vrjvl6ndtpaca5r3ct3piucq7.wrapper.o /home/gasoonjia/executorch/c5cna3r6nfys2tflf6chfc3l6l6rv4a3am2yslkkhyp4e7oaf7ej.kernel.o /home/gasoonjia/executorch/ce5v7wqyagkbtsmdw5kshpjd2t6vrjvl6ndtpaca5r3ct3piucq7/clbguuj2vb7nlf7qm72hrkynyiorwc3udkaj656f3v5xcdaoib67.o -D TORCH_INDUCTOR_CPP_WRAPPER -D STANDALONE_TORCH_HEADER -D  C10_USING_CUSTOM_GENERATED_MACROS -D CPU_CAPABILITY_AVX512 -D  USE_CUDA  -shared -fPIC -O3 -DNDEBUG -fno-trapping-math -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros -fno-math-errno -fexcess-precision=fast -fno-finite-math-only -fno-unsafe-math-optimizations -ffp-contract=off -fno-tree-loop-vectorize -march=native -Wall -std=c++17 -Wno-unused-variable -Wno-unknown-pragmas -pedantic -fopenmp  -I/home/gasoonjia/.conda/envs/aoti/include/python3.10 -I/home/gasoonjia/.conda/envs/aoti/lib/python3.10/site-packages/torch/include -I/home/gasoonjia/.conda/envs/aoti/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/usr/local/cuda-12.6/include   -mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma  -o /home/gasoonjia/executorch/aoti.so  -ltorch -ltorch_cpu -lgomp -lc10 -lc10_cuda -lcuda -ltorch_cuda  -L/home/gasoonjia/.conda/envs/aoti/lib -L/home/gasoonjia/.conda/envs/aoti/lib/python3.10/site-packages/torch/lib -L/usr/local/cuda-12.6/targets/x86_64-linux/lib -L/usr/local/cuda-12.6/targets/x86_64-linux/lib/stubs 
