
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
    CUfunction triton_poi_fused_convolution_0{nullptr};
    CUfunction triton_poi_fused_convolution_1{nullptr};
    CUfunction triton_poi_fused_convolution_2{nullptr};
};
}  // namespace



AOTInductorModel::AOTInductorModel(std::shared_ptr<ConstantMap> constants_map,
                                   std::shared_ptr<std::vector<ConstantHandle>> constants_array,
                                   const std::string& device_str,
                                   std::optional<std::string> cubin_dir)
    : AOTInductorModelBase(1,
                           1,
                           1,
                           device_str,
                           std::move(cubin_dir),
                           true) {
    inputs_info_[0].name = "arg2_1";
    constants_info_[0].name = "conv_weight";
    constants_info_[0].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[0].offset = 0;
    constants_info_[0].data_size = 540;
    constants_info_[0].from_folded = false;
    constants_info_[0].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[0].shape = {5, 3, 3, 3};
    constants_info_[0].stride = {27, 9, 3, 1};
    constants_info_[0].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[0].original_fqn = "conv.weight";
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
        size_hints={'y': 16, 'x': 64}, tile_hint=TileHint.SQUARE,
        filename=__file__,
        triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4F87BAC7C78026030CE21ABCD241F4211145E4ACCDC53C53E0CC97717CB6F329', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'tiling_scores': {'y': 6144, 'x': 3072}},
        min_elem_per_thread=0
    )
    @triton.jit
    def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
        ynumel = 12
        xnumel = 64
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
        tmp0 = tl.load(in_ptr0 + (x2 + 64*y3), xmask & ymask, eviction_policy='evict_last')
        tl.store(out_ptr0 + (y0 + 3*x2 + 192*y1), tmp0, xmask & ymask)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = ((xnumel + (64 - 1)) / (64));
    uint32_t grid_1 = ((ynumel + (16 - 1)) / (16));
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused_convolution_0 == nullptr) {
        kernels_.triton_poi_fused_convolution_0 = loadKernel("/home/gasoonjia/executorch/cuj3mxjkcttcfshkrqr3bbv27ng2dlykmtde7rpiylednxszoer5.cubin", "triton_poi_fused_convolution_0", 4352, cubin_dir_); 
    }
    CUdeviceptr var_0 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_1 = reinterpret_cast<CUdeviceptr>(out_ptr0.data_ptr());
    int var_2 = ynumel;
    int var_3 = xnumel;
    CUdeviceptr global_scratch_4 = 0;
    void* kernel_args_[] = {&var_0, &var_1, &var_2, &var_3, &global_scratch_4};
    launchKernel(kernels_.triton_poi_fused_convolution_0, grid_0, grid_1, grid_2, 4, 4352, kernel_args_, stream_);
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
        size_hints={'y': 16, 'x': 16}, tile_hint=TileHint.SQUARE,
        filename=__file__,
        triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4F87BAC7C78026030CE21ABCD241F4211145E4ACCDC53C53E0CC97717CB6F329', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'tiling_scores': {'y': 1080, 'x': 540}},
        min_elem_per_thread=0
    )
    @triton.jit
    def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
        ynumel = 15
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
    uint32_t grid_1 = ((ynumel + (16 - 1)) / (16));
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused_convolution_1 == nullptr) {
        kernels_.triton_poi_fused_convolution_1 = loadKernel("/home/gasoonjia/executorch/cg7g6znwyjx7worxb7hbjz5rypindv6rgyiqidang4zm47hs6h7u.cubin", "triton_poi_fused_convolution_1", 1088, cubin_dir_); 
    }
    CUdeviceptr var_5 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_6 = reinterpret_cast<CUdeviceptr>(out_ptr0.data_ptr());
    int var_7 = ynumel;
    int var_8 = xnumel;
    CUdeviceptr global_scratch_9 = 0;
    void* kernel_args_[] = {&var_5, &var_6, &var_7, &var_8, &global_scratch_9};
    launchKernel(kernels_.triton_poi_fused_convolution_1, grid_0, grid_1, grid_2, 4, 1088, kernel_args_, stream_);
}

template <typename in_ptr0_type_, typename out_ptr0_type_, typename kernels_type_>
static inline void call_triton_poi_fused_convolution_2(
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
    async_compile.triton('triton_poi_fused_convolution_2', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.pointwise(
        size_hints={'y': 32, 'x': 64}, tile_hint=TileHint.SQUARE,
        filename=__file__,
        triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4F87BAC7C78026030CE21ABCD241F4211145E4ACCDC53C53E0CC97717CB6F329', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'tiling_scores': {'y': 5120, 'x': 10240}},
        min_elem_per_thread=0
    )
    @triton.jit
    def triton_poi_fused_convolution_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
        ynumel = 20
        xnumel = 64
        yoffset = tl.program_id(1) * YBLOCK
        yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
        ymask = yindex < ynumel
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
        xmask = xindex < xnumel
        x2 = xindex
        y0 = (yindex % 5)
        y1 = yindex // 5
        y3 = yindex
        tmp0 = tl.load(in_ptr0 + (y0 + 5*x2 + 320*y1), xmask & ymask, eviction_policy='evict_last')
        tmp1 = y0
        tmp2 = tl.full([1, 1], 2, tl.int64)
        tmp3 = tmp1 < tmp2
        tmp4 = tl.full([1, 1], 1, tl.int64)
        tmp5 = tmp1 < tmp4
        tmp6 = 0.1508762389421463
        tmp7 = -0.15852206945419312
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp9 = tl.full([1, 1], 3, tl.int64)
        tmp10 = tmp1 < tmp9
        tmp11 = tl.full([1, 1], 4, tl.int64)
        tmp12 = tmp1 < tmp11
        tmp13 = -0.047068577259778976
        tmp14 = 0.010523972101509571
        tmp15 = tl.where(tmp12, tmp13, tmp14)
        tmp16 = 0.07869197428226471
        tmp17 = tl.where(tmp10, tmp16, tmp15)
        tmp18 = tl.where(tmp3, tmp8, tmp17)
        tmp19 = tmp0 + tmp18
        tl.store(out_ptr0 + (x2 + 64*y3), tmp19, xmask & ymask)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = ((xnumel + (32 - 1)) / (32));
    uint32_t grid_1 = ((ynumel + (32 - 1)) / (32));
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused_convolution_2 == nullptr) {
        kernels_.triton_poi_fused_convolution_2 = loadKernel("/home/gasoonjia/executorch/ckh2jw4qzbo6bg3d3ft7jfqzeusq2y2hz662iuqm5tpxbodupud4.cubin", "triton_poi_fused_convolution_2", 4608, cubin_dir_); 
    }
    CUdeviceptr var_10 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_11 = reinterpret_cast<CUdeviceptr>(out_ptr0.data_ptr());
    int var_12 = ynumel;
    int var_13 = xnumel;
    CUdeviceptr global_scratch_14 = 0;
    void* kernel_args_[] = {&var_10, &var_11, &var_12, &var_13, &global_scratch_14};
    launchKernel(kernels_.triton_poi_fused_convolution_2, grid_0, grid_1, grid_2, 4, 4608, kernel_args_, stream_);
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
    ConstantHandle arg2_1 = ConstantHandle(input_handles[0]);
    int32_t arg2_1_dtype;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(arg2_1, &arg2_1_dtype));

    int32_t arg2_1_expected_dtype = aoti_torch_dtype_float32();
    if (arg2_1_expected_dtype != arg2_1_dtype) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched dtype, "
           << "expected: " << arg2_1_expected_dtype << "(at::kFloat), "
           << "but got: " << arg2_1_dtype << "\n";
        throw std::runtime_error(ss.str());
    }
    auto arg2_1_size = arg2_1.sizes();

    if (4 != arg2_1_size[0]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched dim value at 0, "
           << "expected: 4, " << "but got: " << arg2_1_size[0]
           << "\n";
        throw std::runtime_error(ss.str());
    }

    if (3 != arg2_1_size[1]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched dim value at 1, "
           << "expected: 3, " << "but got: " << arg2_1_size[1]
           << "\n";
        throw std::runtime_error(ss.str());
    }

    if (8 != arg2_1_size[2]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched dim value at 2, "
           << "expected: 8, " << "but got: " << arg2_1_size[2]
           << "\n";
        throw std::runtime_error(ss.str());
    }

    if (8 != arg2_1_size[3]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched dim value at 3, "
           << "expected: 8, " << "but got: " << arg2_1_size[3]
           << "\n";
        throw std::runtime_error(ss.str());
    }
    auto arg2_1_stride = arg2_1.strides();

    if (192 != arg2_1_stride[0]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched stride value at 0, "
           << "expected: 192, " << "but got: " << arg2_1_stride[0]
           << "\n";
        throw std::runtime_error(ss.str());
    }

    if (64 != arg2_1_stride[1]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched stride value at 1, "
           << "expected: 64, " << "but got: " << arg2_1_stride[1]
           << "\n";
        throw std::runtime_error(ss.str());
    }

    if (8 != arg2_1_stride[2]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched stride value at 2, "
           << "expected: 8, " << "but got: " << arg2_1_stride[2]
           << "\n";
        throw std::runtime_error(ss.str());
    }

    if (1 != arg2_1_stride[3]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched stride value at 3, "
           << "expected: 1, " << "but got: " << arg2_1_stride[3]
           << "\n";
        throw std::runtime_error(ss.str());
    }
    int32_t arg2_1_device_type;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type(arg2_1, &arg2_1_device_type));

    int32_t arg2_1_expected_device_type = 1;
    if (arg2_1_expected_device_type != arg2_1_device_type) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched device type, "
        << "expected: " << arg2_1_expected_device_type << "1(cuda), "
        << "but got: " << arg2_1_device_type << "\n";
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
    auto arg2_1 = std::move(inputs[0]);
    [[maybe_unused]] auto& conv_weight = constants_->at(0);

    if ((long(arg2_1.data_ptr()) & (16 -1)) != 0) {
        AOTI_TORCH_WARN("Input 0 was compiled as 16-bytes aligned, but it is not aligned at run time. Copying to an aligned tensor to guarantee correctness, but expect a performance hit.");
        AtenTensorHandle arg2_1_aligned;
        aoti_torch_clone_preserve_strides(arg2_1, &arg2_1_aligned);
        arg2_1 = std::move(RAIIAtenTensorHandle(arg2_1_aligned));
    }
    inputs.clear();
    [[maybe_unused]] auto& kernels = static_cast<AOTInductorModelKernels&>(*this->kernels_.get());

    AOTICudaStreamGuard stream_guard(stream, this->device_idx_);
    static constexpr int64_t int_array_0[] = {4L, 3L, 8L, 8L};
    static constexpr int64_t int_array_1[] = {192L, 1L, 24L, 3L};
    AtenTensorHandle buf0_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(4, int_array_0, int_array_1, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf0_handle));
    RAIIAtenTensorHandle buf0(buf0_handle);
    // Topologically Sorted Source Nodes: [aten_convolution_default], Original ATen: [aten.convolution]
    call_triton_poi_fused_convolution_0(arg2_1, buf0, 12L, 64L, this->device_idx_, stream, kernels, this->cubin_dir_);
    arg2_1.reset();
    static constexpr int64_t int_array_2[] = {5L, 3L, 3L, 3L};
    static constexpr int64_t int_array_3[] = {27L, 1L, 9L, 3L};
    AtenTensorHandle buf1_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(4, int_array_2, int_array_3, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf1_handle));
    RAIIAtenTensorHandle buf1(buf1_handle);
    // Topologically Sorted Source Nodes: [aten_convolution_default], Original ATen: [aten.convolution]
    call_triton_poi_fused_convolution_1(conv_weight, buf1, 15L, 9L, this->device_idx_, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [aten_convolution_default], Original ATen: [aten.convolution]
    AtenTensorHandle buf2_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_convolution(buf0, buf1, nullptr, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, std::array<int64_t, 2>{1L, 1L}.cbegin(), 2, 0, std::array<int64_t, 2>{0L, 0L}.cbegin(), 2, 1L, &buf2_handle));
    RAIIAtenTensorHandle buf2(buf2_handle);
    buf0.reset();
    buf1.reset();
    static constexpr int64_t int_array_4[] = {4L, 5L, 8L, 8L};
    static constexpr int64_t int_array_5[] = {320L, 64L, 8L, 1L};
    AtenTensorHandle buf3_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(4, int_array_4, int_array_5, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf3_handle));
    RAIIAtenTensorHandle buf3(buf3_handle);
    // Topologically Sorted Source Nodes: [aten_convolution_default], Original ATen: [aten.convolution]
    call_triton_poi_fused_convolution_2(buf2, buf3, 20L, 64L, this->device_idx_, stream, kernels, this->cubin_dir_);
    buf2.reset();
    output_handles[0] = buf3.release();
} // AOTInductorModel::run_impl
} // namespace torch::aot_inductor




// Compile cmd
// g++ /home/gasoonjia/executorch/cwbxrbt67v2w5ftivnaod7qulkhz5pxljkb2gr6xgxxo6t35umkq.wrapper.cpp -D TORCH_INDUCTOR_CPP_WRAPPER -D STANDALONE_TORCH_HEADER -D  C10_USING_CUSTOM_GENERATED_MACROS -D CPU_CAPABILITY_AVX512 -D  USE_CUDA  -fPIC -O1 -DNDEBUG -fno-trapping-math -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros -fno-math-errno -fexcess-precision=fast -fno-finite-math-only -fno-unsafe-math-optimizations -ffp-contract=off -fno-tree-loop-vectorize -march=native -Wall -std=c++17 -Wno-unused-variable -Wno-unknown-pragmas -pedantic -fopenmp  -include /tmp/torchinductor_gasoonjia/precompiled_headers/c4cub4usfsuwqkbp3pfgzit6fkb6qpm3anlkt22y6d2ks3tdluhg.h -I/home/gasoonjia/.conda/envs/aoti/include/python3.10 -I/home/gasoonjia/.conda/envs/aoti/lib/python3.10/site-packages/torch/include -I/home/gasoonjia/.conda/envs/aoti/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/usr/local/cuda-12.6/include   -mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma  -c -o /home/gasoonjia/executorch/cwbxrbt67v2w5ftivnaod7qulkhz5pxljkb2gr6xgxxo6t35umkq.wrapper.o
// Link cmd
// g++ /home/gasoonjia/executorch/cwbxrbt67v2w5ftivnaod7qulkhz5pxljkb2gr6xgxxo6t35umkq.wrapper.o /home/gasoonjia/executorch/c26meop4u3hf2hh76dw6zl4fepetv42wg64xygsadkkb43zczod6.kernel.o /home/gasoonjia/executorch/cwbxrbt67v2w5ftivnaod7qulkhz5pxljkb2gr6xgxxo6t35umkq/c5rhpvrttznyqa5pe725yxk3av45bswzgxcmk7tdg4j7yptcotin.o -D TORCH_INDUCTOR_CPP_WRAPPER -D STANDALONE_TORCH_HEADER -D  C10_USING_CUSTOM_GENERATED_MACROS -D CPU_CAPABILITY_AVX512 -D  USE_CUDA  -shared -fPIC -O3 -DNDEBUG -fno-trapping-math -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros -fno-math-errno -fexcess-precision=fast -fno-finite-math-only -fno-unsafe-math-optimizations -ffp-contract=off -fno-tree-loop-vectorize -march=native -Wall -std=c++17 -Wno-unused-variable -Wno-unknown-pragmas -pedantic -fopenmp  -I/home/gasoonjia/.conda/envs/aoti/include/python3.10 -I/home/gasoonjia/.conda/envs/aoti/lib/python3.10/site-packages/torch/include -I/home/gasoonjia/.conda/envs/aoti/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/usr/local/cuda-12.6/include   -mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma  -o /home/gasoonjia/executorch/aoti.so  -ltorch -ltorch_cpu -lgomp -lc10 -lc10_cuda -lcuda -ltorch_cuda  -L/home/gasoonjia/.conda/envs/aoti/lib -L/home/gasoonjia/.conda/envs/aoti/lib/python3.10/site-packages/torch/lib -L/usr/local/cuda-12.6/targets/x86_64-linux/lib -L/usr/local/cuda-12.6/targets/x86_64-linux/lib/stubs 
