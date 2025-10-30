/*
 * Copyright 2024 NXP
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Implementation of the backend for the NXP Neutron NPU.
 */

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>

#include "NeutronDriver.h"
#include "NeutronErrors.h"

using namespace std;

namespace torch {
namespace executor {
namespace neutron {

// All the memory need to be aligned with 16
#define BUFFER_ALIGNMENT 16
#define ALIGN_SIZE(size) \
  ((size + BUFFER_ALIGNMENT - 1) & (~(BUFFER_ALIGNMENT - 1)))

// clang-format off
/* Header schema:
     +----------------------------+-----------------------------+------------------------+
     | Neutron inputs length (1B) | Neutron outputs length (1B) | Input args length (1B) |
     +----------------------------+-----------+-----------------+------------------------+
     | 1st input tensor format (1B)           | [nth* input tensor format (1B)]          |
     +----------------------------------------+------------------------------------------+
     | 1st output tensor format (1B)          | [nth* output tensor format (1B)]         |
     +----------------------------------------+------------------------------------------+
     | 1st input map (1B)                     | [nth* input map (1B)]                    |
     +----------------------------------------+------------------------------------------+
     | 1st output map (1B)                    | [nth* output map (1B)]                   |
     +----------------------------------------+------------------------------------------+
*/
// clang-format on
#define ITEM_SIZE 1 // 1 Byte
#define INPUT_TENSOR_FORMAT_LEN_POS 0
#define OUTPUT_TENSOR_FORMAT_LEN_POS 1
#define INPUT_ARGS_LEN_POS 2
#define INPUT_TENSOR_FORMAT_ARRAY_ADDR(base) (base + 3 * ITEM_SIZE)
#define OUTPUT_TENSOR_FORMAT_ARRAY_ADDR(base) \
  (base + 3 * ITEM_SIZE + base[INPUT_TENSOR_FORMAT_LEN_POS])
#define INPUT_TENSOR_MAP_ARRAY_ADDR(base)                         \
  (base + 3 * ITEM_SIZE + 1 * base[INPUT_TENSOR_FORMAT_LEN_POS] + \
   1 * base[OUTPUT_TENSOR_FORMAT_LEN_POS])
#define OUTPUT_TENSOR_MAP_ARRAY_ADDR(base)                        \
  (base + 3 * ITEM_SIZE + 2 * base[INPUT_TENSOR_FORMAT_LEN_POS] + \
   1 * base[OUTPUT_TENSOR_FORMAT_LEN_POS])
#define PAYLOAD_ADDR(base)                                     \
  (base +                                                      \
   ALIGN_SIZE(                                                 \
       3 * ITEM_SIZE + 2 * base[INPUT_TENSOR_FORMAT_LEN_POS] + \
       2 * base[OUTPUT_TENSOR_FORMAT_LEN_POS]))

// Aggregate neutron model handle and data structures into one.
typedef struct {
  int numInputs = 0;
  int numOutputs = 0;
  int numInputArgs = 0;
  uint32_t scratchSize = 0;
  NeutronModelConfig mcfg;
  NeutronDataConfig dcfg;
  NeutronModelHandle nmh = NULL;
  const uint8_t* inputTranspositionFlags;
  const uint8_t* outputTranspositionFlags;
  const uint8_t* inputMap;
  const uint8_t* outputMap;
} NeutronConfig;

// Applied on outputs.
template <typename T>
void transposeToChannelFirst(
    const T* src,
    T* dest,
    size_t N,
    size_t C,
    size_t H,
    size_t W) {
  for (size_t n = 0; n < N; n++) {
    for (size_t c = 0; c < C; c++) {
      for (size_t h = 0; h < H; h++) {
        for (size_t w = 0; w < W; w++) {
          dest[n * C * H * W + c * H * W + h * W + w] =
              src[n * H * W * C + h * W * C + w * C + c];
        }
      }
    }
  }
}

// Applied on inputs.
template <typename T>
void transposeToChannelLast(
    const T* src,
    T* dest,
    size_t N,
    size_t C,
    size_t H,
    size_t W) {
  for (size_t n = 0; n < N; n++) {
    for (size_t c = 0; c < C; c++) {
      for (size_t h = 0; h < H; h++) {
        for (size_t w = 0; w < W; w++) {
          dest[n * H * W * C + h * W * C + w * C + c] =
              src[n * C * H * W + c * H * W + h * W + w];
        }
      }
    }
  }
}

// Transpose src buffer in channel first format into dest buffer in channel last
// format, sizes correspond to src dimensions in the Executorch defined tensor
// (which is NCHW), element_size is in Bytes.
void transposeInput(
    const void* src,
    void* dest,
    const ArrayRef<exec_aten::SizesType>& sizes,
    size_t element_size) {
  size_t length = sizes.size();
  if (length < 3) {
    return;
  }
  size_t N = 1;
  size_t C = sizes[length - 3];
  size_t H = sizes[length - 2];
  size_t W = sizes[length - 1];
  for (size_t i = 0; i < length - 3; i++) {
    N *= sizes[i];
  }
  switch (element_size) {
    case 1:
      return transposeToChannelLast<uint8_t>(
          static_cast<const uint8_t*>(src),
          static_cast<uint8_t*>(dest),
          N,
          C,
          H,
          W);
    case 2:
      return transposeToChannelLast<uint16_t>(
          static_cast<const uint16_t*>(src),
          static_cast<uint16_t*>(dest),
          N,
          C,
          H,
          W);
    case 4:
      return transposeToChannelLast<uint32_t>(
          static_cast<const uint32_t*>(src),
          static_cast<uint32_t*>(dest),
          N,
          C,
          H,
          W);
    case 8:
      return transposeToChannelLast<uint64_t>(
          static_cast<const uint64_t*>(src),
          static_cast<uint64_t*>(dest),
          N,
          C,
          H,
          W);
  }
}

// Transpose src buffer in channel last format into dest buffer in channel first
// format, sizes correspond to dest dimensions in the Executorch defined tensor
// (which is NCHW), element_size is in Bytes.
void transposeOutput(
    const void* src,
    void* dest,
    const ArrayRef<exec_aten::SizesType>& sizes,
    size_t element_size) {
  size_t length = sizes.size();
  if (length < 3) {
    return;
  }
  size_t N = 1;
  size_t C = sizes[length - 3];
  size_t H = sizes[length - 2];
  size_t W = sizes[length - 1];
  for (size_t i = 0; i < length - 3; i++) {
    N *= sizes[i];
  }
  switch (element_size) {
    case 1:
      return transposeToChannelFirst<uint8_t>(
          static_cast<const uint8_t*>(src),
          static_cast<uint8_t*>(dest),
          N,
          C,
          H,
          W);
    case 2:
      return transposeToChannelFirst<uint16_t>(
          static_cast<const uint16_t*>(src),
          static_cast<uint16_t*>(dest),
          N,
          C,
          H,
          W);
    case 4:
      return transposeToChannelFirst<uint32_t>(
          static_cast<const uint32_t*>(src),
          static_cast<uint32_t*>(dest),
          N,
          C,
          H,
          W);
    case 8:
      return transposeToChannelFirst<uint64_t>(
          static_cast<const uint64_t*>(src),
          static_cast<uint64_t*>(dest),
          N,
          C,
          H,
          W);
  }
}

bool multipleChannelsPresent(const ArrayRef<exec_aten::SizesType>& sizes) {
  size_t length = sizes.size();
  if (length < 3) {
    return true;
  }
  size_t C = sizes[length - 3];
  return C != 1;
}

class NeutronBackend final : public PyTorchBackendInterface {
 public:
  NeutronBackend() {}

  ~NeutronBackend() = default;

  virtual bool is_available() const override {
    return true;
  }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const override {
    MemoryAllocator* allocator = context.get_runtime_allocator();

    auto* cfg = allocator->allocateInstance<NeutronConfig>();

    // The following data is read from the "processed" data blob.
    //    cfg->numInputs
    //    cfg->numoutputs
    //    cfg->mcfg.microcode
    //    cfg->mcfg.weights
    //    cfg->mcfg.kernels
    const uint8_t* payloadFlags =
        static_cast<const uint8_t*>(processed->data());
    uint32_t numInputs = payloadFlags[INPUT_TENSOR_FORMAT_LEN_POS];
    uint32_t numOutputs = payloadFlags[OUTPUT_TENSOR_FORMAT_LEN_POS];
    cfg->numInputArgs = payloadFlags[INPUT_ARGS_LEN_POS];
    cfg->inputTranspositionFlags = INPUT_TENSOR_FORMAT_ARRAY_ADDR(payloadFlags);
    cfg->outputTranspositionFlags =
        OUTPUT_TENSOR_FORMAT_ARRAY_ADDR(payloadFlags);
    cfg->inputMap = INPUT_TENSOR_MAP_ARRAY_ADDR(payloadFlags);
    cfg->outputMap = OUTPUT_TENSOR_MAP_ARRAY_ADDR(payloadFlags);

    const uint32_t* buffer = static_cast<const uint32_t*>(
        static_cast<const void*> PAYLOAD_ADDR(payloadFlags));
    uint32_t magicWord = buffer[0];
    // Check valid microcode.
    if (magicWord != 0x64434D6E) {
      ET_LOG(
          Error,
          "Preprocessed buffer does not contain a valid Neutron microcode");
      return Error::InvalidProgram;
    }
    uint32_t microcodeSize = buffer[6];
    uint32_t weightsSize = buffer[7];
    cfg->scratchSize = buffer[9];
    cfg->numInputs = buffer[11];
    cfg->numOutputs = buffer[12];
    if (cfg->numInputs != numInputs) {
      ET_LOG(
          Error,
          "Preprocessed buffer does not contain a valid number of inputs");
      return Error::InvalidProgram;
    }
    if (cfg->numOutputs != numOutputs) {
      ET_LOG(
          Error,
          "Preprocessed buffer does not contain a valid number of outputs");
      return Error::InvalidProgram;
    }
    cfg->mcfg.microcode =
        static_cast<const uint8_t*>(static_cast<const void*>(buffer));
    cfg->mcfg.weights = static_cast<const uint8_t*>(cfg->mcfg.microcode) +
        ALIGN_SIZE(microcodeSize);
    cfg->mcfg.kernels = static_cast<const uint8_t*>(cfg->mcfg.weights) +
        ALIGN_SIZE(weightsSize);

#if (NO_HEAP_USAGE == 0)
    // The driver allocates and deallocates place for NeutronModelHandle.
    cfg->nmh = NULL;
#else
    // Allocate place for NeutronModelHandle.
    cfg->nmh = static_cast<NeutronModelHandle>(
        allocator->allocate(neutronGetModelContextSize()));
#endif

    // Prepare data for through neutron driver.
    NeutronError neutronRC =
        neutronModelPrepare((const NeutronModelConfig*)&cfg->mcfg, &cfg->nmh);
    if (neutronRC != ENONE) {
      ET_LOG(
          Error,
          "Neutron model preparation failed with error code %ld",
          neutronRC);
      return Error::InvalidProgram;
    }

    return cfg;
  }

  Error execute(
      BackendExecutionContext& context,
      DelegateHandle* input_handle,
      Span<EValue*> args) const override {
    NeutronConfig* cfg = static_cast<NeutronConfig*>(input_handle);

    // Allocate place for input and output pointers.
    cfg->dcfg.inputs = static_cast<const void**>(
        context.allocate(cfg->numInputs * sizeof(void*)));
    cfg->dcfg.outputs =
        static_cast<void**>(context.allocate(cfg->numOutputs * sizeof(void*)));
    cfg->dcfg.outputs[cfg->numOutputs] =
        static_cast<void*>(context.allocate(cfg->scratchSize, 16));

    // Set inputs from args.
    // Transpose inputs if needed.
    for (int i = 0; i < cfg->numInputs; i++) {
      auto arg = args[cfg->inputMap[i]]->toTensor();
      if (cfg->inputTranspositionFlags[i] &&
          multipleChannelsPresent(arg.sizes())) {
        if (arg.sizes().size() < 3) {
          ET_LOG(Error, "Unable to transpose 1D and 2D input to channel last");
          return Error::InvalidProgram;
        }
        // Allocate buffer, the allocator is reset after each PTE instruction.
        void* buffer = context.allocate(arg.nbytes());
        transposeInput(
            arg.const_data_ptr(), buffer, arg.sizes(), arg.element_size());
        cfg->dcfg.inputs[i] = buffer;
      } else {
        cfg->dcfg.inputs[i] = arg.const_data_ptr();
      }
    }

    // Set outputs from args.
    // Redirect outputs if needed before transposition.
    for (int i = 0; i < cfg->numOutputs; i++) {
      auto arg = args[cfg->numInputArgs + cfg->outputMap[i]]->toTensor();
      if (cfg->outputTranspositionFlags[i] &&
          multipleChannelsPresent(arg.sizes())) {
        // Allocate buffer, the allocator is reset after each PTE instruction.
        void* buffer = context.allocate(arg.nbytes());
        cfg->dcfg.outputs[i] = buffer;
      } else {
        cfg->dcfg.outputs[i] = arg.mutable_data_ptr();
      }
    }

#ifdef NEUTRON_PROFILE
    // TODO: Use trace from BackendExecutionContext.
    NeutronTraceConfig trace_config{.traceConfig = 0};
    neutronSetTrace(cfg->nmh, &trace_config);
#endif

    // Run neutron compute.
    NeutronError neutronRC = neutronRunBlocking(cfg->nmh, &cfg->dcfg);
    if (neutronRC != ENONE) {
      ET_LOG(
          Error,
          "Neutron model evaluation failed with error code %ld",
          neutronRC);
      return Error::InvalidProgram;
    }

    // Transpose outputs.
    for (int i = 0; i < cfg->numOutputs; i++) {
      auto arg = args[cfg->numInputArgs + cfg->outputMap[i]]->toTensor();
      if (cfg->outputTranspositionFlags[i] &&
          multipleChannelsPresent(arg.sizes())) {
        if (arg.sizes().size() < 3) {
          ET_LOG(
              Error, "Unable to transpose 1D and 2D output to channel first");
          return Error::InvalidProgram;
        }
        transposeOutput(
            cfg->dcfg.outputs[i],
            arg.mutable_data_ptr(),
            arg.sizes(),
            arg.element_size());
      }
    }

    return Error::Ok;
  }

  void destroy(DelegateHandle* handle) const override {
    NeutronConfig* cfg = reinterpret_cast<NeutronConfig*>(handle);

    // Unprepare to free resources in neutron driver.
    NeutronError neutronRC = neutronModelUnprepare(cfg->nmh);
    (void)neutronRC;

    // Deallocation is done automatically.
    /*
    delete[] cfg->dcfg.inputs;
    delete[] cfg->dcfg.outputs;
    delete cfg;
    */
    return;
  }
};

namespace {
auto backend = NeutronBackend();
Backend backend_id{"NeutronBackend", &backend};
static auto registered = register_backend(backend_id);
} // namespace

} // namespace neutron
} // namespace executor
} // namespace torch
