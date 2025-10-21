/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/xnnpack/runtime/XNNCompiler.h>
#include <executorch/backends/xnnpack/runtime/XNNHeader.h>
#include <executorch/backends/xnnpack/serialization/schema_generated.h>
#include <executorch/extension/threadpool/threadpool.h>
#include <executorch/runtime/executor/pte_data_map.h>
#include <xnnpack.h>
#include <string>
#include <unordered_map>
#include <vector>

#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wglobal-constructors"

namespace executorch {
namespace backends {
namespace xnnpack {
namespace delegate {

using executorch::ET_RUNTIME_NAMESPACE::NamedDataMap;
using executorch::runtime::Error;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::Result;

/*
 * Provide compile-time allocation.
 */
class CompileAllocator {
 public:
  /*
   * Allocate memory which will be automatically freed at the end
   * of the compilation process.
   */
  void* allocateTemporary(size_t size) {
    auto mem = new uint8_t[size];
    temporaries_.emplace_back(mem);
    return mem;
  }

 private:
  std::vector<std::unique_ptr<uint8_t[]>> temporaries_;
};

// Flatbuffer types
using ValuePtr = const fb_xnnpack::XValue*;
using NodePtr = const fb_xnnpack::XNode*;
using GraphPtr = const fb_xnnpack::XNNGraph*;
using ConstantDataOffsetPtr = const fb_xnnpack::ConstantDataOffset*;
using DataType = fb_xnnpack::XNNDatatype;

// Type for define node function. This is the function signature
// for any function that takes in a flatbuffer node and defines it
// into our xnn_subgraph
using DefineNodeFunc = Error (*)(
    xnn_subgraph_t,
    const std::unordered_map<uint32_t, uint32_t>&,
    NodePtr,
    const fb_xnnpack::XNNGraph*) noexcept;

/*
Convert a tensor from fp32 to bf16.
*/
void convertF32TensorToBF16(
    const float* f32_data,
    uint16_t* bf16_data_out,
    size_t numel) {
  for (auto i = 0u; i < numel; i++) {
    // Adjust the f32 value such that it rounds properly after truncation.
    // Constant factor scales 1+2^-8 to 1+2e-7.
    float f32_adjusted = f32_data[i] * 1.00389105f;
    uint32_t f32_bits;
    memcpy(&f32_bits, &f32_adjusted, sizeof(float));
    bf16_data_out[i] = static_cast<uint16_t>(f32_bits >> 16);
  }
}

/*
Gets the output min and output max for a given node operator
*/
std::pair<float, float> getOutputMinMax(const NodePtr node) noexcept {
  float output_min = -std::numeric_limits<float>::infinity();
  float output_max = std::numeric_limits<float>::infinity();
  auto output_min_max = node->output_min_max();
  if (output_min_max != nullptr) {
    output_min = output_min_max->output_min();
    output_max = output_min_max->output_max();
  }

  return {output_min, output_max};
}

/*
Converts flatbuffer xnn data type to xnnpack data type
*/
xnn_datatype getDataType(const DataType& data_type) {
  switch (data_type) {
    case DataType::xnn_datatype_fp32:
      return xnn_datatype::xnn_datatype_fp32;
    case DataType::xnn_datatype_fp16:
      return xnn_datatype::xnn_datatype_fp16;
    case DataType::xnn_datatype_qint8:
      return xnn_datatype::xnn_datatype_qint8;
    case DataType::xnn_datatype_quint8:
      return xnn_datatype::xnn_datatype_quint8;
    case DataType::xnn_datatype_qint32:
      return xnn_datatype::xnn_datatype_qint32;
    case DataType::xnn_datatype_qcint8:
      return xnn_datatype::xnn_datatype_qcint8;
    case DataType::xnn_datatype_qcint32:
      return xnn_datatype::xnn_datatype_qcint32;
    case DataType::xnn_datatype_qcint4:
      return xnn_datatype::xnn_datatype_qcint4;
    case DataType::xnn_datatype_qdint8:
      return xnn_datatype::xnn_datatype_qdint8;
    case DataType::xnn_datatype_qbint4:
      return xnn_datatype::xnn_datatype_qbint4;
    case DataType::xnn_datatype_qpint8:
      return xnn_datatype::xnn_datatype_qpint8;
    case DataType::xnn_datatype_int32:
      return xnn_datatype::xnn_datatype_int32;
    case DataType::xnn_datatype_pfp32:
      return xnn_datatype::xnn_datatype_pfp32;
    case DataType::xnn_datatype_bf16:
      return xnn_datatype::xnn_datatype_bf16;
    default:
      return xnn_datatype::xnn_datatype_invalid;
  }
}

bool isQuantizedDataType(const xnn_datatype data_type) {
  switch (data_type) {
    case xnn_datatype::xnn_datatype_qint8:
    case xnn_datatype::xnn_datatype_quint8:
    case xnn_datatype::xnn_datatype_qint32:
    case xnn_datatype::xnn_datatype_qcint8:
    case xnn_datatype::xnn_datatype_qcint32:
    case xnn_datatype::xnn_datatype_qcint4:
    case xnn_datatype::xnn_datatype_qdint8:
      return true;
    default:
      return false;
  }
}

/**
Converts dims from uint32 to size_t. Takes in a flatbuffer vector
of uint32_t and returns a std::vector of size_t. XNNPACK takes in
dims of size_t* but tensor shape is serialized in flatbuffer as
int32_t. As a result, we need to static cast the shapes to size_t
*/
template <typename T = size_t>
std::vector<T> flatbufferDimsToVector(
    const flatbuffers::Vector<uint32_t>* fb_dims) {
  std::vector<T> dims_data;
  dims_data.reserve(fb_dims->size());
  for (auto fb_dim : *fb_dims) {
    dims_data.push_back(static_cast<T>(fb_dim));
  }
  return dims_data;
}

/**
Gets the constant data pointer associated with the given tensor value.
Obtaining the constant data pointer can either be from within the flatbuffer
payload (deprecated) or via offsets to the constant_data_ptr. If no constant
data associated with the tensor value, then returns nullptr.
*/
const uint8_t* getConstantDataPtr(
    uint32_t buffer_idx,
    GraphPtr flatbuffer_graph,
    const uint8_t* constant_data_ptr,
    const NamedDataMap* named_data_map,
    std::vector<FreeableBuffer>& freeable_buffers,
    XNNWeightsCache* weights_cache) {
  if (buffer_idx) {
    if (!constant_data_ptr) {
      // TODO(T172265611): Remove constant_buffer in flatbuffer path after BC
      // window
      const auto& constant_buffer = *flatbuffer_graph->constant_buffer();
      return constant_buffer[buffer_idx]->storage()->data();
    } else {
      ConstantDataOffsetPtr constant_data_offset =
          flatbuffer_graph->constant_data()->Get(buffer_idx);
      uint64_t offset = constant_data_offset->offset();

      bool has_named_key = flatbuffers::IsFieldPresent(
          constant_data_offset, fb_xnnpack::ConstantDataOffset::VT_NAMED_KEY);
      // If there is no tensor name
      if (!has_named_key) {
        return constant_data_ptr + offset;
      } else {
        const std::string& data_name = constant_data_offset->named_key()->str();
#ifdef ENABLE_XNNPACK_WEIGHTS_CACHE
        Result<const uint8_t*> data_ptr =
            weights_cache->load_unpacked_data(data_name);
        if (!data_ptr.ok()) {
          ET_LOG(Error, "Failed to load weights from cache");
          return nullptr;
        }
        return data_ptr.get();
#else
        Result<FreeableBuffer> buffer =
            named_data_map->get_data(data_name.c_str());
        if (!buffer.ok()) {
          ET_LOG(
              Error,
              "Failed to get constant data for key %s from named_data_map. Error code: %u",
              data_name.c_str(),
              static_cast<uint32_t>(buffer.error()));
          return nullptr;
        }
        const uint8_t* data_ptr =
            static_cast<const uint8_t*>(buffer.get().data());
        freeable_buffers.push_back(std::move(buffer.get()));
        return data_ptr;
#endif
      }
    }
  }

  return nullptr;
}

const uint8_t* getConstantDataPtr(
    const fb_xnnpack::XNNTensorValue* tensor_value,
    GraphPtr flatbuffer_graph,
    const uint8_t* constant_data_ptr,
    const NamedDataMap* named_data_map,
    std::vector<FreeableBuffer>& freeable_buffers,
    XNNWeightsCache* weights_cache) {
  return getConstantDataPtr(
      tensor_value->constant_buffer_idx(),
      flatbuffer_graph,
      constant_data_ptr,
      named_data_map,
      freeable_buffers,
      weights_cache);
}

/**
Define serialized tensor value into
the subgraph. While also keeping track of the remapped ids from
the serialized id to the newly generated id.
*/
Error defineTensor(
    xnn_subgraph_t subgraph_ptr,
    std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    ValuePtr value,
    GraphPtr flatbuffer_graph,
    const uint8_t* constant_data_ptr,
    std::vector<uint32_t>& input_ids,
    std::vector<uint32_t>& output_ids,
    CompileAllocator& allocator,
    const NamedDataMap* named_data_map,
    std::vector<FreeableBuffer>& freeable_buffers,
    XNNWeightsCache* weights_cache) {
  const fb_xnnpack::XNNTensorValue* tensor_value = nullptr;
  const fb_xnnpack::XNNQuantizedTensorValue* qtensor_value = nullptr;

  switch (value->xvalue_union_type()) {
    case fb_xnnpack::XValueUnion::XNNTensorValue: {
      tensor_value = value->xvalue_union_as_XNNTensorValue();
      break;
    }
    case fb_xnnpack::XValueUnion::XNNQuantizedTensorValue: {
      qtensor_value = value->xvalue_union_as_XNNQuantizedTensorValue();
      tensor_value = qtensor_value->tensor_value();
      break;
    }
    default: {
      ET_CHECK_OR_RETURN_ERROR(
          false,
          NotImplemented,
          "Unhandled value type: %s",
          fb_xnnpack::EnumNameXValueUnion(value->xvalue_union_type()));
    }
  }

  ET_CHECK_OR_RETURN_ERROR(
      tensor_value != nullptr,
      Internal,
      "Deserialized Tensor is Null, this should never happen");

  // Get tensor dims, here we need to use a vector in order
  // to properly convert the uint32_t* to size_t*
  std::vector<size_t> dims_data = flatbufferDimsToVector(tensor_value->dims());

  // XNNPACK Id
  uint32_t id = XNN_INVALID_VALUE_ID;

  // Get Pointer to constant data from flatbuffer, if its non-constant
  // it is a nullptr
  const uint8_t* buffer_ptr = getConstantDataPtr(
      tensor_value,
      flatbuffer_graph,
      constant_data_ptr,
      named_data_map,
      freeable_buffers,
      weights_cache);

  xnn_status status;
  // The type we might have to convert to
  auto dq_datatype = getDataType(tensor_value->dq_datatype());

  if (dq_datatype != xnn_datatype::xnn_datatype_invalid) {
    if (dq_datatype != xnn_datatype::xnn_datatype_qint8) {
      ET_CHECK_OR_RETURN_ERROR(
          false,
          Internal,
          "Only int8_t is supported for dq_datatype for now, got: %d",
          dq_datatype);
    } else {
      ET_CHECK_OR_RETURN_ERROR(
          (tensor_value->flags() & XNN_VALUE_FLAG_EXTERNAL_INPUT),
          Internal,
          "Dynamic quantization of tensor is only allowed for the external input tensor value for now! got flags: %u",
          tensor_value->flags());
    }
  }

  if (qtensor_value == nullptr) {
    // FP32 tensor
    if (!isQuantizedDataType(dq_datatype)) {
      // Define non-quantied tensor
      status = xnn_define_tensor_value(
          /*subgraph=*/subgraph_ptr,
          /*datatype=*/getDataType(tensor_value->datatype()),
          /*num_dims=*/tensor_value->num_dims(),
          /*dims=*/dims_data.data(),
          /*data=*/buffer_ptr,
          /*external_id=*/tensor_value->external_id(),
          /*flags=*/tensor_value->flags(),
          /*id_out=*/&id);
    } else if (dq_datatype != xnn_datatype::xnn_datatype_invalid) {
      ET_CHECK_OR_RETURN_ERROR(
          isQuantizedDataType(dq_datatype),
          Internal,
          "Dynamic quantization can only produce supported quantized dtypes");
      ET_CHECK_OR_RETURN_ERROR(
          tensor_value->external_id() != XNN_INVALID_VALUE_ID,
          Internal,
          "Dynamic quantization can only work with external inputs for now, got an internal ID");
      ET_CHECK_OR_RETURN_ERROR(
          buffer_ptr == nullptr,
          Internal,
          "Dynamic quantization can only work with external inputs for now, got const data");

      switch (dq_datatype) {
        case xnn_datatype::xnn_datatype_qint8: {
          // HACK TO Maintain FC/BC for ASR this will be removed after 01/2024

          // When encountering a dynamically quantized tensor via dq_datatype,
          // which is the old flow for serializing dynamically quantized linear.
          // We replace the definition of a single tensor with a new dynamic
          // Quantization pattern. We change the pattern from:
          //     serialized_qd_input
          //           to
          // (fp32_input --> convert --> qdint8_input)

          status = xnn_define_dynamically_quantized_tensor_value(
              /*subgraph=*/subgraph_ptr,
              /*datatype=*/xnn_datatype_qdint8,
              /*num_dims=*/tensor_value->num_dims(),
              /*num_nonbatch_dims=*/1, // always do per token quantization
              /*dims=*/dims_data.data(),
              /*external_id=*/XNN_INVALID_VALUE_ID, // always internal value id
              /*flags=*/0, // this is netiher external input or output
              /*id_out=*/&id);

          // this is the FP16 or FP32 external value that is being dynamically
          // quantized
          uint32_t float_id;
          enum xnn_datatype fp_datatype = getDataType(tensor_value->datatype());
          status = xnn_define_tensor_value(
              /*subgraph=*/subgraph_ptr,
              /*datatype=*/fp_datatype,
              /*num_dims=*/tensor_value->num_dims(),
              /*dims=*/dims_data.data(),
              /*data=*/buffer_ptr,
              /*external_id=*/tensor_value->external_id(),
              /*flags=*/tensor_value->flags(),
              /*id_out=*/&float_id);

          // Define dynamic conversion from float to qdint8
          status = xnn_define_convert(
              /*subgraph=*/subgraph_ptr,
              /*input_id=*/float_id,
              /*output_id=*/id,
              /*flags=*/0);
          break;
        }
        default:
          ET_CHECK_OR_RETURN_ERROR(
              false,
              NotImplemented,
              "Unhandled Dyanmic Quantization dtype: %d",
              dq_datatype);
      }
    } else {
      ET_CHECK_OR_RETURN_ERROR(false, NotImplemented, "Unhandled fp32 tensor");
    }
  } else {
    // define tensor for quantized
    switch (qtensor_value->quant_params_type()) {
      case fb_xnnpack::XNNQuantParams::PerTensorQuant: {
        auto qparams = qtensor_value->quant_params_as_PerTensorQuant();
        ET_LOG(
            Debug,
            "define quant tensor (per tensor): buffer_ptr: %p, scale: %f, zp: %d\n",
            buffer_ptr,
            qparams->scale(),
            qparams->zero_point());
        status = xnn_define_quantized_tensor_value(
            /*subgraph=*/subgraph_ptr,
            /*datatype=*/getDataType(tensor_value->datatype()),
            /*zero_point=*/qparams->zero_point(),
            /*scale=*/qparams->scale(),
            /*num_dims=*/tensor_value->num_dims(),
            /*dims=*/dims_data.data(),
            /*data=*/buffer_ptr,
            /*external_id=*/tensor_value->external_id(),
            /*flags=*/tensor_value->flags(),
            /*id_out=*/&id);
        break;
      }
      case fb_xnnpack::XNNQuantParams::PerChannelQuant: {
        auto qparams = qtensor_value->quant_params_as_PerChannelQuant();
        enum xnn_datatype dtype = getDataType(tensor_value->datatype());
        int32_t zero_point =
            (dtype == xnn_datatype::xnn_datatype_qcint4 ? 8 : 0);

        ET_LOG(
            Debug,
            "define quant tensor (per channel): buffer_ptr: %p, scale.numel(): %u, channel_dim: %u, dtype: %u, zero_point: %d\n",
            buffer_ptr,
            qparams->scale()->size(),
            qparams->channel_dim(),
            dtype,
            zero_point);

        const float* scale = qparams->scale()->data();

        if (qparams->scale_buffer_idx() != 0) {
          scale = reinterpret_cast<const float*>(getConstantDataPtr(
              qparams->scale_buffer_idx(),
              flatbuffer_graph,
              constant_data_ptr,
              named_data_map,
              freeable_buffers,
              weights_cache));
          ET_CHECK_OR_RETURN_ERROR(
              scale != nullptr, Internal, "Failed to load scale data.");
        }
        status = xnn_define_channelwise_quantized_tensor_value_v2(
            /*subgraph=*/subgraph_ptr,
            /*datatype=*/dtype,
            /*zero_point=*/zero_point,
            /*scale=*/scale,
            /*num_dims=*/tensor_value->num_dims(),
            /*channel_dim*/ qparams->channel_dim(),
            /*dims=*/dims_data.data(),
            /*data=*/buffer_ptr,
            /*external_id=*/tensor_value->external_id(),
            /*flags=*/tensor_value->flags(),
            /*id_out=*/&id);
        break;
      }
      case fb_xnnpack::XNNQuantParams::PerChannelGroupQuant: {
        xnn_datatype datatype = getDataType(tensor_value->datatype());
        ET_CHECK_OR_RETURN_ERROR(
            datatype == xnn_datatype::xnn_datatype_qbint4,
            Internal,
            "Unsupported datatype for per channel group quantization: %d",
            datatype);
        auto qparams = qtensor_value->quant_params_as_PerChannelGroupQuant();
        size_t group_size = qparams->group_size();
        size_t output_channels = tensor_value->dims()->Get(0);
        size_t input_channels = tensor_value->dims()->Get(1);

        const uint16_t* scale_data = nullptr;
        uint32_t scale_numel = 0;

        // Block scales are preferably serialized as bf16 but can also be
        // serialized as fp32 for backwards compatability.
        if (qparams->scale_buffer_idx() != 0) {
          scale_data = reinterpret_cast<const uint16_t*>(getConstantDataPtr(
              qparams->scale_buffer_idx(),
              flatbuffer_graph,
              constant_data_ptr,
              named_data_map,
              freeable_buffers,
              weights_cache));
          ET_CHECK_OR_RETURN_ERROR(
              scale_data != nullptr, Internal, "Failed to load scale data.");
          scale_numel = qparams->num_scales();
        } else {
          // Read fp32 scales, convert to bf16.
          auto conv_buffer = static_cast<uint16_t*>(allocator.allocateTemporary(
              qparams->scale()->size() * sizeof(uint16_t)));
          scale_numel = qparams->scale()->size();
          convertF32TensorToBF16(
              qparams->scale()->data(), conv_buffer, scale_numel);
          scale_data = conv_buffer;
        }

        ET_CHECK_OR_RETURN_ERROR(
            scale_numel == output_channels * input_channels / group_size,
            Internal,
            "scale size %zu != output channels %zu * group size %zu",
            static_cast<size_t>(scale_numel),
            output_channels,
            group_size);
        int32_t zero_point =
            (datatype == xnn_datatype::xnn_datatype_qbint4 ? 8 : 0);
        ET_LOG(
            Debug,
            "define quant tensor (per channel group): buffer_ptr: %p, scale.numel(): %u, channel_dim: %u, grpup_size: %zu, output_channels: %zu, dtype: %u, zero_point: %d, datatype: %d\n",
            buffer_ptr,
            scale_numel,
            qparams->channel_dim(),
            group_size,
            output_channels,
            datatype,
            zero_point,
            datatype);

        status = xnn_define_blockwise_quantized_tensor_value(
            /*subgraph=*/subgraph_ptr,
            /*datatype=*/datatype,
            /*zero_point=*/zero_point,
            /*scale=*/scale_data,
            /*num_dims=*/tensor_value->num_dims(),
            /*channel_dim=*/qparams->channel_dim(),
            /*block_size=*/qparams->group_size(),
            /*dims=*/dims_data.data(),
            /*data=*/buffer_ptr,
            /*external_id=*/tensor_value->external_id(),
            /*flags=*/tensor_value->flags(),
            /*id_out=*/&id);
        break;
      }
      case fb_xnnpack::XNNQuantParams::PerTokenDynamicQuant: {
        auto qparams = qtensor_value->quant_params_as_PerTokenDynamicQuant();
        ET_LOG(
            Debug,
            "define quant tensor (dynamic): num_dims: %i, num_nonbatch_dims: %i\n",
            tensor_value->num_dims(),
            qparams->num_nonbatch_dims());
        ET_CHECK_OR_RETURN_ERROR(
            buffer_ptr == nullptr,
            Internal,
            "Dynamically quantized tensor should not have constant data but found non-nullptr");
        status = xnn_define_dynamically_quantized_tensor_value(
            /*subgraph=*/subgraph_ptr,
            /*datatype=*/getDataType(tensor_value->datatype()),
            /*num_dims=*/tensor_value->num_dims(),
            /*num_nonbatch_dims*/ qparams->num_nonbatch_dims(),
            /*dims=*/dims_data.data(),
            /*external_id=*/tensor_value->external_id(),
            /*flags=*/tensor_value->flags(),
            /*id_out=*/&id);
        break;
      }
      default: {
        ET_CHECK_OR_RETURN_ERROR(
            false,
            NotImplemented,
            "Unhandled Quantization Parameters: %s",
            fb_xnnpack::EnumNameXNNQuantParams(
                qtensor_value->quant_params_type()));
      }
    }
  }

  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to define tensor %i with code: %s",
      tensor_value->id_out(),
      xnn_status_to_string(status));

  // map serialized id to newly generated id
  remapped_ids.emplace(std::make_pair(tensor_value->id_out(), id));

  // Add external ids to either list of input or output ids
  if (tensor_value->flags() & XNN_VALUE_FLAG_EXTERNAL_INPUT) {
    input_ids.push_back(tensor_value->external_id());
  }
  if (tensor_value->flags() & XNN_VALUE_FLAG_EXTERNAL_OUTPUT) {
    output_ids.push_back(tensor_value->external_id());
  }

  return Error::Ok;
};

#define MAYBE_UNUSED(x) (void)(x)

#ifdef ENABLE_XNNPACK_KLEIDI
bool isQP8(const fb_xnnpack::XNNGraph* graph, const NodePtr node) {
  assert(node->xnode_union_type() == fb_xnnpack::XNodeUnion::XNNConvert);
  auto graph_node = node->xnode_union_as_XNNConvert();
  auto cvt_output_id = graph_node->output_id();

  auto check_dtype = [graph](uint32_t id, DataType dtype) -> bool {
    for (auto value : *graph->xvalues()) {
      if (value->xvalue_union_type() !=
          fb_xnnpack::XValueUnion::XNNQuantizedTensorValue) {
        continue;
      }
      auto tensor =
          value->xvalue_union_as_XNNQuantizedTensorValue()->tensor_value();
      if (tensor->id_out() == id) {
        return tensor->datatype() == dtype;
      }
    }
    return false;
  };

  // Check if the output tensor is qint8 else bail early.
  if (!check_dtype(cvt_output_id, DataType::xnn_datatype_qdint8)) {
    return false;
  }

  // XNNPACK dtypes which have qp8 support.
  const std::vector<DataType> supported_filter_dtypes = {
      DataType::xnn_datatype_qbint4,
      DataType::xnn_datatype_qcint4,
      DataType::xnn_datatype_qcint8};

  // Find if the convert output is going to the right linear node.
  // Assuming if we can find one valid linear node, then we can use QP8
  // for all the linear nodes consuming this convert output.
  for (auto node : *graph->xnodes()) {
    if (node->xnode_union_type() == fb_xnnpack::XNodeUnion::XNNFullyConnected) {
      auto linear_node = node->xnode_union_as_XNNFullyConnected();
      if (linear_node->input1_id() == cvt_output_id) {
        for (auto supported_filter_dtype : supported_filter_dtypes) {
          if (check_dtype(linear_node->filter_id(), supported_filter_dtype)) {
            return true;
          }
        }
      }
    }
  }
  return false;
}
#endif // ENABLE_XNNPACK_KLEIDI

/*
Define Convert operator Node into the subgraph
*/
Error defineConvertNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node,
    const fb_xnnpack::XNNGraph* flatbuffer_graph) noexcept {
  MAYBE_UNUSED(flatbuffer_graph);
  auto graph_node = node->xnode_union_as_XNNConvert();

  int32_t flags = graph_node->flags();
#ifdef ENABLE_XNNPACK_KLEIDI
// This is not currently exposed at include/xnnpack.h yet once it is
// we can remove this runtime logic and do this ahead-of-time
#define XNN_FLAG_MAYBE_PACK_FOR_QB4W_GEMM 0x00000100;
  if (isQP8(flatbuffer_graph, node)) {
    flags |= XNN_FLAG_MAYBE_PACK_FOR_QB4W_GEMM;
    ET_LOG(
        Debug,
        "Setting XNN_FLAG_MAYBE_PACK_FOR_QB4W_GEMM flag for convert node %i",
        node->debug_handle());
  }
#endif

  xnn_status status = xnn_define_convert(
      subgraph_ptr,
      remapped_ids.at(graph_node->input_id()),
      remapped_ids.at(graph_node->output_id()),
      flags);

  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create convert node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
};
/*
Define serialized linear(fully-connected) node into the subgraph using
the remapped ids to map the serialized ids, to the new ids generated
when defining the tensor values
*/
Error defineFullyConnectedNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node,
    const fb_xnnpack::XNNGraph* graph) noexcept {
  MAYBE_UNUSED(graph);

  auto graph_node = node->xnode_union_as_XNNFullyConnected();
  std::pair<float, float> min_max = getOutputMinMax(node);
  xnn_status status = xnn_define_fully_connected(
      subgraph_ptr,
      min_max.first,
      min_max.second,
      remapped_ids.at(graph_node->input1_id()),
      remapped_ids.at(graph_node->filter_id()),
      remapped_ids.at(graph_node->bias_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());
  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create linear node %i, with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
};

/*
Define serialized softmax node into the subgraph, using the remapped ids
to map the serialized ids, to the new ids generated when defining
the tensor value
*/
Error defineSoftmaxNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node,
    const fb_xnnpack::XNNGraph* graph) noexcept {
  MAYBE_UNUSED(graph);

  auto graph_node = node->xnode_union_as_XNNSoftmax();
  xnn_status status = xnn_define_softmax(
      subgraph_ptr,
      remapped_ids.at(graph_node->input_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());
  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create softmax node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
}

Error defineGlobalAvgPooling2dNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node,
    const fb_xnnpack::XNNGraph* graph) noexcept {
  MAYBE_UNUSED(graph);

  auto graph_node = node->xnode_union_as_XNNGlobalAvgPooling2d();
  std::pair<float, float> min_max = getOutputMinMax(node);
  xnn_status status = xnn_define_global_average_pooling_2d(
      subgraph_ptr,
      min_max.first,
      min_max.second,
      remapped_ids.at(graph_node->input_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());
  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create global average pooling node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
}

Error defineAvgPooling2dNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node,
    const fb_xnnpack::XNNGraph* graph) noexcept {
  MAYBE_UNUSED(graph);

  auto graph_node = node->xnode_union_as_XNNAvgPooling2d();
  std::pair<float, float> min_max = getOutputMinMax(node);
  xnn_status status = xnn_define_average_pooling_2d(
      subgraph_ptr,
      graph_node->padding_top(),
      graph_node->padding_right(),
      graph_node->padding_bottom(),
      graph_node->padding_left(),
      graph_node->pooling_height(),
      graph_node->pooling_width(),
      graph_node->stride_height(),
      graph_node->stride_width(),
      min_max.first,
      min_max.second,
      remapped_ids.at(graph_node->input_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());
  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create average pooling node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
}

/*
Define serialized conv2d node into the subgraph, using the remapped ids
to map the serialized ids, to the new ids generated when defining the
tensor value
*/
Error defineConv2dNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node,
    const fb_xnnpack::XNNGraph* graph) noexcept {
  MAYBE_UNUSED(graph);

  auto graph_node = node->xnode_union_as_XNNConv2d();
  std::pair<float, float> min_max = getOutputMinMax(node);
  xnn_status status = xnn_define_convolution_2d(
      subgraph_ptr,
      graph_node->padding_top(),
      graph_node->padding_right(),
      graph_node->padding_bottom(),
      graph_node->padding_left(),
      graph_node->kernel_height(),
      graph_node->kernel_width(),
      graph_node->subsampling_height(),
      graph_node->subsampling_width(),
      graph_node->dilation_height(),
      graph_node->dilation_width(),
      graph_node->groups(),
      graph_node->group_input_channels(),
      graph_node->group_output_channels(),
      min_max.first,
      min_max.second,
      remapped_ids.at(graph_node->input1_id()),
      remapped_ids.at(graph_node->filter_id()),
      remapped_ids.at(graph_node->bias_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());
  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create convolution node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
}

/*
Define serialized conv_transpose2d node into the subgraph, using the remapped
ids to map the serialized ids, to the new ids generated when defining the tensor
value
*/
Error defineConvTranspose2dNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node,
    const fb_xnnpack::XNNGraph* graph) noexcept {
  MAYBE_UNUSED(graph);
  auto graph_node = node->xnode_union_as_XNNConvTranspose2d();

  std::pair<float, float> min_max = getOutputMinMax(node);
  xnn_status status = xnn_define_deconvolution_2d(
      subgraph_ptr,
      graph_node->padding_top(),
      graph_node->padding_right(),
      graph_node->padding_bottom(),
      graph_node->padding_left(),
      graph_node->adjustment_height(),
      graph_node->adjustment_width(),
      graph_node->kernel_height(),
      graph_node->kernel_width(),
      graph_node->subsampling_height(),
      graph_node->subsampling_width(),
      graph_node->dilation_height(),
      graph_node->dilation_width(),
      graph_node->groups(),
      graph_node->group_input_channels(),
      graph_node->group_output_channels(),
      min_max.first,
      min_max.second,
      remapped_ids.at(graph_node->input1_id()),
      remapped_ids.at(graph_node->filter_id()),
      remapped_ids.at(graph_node->bias_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());
  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create deconvolution node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
}

/*
Define serialized maxpool2d node into the subgraph, using the remapped ids
to map the serialized ids, to the new ids generated when defining the
tensor value
*/
Error defineMaxPooling2dNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node,
    const fb_xnnpack::XNNGraph* graph) noexcept {
  MAYBE_UNUSED(graph);

  auto graph_node = node->xnode_union_as_XNNMaxPooling2d();
  std::pair<float, float> min_max = getOutputMinMax(node);
  xnn_status status = xnn_define_max_pooling_2d(
      subgraph_ptr,
      graph_node->padding_top(),
      graph_node->padding_right(),
      graph_node->padding_bottom(),
      graph_node->padding_left(),
      graph_node->pooling_height(),
      graph_node->pooling_width(),
      graph_node->stride_height(),
      graph_node->stride_width(),
      graph_node->dilation_height(),
      graph_node->dilation_width(),
      min_max.first,
      min_max.second,
      remapped_ids.at(graph_node->input_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());

  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create maxpool2d node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
}

/*
Define serialized static transpose node into the subgraph, using the remapped
ids to map the serialized ids, to the new ids generated when defining the
tensor value
*/
Error defineStaticTransposeNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node,
    const fb_xnnpack::XNNGraph* graph) noexcept {
  MAYBE_UNUSED(graph);

  auto graph_node = node->xnode_union_as_XNNStaticTranspose();

  // Get tensor dims, we need to convert the uint32_t* to size_t*
  std::vector<size_t> dims_data = flatbufferDimsToVector(graph_node->perm());
  xnn_status status = xnn_define_static_transpose(
      subgraph_ptr,
      graph_node->num_dims(),
      dims_data.data(),
      remapped_ids.at(graph_node->input_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());
  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create static transpose node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
}

/*
Define serialized static resize bilinear 2d node into the subgraph, using the
remapped ids to map the serialized ids, to the new ids generated when defining
the tensor value
*/
Error defineStaticResizeBilinear2DNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node,
    const fb_xnnpack::XNNGraph* graph) noexcept {
  MAYBE_UNUSED(graph);

  const fb_xnnpack::XNNStaticResizeBilinear2D* graph_node =
      node->xnode_union_as_XNNStaticResizeBilinear2D();

  xnn_status status = xnn_define_static_resize_bilinear_2d(
      subgraph_ptr,
      graph_node->new_height(),
      graph_node->new_width(),
      remapped_ids.at(graph_node->input_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());
  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create StaticResizeBilinear2DNode node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
}

/*
Define serialized static constant pad node into the subgraph, using the
remapped ids to map the serialized ids, to the new ids generated when defining
the tensor value
*/
Error defineStaticConstantPadNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node,
    const fb_xnnpack::XNNGraph* graph) noexcept {
  MAYBE_UNUSED(graph);

  const fb_xnnpack::XNNStaticConstantPad* graph_node =
      node->xnode_union_as_XNNStaticConstantPad();

  std::vector<size_t> pre_paddings_dims =
      flatbufferDimsToVector(graph_node->pre_paddings());
  std::vector<size_t> post_paddings_dims =
      flatbufferDimsToVector(graph_node->post_paddings());

  xnn_status status = xnn_define_static_constant_pad(
      subgraph_ptr,
      pre_paddings_dims.data(),
      post_paddings_dims.data(),
      graph_node->padding_value(),
      remapped_ids.at(graph_node->input_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());
  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create StaticConstantPad node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
}

/*
Define serialized depthwise conv2d node into the subgraph, using the remapped
ids to map the serialized ids, to the new ids generated when defining the
tensor value
*/
Error defineDepthwiseConv2dNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node,
    const fb_xnnpack::XNNGraph* graph) noexcept {
  MAYBE_UNUSED(graph);

  auto graph_node = node->xnode_union_as_XNNDepthwiseConv2d();
  std::pair<float, float> min_max = getOutputMinMax(node);
  xnn_status status = xnn_define_depthwise_convolution_2d(
      subgraph_ptr,
      graph_node->padding_top(),
      graph_node->padding_right(),
      graph_node->padding_bottom(),
      graph_node->padding_left(),
      graph_node->kernel_height(),
      graph_node->kernel_width(),
      graph_node->subsampling_height(),
      graph_node->subsampling_width(),
      graph_node->dilation_height(),
      graph_node->dilation_width(),
      graph_node->group_output_channels() /
          graph_node->group_input_channels(), // depth_multiplier
      graph_node->groups(), // input_channels = groups for depthwise conv
      min_max.first,
      min_max.second,
      remapped_ids.at(graph_node->input1_id()),
      remapped_ids.at(graph_node->filter_id()),
      remapped_ids.at(graph_node->bias_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());

  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create depthwise convolution node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
}

Error defineStaticReshapeNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node,
    const fb_xnnpack::XNNGraph* graph) noexcept {
  MAYBE_UNUSED(graph);

  auto graph_node = node->xnode_union_as_XNNStaticReshape();

  // Get tensor dims, we need to convert the uint32_t* to size_t*
  std::vector<size_t> dims_data =
      flatbufferDimsToVector(graph_node->new_shape());
  xnn_status status = xnn_define_static_reshape(
      subgraph_ptr,
      graph_node->num_dims(),
      dims_data.data(),
      remapped_ids.at(graph_node->input_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());
  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create squeeze node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
}

/*
Define serialized maxpool2d node into the subgraph, using the remapped ids
to map the serialized ids, to the new ids generated when defining the
tensor value
*/
Error defineArgMaxPooling2dNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node,
    const fb_xnnpack::XNNGraph* graph) noexcept {
  MAYBE_UNUSED(graph);

  auto graph_node = node->xnode_union_as_XNNArgMaxPooling2d();

  xnn_status status = xnn_define_argmax_pooling_2d(
      subgraph_ptr,
      graph_node->padding_top(),
      graph_node->padding_right(),
      graph_node->padding_bottom(),
      graph_node->padding_left(),
      graph_node->pooling_height(),
      graph_node->pooling_width(),
      remapped_ids.at(graph_node->input_id()),
      remapped_ids.at(graph_node->output_value_id()),
      remapped_ids.at(graph_node->output_index_id()),
      graph_node->flags());

  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create argmaxpool2d node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
}

/*
Define serialized exp node into the subgraph, using the remapped ids
to map the serialized ids, to the new ids generated when defining the
tensor value
*/
Error defineExpNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node,
    const fb_xnnpack::XNNGraph* graph) noexcept {
  MAYBE_UNUSED(graph);

  auto graph_node = node->xnode_union_as_XNNExp();

  xnn_status status = xnn_define_exp(
      subgraph_ptr,
      remapped_ids.at(graph_node->input_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());

  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create exp node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
}

/*
Define serialized tanh node into the subgraph, using the remapped ids
to map the serialized ids, to the new ids generated when defining the
tensor value
*/
Error defineTanhNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node,
    const fb_xnnpack::XNNGraph* graph) noexcept {
  MAYBE_UNUSED(graph);

  auto graph_node = node->xnode_union_as_XNNTanh();

  xnn_status status = xnn_define_tanh(
      subgraph_ptr,
      remapped_ids.at(graph_node->input_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());

  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create tanh node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
}

/*
Defines serialized prelu node into the subgraph,
using the remapped ids to map the serialized ids,
to the new ids generated when defining the tensor value
*/
Error definePReLUNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node,
    const fb_xnnpack::XNNGraph* graph) noexcept {
  MAYBE_UNUSED(graph);

  auto graph_node = node->xnode_union_as_XNNPReLU();

  xnn_status status = xnn_define_prelu(
      subgraph_ptr,
      remapped_ids.at(graph_node->input1_id()),
      remapped_ids.at(graph_node->input2_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());

  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create prelu node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
}

/*
Defines serialized concatenate2 node into the subgraph,
using the remapped ids to map the serialized ids,
to the new ids generated when defining the tensor value
*/
Error defineConcatenate2Node(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node,
    const fb_xnnpack::XNNGraph* graph) noexcept {
  MAYBE_UNUSED(graph);

  auto graph_node = node->xnode_union_as_XNNConcatenate2();

  xnn_status status = xnn_define_concatenate2(
      subgraph_ptr,
      graph_node->axis(),
      remapped_ids.at(graph_node->input1_id()),
      remapped_ids.at(graph_node->input2_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());

  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create cat2 node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
}

/*
Defines serialized concatenate3 node into the subgraph,
using the remapped ids to map the serialized ids,
to the new ids generated when defining the tensor value
*/
Error defineConcatenate3Node(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node,
    const fb_xnnpack::XNNGraph* graph) noexcept {
  MAYBE_UNUSED(graph);

  auto graph_node = node->xnode_union_as_XNNConcatenate3();

  xnn_status status = xnn_define_concatenate3(
      subgraph_ptr,
      graph_node->axis(),
      remapped_ids.at(graph_node->input1_id()),
      remapped_ids.at(graph_node->input2_id()),
      remapped_ids.at(graph_node->input3_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());

  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create cat3 node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
}

/*
Defines serialized concatenate4 node into the subgraph,
using the remapped ids to map the serialized ids,
to the new ids generated when defining the tensor value
*/
Error defineConcatenate4Node(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node,
    const fb_xnnpack::XNNGraph* graph) noexcept {
  MAYBE_UNUSED(graph);

  auto graph_node = node->xnode_union_as_XNNConcatenate4();

  xnn_status status = xnn_define_concatenate4(
      subgraph_ptr,
      graph_node->axis(),
      remapped_ids.at(graph_node->input1_id()),
      remapped_ids.at(graph_node->input2_id()),
      remapped_ids.at(graph_node->input3_id()),
      remapped_ids.at(graph_node->input4_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());

  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create cat4 node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
}

/*
Defines serialized concatenate5 node into the subgraph,
using the remapped ids to map the serialized ids,
to the new ids generated when defining the tensor value
*/
Error defineConcatenate5Node(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node,
    const fb_xnnpack::XNNGraph* graph) noexcept {
  MAYBE_UNUSED(graph);

  auto graph_node = node->xnode_union_as_XNNConcatenate5();

  xnn_status status = xnn_define_concatenate5(
      subgraph_ptr,
      graph_node->axis(),
      remapped_ids.at(graph_node->input1_id()),
      remapped_ids.at(graph_node->input2_id()),
      remapped_ids.at(graph_node->input3_id()),
      remapped_ids.at(graph_node->input4_id()),
      remapped_ids.at(graph_node->input5_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());

  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create cat5 node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
}

/*
Defines serialized static_slice node into the subgraph,
using the remapped ids to map the serialized ids,
to the new ids generated when defining the tensor value
*/
Error defineStaticSliceNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node,
    const fb_xnnpack::XNNGraph* graph) noexcept {
  MAYBE_UNUSED(graph);

  auto graph_node = node->xnode_union_as_XNNStaticSlice();

  std::vector<size_t> offsets = flatbufferDimsToVector(graph_node->offsets());
  std::vector<size_t> sizes = flatbufferDimsToVector(graph_node->sizes());

  xnn_status status = xnn_define_static_slice(
      subgraph_ptr,
      graph_node->num_dims(),
      offsets.data(),
      sizes.data(),
      remapped_ids.at(graph_node->input_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());

  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create static slice node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
}

/*
Defines batch matrix multiply node into the subgraph,
using the remapped ids to map the serialized ids,
to the new ids generated when defining the tensor value
*/
Error defineBatchMatrixMultiplyNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node,
    const fb_xnnpack::XNNGraph* graph) noexcept {
  MAYBE_UNUSED(graph);

  auto graph_node = node->xnode_union_as_XNNBatchMatrixMultiply();

  xnn_status status = xnn_define_batch_matrix_multiply(
      subgraph_ptr,
      remapped_ids.at(graph_node->input1_id()),
      remapped_ids.at(graph_node->input2_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());

  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create BMM node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
}

/*
Returns not Implemented Error code. This function is meant to be
called when the compiler encountes a XNodeType from the flatbuffer
that has not yet been implemented
*/
Error defineNotImplementedNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node,
    const fb_xnnpack::XNNGraph* graph) noexcept {
  MAYBE_UNUSED(graph);

  ET_CHECK_OR_RETURN_ERROR(
      false,
      NotImplemented,
      "Unhandled node type: %s",
      fb_xnnpack::EnumNameXNodeUnion(node->xnode_union_type()));
}

// Generic helper function for unary operations
Error defineGenericUnaryNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    uint32_t input_id,
    uint32_t output_id,
    uint32_t flags,
    xnn_unary_operator op_type,
    const union xnn_unary_params* params,
    fb_xnnpack::XNodeUnion node_type,
    uint32_t debug_handle) noexcept {
  xnn_status status = xnn_define_unary(
      subgraph_ptr,
      op_type,
      params,
      remapped_ids.at(input_id),
      remapped_ids.at(output_id),
      flags);

  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create %s node %i with code: %s",
      fb_xnnpack::EnumNameXNodeUnion(node_type),
      debug_handle,
      xnn_status_to_string(status));

  return Error::Ok;
}

// Macro for unary operations with no parameters
#define _DEFINE_UNARY_NODE_NO_PARAMS(name, op_type)               \
  Error define##name##Node(                                       \
      xnn_subgraph_t subgraph_ptr,                                \
      const std::unordered_map<uint32_t, uint32_t>& remapped_ids, \
      const NodePtr node,                                         \
      const fb_xnnpack::XNNGraph* graph) noexcept {               \
    MAYBE_UNUSED(graph);                                          \
    auto graph_node = node->xnode_union_as_XNN##name();           \
    return defineGenericUnaryNode(                                \
        subgraph_ptr,                                             \
        remapped_ids,                                             \
        graph_node->input_id(),                                   \
        graph_node->output_id(),                                  \
        graph_node->flags(),                                      \
        op_type,                                                  \
        nullptr,                                                  \
        node->xnode_union_type(),                                 \
        node->debug_handle());                                    \
  }

// Macro for unary operations with min/max parameters
#define _DEFINE_UNARY_NODE_WITH_MINMAX(name, op_type)             \
  Error define##name##Node(                                       \
      xnn_subgraph_t subgraph_ptr,                                \
      const std::unordered_map<uint32_t, uint32_t>& remapped_ids, \
      const NodePtr node,                                         \
      const fb_xnnpack::XNNGraph* graph) noexcept {               \
    MAYBE_UNUSED(graph);                                          \
    auto graph_node = node->xnode_union_as_XNN##name();           \
    std::pair<float, float> min_max = getOutputMinMax(node);      \
    union xnn_unary_params params;                                \
    params.clamp.min = min_max.first;                             \
    params.clamp.max = min_max.second;                            \
    return defineGenericUnaryNode(                                \
        subgraph_ptr,                                             \
        remapped_ids,                                             \
        graph_node->input_id(),                                   \
        graph_node->output_id(),                                  \
        graph_node->flags(),                                      \
        op_type,                                                  \
        &params,                                                  \
        node->xnode_union_type(),                                 \
        node->debug_handle());                                    \
  }

// Macro for unary operations with leaky_relu parameters
#define _DEFINE_UNARY_NODE_WITH_LEAKY_RELU(name)                     \
  Error define##name##Node(                                          \
      xnn_subgraph_t subgraph_ptr,                                   \
      const std::unordered_map<uint32_t, uint32_t>& remapped_ids,    \
      const NodePtr node,                                            \
      const fb_xnnpack::XNNGraph* graph) noexcept {                  \
    MAYBE_UNUSED(graph);                                             \
    auto graph_node = node->xnode_union_as_XNNLeakyReLU();           \
    union xnn_unary_params params;                                   \
    params.leaky_relu.negative_slope = graph_node->negative_slope(); \
    return defineGenericUnaryNode(                                   \
        subgraph_ptr,                                                \
        remapped_ids,                                                \
        graph_node->input_id(),                                      \
        graph_node->output_id(),                                     \
        graph_node->flags(),                                         \
        xnn_unary_leaky_relu,                                        \
        &params,                                                     \
        node->xnode_union_type(),                                    \
        node->debug_handle());                                       \
  }

// Macro for unary operations with elu parameters
#define _DEFINE_UNARY_NODE_WITH_ELU(name)                         \
  Error define##name##Node(                                       \
      xnn_subgraph_t subgraph_ptr,                                \
      const std::unordered_map<uint32_t, uint32_t>& remapped_ids, \
      const NodePtr node,                                         \
      const fb_xnnpack::XNNGraph* graph) noexcept {               \
    MAYBE_UNUSED(graph);                                          \
    auto graph_node = node->xnode_union_as_XNNELU();              \
    union xnn_unary_params params;                                \
    params.elu.alpha = graph_node->alpha();                       \
    return defineGenericUnaryNode(                                \
        subgraph_ptr,                                             \
        remapped_ids,                                             \
        graph_node->input_id(),                                   \
        graph_node->output_id(),                                  \
        graph_node->flags(),                                      \
        xnn_unary_elu,                                            \
        &params,                                                  \
        node->xnode_union_type(),                                 \
        node->debug_handle());                                    \
  }

// Generic helper function for binary operations
Error defineGenericBinaryNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const fb_xnnpack::_XNNNode2x1* graph_node,
    xnn_binary_operator op_type,
    const struct xnn_binary_params* params,
    fb_xnnpack::XNodeUnion node_type,
    uint32_t debug_handle) noexcept {
  xnn_status status = xnn_define_binary(
      subgraph_ptr,
      op_type,
      params,
      remapped_ids.at(graph_node->input1_id()),
      remapped_ids.at(graph_node->input2_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());

  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create %s node %i with code: %s",
      fb_xnnpack::EnumNameXNodeUnion(node_type),
      debug_handle,
      xnn_status_to_string(status));

  return Error::Ok;
}

// Macro for binary operations with min/max parameters
#define _DEFINE_BINARY_NODE_WITH_MINMAX(name, op_type)            \
  Error define##name##Node(                                       \
      xnn_subgraph_t subgraph_ptr,                                \
      const std::unordered_map<uint32_t, uint32_t>& remapped_ids, \
      const NodePtr node,                                         \
      const fb_xnnpack::XNNGraph* graph) noexcept {               \
    MAYBE_UNUSED(graph);                                          \
    auto graph_node = node->xnode_union_as_XNN##name();           \
    std::pair<float, float> min_max = getOutputMinMax(node);      \
    struct xnn_binary_params params;                              \
    params.output_min = min_max.first;                            \
    params.output_max = min_max.second;                           \
    return defineGenericBinaryNode(                               \
        subgraph_ptr,                                             \
        remapped_ids,                                             \
        graph_node,                                               \
        op_type,                                                  \
        &params,                                                  \
        node->xnode_union_type(),                                 \
        node->debug_handle());                                    \
  }

// Macro for binary operations without parameters
#define _DEFINE_BINARY_NODE_NO_PARAMS(name, op_type)              \
  Error define##name##Node(                                       \
      xnn_subgraph_t subgraph_ptr,                                \
      const std::unordered_map<uint32_t, uint32_t>& remapped_ids, \
      const NodePtr node,                                         \
      const fb_xnnpack::XNNGraph* graph) noexcept {               \
    MAYBE_UNUSED(graph);                                          \
    auto graph_node = node->xnode_union_as_XNN##name();           \
    return defineGenericBinaryNode(                               \
        subgraph_ptr,                                             \
        remapped_ids,                                             \
        graph_node,                                               \
        op_type,                                                  \
        nullptr,                                                  \
        node->xnode_union_type(),                                 \
        node->debug_handle());                                    \
  }

/*
Returns the pointer to the defineNode function that handles the given
XNode type
*/
#define _DEFINE(name)                     \
  case fb_xnnpack::XNodeUnion::XNN##name: \
    return &define##name##Node;

// Unary Ops with no params
_DEFINE_UNARY_NODE_NO_PARAMS(Sigmoid, xnn_unary_sigmoid)
_DEFINE_UNARY_NODE_NO_PARAMS(Floor, xnn_unary_floor)
_DEFINE_UNARY_NODE_NO_PARAMS(SquareRoot, xnn_unary_square_root)
_DEFINE_UNARY_NODE_NO_PARAMS(
    ReciprocalSquareRoot,
    xnn_unary_reciprocal_square_root)
_DEFINE_UNARY_NODE_NO_PARAMS(Ceiling, xnn_unary_ceiling)
_DEFINE_UNARY_NODE_NO_PARAMS(Gelu, xnn_unary_gelu)
_DEFINE_UNARY_NODE_NO_PARAMS(Hardswish, xnn_unary_hardswish)
_DEFINE_UNARY_NODE_NO_PARAMS(Log, xnn_unary_log)
_DEFINE_UNARY_NODE_NO_PARAMS(Negate, xnn_unary_negate)
_DEFINE_UNARY_NODE_NO_PARAMS(Square, xnn_unary_square)
_DEFINE_UNARY_NODE_NO_PARAMS(Abs, xnn_unary_abs)
_DEFINE_UNARY_NODE_NO_PARAMS(Sin, xnn_unary_sine)

// Unary Ops with min/max params
_DEFINE_UNARY_NODE_WITH_MINMAX(Clamp, xnn_unary_clamp)

// Unary Ops with specific params
_DEFINE_UNARY_NODE_WITH_LEAKY_RELU(LeakyReLU)
_DEFINE_UNARY_NODE_WITH_ELU(ELU)

// Binary Ops with params
_DEFINE_BINARY_NODE_WITH_MINMAX(Add, xnn_binary_add)
_DEFINE_BINARY_NODE_WITH_MINMAX(Subtract, xnn_binary_subtract)
_DEFINE_BINARY_NODE_WITH_MINMAX(Multiply, xnn_binary_multiply)
_DEFINE_BINARY_NODE_WITH_MINMAX(Div, xnn_binary_divide)

// Binary Ops without params
_DEFINE_BINARY_NODE_NO_PARAMS(Minimum, xnn_binary_minimum)
_DEFINE_BINARY_NODE_NO_PARAMS(Maximum, xnn_binary_maximum)

DefineNodeFunc getDefineNodeFunc(fb_xnnpack::XNodeUnion nodeType) {
  switch (nodeType) {
    // Binary ops
    _DEFINE(Add)
    _DEFINE(Subtract)
    _DEFINE(Multiply)
    _DEFINE(Div)
    _DEFINE(Minimum)
    _DEFINE(Maximum)

    // Unary ops
    _DEFINE(Softmax)
    _DEFINE(SquareRoot)
    _DEFINE(ReciprocalSquareRoot)
    _DEFINE(Ceiling)
    _DEFINE(Gelu)
    _DEFINE(Hardswish)
    _DEFINE(Log)
    _DEFINE(Tanh)
    _DEFINE(Negate)
    _DEFINE(Square)
    _DEFINE(Clamp)
    _DEFINE(LeakyReLU)
    _DEFINE(ELU)
    _DEFINE(Exp)
    _DEFINE(Abs)
    _DEFINE(Floor)
    _DEFINE(PReLU)
    _DEFINE(Sigmoid)
    _DEFINE(Sin)

    // Others
    _DEFINE(FullyConnected)
    _DEFINE(StaticTranspose)
    _DEFINE(Conv2d)
    _DEFINE(ConvTranspose2d)
    _DEFINE(StaticResizeBilinear2D)
    _DEFINE(StaticConstantPad)
    _DEFINE(AvgPooling2d)
    _DEFINE(DepthwiseConv2d)
    _DEFINE(MaxPooling2d)
    _DEFINE(Convert)
    _DEFINE(GlobalAvgPooling2d)
    _DEFINE(StaticReshape)
    _DEFINE(ArgMaxPooling2d)
    _DEFINE(Concatenate2)
    _DEFINE(Concatenate3)
    _DEFINE(Concatenate4)
    _DEFINE(Concatenate5)
    _DEFINE(StaticSlice)
    _DEFINE(BatchMatrixMultiply)
    case fb_xnnpack::XNodeUnion::NONE:
    default: // Adding here as a catch all, just in case
      return &defineNotImplementedNode;
  }
}
#undef _DEFINE

/*
Builds the xnnpack runtime object using the buffer pointer. The buffer pointer
must be a valid pointer to the serialized xnnpack object. It also fills the
XNNExecutor object with the built xnn_runtime and the input/output ids.
*/
ET_NODISCARD Error XNNCompiler::compileModel(
    const void* buffer_pointer,
    size_t num_bytes,
    XNNExecutor* executor,
    XNNWeightsCache* weights_cache,
    xnn_workspace_t workspace,
    const NamedDataMap* named_data_map) {
  Result<XNNHeader> header = XNNHeader::Parse(buffer_pointer, num_bytes);
  const uint8_t* flatbuffer_data = nullptr;
  const uint8_t* constant_data = nullptr;
  CompileAllocator compile_allocator;

  // Header status can only either be Error::Ok or Error::NotFound
  if (header.ok()) {
    flatbuffer_data = reinterpret_cast<const uint8_t*>(buffer_pointer) +
        header->flatbuffer_offset;
    constant_data = reinterpret_cast<const uint8_t*>(buffer_pointer) +
        header->constant_data_offset;
  } else if (header.error() == Error::NotFound) {
    flatbuffer_data = reinterpret_cast<const uint8_t*>(buffer_pointer);
  } else {
    ET_LOG(Error, "XNNHeader may be corrupt");
    return header.error();
  }

  // Temporarily support identifier XN00 and XN01
  bool is_supported_version =
      strncmp(flatbuffers::GetBufferIdentifier(flatbuffer_data), "XN00", 4) ==
          0 ||
      strncmp(flatbuffers::GetBufferIdentifier(flatbuffer_data), "XN01", 4) ==
          0;
  ET_CHECK_OR_RETURN_ERROR(
      is_supported_version,
      DelegateInvalidCompatibility,
      "XNNPACK Delegate Serialization Format version identifier '%.4s' != expected XN00 or XN01'",
      flatbuffers::GetBufferIdentifier(flatbuffer_data));

  auto flatbuffer_graph = fb_xnnpack::GetXNNGraph(flatbuffer_data);
  // initialize xnnpack
  xnn_status status = xnn_initialize(/*allocator =*/nullptr);
  ET_CHECK_OR_RETURN_ERROR(
      xnn_status_success == status,
      Internal,
      "XNN Initialize failed with code: %s",
      xnn_status_to_string(status));

  // create xnnpack subgraph
  xnn_subgraph_t subgraph_ptr = nullptr;
  status = xnn_create_subgraph(
      /*external_value_ids=*/flatbuffer_graph->num_externs(),
      /*flags=*/0,
      &subgraph_ptr);
  ET_CHECK_OR_RETURN_ERROR(
      xnn_status_success == status,
      Internal,
      "XNN Subgraph creation failed with code: %s",
      xnn_status_to_string(status));

  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> subgraph(
      subgraph_ptr, &xnn_delete_subgraph);

  // mapping from old ids to new created value ids
  // The old ids that were serialied were generated AoT, since
  // we are re-defining tensor values, the defined IDs could be
  // different from the ones generated AoT, as a result, we need
  // a new mapping from the old ids to the newly created ones
  std::unordered_map<uint32_t, uint32_t> remapped_ids;
  // Invalid ids do not need to be remapped
  remapped_ids.emplace(XNN_INVALID_VALUE_ID, XNN_INVALID_VALUE_ID);

  // If weight cache is not on we hold onto all the unpacked buffers
  // and we free them at the end
  std::vector<FreeableBuffer> unpacked_buffers;

  // External Ids for inputs and outputs
  std::vector<uint32_t> input_ids;
  std::vector<uint32_t> output_ids;
  Error err = Error::Ok;
  for (auto value : *flatbuffer_graph->xvalues()) {
    err = defineTensor(
        subgraph.get(),
        remapped_ids,
        value,
        flatbuffer_graph,
        constant_data,
        input_ids,
        output_ids,
        compile_allocator,
        named_data_map,
        unpacked_buffers,
        weights_cache);

    if (err != Error::Ok) {
      return err;
    }
  }

  for (auto node : *flatbuffer_graph->xnodes()) {
    err = getDefineNodeFunc(node->xnode_union_type())(
        subgraph.get(), remapped_ids, node, flatbuffer_graph);
    if (err != Error::Ok) {
      return err;
    }
  }
  uint32_t runtime_flags = 0;

#if defined(ENABLE_XNNPACK_PROFILING) || defined(ET_EVENT_TRACER_ENABLED)
  runtime_flags |= XNN_FLAG_BASIC_PROFILING;
#endif

  xnn_runtime_t runtime_ptr = nullptr;

  // XNNWeightsCache if weights cache is not enabled, then XNNWeightsCache
  // just manages the unpacked weights until the runtime is created.
#ifdef ENABLE_XNNPACK_WEIGHTS_CACHE
  ET_CHECK_OR_RETURN_ERROR(
      unpacked_buffers.size() == 0,
      Internal,
      "Weight Cache is enabled, which means unpacked buffers should be owned by the cache");
  xnn_weights_cache_t weights_cache_ptr =
      weights_cache->get_num_unpacked_data() > 0 ? weights_cache->get()
                                                 : nullptr;
#else
  xnn_weights_cache_t weights_cache_ptr = nullptr;
#endif

  // NOLINTBEGIN(facebook-hte-NullableDereference) - weights cache is allowed to
  // be null
  status = xnn_create_runtime_v4(
      subgraph.get(),
      weights_cache_ptr,
      workspace,
      ::executorch::extension::threadpool::get_pthreadpool(),
      runtime_flags,
      &runtime_ptr);
  // NOLINTEND(facebook-hte-NullableDereference)

  ET_CHECK_OR_RETURN_ERROR(
      xnn_status_success == status,
      Internal,
      "XNN Runtime creation failed with code: %s",
      xnn_status_to_string(status));

#ifdef ENABLE_XNNPACK_WEIGHTS_CACHE
  auto packed_weights_names = weights_cache->finalize_for_runtime();
  ET_CHECK_OR_RETURN_ERROR(
      packed_weights_names.ok(),
      Internal,
      "Failed to finalize weights cache after creating the xnn runtime")
#else
  for (auto& buffer : unpacked_buffers) {
    buffer.Free();
  }
  Result<std::vector<std::string>> packed_weights_names =
      std::vector<std::string>();
#endif

  err = executor->initialize( // NOLINT: runtime_ptr is non-null
      runtime_ptr,
      std::move(input_ids),
      std::move(output_ids),
      std::move(packed_weights_names.get()));

  return err;
};

} // namespace delegate
} // namespace xnnpack
} // namespace backends
} // namespace executorch
