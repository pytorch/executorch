/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/xnnpack/runtime/XNNCompiler.h>
#include <executorch/backends/xnnpack/schema_generated.h>
#include <executorch/backends/xnnpack/threadpool/threadpool.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <unordered_map>

#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wglobal-constructors"

namespace torch {
namespace executor {
namespace xnnpack {
namespace delegate {

// Flatbuffer types
using ValuePtr = const fb_xnnpack::XValue*;
using NodePtr = const fb_xnnpack::XNode*;
using GraphPtr = const fb_xnnpack::XNNGraph*;
using DataType = fb_xnnpack::XNNDatatype;

// Type for define node function. This is the function signature
// for any function that takes in a flatbuffer node and defines it
// into our xnn_subgraph
using DefineNodeFunc = Error (*)(
    xnn_subgraph_t,
    const std::unordered_map<uint32_t, uint32_t>&,
    NodePtr) noexcept;

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
Define serialized tensor value into
the subgraph. While also keeping track of the remapped ids from
the serialized id to the newly generated id.
*/
Error defineTensor(
    xnn_subgraph_t subgraph_ptr,
    std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    ValuePtr value,
    GraphPtr flatbuffer_graph,
    XNNExecutor* executor,
    MemoryAllocator* runtime_allocator) {
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
  const auto& constant_buffer = *flatbuffer_graph->constant_buffer();
  auto buffer_idx = tensor_value->constant_buffer_idx();
  const auto buffer_ptr = buffer_idx == 0
      ? nullptr
      : constant_buffer[buffer_idx]->storage()->data();
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
              /*num_nonbatch_dims=*/1, // This is always for fully connected
              /*dims=*/dims_data.data(),
              /*external_id=*/XNN_INVALID_VALUE_ID, // always internal value id
              /*flags=*/0, // this is netiher external input or output
              /*id_out=*/&id);

          // this is the FP32 external value that is dynamically quantized
          uint32_t fp32_id;
          status = xnn_define_tensor_value(
              /*subgraph=*/subgraph_ptr,
              /*datatype=*/xnn_datatype_fp32, // always fp32
              /*num_dims=*/tensor_value->num_dims(),
              /*dims=*/dims_data.data(),
              /*data=*/buffer_ptr,
              /*external_id=*/tensor_value->external_id(),
              /*flags=*/tensor_value->flags(),
              /*id_out=*/&fp32_id);
          executor->addDynamicQinput(fp32_id);

          // Define dynamic conversion from fp32 to qdint8
          status = xnn_define_convert(
              /*subgraph=*/subgraph_ptr,
              /*input_id=*/fp32_id,
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
            "define quant tensor (per tensor): buffer_ptr: %p, scale: %f, zp: %u\n",
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
        ET_LOG(
            Debug,
            "define quant tensor (per channel): buffer_ptr: %p, scale.numel(): %u, channel_dim: %u\n",
            buffer_ptr,
            qparams->scale()->size(),
            qparams->channel_dim());
        status = xnn_define_channelwise_quantized_tensor_value(
            /*subgraph=*/subgraph_ptr,
            /*datatype=*/getDataType(tensor_value->datatype()),
            /*scale=*/qparams->scale()->data(),
            /*num_dims=*/tensor_value->num_dims(),
            /*channel_dim*/ qparams->channel_dim(),
            /*dims=*/dims_data.data(),
            /*data=*/buffer_ptr,
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
  // Append this external id to the arg list for execute(*args) to extract from
  // as args[external_id]
  if (tensor_value->external_id() != XNN_INVALID_VALUE_ID) {
    executor->append_arg(tensor_value->external_id());
  }

  return Error::Ok;
};

/*
Define serialized add node into the subgraph, using the remapped ids
to map the serialized ids, to the new ids generated when defining
the tensor value
*/
Error defineAddNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node) noexcept {
  std::pair<float, float> min_max = getOutputMinMax(node);
  auto graph_node = node->xnode_union_as_XNNAdd();
  xnn_status status = xnn_define_add2(
      subgraph_ptr,
      min_max.first,
      min_max.second,
      remapped_ids.at(graph_node->input1_id()),
      remapped_ids.at(graph_node->input2_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());
  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create add node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
};

/*
Define Minimum operator Node into the subgraph
*/
Error defineMinimumNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node) noexcept {
  auto graph_node = node->xnode_union_as_XNNMinimum();
  xnn_status status = xnn_define_minimum2(
      subgraph_ptr,
      remapped_ids.at(graph_node->input1_id()),
      remapped_ids.at(graph_node->input2_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());

  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create minumum node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
};

/*
Define subtract operator Node into the subgraph
*/
Error defineSubtractNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node) noexcept {
  auto graph_node = node->xnode_union_as_XNNSubtract();
  std::pair<float, float> min_max = getOutputMinMax(node);
  xnn_status status = xnn_define_subtract(
      subgraph_ptr,
      min_max.first,
      min_max.second,
      remapped_ids.at(graph_node->input1_id()),
      remapped_ids.at(graph_node->input2_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());

  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create subtract node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
};

/*
Define Multiply operator Node into the subgraph
*/
Error defineMultiplyNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node) noexcept {
  auto graph_node = node->xnode_union_as_XNNMultiply();
  std::pair<float, float> min_max = getOutputMinMax(node);
  xnn_status status = xnn_define_multiply2(
      subgraph_ptr,
      min_max.first,
      min_max.second,
      remapped_ids.at(graph_node->input1_id()),
      remapped_ids.at(graph_node->input2_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());

  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create multiply node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
};

/*
Define Convert operator Node into the subgraph
*/
Error defineConvertNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node) noexcept {
  auto graph_node = node->xnode_union_as_XNNConvert();
  xnn_status status = xnn_define_convert(
      subgraph_ptr,
      remapped_ids.at(graph_node->input_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());

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
    const NodePtr node) noexcept {
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
Define serialized clamp node into the subgraph, using the remapped ids
to map the serialized ids, to the new ids generated when defining
the tensor value
*/
Error defineClampNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node) noexcept {
  std::pair<float, float> min_max = getOutputMinMax(node);
  auto graph_node = node->xnode_union_as_XNNClamp();
  xnn_status status = xnn_define_clamp(
      subgraph_ptr,
      min_max.first,
      min_max.second,
      remapped_ids.at(graph_node->input_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());

  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create hardtanh node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
}

/*
Define serialized softmax node into the subgraph, using the remapped ids
to map the serialized ids, to the new ids generated when defining
the tensor value
*/
Error defineSoftmaxNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node) noexcept {
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

/*
Define serialized sigmoid node into the subgraph, using the remapped ids
to map the serialized ids, to the new ids generated when defining
the tensor value
*/
Error defineSigmoidNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node) noexcept {
  auto graph_node = node->xnode_union_as_XNNSigmoid();
  xnn_status status = xnn_define_sigmoid(
      subgraph_ptr,
      remapped_ids.at(graph_node->input_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());
  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create sigmoid node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
}

/*
Define serialized floor node into the subgraph, using the remapped ids
to map the serialized ids, to the new ids generated when defining
the tensor value
*/
Error defineFloorNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node) noexcept {
  auto graph_node = node->xnode_union_as_XNNFloor();
  xnn_status status = xnn_define_floor(
      subgraph_ptr,
      remapped_ids.at(graph_node->input_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());
  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create floor node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
}

Error defineGlobalAvgPooling2dNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node) noexcept {
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
    const NodePtr node) noexcept {
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
    const NodePtr node) noexcept {
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
Define serialized maxpool2d node into the subgraph, using the remapped ids
to map the serialized ids, to the new ids generated when defining the
tensor value
*/
Error defineMaxPooling2dNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node) noexcept {
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
Define serialized div node into the subgraph
*/
Error defineDivNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node) noexcept {
  auto graph_node = node->xnode_union_as_XNNDiv();
  std::pair<float, float> min_max = getOutputMinMax(node);
  xnn_status status = xnn_define_divide(
      subgraph_ptr,
      min_max.first,
      min_max.second,
      remapped_ids.at(graph_node->input1_id()),
      remapped_ids.at(graph_node->input2_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());
  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create div node %i with code: %s",
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
    const NodePtr node) noexcept {
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
      "Failed to create sigmoid node %i with code: %s",
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
    const NodePtr node) noexcept {
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
    const NodePtr node) noexcept {
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
    const NodePtr node) noexcept {
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
    const NodePtr node) noexcept {
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
    const NodePtr node) noexcept {
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
Define serialized square root node into the subgraph, using the remapped ids
to map the serialized ids, to the new ids generated when defining the
tensor value
*/
Error defineSquareRootNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node) noexcept {
  auto graph_node = node->xnode_union_as_XNNSquareRoot();

  xnn_status status = xnn_define_square_root(
      subgraph_ptr,
      remapped_ids.at(graph_node->input_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());

  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create square root node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
}

/*
Define serialized ceiling node into the subgraph, using the remapped ids
to map the serialized ids, to the new ids generated when defining the
tensor value
*/
Error defineCeilingNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node) noexcept {
  auto graph_node = node->xnode_union_as_XNNCeiling();

  xnn_status status = xnn_define_ceiling(
      subgraph_ptr,
      remapped_ids.at(graph_node->input_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());

  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create ceiling node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
}

/*
Define serialized hardswish node into the subgraph, using the remapped ids
to map the serialized ids, to the new ids generated when defining the
tensor value
*/
Error defineHardswishNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node) noexcept {
  auto graph_node = node->xnode_union_as_XNNHardswish();

  xnn_status status = xnn_define_hardswish(
      subgraph_ptr,
      remapped_ids.at(graph_node->input_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());

  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create hardswish node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
}

/*
Define serialized leaky relu node into the subgraph, using the remapped ids
to map the serialized ids, to the new ids generated when defining the
tensor value
*/
Error defineLeakyReLUNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node) noexcept {
  auto graph_node = node->xnode_union_as_XNNLeakyReLU();

  xnn_status status = xnn_define_leaky_relu(
      subgraph_ptr,
      graph_node->negative_slope(),
      remapped_ids.at(graph_node->input_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());

  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create leaky relu node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
}

/*
Define serialized maximum node into the subgraph, using the remapped ids
to map the serialized ids, to the new ids generated when defining the
tensor value
*/
Error defineMaximumNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node) noexcept {
  auto graph_node = node->xnode_union_as_XNNMaximum();

  xnn_status status = xnn_define_maximum2(
      subgraph_ptr,
      remapped_ids.at(graph_node->input1_id()),
      remapped_ids.at(graph_node->input2_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());

  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create maximum node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
}

/*
Define Negate node into subgraph, using the remapped ids to map the
serialized ids, to the new ids generated when defining the tensor value
*/
Error defineNegateNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node) noexcept {
  auto graph_node = node->xnode_union_as_XNNNegate();

  xnn_status status = xnn_define_negate(
      subgraph_ptr,
      remapped_ids.at(graph_node->input_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());

  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create negate node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
}

/*
Defines square node into subgraph using the remapped ids to map the
serialized ids to the new ids generated when defining the tensor value
*/
Error defineSquareNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node) noexcept {
  auto graph_node = node->xnode_union_as_XNNSquare();

  xnn_status status = xnn_define_square(
      subgraph_ptr,
      remapped_ids.at(graph_node->input_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());

  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create square node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
}

/*
Defines square node into subgraph using the remapped ids to map the
serialized ids to the new ids generated when defining the tensor value
*/
Error defineELUNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node) noexcept {
  auto graph_node = node->xnode_union_as_XNNELU();

  xnn_status status = xnn_define_elu(
      subgraph_ptr,
      graph_node->alpha(),
      remapped_ids.at(graph_node->input_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());

  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create ELU node %i with code: %s",
      node->debug_handle(),
      xnn_status_to_string(status));

  return Error::Ok;
}

/*
Defines absolute value node into subgraph using the remapped ids to map the
serialized ids to the new ids generated when defining the tensor value
*/
Error defineAbsNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node) noexcept {
  auto graph_node = node->xnode_union_as_XNNAbs();

  xnn_status status = xnn_define_abs(
      subgraph_ptr,
      remapped_ids.at(graph_node->input_id()),
      remapped_ids.at(graph_node->output_id()),
      graph_node->flags());

  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Failed to create abs node %i with code: %s",
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
    const NodePtr node) noexcept {
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
    const NodePtr node) noexcept {
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
Defines serialized concatenate2 node into the subgraph,
using the remapped ids to map the serialized ids,
to the new ids generated when defining the tensor value
*/
Error defineConcatenate3Node(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node) noexcept {
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
Defines serialized concatenate2 node into the subgraph,
using the remapped ids to map the serialized ids,
to the new ids generated when defining the tensor value
*/
Error defineConcatenate4Node(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node) noexcept {
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
Defines serialized static_slice node into the subgraph,
using the remapped ids to map the serialized ids,
to the new ids generated when defining the tensor value
*/
Error defineStaticSliceNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node) noexcept {
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
Returns not Implemented Error code. This function is meant to be
called when the compiler encountes a XNodeType from the flatbuffer
that has not yet been implemented
*/
Error defineNotImplementedNode(
    xnn_subgraph_t subgraph_ptr,
    const std::unordered_map<uint32_t, uint32_t>& remapped_ids,
    const NodePtr node) noexcept {
  ET_CHECK_OR_RETURN_ERROR(
      false,
      NotImplemented,
      "Unhandled node type: %s",
      fb_xnnpack::EnumNameXNodeUnion(node->xnode_union_type()));
}

/*
Returns the pointer to the defineNode function that handles the given
XNode type
*/
#define _DEFINE(name)                     \
  case fb_xnnpack::XNodeUnion::XNN##name: \
    return &define##name##Node;

DefineNodeFunc getDefineNodeFunc(fb_xnnpack::XNodeUnion nodeType) {
  switch (nodeType) {
    _DEFINE(Add)
    _DEFINE(FullyConnected)
    _DEFINE(Softmax)
    _DEFINE(Sigmoid)
    _DEFINE(StaticTranspose)
    _DEFINE(Clamp)
    _DEFINE(Conv2d)
    _DEFINE(Div)
    _DEFINE(StaticResizeBilinear2D)
    _DEFINE(StaticConstantPad)
    _DEFINE(AvgPooling2d)
    _DEFINE(Minimum)
    _DEFINE(DepthwiseConv2d)
    _DEFINE(MaxPooling2d)
    _DEFINE(Multiply)
    _DEFINE(Subtract)
    _DEFINE(Floor)
    _DEFINE(Convert)
    _DEFINE(GlobalAvgPooling2d)
    _DEFINE(StaticReshape)
    _DEFINE(ArgMaxPooling2d)
    _DEFINE(SquareRoot)
    _DEFINE(Ceiling)
    _DEFINE(Hardswish)
    _DEFINE(LeakyReLU)
    _DEFINE(Maximum)
    _DEFINE(Negate)
    _DEFINE(Square)
    _DEFINE(ELU)
    _DEFINE(Abs)
    _DEFINE(PReLU)
    _DEFINE(Concatenate2)
    _DEFINE(Concatenate3)
    _DEFINE(Concatenate4)
    _DEFINE(StaticSlice)
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
__ET_NODISCARD Error XNNCompiler::compileModel(
    const void* buffer_pointer,
    size_t num_bytes,
    XNNExecutor* executor,
    MemoryAllocator* runtime_allocator) {
  ET_CHECK_OR_RETURN_ERROR(
      fb_xnnpack::XNNGraphBufferHasIdentifier(buffer_pointer),
      DelegateInvalidCompatibility,
      "XNNPACK Delegate Serialization Format version identifier '%.4s' != expected '%.4s'",
      flatbuffers::GetBufferIdentifier(buffer_pointer),
      fb_xnnpack::XNNGraphIdentifier());

  auto flatbuffer_graph = fb_xnnpack::GetXNNGraph(buffer_pointer);
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

  Error err = Error::Ok;
  for (auto value : *flatbuffer_graph->xvalues()) {
    err = defineTensor(
        subgraph.get(),
        remapped_ids,
        value,
        flatbuffer_graph,
        executor,
        runtime_allocator);

    if (err != Error::Ok) {
      return err;
    }
  }

  for (auto node : *flatbuffer_graph->xnodes()) {
    err = getDefineNodeFunc(node->xnode_union_type())(
        subgraph.get(), remapped_ids, node);
    if (err != Error::Ok) {
      return err;
    }
  }
  uint32_t runtime_flags = 0;

#ifdef ENABLE_XNNPACK_PROFILING
  runtime_flags |= XNN_FLAG_BASIC_PROFILING;
#endif

  xnn_runtime_t runtime_ptr = nullptr;
  status = xnn_create_runtime_v2(
      subgraph.get(),
      torch::executorch::threadpool::get_pthreadpool(),
      runtime_flags,
      &runtime_ptr);
  ET_CHECK_OR_RETURN_ERROR(
      xnn_status_success == status,
      Internal,
      "XNN Runtime creation failed with code: %s",
      xnn_status_to_string(status));

  executor->runtime_ =
      std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)>(
          runtime_ptr, xnn_delete_runtime);

#ifdef ENABLE_XNNPACK_PROFILING
  executor->init_profiler();
#endif
  // HACK FOR FC/BC this is only to support old dq_datatype
  if (executor->qinputs_.size() > 0) {
    // qinputs_ is only set when using the old dq linear path. At which point
    // We need to overide the input_ids_ This workse based off the assumption
    // old dqlinear path will be single node single input delegate
    for (uint32_t id : executor->qinputs_) {
      executor->input_ids_.emplace_back(id);
    }
  } else {
    for (auto old_id : *flatbuffer_graph->input_ids()) {
      executor->input_ids_.emplace_back(remapped_ids.at(old_id));
    }
  }
  // External ids need to be in order for wiring with args
  std::sort(executor->input_ids_.begin(), executor->input_ids_.end());

  for (auto old_id : *flatbuffer_graph->output_ids()) {
    executor->output_ids_.emplace_back(remapped_ids.at(old_id));
  }
  // External ids need to be in order for wiring with args
  std::sort(executor->output_ids_.begin(), executor->output_ids_.end());

  if (!executor->qinputs_.empty() && flatbuffer_graph->xnodes()->size() > 0 &&
      flatbuffer_graph->xnodes()->Get(0)->xnode_union_type() ==
          fb_xnnpack::XNodeUnion::XNNFullyConnected) {
#ifdef ENABLE_DYNAMIC_QUANTIZATION
    // This delegate is for DQLinear which supports dynamic input shapes
    if (executor->getNumInputs() < 1 || executor->getNumOutputs() != 1) {
      ET_LOG(
          Error,
          "DQLinear should have at least one input and exactly one output");
      return Error::NotSupported;
    }
#else
    ET_LOG(Error, "DQ Linear is not supported");
    return Error::NotSupported;
#endif
  }

  return err;
};

} // namespace delegate
} // namespace xnnpack
} // namespace executor
} // namespace torch
