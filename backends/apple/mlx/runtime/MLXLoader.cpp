//
// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// ============================================================================
// AUTO-GENERATED FILE - DO NOT EDIT MANUALLY
// ============================================================================
//
// This file was generated from schema.fbs by the MLX delegate code generator.
//
// Source:    backends/apple/mlx/serialization/schema.fbs
// Generator: backends/apple/mlx/serialization/generate.py
//
// To regenerate, run from the executorch root:
//     python backends/apple/mlx/serialization/generate.py
//
// ============================================================================
//

#include "MLXLoader.h"

#include <cstring>
#include <stdexcept>

namespace executorch {
namespace backends {
namespace mlx {
namespace loader {

namespace {

// Header structure for MLX payload
constexpr size_t kHeaderSize = 24;
constexpr uint32_t kMagic = 0x30584C4D;  // "MLX0" in little-endian

struct MLXHeader {
  uint32_t padding;
  uint32_t magic;
  uint64_t data_offset;
  uint64_t data_size;
};

bool parse_header(const void* data, size_t size, MLXHeader& header) {
  if (size < kHeaderSize) {
    return false;
  }
  std::memcpy(&header, data, sizeof(MLXHeader));
  if (header.magic != kMagic) {
    return false;
  }
  return true;
}

// Helper to convert FlatBuffer vectors to std::vector
template <typename T>
std::vector<T> to_vector(const flatbuffers::Vector<T>* fb_vec) {
  if (!fb_vec) {
    return {};
  }
  return std::vector<T>(fb_vec->begin(), fb_vec->end());
}

}  // namespace

// =============================================================================
// load_instruction - AUTO-GENERATED switch statement
// =============================================================================

Instruction load_instruction(const mlx_delegate::Instruction* fb_instr) {
  Instruction instr;

  if (!fb_instr || !fb_instr->op()) {
    instr.op = OpCode::NOOP;
    instr.node = NoopNode{};
    return instr;
  }

  auto op_type = fb_instr->op_type();

  switch (op_type) {
    case mlx_delegate::OpNode_NoopNode: {
      instr.op = OpCode::NOOP;
      instr.node = NoopNode{};
      break;
    }

    case mlx_delegate::OpNode_LinearNode: {
      auto fb = fb_instr->op_as_LinearNode();
      LinearNode node;
      node.x = convert_tid(fb->x());
      node.weight = convert_tid(fb->weight());
      node.out = convert_tid(fb->out());
      if (fb->bias()) {
        node.bias = convert_tid(fb->bias());
      }
      instr.op = OpCode::LINEAR;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ItemIntNode: {
      auto fb = fb_instr->op_as_ItemIntNode();
      ItemIntNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_vid(fb->out());
      instr.op = OpCode::ITEM_INT;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ExpandDimsNode: {
      auto fb = fb_instr->op_as_ExpandDimsNode();
      ExpandDimsNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.axis = fb->axis();
      instr.op = OpCode::EXPAND_DIMS;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_TileNode: {
      auto fb = fb_instr->op_as_TileNode();
      TileNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::TILE;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_TakeAlongAxisNode: {
      auto fb = fb_instr->op_as_TakeAlongAxisNode();
      TakeAlongAxisNode node;
      node.x = convert_tid(fb->x());
      node.indices = convert_tid(fb->indices());
      node.out = convert_tid(fb->out());
      node.axis = fb->axis();
      instr.op = OpCode::TAKE_ALONG_AXIS;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_RMSNormNode: {
      auto fb = fb_instr->op_as_RMSNormNode();
      RMSNormNode node;
      node.x = convert_tid(fb->x());
      node.weight = convert_tid(fb->weight());
      node.out = convert_tid(fb->out());
      node.eps = fb->eps();
      instr.op = OpCode::RMS_NORM;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_LayerNormNode: {
      auto fb = fb_instr->op_as_LayerNormNode();
      LayerNormNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      if (fb->weight()) {
        node.weight = convert_tid(fb->weight());
      }
      if (fb->bias()) {
        node.bias = convert_tid(fb->bias());
      }
      node.eps = fb->eps();
      instr.op = OpCode::LAYER_NORM;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_RopeNode: {
      auto fb = fb_instr->op_as_RopeNode();
      RopeNode node;
      node.q_in = convert_tid(fb->q_in());
      node.k_in = convert_tid(fb->k_in());
      node.q_out = convert_tid(fb->q_out());
      node.k_out = convert_tid(fb->k_out());
      node.head_dim = fb->head_dim();
      node.pos = convert_vid(fb->pos());
      if (fb->freqs()) {
        node.freqs = convert_tid(fb->freqs());
      }
      node.traditional = fb->traditional();
      if (fb->base_is_set()) {
        node.base = fb->base();
      }
      node.scale = fb->scale();
      instr.op = OpCode::ROPE;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_SdpaNode: {
      auto fb = fb_instr->op_as_SdpaNode();
      SdpaNode node;
      node.q = convert_tid(fb->q());
      node.k = convert_tid(fb->k());
      node.v = convert_tid(fb->v());
      node.out = convert_tid(fb->out());
      node.scale = fb->scale();
      if (fb->mask()) {
        node.mask = convert_tid(fb->mask());
      }
      node.causal = fb->causal();
      instr.op = OpCode::SDPA;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_AddNode: {
      auto fb = fb_instr->op_as_AddNode();
      AddNode node;
      node.a = convert_tid(fb->a());
      node.b = convert_tid(fb->b());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::ADD;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_AddScalarNode: {
      auto fb = fb_instr->op_as_AddScalarNode();
      AddScalarNode node;
      node.a = convert_int_or_vid(fb->a());
      node.b = convert_int_or_vid(fb->b());
      node.out = convert_vid(fb->out());
      instr.op = OpCode::ADD_SCALAR;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_SymSizeNode: {
      auto fb = fb_instr->op_as_SymSizeNode();
      SymSizeNode node;
      node.a = convert_tid(fb->a());
      node.dim = fb->dim();
      node.out = convert_vid(fb->out());
      instr.op = OpCode::SYM_SIZE;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_MulNode: {
      auto fb = fb_instr->op_as_MulNode();
      MulNode node;
      node.a = convert_tid(fb->a());
      node.b = convert_tid(fb->b());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::MUL;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_Conv1DNode: {
      auto fb = fb_instr->op_as_Conv1DNode();
      Conv1DNode node;
      node.x = convert_tid(fb->x());
      node.w = convert_tid(fb->w());
      node.out = convert_tid(fb->out());
      node.stride = fb->stride();
      node.padding = fb->padding();
      node.dilation = fb->dilation();
      node.groups = fb->groups();
      instr.op = OpCode::CONV1D;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_GeluNode: {
      auto fb = fb_instr->op_as_GeluNode();
      GeluNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::GELU;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ARangeNode: {
      auto fb = fb_instr->op_as_ARangeNode();
      ARangeNode node;
      node.out = convert_tid(fb->out());
      node.start = fb->start();
      node.stop = fb->stop();
      node.step = fb->step();
      if (fb->dtype_is_set()) {
        node.dtype = convert_dtype(fb->dtype());
      }
      instr.op = OpCode::ARANGE;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_SiluNode: {
      auto fb = fb_instr->op_as_SiluNode();
      SiluNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::SILU;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ReshapeNode: {
      auto fb = fb_instr->op_as_ReshapeNode();
      ReshapeNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::RESHAPE;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_TransposeNode: {
      auto fb = fb_instr->op_as_TransposeNode();
      TransposeNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::TRANSPOSE;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ContiguousNode: {
      auto fb = fb_instr->op_as_ContiguousNode();
      ContiguousNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::CONTIGUOUS;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_IdCopyNode: {
      auto fb = fb_instr->op_as_IdCopyNode();
      IdCopyNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::ID_COPY;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_GatherNode: {
      auto fb = fb_instr->op_as_GatherNode();
      GatherNode node;
      node.table_ = convert_tid(fb->table_());
      node.ids = convert_tid(fb->ids());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::GATHER;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_SliceNode: {
      auto fb = fb_instr->op_as_SliceNode();
      SliceNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.axis = convert_int_or_vid(fb->axis());
      node.start = convert_int_or_vid(fb->start());
      node.end = convert_int_or_vid(fb->end());
      instr.op = OpCode::SLICE;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_CastNode: {
      auto fb = fb_instr->op_as_CastNode();
      CastNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.dtype = convert_dtype(fb->dtype());
      instr.op = OpCode::CAST;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_QuantizedLinearNode: {
      auto fb = fb_instr->op_as_QuantizedLinearNode();
      QuantizedLinearNode node;
      node.x = convert_tid(fb->x());
      node.w = convert_tid(fb->w());
      node.scales = convert_tid(fb->scales());
      node.out = convert_tid(fb->out());
      if (fb->biases()) {
        node.biases = convert_tid(fb->biases());
      }
      if (fb->bias()) {
        node.bias = convert_tid(fb->bias());
      }
      node.group_size = fb->group_size();
      node.bits = fb->bits();
      node.mode = fb->mode() ? fb->mode()->str() : "";
      node.out_dtype = convert_dtype(fb->out_dtype());
      instr.op = OpCode::QUANTIZED_LINEAR;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ConcatNode: {
      auto fb = fb_instr->op_as_ConcatNode();
      ConcatNode node;
      node.a = convert_tid(fb->a());
      node.b = convert_tid(fb->b());
      node.out = convert_tid(fb->out());
      node.axis = fb->axis();
      instr.op = OpCode::CONCAT;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_FullNode: {
      auto fb = fb_instr->op_as_FullNode();
      FullNode node;
      node.out = convert_tid(fb->out());
      node.v = fb->v();
      node.dtype = convert_dtype(fb->dtype());
      instr.op = OpCode::FULL;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ZerosNode: {
      auto fb = fb_instr->op_as_ZerosNode();
      ZerosNode node;
      node.out = convert_tid(fb->out());
      node.dtype = convert_dtype(fb->dtype());
      instr.op = OpCode::ZEROS;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_OnesNode: {
      auto fb = fb_instr->op_as_OnesNode();
      OnesNode node;
      node.out = convert_tid(fb->out());
      node.dtype = convert_dtype(fb->dtype());
      instr.op = OpCode::ONES;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ArgmaxNode: {
      auto fb = fb_instr->op_as_ArgmaxNode();
      ArgmaxNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.axis = fb->axis();
      instr.op = OpCode::ARGMAX;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_SliceUpdateNode: {
      auto fb = fb_instr->op_as_SliceUpdateNode();
      SliceUpdateNode node;
      node.dst = convert_tid(fb->dst());
      node.update = convert_tid(fb->update());
      node.axis = convert_int_or_vid(fb->axis());
      node.start = convert_int_or_vid(fb->start());
      node.stop = convert_int_or_vid(fb->stop());
      instr.op = OpCode::SLICE_UPDATE;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_QuantizedGatherNode: {
      auto fb = fb_instr->op_as_QuantizedGatherNode();
      QuantizedGatherNode node;
      node.table_q = convert_tid(fb->table_q());
      node.scales = convert_tid(fb->scales());
      node.ids = convert_tid(fb->ids());
      node.out = convert_tid(fb->out());
      if (fb->biases()) {
        node.biases = convert_tid(fb->biases());
      }
      node.group_size = fb->group_size();
      node.bits = fb->bits();
      node.mode = fb->mode() ? fb->mode()->str() : "";
      node.out_dtype = convert_dtype(fb->out_dtype());
      instr.op = OpCode::QUANTIZED_GATHER;
      instr.node = std::move(node);
      break;
    }

    default: {
      instr.op = OpCode::NOOP;
      instr.node = NoopNode{};
      break;
    }
  }

  return instr;
}

// =============================================================================
// load_program
// =============================================================================

MLXProgram load_program(const void* data, size_t size) {
  MLXHeader header;
  if (!parse_header(data, size, header)) {
    throw std::runtime_error("Invalid MLX header");
  }

  const uint8_t* fb_data = static_cast<const uint8_t*>(data) + kHeaderSize;
  size_t fb_size = header.data_offset - kHeaderSize;

  flatbuffers::Verifier verifier(fb_data, fb_size);
  if (!mlx_delegate::VerifyMLXGraphBuffer(verifier)) {
    throw std::runtime_error("Invalid FlatBuffer data");
  }

  const auto* fb_graph = mlx_delegate::GetMLXGraph(fb_data);
  if (!fb_graph) {
    throw std::runtime_error("Failed to parse MLXGraph");
  }

  MLXProgram program;

  if (fb_graph->version()) {
    program.version = fb_graph->version()->str();
  }

  program.num_constant_tensors = fb_graph->num_constant_tensors();
  program.num_non_constant_tensors = fb_graph->num_non_constant_tensors();
  program.num_non_constant_values = fb_graph->num_non_constant_values();

  if (fb_graph->instructions()) {
    program.instructions.reserve(fb_graph->instructions()->size());
    for (size_t i = 0; i < fb_graph->instructions()->size(); ++i) {
      const auto* fb_instr = fb_graph->instructions()->Get(i);
      program.instructions.push_back(load_instruction(fb_instr));
    }
  }

  if (fb_graph->input_map()) {
    for (size_t i = 0; i < fb_graph->input_map()->size(); ++i) {
      const auto* slot = fb_graph->input_map()->Get(i);
      program.input_map.push_back(convert_slot_variant(slot));
    }
  }

  if (fb_graph->output_map()) {
    for (size_t i = 0; i < fb_graph->output_map()->size(); ++i) {
      const auto* slot = fb_graph->output_map()->Get(i);
      program.output_map.push_back(convert_slot_variant(slot));
    }
  }

  if (fb_graph->mutable_buffer_map()) {
    for (size_t i = 0; i < fb_graph->mutable_buffer_map()->size(); ++i) {
      const auto* slot = fb_graph->mutable_buffer_map()->Get(i);
      program.mutable_buffer_map.push_back(convert_slot_variant(slot));
    }
  }

  if (fb_graph->named_slots()) {
    for (size_t i = 0; i < fb_graph->named_slots()->size(); ++i) {
      const auto* fb_slot = fb_graph->named_slots()->Get(i);
      NamedSlot slot;
      slot.name = fb_slot->name() ? fb_slot->name()->str() : "";
      slot.slot = convert_slot_variant(fb_slot->slot());
      program.named_slots.push_back(std::move(slot));
    }
  }

  if (fb_graph->tensor_meta()) {
    for (size_t i = 0; i < fb_graph->tensor_meta()->size(); ++i) {
      const auto* fb_meta = fb_graph->tensor_meta()->Get(i);
      if (fb_meta) {
        TensorMeta meta;
        if (fb_meta->shape()) {
          for (size_t j = 0; j < fb_meta->shape()->size(); ++j) {
            const auto* iov = fb_meta->shape()->Get(j);
            meta.shape.push_back(convert_int_or_vid(iov));
          }
        }
        meta.dtype = convert_dtype(fb_meta->dtype());
        meta.strides = to_vector(fb_meta->strides());
        program.tensor_meta.push_back(std::move(meta));
      } else {
        program.tensor_meta.push_back(std::nullopt);
      }
    }
  }

  if (fb_graph->constant_segment()) {
    program.constant_segment.offset = fb_graph->constant_segment()->offset();
    program.constant_segment.size = fb_graph->constant_segment()->size();
  }

  program.constant_data =
      static_cast<const uint8_t*>(data) + header.data_offset;

  return program;
}

}  // namespace loader
}  // namespace mlx
}  // namespace backends
}  // namespace executorch