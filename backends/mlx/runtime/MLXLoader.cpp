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
// Source:    backends/mlx/serialization/schema.fbs
// Generator: backends/mlx/serialization/generate.py
//
// To regenerate, run from the executorch root:
//     python backends/mlx/serialization/generate.py
//
// ============================================================================
// -*- c++ -*-

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
static_assert(sizeof(MLXHeader) == kHeaderSize, "MLXHeader size mismatch");

bool parse_header(const void* data, size_t size, MLXHeader& header) {
  if (size < kHeaderSize) {
    return false;
  }
  std::memcpy(&header, data, sizeof(MLXHeader));
  if (header.magic != kMagic) {
    return false;
  }
  // Validate data_offset: must be strictly greater than kHeaderSize (so the
  // FlatBuffer region is non-empty) and must not exceed the total buffer size.
  if (header.data_offset <= kHeaderSize || header.data_offset > size) {
    return false;
  }
  return true;
}

// Helper to convert FlatBuffer vectors to std::vector.
// Caps size to prevent unbounded allocations from malformed payloads.
template <typename T>
std::vector<T> to_vector(const flatbuffers::Vector<T>* fb_vec) {
  if (!fb_vec) {
    return {};
  }
  constexpr size_t kMaxVectorSize = 1'000'000;
  if (fb_vec->size() > kMaxVectorSize) {
    throw std::runtime_error(
        "FlatBuffer vector size " + std::to_string(fb_vec->size()) +
        " exceeds maximum of " + std::to_string(kMaxVectorSize));
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

    case mlx_delegate::OpNode_IdCopyNode: {
      auto fb = fb_instr->op_as_IdCopyNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      IdCopyNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::ID_COPY;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_AddmmNode: {
      auto fb = fb_instr->op_as_AddmmNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      AddmmNode node;
      node.mat1 = convert_tid(fb->mat1());
      node.mat2 = convert_tid(fb->mat2());
      node.out = convert_tid(fb->out());
      if (fb->bias()) {
        node.bias = convert_tid(fb->bias());
      }
      node.alpha = fb->alpha();
      node.beta = fb->beta();
      instr.op = OpCode::ADDMM;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ItemIntNode: {
      auto fb = fb_instr->op_as_ItemIntNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      ItemIntNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_vid(fb->out());
      instr.op = OpCode::ITEM_INT;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ExpandDimsNode: {
      auto fb = fb_instr->op_as_ExpandDimsNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
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
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      TileNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      if (fb->reps()) {
        for (size_t i = 0; i < fb->reps()->size(); ++i) {
          node.reps.push_back(convert_int_or_vid(fb->reps()->Get(static_cast<flatbuffers::uoffset_t>(i))));
        }
      }
      instr.op = OpCode::TILE;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_TakeAlongAxisNode: {
      auto fb = fb_instr->op_as_TakeAlongAxisNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      TakeAlongAxisNode node;
      node.x = convert_tid(fb->x());
      node.indices = convert_tid(fb->indices());
      node.out = convert_tid(fb->out());
      node.axis = fb->axis();
      instr.op = OpCode::TAKE_ALONG_AXIS;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_TakeNode: {
      auto fb = fb_instr->op_as_TakeNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      TakeNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.index = convert_int_or_vid_or_tid(fb->index());
      node.axis = fb->axis();
      instr.op = OpCode::TAKE;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_RMSNormNode: {
      auto fb = fb_instr->op_as_RMSNormNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      RMSNormNode node;
      node.x = convert_tid(fb->x());
      if (fb->weight()) {
        node.weight = convert_tid(fb->weight());
      }
      node.out = convert_tid(fb->out());
      node.eps = fb->eps();
      instr.op = OpCode::RMS_NORM;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_LayerNormNode: {
      auto fb = fb_instr->op_as_LayerNormNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
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
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      RopeNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.dims = fb->dims();
      node.offset = convert_vid_or_tid(fb->offset());
      if (fb->freqs()) {
        node.freqs = convert_tid(fb->freqs());
      }
      node.traditional = fb->traditional();
      node.base = fb->base();
      node.scale = fb->scale();
      instr.op = OpCode::ROPE;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_SdpaNode: {
      auto fb = fb_instr->op_as_SdpaNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
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
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      AddNode node;
      node.a = convert_tid(fb->a());
      node.b = convert_tid(fb->b());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::ADD;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_AddIntNode: {
      auto fb = fb_instr->op_as_AddIntNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      AddIntNode node;
      node.a = convert_int_or_vid(fb->a());
      node.b = convert_int_or_vid(fb->b());
      node.out = convert_vid(fb->out());
      instr.op = OpCode::ADD_INT;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_SubtractIntNode: {
      auto fb = fb_instr->op_as_SubtractIntNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      SubtractIntNode node;
      node.a = convert_int_or_vid(fb->a());
      node.b = convert_int_or_vid(fb->b());
      node.out = convert_vid(fb->out());
      instr.op = OpCode::SUBTRACT_INT;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_MultiplyIntNode: {
      auto fb = fb_instr->op_as_MultiplyIntNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      MultiplyIntNode node;
      node.a = convert_int_or_vid(fb->a());
      node.b = convert_int_or_vid(fb->b());
      node.out = convert_vid(fb->out());
      instr.op = OpCode::MULTIPLY_INT;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_FloorDivideIntNode: {
      auto fb = fb_instr->op_as_FloorDivideIntNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      FloorDivideIntNode node;
      node.a = convert_int_or_vid(fb->a());
      node.b = convert_int_or_vid(fb->b());
      node.out = convert_vid(fb->out());
      instr.op = OpCode::FLOOR_DIVIDE_INT;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ModIntNode: {
      auto fb = fb_instr->op_as_ModIntNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      ModIntNode node;
      node.a = convert_int_or_vid(fb->a());
      node.b = convert_int_or_vid(fb->b());
      node.out = convert_vid(fb->out());
      instr.op = OpCode::MOD_INT;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_SymSizeNode: {
      auto fb = fb_instr->op_as_SymSizeNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      SymSizeNode node;
      node.a = convert_tid(fb->a());
      node.dim = fb->dim();
      node.out = convert_vid(fb->out());
      instr.op = OpCode::SYM_SIZE;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_MultiplyNode: {
      auto fb = fb_instr->op_as_MultiplyNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      MultiplyNode node;
      node.a = convert_tid(fb->a());
      node.b = convert_tid(fb->b());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::MULTIPLY;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_DivideNode: {
      auto fb = fb_instr->op_as_DivideNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      DivideNode node;
      node.a = convert_tid(fb->a());
      node.b = convert_tid(fb->b());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::DIVIDE;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_SubtractNode: {
      auto fb = fb_instr->op_as_SubtractNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      SubtractNode node;
      node.a = convert_tid(fb->a());
      node.b = convert_tid(fb->b());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::SUBTRACT;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_Conv1DNode: {
      auto fb = fb_instr->op_as_Conv1DNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
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

    case mlx_delegate::OpNode_Conv2DNode: {
      auto fb = fb_instr->op_as_Conv2DNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      Conv2DNode node;
      node.x = convert_tid(fb->x());
      node.w = convert_tid(fb->w());
      node.out = convert_tid(fb->out());
      node.stride_h = fb->stride_h();
      node.stride_w = fb->stride_w();
      node.padding_h = fb->padding_h();
      node.padding_w = fb->padding_w();
      node.dilation_h = fb->dilation_h();
      node.dilation_w = fb->dilation_w();
      node.groups = fb->groups();
      instr.op = OpCode::CONV2D;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_Conv3DNode: {
      auto fb = fb_instr->op_as_Conv3DNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      Conv3DNode node;
      node.x = convert_tid(fb->x());
      node.w = convert_tid(fb->w());
      node.out = convert_tid(fb->out());
      node.stride_d = fb->stride_d();
      node.stride_h = fb->stride_h();
      node.stride_w = fb->stride_w();
      node.padding_d = fb->padding_d();
      node.padding_h = fb->padding_h();
      node.padding_w = fb->padding_w();
      node.dilation_d = fb->dilation_d();
      node.dilation_h = fb->dilation_h();
      node.dilation_w = fb->dilation_w();
      node.groups = fb->groups();
      instr.op = OpCode::CONV3D;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ConvTranspose1DNode: {
      auto fb = fb_instr->op_as_ConvTranspose1DNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      ConvTranspose1DNode node;
      node.x = convert_tid(fb->x());
      node.w = convert_tid(fb->w());
      node.out = convert_tid(fb->out());
      node.stride = fb->stride();
      node.padding = fb->padding();
      node.dilation = fb->dilation();
      node.output_padding = fb->output_padding();
      node.groups = fb->groups();
      instr.op = OpCode::CONV_TRANSPOSE1D;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ConvTranspose2DNode: {
      auto fb = fb_instr->op_as_ConvTranspose2DNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      ConvTranspose2DNode node;
      node.x = convert_tid(fb->x());
      node.w = convert_tid(fb->w());
      node.out = convert_tid(fb->out());
      node.stride_h = fb->stride_h();
      node.stride_w = fb->stride_w();
      node.padding_h = fb->padding_h();
      node.padding_w = fb->padding_w();
      node.dilation_h = fb->dilation_h();
      node.dilation_w = fb->dilation_w();
      node.output_padding_h = fb->output_padding_h();
      node.output_padding_w = fb->output_padding_w();
      node.groups = fb->groups();
      instr.op = OpCode::CONV_TRANSPOSE2D;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ConvTranspose3DNode: {
      auto fb = fb_instr->op_as_ConvTranspose3DNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      ConvTranspose3DNode node;
      node.x = convert_tid(fb->x());
      node.w = convert_tid(fb->w());
      node.out = convert_tid(fb->out());
      node.stride_d = fb->stride_d();
      node.stride_h = fb->stride_h();
      node.stride_w = fb->stride_w();
      node.padding_d = fb->padding_d();
      node.padding_h = fb->padding_h();
      node.padding_w = fb->padding_w();
      node.dilation_d = fb->dilation_d();
      node.dilation_h = fb->dilation_h();
      node.dilation_w = fb->dilation_w();
      node.output_padding_d = fb->output_padding_d();
      node.output_padding_h = fb->output_padding_h();
      node.output_padding_w = fb->output_padding_w();
      node.groups = fb->groups();
      instr.op = OpCode::CONV_TRANSPOSE3D;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_GeluNode: {
      auto fb = fb_instr->op_as_GeluNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      GeluNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.approximate = fb->approximate() ? fb->approximate()->str() : "";
      instr.op = OpCode::GELU;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ARangeNode: {
      auto fb = fb_instr->op_as_ARangeNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      ARangeNode node;
      node.out = convert_tid(fb->out());
      node.start = convert_int_or_vid(fb->start());
      node.stop = convert_int_or_vid(fb->stop());
      node.step = convert_int_or_vid(fb->step());
      auto scalar_type_opt = fb->scalar_type();
      if (scalar_type_opt.has_value()) {
        node.scalar_type = scalar_type_opt.value();
      }
      instr.op = OpCode::ARANGE;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_SiluNode: {
      auto fb = fb_instr->op_as_SiluNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      SiluNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::SILU;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_SigmoidNode: {
      auto fb = fb_instr->op_as_SigmoidNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      SigmoidNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::SIGMOID;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_TanhNode: {
      auto fb = fb_instr->op_as_TanhNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      TanhNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::TANH;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_SqueezeNode: {
      auto fb = fb_instr->op_as_SqueezeNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      SqueezeNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.dims = to_vector(fb->dims());
      instr.op = OpCode::SQUEEZE;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_SplitNode: {
      auto fb = fb_instr->op_as_SplitNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      SplitNode node;
      node.x = convert_tid(fb->x());
      if (fb->outs()) {
        for (auto fb_tid : *fb->outs()) {
          node.outs.push_back(convert_tid(fb_tid));
        }
      }
      if (fb->sizes()) {
        for (size_t i = 0; i < fb->sizes()->size(); ++i) {
          node.sizes.push_back(convert_int_or_vid(fb->sizes()->Get(static_cast<flatbuffers::uoffset_t>(i))));
        }
      }
      node.axis = fb->axis();
      instr.op = OpCode::SPLIT;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_RsqrtNode: {
      auto fb = fb_instr->op_as_RsqrtNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      RsqrtNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::RSQRT;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_MaximumNode: {
      auto fb = fb_instr->op_as_MaximumNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      MaximumNode node;
      node.a = convert_tid(fb->a());
      node.b = convert_tid(fb->b());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::MAXIMUM;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_MinimumNode: {
      auto fb = fb_instr->op_as_MinimumNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      MinimumNode node;
      node.a = convert_tid(fb->a());
      node.b = convert_tid(fb->b());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::MINIMUM;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_LogNode: {
      auto fb = fb_instr->op_as_LogNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      LogNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::LOG;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_SoftmaxNode: {
      auto fb = fb_instr->op_as_SoftmaxNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      SoftmaxNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.axis = fb->axis();
      node.precise = fb->precise();
      instr.op = OpCode::SOFTMAX;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_BroadcastToNode: {
      auto fb = fb_instr->op_as_BroadcastToNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      BroadcastToNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      if (fb->shape()) {
        for (size_t i = 0; i < fb->shape()->size(); ++i) {
          node.shape.push_back(convert_int_or_vid(fb->shape()->Get(static_cast<flatbuffers::uoffset_t>(i))));
        }
      }
      instr.op = OpCode::BROADCAST_TO;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_PadNode: {
      auto fb = fb_instr->op_as_PadNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      PadNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      if (fb->pad_width()) {
        for (size_t i = 0; i < fb->pad_width()->size(); ++i) {
          node.pad_width.push_back(convert_int_or_vid(fb->pad_width()->Get(static_cast<flatbuffers::uoffset_t>(i))));
        }
      }
      node.mode = fb->mode() ? fb->mode()->str() : "";
      node.constant_value = fb->constant_value();
      instr.op = OpCode::PAD;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_WhereNode: {
      auto fb = fb_instr->op_as_WhereNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      WhereNode node;
      node.condition = convert_tid(fb->condition());
      node.x = convert_tid(fb->x());
      node.y = convert_tid(fb->y());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::WHERE;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ReshapeNode: {
      auto fb = fb_instr->op_as_ReshapeNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      ReshapeNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      if (fb->shape()) {
        for (size_t i = 0; i < fb->shape()->size(); ++i) {
          node.shape.push_back(convert_int_or_vid(fb->shape()->Get(static_cast<flatbuffers::uoffset_t>(i))));
        }
      }
      instr.op = OpCode::RESHAPE;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_TransposeNode: {
      auto fb = fb_instr->op_as_TransposeNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      TransposeNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.perm = to_vector(fb->perm());
      instr.op = OpCode::TRANSPOSE;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_AsStridedNode: {
      auto fb = fb_instr->op_as_AsStridedNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      AsStridedNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      if (fb->shape()) {
        for (size_t i = 0; i < fb->shape()->size(); ++i) {
          node.shape.push_back(convert_int_or_vid(fb->shape()->Get(static_cast<flatbuffers::uoffset_t>(i))));
        }
      }
      if (fb->strides()) {
        for (size_t i = 0; i < fb->strides()->size(); ++i) {
          node.strides.push_back(convert_int_or_vid(fb->strides()->Get(static_cast<flatbuffers::uoffset_t>(i))));
        }
      }
      node.offset = fb->offset();
      instr.op = OpCode::AS_STRIDED;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ContiguousNode: {
      auto fb = fb_instr->op_as_ContiguousNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      ContiguousNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::CONTIGUOUS;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_GatherNode: {
      auto fb = fb_instr->op_as_GatherNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      GatherNode node;
      node.x = convert_tid(fb->x());
      if (fb->indices()) {
        for (auto fb_tid : *fb->indices()) {
          node.indices.push_back(convert_tid(fb_tid));
        }
      }
      node.out = convert_tid(fb->out());
      node.axes = to_vector(fb->axes());
      node.slice_sizes = to_vector(fb->slice_sizes());
      instr.op = OpCode::GATHER;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_SliceNode: {
      auto fb = fb_instr->op_as_SliceNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      SliceNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.axis = convert_int_or_vid(fb->axis());
      node.start = convert_int_or_vid(fb->start());
      node.stop = convert_int_or_vid(fb->stop());
      node.step = fb->step();
      instr.op = OpCode::SLICE;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_AsTypeNode: {
      auto fb = fb_instr->op_as_AsTypeNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      AsTypeNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.scalar_type = fb->scalar_type();
      instr.op = OpCode::ASTYPE;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_QuantizedMatmulNode: {
      auto fb = fb_instr->op_as_QuantizedMatmulNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      QuantizedMatmulNode node;
      node.x = convert_tid(fb->x());
      node.w = convert_tid(fb->w());
      node.scales = convert_tid(fb->scales());
      node.out = convert_tid(fb->out());
      if (fb->biases()) {
        node.biases = convert_tid(fb->biases());
      }
      node.group_size = fb->group_size();
      node.bits = fb->bits();
      node.mode = fb->mode() ? fb->mode()->str() : "";
      node.transpose = fb->transpose();
      instr.op = OpCode::QUANTIZED_MATMUL;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ScatterAddNode: {
      auto fb = fb_instr->op_as_ScatterAddNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      ScatterAddNode node;
      node.x = convert_tid(fb->x());
      node.indices = convert_tid(fb->indices());
      node.updates = convert_tid(fb->updates());
      node.out = convert_tid(fb->out());
      node.axis = fb->axis();
      instr.op = OpCode::SCATTER_ADD;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ConcatenateNode: {
      auto fb = fb_instr->op_as_ConcatenateNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      ConcatenateNode node;
      if (fb->tensors()) {
        for (auto fb_tid : *fb->tensors()) {
          node.tensors.push_back(convert_tid(fb_tid));
        }
      }
      node.out = convert_tid(fb->out());
      node.axis = fb->axis();
      instr.op = OpCode::CONCATENATE;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_FullNode: {
      auto fb = fb_instr->op_as_FullNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      FullNode node;
      node.out = convert_tid(fb->out());
      if (fb->shape()) {
        for (size_t i = 0; i < fb->shape()->size(); ++i) {
          node.shape.push_back(convert_int_or_vid(fb->shape()->Get(static_cast<flatbuffers::uoffset_t>(i))));
        }
      }
      node.v = convert_float_or_vid(fb->v());
      node.scalar_type = fb->scalar_type();
      instr.op = OpCode::FULL;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_FullLikeNode: {
      auto fb = fb_instr->op_as_FullLikeNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      FullLikeNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.v = convert_float_or_vid(fb->v());
      auto scalar_type_opt = fb->scalar_type();
      if (scalar_type_opt.has_value()) {
        node.scalar_type = scalar_type_opt.value();
      }
      instr.op = OpCode::FULL_LIKE;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ArgmaxNode: {
      auto fb = fb_instr->op_as_ArgmaxNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      ArgmaxNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.axis = fb->axis();
      node.keepdims = fb->keepdims();
      instr.op = OpCode::ARGMAX;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_SliceUpdateNode: {
      auto fb = fb_instr->op_as_SliceUpdateNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      SliceUpdateNode node;
      node.dst = convert_tid(fb->dst());
      node.update = convert_tid(fb->update());
      node.out = convert_tid(fb->out());
      node.axis = convert_int_or_vid(fb->axis());
      node.start = convert_int_or_vid(fb->start());
      node.stop = convert_int_or_vid(fb->stop());
      node.step = fb->step();
      instr.op = OpCode::SLICE_UPDATE;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_IndexCopyNode: {
      auto fb = fb_instr->op_as_IndexCopyNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      IndexCopyNode node;
      node.dst = convert_tid(fb->dst());
      node.update = convert_tid(fb->update());
      node.indices = convert_tid(fb->indices());
      node.out = convert_tid(fb->out());
      node.axis = fb->axis();
      instr.op = OpCode::INDEX_COPY;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_DequantizeNode: {
      auto fb = fb_instr->op_as_DequantizeNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      DequantizeNode node;
      node.w = convert_tid(fb->w());
      node.scales = convert_tid(fb->scales());
      node.out = convert_tid(fb->out());
      if (fb->biases()) {
        node.biases = convert_tid(fb->biases());
      }
      node.group_size = fb->group_size();
      node.bits = fb->bits();
      node.mode = fb->mode() ? fb->mode()->str() : "";
      if (fb->global_scale()) {
        node.global_scale = convert_tid(fb->global_scale());
      }
      auto dtype_opt = fb->dtype();
      if (dtype_opt.has_value()) {
        node.dtype = dtype_opt.value();
      }
      instr.op = OpCode::DEQUANTIZE;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_LessNode: {
      auto fb = fb_instr->op_as_LessNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      LessNode node;
      node.a = convert_tid(fb->a());
      node.b = convert_tid(fb->b());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::LESS;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_LessEqualNode: {
      auto fb = fb_instr->op_as_LessEqualNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      LessEqualNode node;
      node.a = convert_tid(fb->a());
      node.b = convert_tid(fb->b());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::LESS_EQUAL;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_GreaterNode: {
      auto fb = fb_instr->op_as_GreaterNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      GreaterNode node;
      node.a = convert_tid(fb->a());
      node.b = convert_tid(fb->b());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::GREATER;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_GreaterEqualNode: {
      auto fb = fb_instr->op_as_GreaterEqualNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      GreaterEqualNode node;
      node.a = convert_tid(fb->a());
      node.b = convert_tid(fb->b());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::GREATER_EQUAL;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_EqualNode: {
      auto fb = fb_instr->op_as_EqualNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      EqualNode node;
      node.a = convert_tid(fb->a());
      node.b = convert_tid(fb->b());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::EQUAL;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_NotEqualNode: {
      auto fb = fb_instr->op_as_NotEqualNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      NotEqualNode node;
      node.a = convert_tid(fb->a());
      node.b = convert_tid(fb->b());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::NOT_EQUAL;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_LogicalNotNode: {
      auto fb = fb_instr->op_as_LogicalNotNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      LogicalNotNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::LOGICAL_NOT;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_LogicalAndNode: {
      auto fb = fb_instr->op_as_LogicalAndNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      LogicalAndNode node;
      node.a = convert_tid(fb->a());
      node.b = convert_tid(fb->b());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::LOGICAL_AND;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_LogicalOrNode: {
      auto fb = fb_instr->op_as_LogicalOrNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      LogicalOrNode node;
      node.a = convert_tid(fb->a());
      node.b = convert_tid(fb->b());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::LOGICAL_OR;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_TriNode: {
      auto fb = fb_instr->op_as_TriNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      TriNode node;
      node.out = convert_tid(fb->out());
      node.n = convert_int_or_vid(fb->n());
      node.m = convert_int_or_vid(fb->m());
      node.k = fb->k();
      node.scalar_type = fb->scalar_type();
      instr.op = OpCode::TRI;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_TrilNode: {
      auto fb = fb_instr->op_as_TrilNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      TrilNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.k = fb->k();
      instr.op = OpCode::TRIL;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_TriuNode: {
      auto fb = fb_instr->op_as_TriuNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      TriuNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.k = fb->k();
      instr.op = OpCode::TRIU;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ClipNode: {
      auto fb = fb_instr->op_as_ClipNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      ClipNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      if (fb->a_min()) {
        node.a_min = convert_tid(fb->a_min());
      }
      if (fb->a_max()) {
        node.a_max = convert_tid(fb->a_max());
      }
      instr.op = OpCode::CLIP;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_CumsumNode: {
      auto fb = fb_instr->op_as_CumsumNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      CumsumNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.axis = fb->axis();
      node.reverse = fb->reverse();
      node.inclusive = fb->inclusive();
      instr.op = OpCode::CUMSUM;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_StackNode: {
      auto fb = fb_instr->op_as_StackNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      StackNode node;
      if (fb->tensors()) {
        for (auto fb_tid : *fb->tensors()) {
          node.tensors.push_back(convert_tid(fb_tid));
        }
      }
      node.out = convert_tid(fb->out());
      node.axis = fb->axis();
      instr.op = OpCode::STACK;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_SignNode: {
      auto fb = fb_instr->op_as_SignNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      SignNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::SIGN;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_AnyNode: {
      auto fb = fb_instr->op_as_AnyNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      AnyNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.axes = to_vector(fb->axes());
      node.keepdims = fb->keepdims();
      instr.op = OpCode::ANY;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_AllNode: {
      auto fb = fb_instr->op_as_AllNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      AllNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.axes = to_vector(fb->axes());
      node.keepdims = fb->keepdims();
      instr.op = OpCode::ALL;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_RepeatNode: {
      auto fb = fb_instr->op_as_RepeatNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      RepeatNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.repeats = convert_int_or_vid(fb->repeats());
      node.axis = fb->axis();
      instr.op = OpCode::REPEAT;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_SortNode: {
      auto fb = fb_instr->op_as_SortNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      SortNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.axis = fb->axis();
      instr.op = OpCode::SORT;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ArgsortNode: {
      auto fb = fb_instr->op_as_ArgsortNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      ArgsortNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.axis = fb->axis();
      instr.op = OpCode::ARGSORT;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_PartitionNode: {
      auto fb = fb_instr->op_as_PartitionNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      PartitionNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.kth = convert_int_or_vid(fb->kth());
      node.axis = fb->axis();
      instr.op = OpCode::PARTITION;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ArgPartitionNode: {
      auto fb = fb_instr->op_as_ArgPartitionNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      ArgPartitionNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.kth = convert_int_or_vid(fb->kth());
      node.axis = fb->axis();
      instr.op = OpCode::ARG_PARTITION;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_FloorNode: {
      auto fb = fb_instr->op_as_FloorNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      FloorNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::FLOOR;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_CeilNode: {
      auto fb = fb_instr->op_as_CeilNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      CeilNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::CEIL;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_SquareNode: {
      auto fb = fb_instr->op_as_SquareNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      SquareNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::SQUARE;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ExpNode: {
      auto fb = fb_instr->op_as_ExpNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      ExpNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::EXP;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_SinNode: {
      auto fb = fb_instr->op_as_SinNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      SinNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::SIN;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_CosNode: {
      auto fb = fb_instr->op_as_CosNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      CosNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::COS;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_TanNode: {
      auto fb = fb_instr->op_as_TanNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      TanNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::TAN;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ArcsinNode: {
      auto fb = fb_instr->op_as_ArcsinNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      ArcsinNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::ARCSIN;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ArccosNode: {
      auto fb = fb_instr->op_as_ArccosNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      ArccosNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::ARCCOS;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ArctanNode: {
      auto fb = fb_instr->op_as_ArctanNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      ArctanNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::ARCTAN;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_SinhNode: {
      auto fb = fb_instr->op_as_SinhNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      SinhNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::SINH;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_CoshNode: {
      auto fb = fb_instr->op_as_CoshNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      CoshNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::COSH;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ArcsinhNode: {
      auto fb = fb_instr->op_as_ArcsinhNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      ArcsinhNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::ARCSINH;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ArccoshNode: {
      auto fb = fb_instr->op_as_ArccoshNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      ArccoshNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::ARCCOSH;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ArctanhNode: {
      auto fb = fb_instr->op_as_ArctanhNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      ArctanhNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::ARCTANH;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_Log2Node: {
      auto fb = fb_instr->op_as_Log2Node();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      Log2Node node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::LOG2;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_Log10Node: {
      auto fb = fb_instr->op_as_Log10Node();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      Log10Node node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::LOG10;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_Log1pNode: {
      auto fb = fb_instr->op_as_Log1pNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      Log1pNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::LOG1P;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ErfNode: {
      auto fb = fb_instr->op_as_ErfNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      ErfNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::ERF;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_Expm1Node: {
      auto fb = fb_instr->op_as_Expm1Node();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      Expm1Node node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::EXPM1;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_RoundNode: {
      auto fb = fb_instr->op_as_RoundNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      RoundNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.decimals = fb->decimals();
      instr.op = OpCode::ROUND;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ReciprocalNode: {
      auto fb = fb_instr->op_as_ReciprocalNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      ReciprocalNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::RECIPROCAL;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_SqrtNode: {
      auto fb = fb_instr->op_as_SqrtNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      SqrtNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::SQRT;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_AbsNode: {
      auto fb = fb_instr->op_as_AbsNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      AbsNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::ABS;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_NegNode: {
      auto fb = fb_instr->op_as_NegNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      NegNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::NEG;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_Atan2Node: {
      auto fb = fb_instr->op_as_Atan2Node();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      Atan2Node node;
      node.a = convert_tid(fb->a());
      node.b = convert_tid(fb->b());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::ATAN2;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_LogAddExpNode: {
      auto fb = fb_instr->op_as_LogAddExpNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      LogAddExpNode node;
      node.a = convert_tid(fb->a());
      node.b = convert_tid(fb->b());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::LOG_ADD_EXP;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_FloorDivideNode: {
      auto fb = fb_instr->op_as_FloorDivideNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      FloorDivideNode node;
      node.a = convert_tid(fb->a());
      node.b = convert_tid(fb->b());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::FLOOR_DIVIDE;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_RemainderNode: {
      auto fb = fb_instr->op_as_RemainderNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      RemainderNode node;
      node.a = convert_tid(fb->a());
      node.b = convert_tid(fb->b());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::REMAINDER;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_PowerNode: {
      auto fb = fb_instr->op_as_PowerNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      PowerNode node;
      node.a = convert_tid(fb->a());
      node.b = convert_tid(fb->b());
      node.out = convert_tid(fb->out());
      instr.op = OpCode::POWER;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_LogSumExpNode: {
      auto fb = fb_instr->op_as_LogSumExpNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      LogSumExpNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.axes = to_vector(fb->axes());
      node.keepdims = fb->keepdims();
      instr.op = OpCode::LOG_SUM_EXP;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_SumNode: {
      auto fb = fb_instr->op_as_SumNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      SumNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.axes = to_vector(fb->axes());
      node.keepdims = fb->keepdims();
      instr.op = OpCode::SUM;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_MeanNode: {
      auto fb = fb_instr->op_as_MeanNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      MeanNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.axes = to_vector(fb->axes());
      node.keepdims = fb->keepdims();
      instr.op = OpCode::MEAN;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_VarNode: {
      auto fb = fb_instr->op_as_VarNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      VarNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.axes = to_vector(fb->axes());
      node.keepdims = fb->keepdims();
      node.ddof = fb->ddof();
      instr.op = OpCode::VAR;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_StdNode: {
      auto fb = fb_instr->op_as_StdNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      StdNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.axes = to_vector(fb->axes());
      node.keepdims = fb->keepdims();
      node.ddof = fb->ddof();
      instr.op = OpCode::STD;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ProdNode: {
      auto fb = fb_instr->op_as_ProdNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      ProdNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.axes = to_vector(fb->axes());
      node.keepdims = fb->keepdims();
      instr.op = OpCode::PROD;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_MaxNode: {
      auto fb = fb_instr->op_as_MaxNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      MaxNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.axes = to_vector(fb->axes());
      node.keepdims = fb->keepdims();
      instr.op = OpCode::MAX;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_MinNode: {
      auto fb = fb_instr->op_as_MinNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      MinNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.axes = to_vector(fb->axes());
      node.keepdims = fb->keepdims();
      instr.op = OpCode::MIN;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ArgminNode: {
      auto fb = fb_instr->op_as_ArgminNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      ArgminNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.axis = fb->axis();
      node.keepdims = fb->keepdims();
      instr.op = OpCode::ARGMIN;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_MedianNode: {
      auto fb = fb_instr->op_as_MedianNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      MedianNode node;
      node.x = convert_tid(fb->x());
      node.out = convert_tid(fb->out());
      node.axes = to_vector(fb->axes());
      node.keepdims = fb->keepdims();
      instr.op = OpCode::MEDIAN;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_GatherMmNode: {
      auto fb = fb_instr->op_as_GatherMmNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      GatherMmNode node;
      node.a = convert_tid(fb->a());
      node.b = convert_tid(fb->b());
      node.out = convert_tid(fb->out());
      if (fb->lhs_indices()) {
        node.lhs_indices = convert_tid(fb->lhs_indices());
      }
      if (fb->rhs_indices()) {
        node.rhs_indices = convert_tid(fb->rhs_indices());
      }
      node.sorted_indices = fb->sorted_indices();
      instr.op = OpCode::GATHER_MM;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_GatherQmmNode: {
      auto fb = fb_instr->op_as_GatherQmmNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      GatherQmmNode node;
      node.x = convert_tid(fb->x());
      node.w = convert_tid(fb->w());
      node.scales = convert_tid(fb->scales());
      node.out = convert_tid(fb->out());
      node.mode = fb->mode() ? fb->mode()->str() : "";
      if (fb->biases()) {
        node.biases = convert_tid(fb->biases());
      }
      if (fb->lhs_indices()) {
        node.lhs_indices = convert_tid(fb->lhs_indices());
      }
      if (fb->rhs_indices()) {
        node.rhs_indices = convert_tid(fb->rhs_indices());
      }
      node.transpose = fb->transpose();
      node.group_size = fb->group_size();
      node.bits = fb->bits();
      node.sorted_indices = fb->sorted_indices();
      instr.op = OpCode::GATHER_QMM;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_ScanNode: {
      auto fb = fb_instr->op_as_ScanNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      ScanNode node;
      if (fb->originals()) {
        for (auto fb_tid : *fb->originals()) {
          node.originals.push_back(convert_tid(fb_tid));
        }
      }
      if (fb->sliced()) {
        for (auto fb_tid : *fb->sliced()) {
          node.sliced.push_back(convert_tid(fb_tid));
        }
      }
      if (fb->outputs()) {
        for (auto fb_tid : *fb->outputs()) {
          node.outputs.push_back(convert_tid(fb_tid));
        }
      }
      if (fb->carry()) {
        for (auto fb_tid : *fb->carry()) {
          node.carry.push_back(convert_tid(fb_tid));
        }
      }
      node.body_chain_idx = fb->body_chain_idx();
      node.scan_axis = fb->scan_axis();
      instr.op = OpCode::SCAN;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_MetalKernelNode: {
      auto fb = fb_instr->op_as_MetalKernelNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      MetalKernelNode node;
      node.name = fb->name() ? fb->name()->str() : "";
      node.source = fb->source() ? fb->source()->str() : "";
      if (fb->inputs()) {
        for (auto fb_tid : *fb->inputs()) {
          node.inputs.push_back(convert_tid(fb_tid));
        }
      }
      if (fb->outputs()) {
        for (auto fb_tid : *fb->outputs()) {
          node.outputs.push_back(convert_tid(fb_tid));
        }
      }
      if (fb->grid()) {
        for (size_t i = 0; i < fb->grid()->size(); ++i) {
          node.grid.push_back(convert_int_or_vid(fb->grid()->Get(static_cast<flatbuffers::uoffset_t>(i))));
        }
      }
      if (fb->threadgroup()) {
        for (size_t i = 0; i < fb->threadgroup()->size(); ++i) {
          node.threadgroup.push_back(convert_int_or_vid(fb->threadgroup()->Get(static_cast<flatbuffers::uoffset_t>(i))));
        }
      }
      if (fb->header()) {
        node.header = fb->header()->str();
      }
      if (fb->input_names()) {
        for (const auto* s : *fb->input_names()) {
          node.input_names.push_back(s ? s->str() : std::string{});
        }
      }
      if (fb->output_names()) {
        for (const auto* s : *fb->output_names()) {
          node.output_names.push_back(s ? s->str() : std::string{});
        }
      }
      node.ensure_row_contiguous = fb->ensure_row_contiguous();
      node.atomic_outputs = fb->atomic_outputs();
      if (fb->output_shapes_flat()) {
        for (size_t i = 0; i < fb->output_shapes_flat()->size(); ++i) {
          node.output_shapes_flat.push_back(convert_int_or_vid(fb->output_shapes_flat()->Get(static_cast<flatbuffers::uoffset_t>(i))));
        }
      }
      node.output_shape_lengths = to_vector(fb->output_shape_lengths());
      node.output_dtypes = to_vector(fb->output_dtypes());
      if (fb->template_arg_names()) {
        for (const auto* s : *fb->template_arg_names()) {
          node.template_arg_names.push_back(s ? s->str() : std::string{});
        }
      }
      node.template_arg_kinds = to_vector(fb->template_arg_kinds());
      node.template_arg_values = to_vector(fb->template_arg_values());
      auto init_value_opt = fb->init_value();
      if (init_value_opt.has_value()) {
        node.init_value = init_value_opt.value();
      }
      instr.op = OpCode::METAL_KERNEL;
      instr.node = std::move(node);
      break;
    }

    case mlx_delegate::OpNode_BitwiseXorNode: {
      auto fb = fb_instr->op_as_BitwiseXorNode();
      if (!fb) {{
        throw std::runtime_error("FlatBuffer op_type/payload mismatch for {class_name}");
      }}
      BitwiseXorNode node;
      if (fb->a()) {
        node.a = convert_tid(fb->a());
      }
      if (fb->b()) {
        node.b = convert_tid(fb->b());
      }
      if (fb->out()) {
        node.out = convert_tid(fb->out());
      }
      instr.op = OpCode::BITWISE_XOR;
      instr.node = std::move(node);
      break;
    }

    default:
      throw std::runtime_error(
          "Unknown op_type in load_instruction: " +
          std::to_string(static_cast<int>(op_type)) +
          ". The .pte was built with a newer schema than this binary. "
          "Rebuild with the latest runtime.");
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

  // Defense-in-depth: parse_header already validates this, but guard the
  // unsigned subtraction against underflow in case the call site ever changes.
  if (header.data_offset <= kHeaderSize || header.data_offset > size) {
    throw std::runtime_error("data_offset out of range");
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
  program.num_input_tensors = fb_graph->num_input_tensors();
  program.num_output_tensors = fb_graph->num_output_tensors();
  program.num_mutable_buffer_tensors = fb_graph->num_mutable_buffer_tensors();
  program.num_temp_tensors = fb_graph->num_temp_tensors();
  program.num_values = fb_graph->num_values();

  // Cap all counts/collection sizes to prevent unbounded allocations from
  // malformed FlatBuffer payloads
  constexpr size_t kMaxCollectionSize = 1'000'000;
  auto check_collection_size = [](size_t sz, const char* name) {
    if (sz > kMaxCollectionSize) {
      throw std::runtime_error(
          std::string("Malformed program: ") + name + " size " +
          std::to_string(sz) + " exceeds maximum of " +
          std::to_string(kMaxCollectionSize));
    }
  };

  check_collection_size(program.num_tensors(), "num_tensors()");
  check_collection_size(program.num_values, "num_values");

  if (fb_graph->instruction_chains()) {
    check_collection_size(fb_graph->instruction_chains()->size(), "instruction_chains");
    program.instruction_chains.reserve(fb_graph->instruction_chains()->size());
    for (size_t c = 0; c < fb_graph->instruction_chains()->size(); ++c) {
      const auto* fb_chain = fb_graph->instruction_chains()->Get(static_cast<flatbuffers::uoffset_t>(c));
      std::vector<Instruction> chain;
      if (fb_chain && fb_chain->instructions()) {
        check_collection_size(fb_chain->instructions()->size(), "instructions in chain");
        chain.reserve(fb_chain->instructions()->size());
        for (size_t i = 0; i < fb_chain->instructions()->size(); ++i) {
          chain.push_back(load_instruction(fb_chain->instructions()->Get(static_cast<flatbuffers::uoffset_t>(i))));
        }
      }
      program.instruction_chains.push_back(std::move(chain));
    }
  }

  program.main_chain_idx = fb_graph->main_chain_idx();
  program.init_chain_idx = fb_graph->init_chain_idx();

  // Validate chain indices against actual instruction_chains size.
  if (program.main_chain_idx >= program.instruction_chains.size()) {
    throw std::runtime_error(
        "Invalid main_chain_idx " +
        std::to_string(program.main_chain_idx) +
        " (only " + std::to_string(program.instruction_chains.size()) +
        " chains loaded)");
  }
  if (program.init_chain_idx >= 0 &&
      static_cast<uint32_t>(program.init_chain_idx) >=
          program.instruction_chains.size()) {
    throw std::runtime_error(
        "Invalid init_chain_idx " +
        std::to_string(program.init_chain_idx) +
        " (only " + std::to_string(program.instruction_chains.size()) +
        " chains loaded)");
  }

  if (fb_graph->input_map()) {
    check_collection_size(fb_graph->input_map()->size(), "input_map");
    for (size_t i = 0; i < fb_graph->input_map()->size(); ++i) {
      const auto* slot = fb_graph->input_map()->Get(static_cast<flatbuffers::uoffset_t>(i));
      auto sv = convert_slot_variant(slot);
      if (sv.slot_type == SlotType::TensorSlot &&
          sv.idx >= program.num_tensors()) {
        throw std::runtime_error(
            "input_map: slot index " + std::to_string(sv.idx) +
            " exceeds num_tensors " +
            std::to_string(program.num_tensors()));
      }
      program.input_map.push_back(sv);
    }
  }

  if (fb_graph->output_map()) {
    check_collection_size(fb_graph->output_map()->size(), "output_map");
    for (size_t i = 0; i < fb_graph->output_map()->size(); ++i) {
      const auto* slot = fb_graph->output_map()->Get(static_cast<flatbuffers::uoffset_t>(i));
      auto sv = convert_slot_variant(slot);
      if (sv.slot_type == SlotType::TensorSlot &&
          sv.idx >= program.num_tensors()) {
        throw std::runtime_error(
            "output_map: slot index " + std::to_string(sv.idx) +
            " exceeds num_tensors " +
            std::to_string(program.num_tensors()));
      }
      program.output_map.push_back(sv);
    }
  }

  if (fb_graph->mutable_buffer_map()) {
    check_collection_size(fb_graph->mutable_buffer_map()->size(), "mutable_buffer_map");
    for (size_t i = 0; i < fb_graph->mutable_buffer_map()->size(); ++i) {
      const auto* slot = fb_graph->mutable_buffer_map()->Get(static_cast<flatbuffers::uoffset_t>(i));
      auto sv = convert_slot_variant(slot);
      if (sv.slot_type == SlotType::TensorSlot &&
          sv.idx >= program.num_tensors()) {
        throw std::runtime_error(
            "mutable_buffer_map: slot index " + std::to_string(sv.idx) +
            " exceeds num_tensors " +
            std::to_string(program.num_tensors()));
      }
      program.mutable_buffer_map.push_back(sv);
    }
  }

  if (fb_graph->named_slots()) {
    check_collection_size(fb_graph->named_slots()->size(), "named_slots");
    for (size_t i = 0; i < fb_graph->named_slots()->size(); ++i) {
      const auto* fb_slot = fb_graph->named_slots()->Get(static_cast<flatbuffers::uoffset_t>(i));
      if (!fb_slot || !fb_slot->name()) {
        throw std::runtime_error(
            "Malformed program: named_slot at index " + std::to_string(i) +
            " is null or has null name");
      }
      NamedSlot slot;
      slot.name = fb_slot->name()->str();
      slot.slot = convert_slot_variant(fb_slot->slot());
      program.named_slots.push_back(std::move(slot));
    }
  }

  if (fb_graph->tensor_meta()) {
    check_collection_size(fb_graph->tensor_meta()->size(), "tensor_meta");
    for (size_t i = 0; i < fb_graph->tensor_meta()->size(); ++i) {
      const auto* fb_meta = fb_graph->tensor_meta()->Get(static_cast<flatbuffers::uoffset_t>(i));
      if (fb_meta) {
        TensorMeta meta;
        if (fb_meta->shape()) {
          // Validate tensor rank against kTensorDimensionLimit to prevent
          // stack overflows from unchecked rank
          constexpr size_t kTensorDimensionLimit = 16;
          if (fb_meta->shape()->size() > kTensorDimensionLimit) {
            throw std::runtime_error(
                "Tensor at index " + std::to_string(i) +
                " has rank " + std::to_string(fb_meta->shape()->size()) +
                " exceeding kTensorDimensionLimit (" +
                std::to_string(kTensorDimensionLimit) + ")");
          }
          for (size_t j = 0; j < fb_meta->shape()->size(); ++j) {
            const auto* fb_dim = fb_meta->shape()->Get(static_cast<flatbuffers::uoffset_t>(j));
            if (!fb_dim) {
              throw std::runtime_error(
                  "Null ShapeDim at index " + std::to_string(j) +
                  " in tensor_meta " + std::to_string(i));
            }
            ShapeDim dim;
            dim.value = fb_dim->value();
            dim.min_value = fb_dim->min_value();
            dim.max_value = fb_dim->max_value();
            if (dim.value < -1) {
              throw std::runtime_error(
                  "Invalid ShapeDim value " + std::to_string(dim.value) +
                  " at index " + std::to_string(j) +
                  " in tensor_meta " + std::to_string(i));
            }
            if (dim.is_dynamic()) {
              if (dim.min_value < 0) {
                throw std::runtime_error(
                    "Invalid ShapeDim min_value " + std::to_string(dim.min_value) +
                    " at index " + std::to_string(j) +
                    " in tensor_meta " + std::to_string(i));
              }
              if (dim.max_value != -1 && dim.max_value < dim.min_value) {
                throw std::runtime_error(
                    "ShapeDim max_value " + std::to_string(dim.max_value) +
                    " < min_value " + std::to_string(dim.min_value) +
                    " at index " + std::to_string(j) +
                    " in tensor_meta " + std::to_string(i));
              }
            }
            meta.shape.push_back(dim);
          }
        }
        auto raw_scalar_type = fb_meta->scalar_type();
        if (raw_scalar_type < 0 ||
            raw_scalar_type >=
                static_cast<int8_t>(ScalarType::NumOptions)) {
          throw std::runtime_error(
              "Invalid scalar_type " + std::to_string(raw_scalar_type) +
              " in tensor_meta at index " + std::to_string(i));
        }
        meta.scalar_type = static_cast<ScalarType>(raw_scalar_type);
        if (fb_meta->dim_order()) {
          meta.dim_order = to_vector(fb_meta->dim_order());
        }
        program.tensor_meta.push_back(std::move(meta));
      } else {
        program.tensor_meta.push_back(std::nullopt);
      }
    }
  }

  return program;
}

}  // namespace loader
}  // namespace mlx
}  // namespace backends
}  // namespace executorch
