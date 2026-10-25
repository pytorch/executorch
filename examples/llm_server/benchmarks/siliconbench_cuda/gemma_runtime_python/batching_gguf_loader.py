# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license in the root LICENSE.

"""Q4_0/Q4_1 block decoding into the existing portable INT4 representation."""

import torch
from executorch.extension.llm.export.gguf import ExportableGGUFTensor
from executorch.extension.llm.export.int4 import ExportableInt4Tensor


def decode_q4_blocks(raw, quant_type, shape):
    block_bytes = 18 if quant_type == 'q4_0' else 20
    blocks = raw.reshape(-1, block_bytes)
    scale = blocks[:, :2].contiguous().view(torch.float16).float().reshape(-1)
    offset = 2 if quant_type == 'q4_0' else 4
    packed = blocks[:, offset:]
    q = torch.cat((packed & 15, packed >> 4), dim=1)
    negative = scale < 0
    if quant_type == 'q4_0':
        zero = torch.where(negative, 7.0, 8.0)
    elif quant_type == 'q4_1':
        minimum = blocks[:, 2:4].contiguous().view(torch.float16).float().reshape(-1)
        if torch.any((scale == 0) & (minimum != 0)):
            raise ValueError('Nonzero constant Q4_1 block is not supported')
        zero = torch.where(scale != 0, -minimum / scale, 0.0)
        zero = torch.where(negative, 15.0 - zero, zero)
    else:
        raise ValueError(f'Unsupported legacy quant type: {quant_type}')
    q = torch.where(negative[:, None], 15 - q, q).reshape(shape)
    scale = scale.abs().reshape(shape[0], shape[1] // 32)
    zero = zero.reshape_as(scale)
    if torch.any(zero < 0):
        raise ValueError('Negative zero point cannot use the CUDA coalesced layout')
    qdata = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()
    return ExportableInt4Tensor(qdata, scale.t().contiguous(), zero.t().contiguous(),
                               32, torch.bfloat16)


def iter_gemma_gguf(path):
    from gguf import GGMLQuantizationType, GGUFReader
    types = {GGMLQuantizationType.Q4_K: 'q4_k',
             GGMLQuantizationType.Q5_K: 'q5_k', GGMLQuantizationType.Q6_K: 'q6_k'}
    legacy = {GGMLQuantizationType.Q4_0: 'q4_0', GGMLQuantizationType.Q4_1: 'q4_1'}
    floats = {GGMLQuantizationType.F32: torch.float32,
              GGMLQuantizationType.F16: torch.float16, GGMLQuantizationType.BF16: torch.bfloat16}
    reader = GGUFReader(path)
    for tensor in reader.tensors:
        shape = tuple(reversed(tensor.shape.tolist()))
        raw = torch.frombuffer(memoryview(tensor.data), dtype=torch.uint8)
        if tensor.tensor_type in legacy:
            value = decode_q4_blocks(raw, legacy[tensor.tensor_type], shape)
        elif tensor.tensor_type in types:
            value = ExportableGGUFTensor.from_raw(
                raw.reshape(shape[0], -1).clone(), types[tensor.tensor_type])
        elif tensor.tensor_type in floats:
            value = raw.view(floats[tensor.tensor_type]).reshape(shape).clone()
        else:
            raise ValueError(f'Unsupported GGUF quant type: {tensor.tensor_type}')
        yield tensor.name, value
