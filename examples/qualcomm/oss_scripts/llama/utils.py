# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from typing import Iterator, Optional

from executorch.exir._serialize._program import deserialize_pte_binary
from torch.utils.data import DataLoader


def safe_dataloader_iter(dataloader: Optional[DataLoader]) -> Iterator:
    """Infinite iterator over a DataLoader; yields empty lists when dataloader is None."""
    if dataloader is not None:
        return itertools.chain(dataloader, itertools.repeat([]))
    return itertools.repeat([])


def retrieve_info_from_pte(pte_path: str) -> dict:
    """Read quantization metadata from a compiled .pte binary."""
    output_vocab_size = None
    pte_max_context_len = None
    pte_max_seq_len = None
    logits_scale = None
    logits_zero_point = None
    kv_io_bit_width = 32

    with open(pte_path, "rb") as f:
        program_data = f.read()
        program = deserialize_pte_binary(program_data).program

    for method in program.execution_plan:
        if method.name == "get_vocab_size":
            # pyre-ignore
            output_vocab_size = method.values[0].val.int_val
        if method.name == "get_max_seq_len":
            # pyre-ignore
            pte_max_seq_len = method.values[0].val.int_val
        if method.name == "get_max_context_len":
            # pyre-ignore
            pte_max_context_len = method.values[0].val.int_val
        if method.name == "get_logits_scale":
            logits_scale = method.values[0].val.double_val
        if method.name == "get_logits_zero_point":
            logits_zero_point = method.values[0].val.int_val
        if method.name == "get_kv_io_bit_width":
            kv_io_bit_width = method.values[0].val.int_val
    if pte_max_context_len is None:
        pte_max_context_len = pte_max_seq_len

    # FP has no scale/zero_point; use identity values (no dequantize effect).
    if kv_io_bit_width == 32:
        logits_scale = 1
        logits_zero_point = 0
    elif logits_scale is None or logits_zero_point is None:
        raise RuntimeError(
            "Unable to find scale/offset. The .pte file might be deprecated. Please generate a new .pte file"
        )
    assert output_vocab_size is not None, "Couldn't find the vocab size"
    assert pte_max_seq_len is not None, "Couldn't find the max_seq_len from pte"
    return {
        "output_vocab_size": output_vocab_size,
        "pte_max_context_len": pte_max_context_len,
        "pte_max_seq_len": pte_max_seq_len,
        "logits_scale": logits_scale,
        "logits_zero_point": logits_zero_point,
        "kv_io_bit_width": kv_io_bit_width,
    }
