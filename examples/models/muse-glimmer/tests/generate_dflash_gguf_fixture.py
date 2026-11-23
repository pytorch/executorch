# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generate a reduced-width DFlash2 checkpoint with synthetic K-quant weights."""

import argparse
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFWriter


def write_dflash_gguf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(str(path), "dflash")
    for key, value in {
        "embedding_length": 256,
        "block_count": 2,
        "attention.head_count": 4,
        "attention.head_count_kv": 1,
        "attention.key_length": 128,
        "attention.value_length": 128,
        "feed_forward_length": 512,
        "block_size": 16,
        "context_length": 64,
        "attention.sliding_window": 16,
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "selector_rank": 256,
        "selector_top_k": 16,
    }.items():
        writer.add_uint32("dflash." + key, value)
    for key, value in {
        "rope.freq_base": 500000.0,
        "attention.layer_norm_rms_epsilon": 1e-5,
        "logit_scale": 0.196116135,
        "final_logit_softcapping": 20.0,
    }.items():
        writer.add_float32("dflash." + key, value)
    writer.add_bool("dflash.attention.causal", False)
    writer.add_array("dflash.attention.sliding_window_pattern", [True, True])
    writer.add_array("dflash.target_layers", [0, 1])
    writer.add_uint32("tokenizer.ggml.mask_token_id", 201818)
    rng = np.random.default_rng(73)

    def quantized(name, shape, qtype=GGMLQuantizationType.Q4_K, scale=0.001):
        rows, columns = shape
        assert columns % 256 == 0
        block_bytes = 144 if qtype == GGMLQuantizationType.Q4_K else 210
        raw = rng.integers(
            0, 256, size=(rows * columns // 256, block_bytes), dtype=np.uint8
        )
        if qtype == GGMLQuantizationType.Q4_K:
            # Uniform sub-scales/mins of eight give centered, nonzero weights.
            raw[:, :2] = np.frombuffer(np.array(scale, dtype="<f2").tobytes(), np.uint8)
            raw[:, 2:4] = np.frombuffer(
                np.array(7.5 * scale, dtype="<f2").tobytes(), np.uint8
            )
            raw[:, 4:12] = 8
            raw[:, 12:16] = 0x88
        else:
            # Nonuniform Q6_K sub-scales exercise the native GGUF kernel path.
            raw[:, 192:208] = np.arange(1, 17, dtype=np.uint8)
            raw[:, 208:210] = np.frombuffer(
                np.array(scale / 2, dtype="<f2").tobytes(), np.uint8
            )
        writer.add_tensor(name, raw.reshape(rows, -1), raw_dtype=qtype)

    quantized("fc.weight", (256, 512))
    quantized("selector_hidden.weight", (256, 256))
    for side in ("predecessor", "successor"):
        quantized(f"selector_{side}.weight", (202048, 256), scale=0.002)
    for name in ("enc.output_norm.weight", "output_norm.weight"):
        writer.add_tensor(name, 1 + 0.02 * rng.standard_normal(256, dtype=np.float32))
    for layer in range(2):
        prefix = f"blk.{layer}."
        for name, shape in {
            "attn_q.weight": (512, 256),
            "attn_k.weight": (128, 256),
            "attn_output.weight": (256, 512),
            "ffn_gate.weight": (512, 256),
            "ffn_up.weight": (512, 256),
        }.items():
            quantized(prefix + name, shape)
        for name, shape, q6 in (
            ("attn_v.weight", (128, 256), layer == 0),
            ("ffn_down.weight", (256, 512), layer == 1),
        ):
            quantized(
                prefix + name,
                shape,
                GGMLQuantizationType.Q6_K if q6 else GGMLQuantizationType.Q4_K,
            )
        for name in ("attn_conv", "ffn_conv"):
            quantized(prefix + name + "_proj.weight", (64, 256), scale=0.00025)
            base = 0.04 * rng.standard_normal((2, 2, 256), dtype=np.float32)
            base[:, 0] += 1
            writer.add_tensor(prefix + name + "_base", base)
        for name, size in (
            ("attn_norm.weight", 256),
            ("ffn_norm.weight", 256),
            ("attn_q_norm.weight", 128),
            ("attn_k_norm.weight", 128),
        ):
            writer.add_tensor(
                prefix + name, 1 + 0.02 * rng.standard_normal(size, dtype=np.float32)
            )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    write_dflash_gguf(parser.parse_args().output)
