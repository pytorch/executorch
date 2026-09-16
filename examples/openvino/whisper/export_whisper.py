# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

# mypy: disable-error-code="import-not-found"

"""
Export Whisper as a three-way split and lower each program to the
OpenVINO backend:

  * encoder.pte  : mel features -> encoder hidden states
  * cross_kv.pte : encoder hidden states -> per-layer cross-attention K/V
  * decoder.pte  : token-by-token generation with a self-attention KV cache,
                   consuming the pre-computed cross K/V as inputs.

Splitting the cross-attention projections out of the decoder graph means they
run once per utterance instead of once per generated token.

Usage:
    python export_whisper.py \
        --model_id openai/whisper-small \
        --output_dir ./whisper_ov \
        --device GPU \
        --max_cache_length 448
"""

import argparse
import json
import logging
import os

import torch
from torch.nn.attention import SDPBackend

from executorch.backends.openvino.partitioner import OpenvinoPartitioner
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from transformers import AutoModelForSpeechSeq2Seq

from whisper_model import (
    RemovePaddingIdxEmbeddingPass,
    WhisperCrossKVProjection,
    WhisperDecoderWithCache,
    WhisperEncoderModule,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def lower(exported_program, device: str):
    partitioner = OpenvinoPartitioner(
        compile_spec=[CompileSpec("device", device.encode())]
    )
    edge = to_edge_transform_and_lower(
        exported_program,
        partitioner=[partitioner],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False, _skip_dim_order=True
        ),
        transform_passes=[RemovePaddingIdxEmbeddingPass()],
    )
    return edge.to_executorch(
        config=ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            extract_delegate_segments=True,
        )
    )


def export_and_save(module, example_inputs, device, out_path, name):
    with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
        ep = torch.export.export(module, example_inputs, strict=True)
    et = lower(ep, device)
    with open(out_path, "wb") as f:
        et.write_to_file(f)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    logger.info(f"  {name}: {out_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Export Whisper to the OpenVINO backend as a 3-way split"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="openai/whisper-small",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="whisper_ov",
        help="Output directory for the exported .pte files",
    )
    parser.add_argument(
        "--device",
        choices=["CPU", "GPU", "NPU"],
        default="CPU",
        help="Target OpenVINO device",
    )
    parser.add_argument(
        "--max_cache_length",
        type=int,
        default=448,
        help="Maximum decoder sequence length (self-attention cache size)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Loading {args.model_id}")
    whisper = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_id).to("cpu").eval()
    cfg = whisper.config

    encoder_mod = WhisperEncoderModule(
        whisper.get_encoder(),
        feature_size=cfg.num_mel_bins,
        num_frames=cfg.max_source_positions * 2,
    ).eval()
    enc_example = encoder_mod.get_example_inputs()

    cross_kv_mod = WhisperCrossKVProjection(whisper).eval()
    decoder_mod = WhisperDecoderWithCache(
        whisper, max_decoder_seq_len=args.max_cache_length
    ).eval()

    logger.info("Exporting encoder ...")
    export_and_save(
        encoder_mod,
        enc_example,
        args.device,
        os.path.join(args.output_dir, "encoder.pte"),
        "encoder",
    )

    logger.info("Exporting cross_kv ...")
    with torch.no_grad():
        enc_hidden = encoder_mod(*enc_example)
    export_and_save(
        cross_kv_mod,
        (enc_hidden,),
        args.device,
        os.path.join(args.output_dir, "cross_kv.pte"),
        "cross_kv",
    )

    logger.info("Exporting decoder ...")
    with torch.no_grad():
        example_cross_k, example_cross_v = cross_kv_mod(enc_hidden)
    decoder_input_ids = torch.tensor(
        [[cfg.decoder_start_token_id or 0]], dtype=torch.long
    )
    cache_position = torch.tensor([0], dtype=torch.long)
    attn_mask = torch.zeros(1, 1, 1, args.max_cache_length)
    export_and_save(
        decoder_mod,
        (
            decoder_input_ids,
            cache_position,
            attn_mask,
            example_cross_k,
            example_cross_v,
        ),
        args.device,
        os.path.join(args.output_dir, "decoder.pte"),
        "decoder",
    )

    meta = {
        "model_id": args.model_id,
        "num_decoder_layers": decoder_mod.num_layers,
        "max_cache_length": args.max_cache_length,
        "decoder_start_token_id": cfg.decoder_start_token_id,
        "eos_token_id": cfg.eos_token_id,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata: {meta}")


if __name__ == "__main__":
    main()
