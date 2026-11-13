# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export the LFM3-Omni speech-output components to QNN fp16:

  depth.pte        hidden [1, 2048], codes [12], masked [12] -> logits [12, 2017]
  embed_frame.pte  codes [12] -> decoder input embedding [1, 1, 2048]
  codec.pte        codes [12, frames] -> 24 kHz audio [frames * 1920]

The audio encoder and decoder come from llama.py (--decoder_model lfm3-2b_omni_speech).
"""

import argparse
import json
import os

import torch
from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.examples.qualcomm.oss_scripts.lfm3.codec import (
    codec_state_dict,
    StaticCodecDecoder,
)
from executorch.examples.qualcomm.oss_scripts.lfm3.depth import (
    depth_state_dict,
    frame_embedding_state_dict,
    StaticDepthDecoder,
    StaticFrameEmbedding,
)
from executorch.examples.qualcomm.oss_scripts.lfm3.speech_policy import (
    speech_sampling_metadata,
)
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


def build(snap: str, codec_frames: int, generation_config_path: str | None = None):
    cfg = json.load(open(f"{snap}/config.json"))
    with open(generation_config_path or f"{snap}/generation_config.json") as f:
        sampling_meta = speech_sampling_metadata(json.load(f))
    sd = load_file(f"{snap}/model.safetensors")
    n_cb, cb = cfg["num_codebooks"], cfg["codebook_size"]
    hidden = cfg["text_config"]["hidden_size"]

    depth = StaticDepthDecoder(cfg["depth_config"], hidden, n_cb, cb)
    depth.load_state_dict(depth_state_dict(sd, n_cb), strict=True)
    embed = StaticFrameEmbedding(n_cb, cb, hidden)
    embed.load_state_dict(frame_embedding_state_dict(sd), strict=True)

    ccfg = json.load(open(f"{snap}/audio_tokenizer/config.json"))
    csd = load_file(f"{snap}/audio_tokenizer/model.safetensors")
    codec = StaticCodecDecoder(ccfg, codec_frames, csd["decoder.istft.window"])
    codec.load_state_dict(codec_state_dict(csd), strict=True)

    codes = torch.zeros(n_cb, dtype=torch.int32)
    audio_eos = cfg["audio_eos_code"]
    common = {
        "get_speech_contract_version": 1,
        "get_num_codebooks": n_cb,
        "get_codebook_size": cb,
        "get_audio_eos_code": audio_eos,
    }
    codec_meta = codec.get_metadata()
    if (
        n_cb != ccfg["n_q"]
        or cb != codec.codebook_size
        or audio_eos != codec.audio_eos_code
        or codec_frames <= codec.left_context + codec.right_context
        or cfg["maskgit_schedule"] != [4, 4, 4]
        or cfg["interleaved_n_text"] != 6
        or cfg["interleaved_n_audio"] != 12
    ):
        raise ValueError(
            "Unsupported speech model contract or insufficient codec window"
        )
    return {
        "depth": (
            depth,
            (torch.zeros(1, hidden), codes, torch.ones(n_cb)),
            {**common, **sampling_meta},
        ),
        "embed_frame": (embed, (codes,), common),
        "codec": (
            codec,
            (
                torch.zeros(ccfg["n_q"], codec_frames, dtype=torch.int32),
                torch.tensor([codec_frames], dtype=torch.int32),
            ),
            {
                **codec_meta,
                "get_sample_rate": ccfg["sampling_rate"],
            },
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="LiquidAI/LFM3-Omni-2B")
    parser.add_argument("--soc_model", default="SM8750")
    parser.add_argument("--codec_frames", type=int, default=128)
    parser.add_argument(
        "--generation_config",
        default=None,
        help="Explicit generation_config JSON; otherwise use the checkpoint file.",
    )
    parser.add_argument("-a", "--artifact", required=True)
    args = parser.parse_args()

    snap = snapshot_download(
        args.repo_id,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "audio_tokenizer/*",
        ],
    )
    specs = generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, args.soc_model),
        backend_options=generate_htp_compiler_spec(use_fp16=True),
    )
    os.makedirs(args.artifact, exist_ok=True)
    for name, (module, example_inputs, metadata) in build(
        snap, args.codec_frames, args.generation_config
    ).items():
        with torch.no_grad():
            program = to_edge_transform_and_lower_to_qnn(
                module.eval(), example_inputs, specs, constant_methods=metadata
            ).to_executorch()
        path = os.path.join(args.artifact, f"{name}.pte")
        with open(path, "wb") as f:
            f.write(program.buffer)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
