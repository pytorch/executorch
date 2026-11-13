# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Record the decoder input-embedding sequences HF LFM3-Omni feeds its language
model while generating speech (text, speech-encoder embeddings, and the summed
codebook embeddings of generated speech frames), for llama.py --calib_embeddings.

Needs the checkpoint's remote code (transformers >= 5.20.2, trust_remote_code).

  python capture_speech_calibration.py --audio question.wav --audio other.wav -o calib_embeds.pt
"""

import argparse

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

TTS = [
    (
        "Perform TTS in American English using a female voice.",
        "ExecuTorch runs Liquid models on the Snapdragon NPU.",
    ),
    (
        "Perform TTS in American English using a male voice.",
        "The weather today is sunny with a light breeze.",
    ),
    (
        "Perform TTS in British English using a female voice.",
        "Please remember to bring your umbrella tomorrow.",
    ),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="LiquidAI/LFM3-Omni-2B")
    parser.add_argument(
        "--audio", action="append", default=[], help="Speech clips (wav)"
    )
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    torch.manual_seed(0)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.repo_id, dtype=torch.float32, trust_remote_code=True
    ).eval()
    processor = AutoProcessor.from_pretrained(args.repo_id, trust_remote_code=True)
    lm = model.model.language_model
    forward, captured = lm.forward, []

    def record(*a, **kw):
        if kw.get("inputs_embeds") is not None:
            captured.append(kw["inputs_embeds"].detach().float())
        return forward(*a, **kw)

    lm.forward = record

    def run(conversation, **kwargs):
        captured.clear()
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            model.generate(**inputs, **kwargs)
        return torch.cat(captured, dim=1)[:, : args.max_len].contiguous()

    speech = {"audio_temperature": 0.4, "audio_repetition_penalty": 1.2}
    sequences = [
        run(
            [{"role": "system", "content": system}, {"role": "user", "content": text}],
            max_new_tokens=160,
            generation_mode="audio",
            **speech,
        )
        for system, text in TTS
    ]
    for clip in args.audio:
        audio = [{"type": "audio", "audio": clip}]
        sequences.append(
            run(
                [
                    {"role": "system", "content": "Perform ASR in English."},
                    {"role": "user", "content": audio},
                ],
                max_new_tokens=64,
                generation_mode="text",
            )
        )
        sequences.append(
            run(
                [
                    {"role": "system", "content": "Respond with speech."},
                    {"role": "user", "content": audio},
                ],
                max_new_tokens=200,
                **speech,
            )
        )
    torch.save(sequences, args.output)
    print(f"wrote {len(sequences)} sequences to {args.output}")


if __name__ == "__main__":
    main()
