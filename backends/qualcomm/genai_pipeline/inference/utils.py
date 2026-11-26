# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inference helpers: locate the deployed artifacts, build runner flags, and decode runner outputs."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import torch

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_AUDIO_ENCODER,
    ARTIFACT_TEXT_DECODER,
    ARTIFACT_VISION_ENCODER,
)
from executorch.backends.qualcomm.genai_pipeline.configs.inference_input_config import (
    InferenceInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.control_args import ControlArgs
from executorch.backends.qualcomm.genai_pipeline.inference.runner_constants import (
    DECODER_MODEL_VERSION,
    EVAL_MODE,
)
from executorch.backends.qualcomm.genai_pipeline.inference.runner_io import (
    GenerateOutput,
    RunnerMetadata,
)
from executorch.exir._serialize._program import (
    _ExtendedHeader,
    _flatbuffer_to_program,
    _get_extended_header,
)

if TYPE_CHECKING:
    from executorch.backends.qualcomm.genai_pipeline.inference.platforms import (
        RunnerPlatform,
    )
    from executorch.examples.qualcomm.oss_scripts.llama import LLMModelConfig

# Files the llama-family runners read and write, named by the flags that take
# them.
RUNNER_OUTPUTS = "outputs.txt"
RUNNER_INFERENCE_SPEED = "inference_speed.txt"
RUNNER_LOGITS = "all_logits.raw"
RUNNER_INPUT_TOKENS = "input_tokens.raw"


def get_pte_paths(
    pte_paths: Dict[str, Path],
    artifact_keys: Sequence[str] = (ARTIFACT_TEXT_DECODER,),
) -> Dict[str, Path]:
    """Select the ``.pte`` of each artifact key the runner needs.

    Args:
        pte_paths: Compiled ``.pte`` files keyed by artifact name, resolved by
            the inference strategy from the compilation output or
            ``--pre-gen-pte``.
        artifact_keys: The artifacts to select.

    Returns:
        The ``.pte`` path of each key in ``artifact_keys``.

    Raises:
        ValueError: If any artifact is missing or not a file.
    """
    missing = [key for key in artifact_keys if key not in pte_paths]
    if missing:
        raise ValueError(
            f"Missing PTE artifact(s) {missing}; available: {sorted(pte_paths)}"
        )
    not_files = [key for key in artifact_keys if not Path(pte_paths[key]).is_file()]
    if not_files:
        raise ValueError(
            "; ".join(
                f"{key} artifact is not a file: {pte_paths[key]}" for key in not_files
            )
        )
    return {key: Path(pte_paths[key]) for key in artifact_keys}


def get_runtime_tokenizer_path(input_config: InferenceInputConfig) -> Path:
    """Return the tokenizer file a runner that loads one requires.

    The inference strategy has already settled ``runtime_tokenizer_path``;
    this only checks that the runner can load it.

    Args:
        input_config: Inference configuration carrying the
            ``runtime_tokenizer_path``.

    Returns:
        The runtime tokenizer path.

    Raises:
        ValueError: If no runtime tokenizer is available, or it is not an
            existing file.
    """
    if input_config.runtime_tokenizer_path is None:
        raise ValueError(
            "No runtime tokenizer is available; pass --runtime-tokenizer-path"
        )
    tokenizer_path = Path(input_config.runtime_tokenizer_path)
    if not tokenizer_path.is_file():
        raise ValueError(
            f"runtime_tokenizer_path is not an existing file: {tokenizer_path}"
        )
    return tokenizer_path


def retrieve_info_from_pte(pte_path: Path) -> RunnerMetadata:
    """Read the metadata the decoder PTE was compiled with.

    The only place that knows the PTE's metadata layout; everything else reads
    the returned ``RunnerMetadata``.

    Args:
        pte_path: The text-decoder PTE.

    Returns:
        The decoder's metadata.

    Raises:
        RuntimeError: If the PTE lacks the vocab size or max sequence length,
            or a quantized PTE lacks the logits scale or zero point.
    """
    output_vocab_size = None
    pte_max_context_len = None
    pte_max_seq_len = None
    logits_scale = None
    logits_zero_point = None
    kv_io_bit_width = 32

    # Only the program flatbuffer: the delegate segments after it can be GBs.
    with open(pte_path, "rb") as f:
        header = _get_extended_header(f.read(8 + _ExtendedHeader.EXPECTED_LENGTH))
        f.seek(0)
        program = _flatbuffer_to_program(f.read(header.program_size if header else -1))

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
    if output_vocab_size is None or pte_max_seq_len is None:
        raise RuntimeError(
            f"Unable to find the vocab size or max_seq_len in {pte_path}. The "
            ".pte file might be deprecated. Please generate a new .pte file"
        )
    return RunnerMetadata(
        max_context_len=pte_max_context_len,
        max_seq_len=pte_max_seq_len,
        vocab_size=output_vocab_size,
        logits_scale=logits_scale,
        logits_zero_point=logits_zero_point,
        kv_io_bit_width=kv_io_bit_width,
    )


def post_process_logits(
    logits_path: Path,
    metadata: RunnerMetadata,
    num_tokens: int,
) -> torch.Tensor:
    """Read the runner's dumped logits and dequantize the prompt positions.

    The runner dumps float32 logits from a float PTE and uint16 ones otherwise.

    Args:
        logits_path: Host path of the ``--dump_logits_path`` output.
        metadata: Metadata of the decoder that dumped the logits.
        num_tokens: Number of prompt tokens.

    Returns:
        Logits of shape ``(1, num_tokens, vocab)``.
    """
    dtype = np.float32 if metadata.kv_io_bit_width == 32 else np.uint16
    logits = torch.from_numpy(
        np.fromfile(logits_path, dtype=dtype).reshape(1, -1, metadata.vocab_size)
    )[:, :num_tokens, :]
    return (
        logits.to(torch.float32) - metadata.logits_zero_point
    ) * metadata.logits_scale


def preprocess_encoder_inputs(
    config: "LLMModelConfig",
    decoder_model: str,
    audio_paths: List[str],
    image_paths: List[str],
) -> Dict[str, List[Tuple[torch.Tensor, ...]]]:
    """Preprocess raw encoder inputs for runtime inference.

    The processor, media loaders, modality configs, and special tokens it uses
    still live under ``examples/`` and are imported lazily.

    Args:
        config: The multimodal model's config, whose modality attributes select
            what is preprocessed.
        decoder_model: The decoder model, which selects the modality token.
        audio_paths: Audio files for an audio encoder.
        image_paths: Image files for a vision encoder.

    Returns:
        The encoder inputs keyed by modality artifact key; a modality the model
        lacks or has no paths for is absent.
    """
    from executorch.examples.qualcomm.oss_scripts.llama.dataset.loaders import (
        load_audio_file,
    )
    from executorch.examples.qualcomm.oss_scripts.llama.dataset.preprocessors import (
        ModalityPreprocessor,
    )
    from executorch.examples.qualcomm.oss_scripts.llama.encoder.encoder_config import (
        AudioModalityConfig,
        VisionModalityConfig,
    )
    from executorch.examples.qualcomm.oss_scripts.llama.tokenizer import (
        ALM_SPECIAL_TOKENS,
        AUDIO_TOKEN,
        IMG_TOKEN,
        VLM_SPECIAL_TOKENS,
    )
    from transformers.image_utils import load_image

    preprocessor = ModalityPreprocessor(config.repo_id)
    result = {}
    for modality, paths in (
        (ARTIFACT_AUDIO_ENCODER, audio_paths),
        (ARTIFACT_VISION_ENCODER, image_paths),
    ):
        if not hasattr(config, modality) or not paths:
            continue
        modality_config = getattr(config, modality)
        if issubclass(modality_config, AudioModalityConfig):
            token = ALM_SPECIAL_TOKENS[decoder_model][AUDIO_TOKEN]
            normalized_prompt = token * len(paths)
            result[modality] = [
                preprocessor.preprocess_audio(
                    modality_config,
                    normalized_prompt,
                    load_audio_file(p, config.repo_id),
                )
                for p in paths
            ]
        elif issubclass(modality_config, VisionModalityConfig):
            token = VLM_SPECIAL_TOKENS[decoder_model][IMG_TOKEN]
            normalized_prompt = token * len(paths)
            result[modality] = preprocessor.preprocess_images(
                modality_config, normalized_prompt, [load_image(p) for p in paths]
            )
    return result


def build_runner_args(
    control_args: ControlArgs,
    platform: "RunnerPlatform",
    runtime_tokenizer_path: Path,
) -> List[str]:
    """Build the runner flags every llama-family runner request takes.

    The text and multimodal runners share them; each adapter adds its own PTE
    flags and the flags of the request. Paths are shell-quoted, since the
    platform's command is parsed by a shell once.

    Args:
        control_args: Control args supplying the decoder model.
        platform: Where the runner runs, which decides the paths and the
            target flags.
        runtime_tokenizer_path: Host path of the runtime tokenizer.

    Returns:
        The shared flags, each as ``"--flag value"``.
    """
    output_dir = platform.output_dir()
    return [
        f"--decoder_model_version {DECODER_MODEL_VERSION[control_args.decoder_model]}",
        f"--tokenizer_path {shlex.quote(platform.runner_path(runtime_tokenizer_path))}",
        f"--output_path {shlex.quote(f'{output_dir}/{RUNNER_OUTPUTS}')}",
        f"--performance_output_path {shlex.quote(f'{output_dir}/{RUNNER_INFERENCE_SPEED}')}",
        *platform.target_flags,
    ]


def build_generate_args(
    control_args: ControlArgs,
    prompts: Sequence[str],
    platform: "RunnerPlatform",
    limited_seq_len: int,
) -> List[str]:
    """Build the runner flags of a generation request.

    The text and multimodal runners take the same generation flags. Values a
    user supplies are shell-quoted, since the platform's command is parsed by a
    shell once.

    Args:
        control_args: Control args supplying the mode and sampling settings.
        prompts: The prompts to generate from, one ``--prompt`` each.
        platform: Where the runner runs; one that limits generation caps the
            decode length.
        limited_seq_len: The decode-length cap on such a platform, which
            depends on the model, as the legacy flow's x86 caps do.

    Returns:
        The generation flags, each as ``"--flag value"``.
    """
    seq_len = control_args.max_seq_len
    if platform.limits_generation:
        seq_len = min(seq_len, limited_seq_len)
    args = [
        f"--eval_mode {EVAL_MODE[control_args.model_mode]}",
        f"--temperature {control_args.temperature}",
        f"--system_prompt {shlex.quote(control_args.system_prompt)}",
        f"--seq_len {seq_len}",
        *(f"--prompt {shlex.quote(prompt)}" for prompt in prompts),
    ]
    if control_args.model_mode == "lookahead":
        args.extend(
            (
                f"--window {control_args.window}",
                f"--gcap {control_args.gcap}",
                f"--ngram {control_args.ngram}",
            )
        )
    return args


def read_generate_output(
    outputs_path: Path, inference_speed_path: Path
) -> GenerateOutput:
    """Read the runner's generated text and decode speed.

    Args:
        outputs_path: Host path of the ``RUNNER_OUTPUTS`` file.
        inference_speed_path: Host path of the ``RUNNER_INFERENCE_SPEED`` file.

    Returns:
        The generated text, decode speed, and both files.
    """
    return GenerateOutput(
        texts=[outputs_path.read_text()],
        tokens_per_sec=float(inference_speed_path.read_text()),
        files=[outputs_path, inference_speed_path],
    )
