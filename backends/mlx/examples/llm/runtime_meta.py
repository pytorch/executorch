# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared runtime helpers for the MLX LLM example runners.

Exports publish their limits as constant methods (``get_max_ctx_len``,
``get_prefill_chunk_size``) so runners do not have to be told what a .pte
supports. This mirrors ``const_int`` in run_llm_hf.cpp.

Prompt handling (processor loading, chat templating, EOS lookup) lives here too:
it is driven by the checkpoint's own tokenizer config, so it is the same for
every model and every runner.
"""

import json
import logging
from typing import Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)


def read_const_int(program, name: str) -> Optional[int]:
    """Read an int constant method from a loaded program, or None if absent."""
    try:
        result = program.load_method(name).execute([])
    except Exception:
        return None
    if not result:
        return None
    value = result[0]
    return int(value) if isinstance(value, int) else int(value.item())


def read_model_limits(program) -> Tuple[Optional[int], Optional[int]]:
    """Return (max_ctx_len, prefill_chunk_size) as published by the export."""
    return (
        read_const_int(program, "get_max_ctx_len"),
        read_const_int(program, "get_prefill_chunk_size"),
    )


def chunked_prefill(
    method,
    input_ids: torch.Tensor,
    chunk_size: int,
    start_pos: int = 0,
    concat_outputs: Sequence[int] = (),
):
    """Prefill ``input_ids`` in steps of at most ``chunk_size`` tokens.

    Returns the final chunk's outputs, except that each index in
    ``concat_outputs`` is replaced by that output concatenated along dim 1
    across every chunk — used for per-token outputs like tapped hidden states,
    where the caller needs the whole prompt rather than the last step.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    seq_len = input_ids.shape[1]
    collected = {i: [] for i in concat_outputs}
    outputs = None

    for off in range(0, seq_len, chunk_size):
        end = min(off + chunk_size, seq_len)
        tokens = input_ids[:, off:end].contiguous()
        cache_position = torch.arange(
            start_pos + off, start_pos + end, dtype=torch.long
        )
        outputs = method.execute([tokens, cache_position.contiguous()])
        for i in collected:
            collected[i].append(outputs[i])

    if outputs is None:
        raise ValueError("input_ids is empty; nothing to prefill")

    outputs = list(outputs)
    for i, parts in collected.items():
        outputs[i] = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
    return outputs


def load_text_processor(
    model_id: str,
    revision: Optional[str] = None,
    local_files_only: bool = False,
):
    """Load the tokenizer for ``model_id``, falling back to its processor.

    Prefer AutoTokenizer for text-only prompting, even for checkpoints that also
    ship an AutoProcessor. Some hybrid checkpoints (for example Gemma 4) expose
    both, but the tokenizer path is the more stable interface for plain text
    generation.
    """
    from transformers import AutoProcessor, AutoTokenizer

    logger.info(f"Loading tokenizer from HuggingFace: {model_id}...")
    processor = None
    try:
        processor = AutoTokenizer.from_pretrained(
            model_id, revision=revision, local_files_only=local_files_only
        )
    except Exception as exc:
        logger.info(f"AutoTokenizer unavailable for {model_id}: {exc}")

    if processor is None:
        try:
            candidate = AutoProcessor.from_pretrained(
                model_id, revision=revision, local_files_only=local_files_only
            )
            if hasattr(candidate, "apply_chat_template") and hasattr(
                candidate, "decode"
            ):
                logger.info(f"Loaded processor from HuggingFace: {model_id}")
                processor = candidate
        except Exception as exc:
            logger.info(f"AutoProcessor unavailable for {model_id}: {exc}")

    if processor is None:
        raise RuntimeError(f"Could not load tokenizer or processor for {model_id}")

    _repair_chat_template(processor, model_id)
    return processor


def _repair_chat_template(processor, model_id: str) -> None:
    """Backfill chat_template from tokenizer_config.json when it did not load.

    Some transformers versions expect a standalone chat_template.jinja and do
    not fall back to the embedded tokenizer_config.json field.
    """
    if getattr(processor, "chat_template", None) is not None:
        return
    try:
        from pathlib import Path

        from huggingface_hub import hf_hub_download

        cfg_path = hf_hub_download(model_id, "tokenizer_config.json")
        cfg = json.loads(Path(cfg_path).read_text())
    except Exception as exc:
        logger.info(f"Could not backfill chat_template for {model_id}: {exc}")
        return
    if "chat_template" in cfg:
        processor.chat_template = cfg["chat_template"]


def apply_chat_template(
    processor, prompt: str, enable_thinking: bool = False
) -> torch.Tensor:
    """Render ``prompt`` through the checkpoint's chat template into input ids.

    ``enable_thinking`` is a template variable rather than a tokenizer argument;
    checkpoints whose template does not declare it are retried without it.
    """
    messages = [{"role": "user", "content": prompt}]
    kwargs = {"add_generation_prompt": True, "tokenize": True, "return_tensors": "pt"}
    try:
        out = processor.apply_chat_template(
            messages, enable_thinking=enable_thinking, **kwargs
        )
    except TypeError:
        out = processor.apply_chat_template(messages, **kwargs)
    # Different transformers versions return either a BatchEncoding or a tensor.
    return out.input_ids if hasattr(out, "input_ids") else out


def get_eos_token_ids(processor, model_id=None, local_files_only=False):
    """Collect every id that should stop generation.

    A checkpoint can declare several: Qwen3 stops on both ``<|im_end|>`` and
    ``<|endoftext|>``, but only the former is ``tokenizer.eos_token_id``. The
    rest live in generation_config, so pass ``model_id`` to pick them up.
    """
    eos_ids = set()

    candidate = getattr(processor, "eos_token_id", None)
    if candidate is None:
        candidate = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None)
    eos_ids.update(_as_id_set(candidate))

    if model_id is not None:
        try:
            from transformers import GenerationConfig

            generation_config = GenerationConfig.from_pretrained(
                model_id, local_files_only=local_files_only
            )
            eos_ids.update(_as_id_set(generation_config.eos_token_id))
        except Exception as exc:
            logger.info(f"No generation_config eos ids for {model_id}: {exc}")

    return eos_ids


def _as_id_set(value):
    """Normalize an int / list / None token-id field into a set of ints."""
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(v) for v in value if v is not None}
