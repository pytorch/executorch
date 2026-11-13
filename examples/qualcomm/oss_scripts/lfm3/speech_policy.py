# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sampling metadata for the supported Omni generation policy (no weights)."""

import math
from collections.abc import Mapping


def require_speech_calibration(
    output_hidden, calib_embeddings, pre_gen_pte=None, use_fp16=False
):
    if output_hidden and not pre_gen_pte and not use_fp16 and not calib_embeddings:
        raise ValueError(
            "Speech-output decoder export requires --calib_embeddings with "
            "captured speech-frame input sequences."
        )


def _number(config, name, default, minimum, maximum=None, integer=False):
    value = config.get(name, default)
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid generation setting {name}: {value!r}")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"Invalid generation setting {name}: {value!r}")
    if maximum is not None and value > maximum:
        raise ValueError(f"Invalid generation setting {name}: {value!r}")
    if integer and not isinstance(value, int):
        raise ValueError(f"Generation setting {name} must be an integer")
    return int(value) if integer else float(value)


def speech_sampling_metadata(config: Mapping) -> dict:
    # Pinned Omni generate uses generated-only deques, defaulting to 32/64.
    metadata = {"get_speech_sampling_version": 1}
    for name in ("do_sample", "temperature", "top_k", "repetition_penalty"):
        if name not in config or config[name] is None:
            raise ValueError(f"Generation configuration must explicitly set {name}")
    do_sample = config["do_sample"]
    if not isinstance(do_sample, bool):
        raise ValueError("do_sample must be a boolean")
    for prefix in ("text", "audio"):
        key = "" if prefix == "text" else "audio_"
        top_k = _number(config, key + "top_k", 0, 0, 2**31 - 1, True)
        temperature = config.get(key + "temperature")
        if temperature is None:
            temperature = 1.0 if top_k else 0.0
        temperature = _number({"temperature": temperature}, "temperature", 0.0, 0)
        if prefix == "text" and not do_sample:
            temperature, top_k = 0.0, 0
        metadata[f"get_{prefix}_temperature"] = temperature
        metadata[f"get_{prefix}_top_k"] = top_k
        penalty = _number(config, key + "repetition_penalty", 1.0, 0)
        if penalty == 0:
            raise ValueError("Repetition penalties must be positive")
        metadata[f"get_{prefix}_repetition_penalty"] = penalty
        metadata[f"get_{prefix}_repetition_window"] = _number(
            config,
            prefix + "_repetition_window",
            32 if not key else 64,
            0,
            2**31 - 1,
            True,
        )
    metadata["get_text_top_p"] = _number(config, "top_p", 1.0, 0, 1)
    metadata["get_text_min_p"] = _number(config, "min_p", 0.0, 0, 1)
    return metadata
