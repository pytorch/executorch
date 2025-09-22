# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Python bindings for ExecuTorch MultimodalRunner.

This module provides a Python interface to the ExecuTorch multimodal LLM runner,
enabling processing of mixed inputs (text, images, audio) and text generation.
"""

try:
    # Import shared components from the compiled C++ extension
    from executorch.extension.llm.runner._llm_runner import (  # noqa: F401
        GenerationConfig,
        Image,
        make_audio_input,
        make_image_input,
        make_raw_audio_input,
        make_text_input,
        make_token_input,
        MultimodalInput,
        MultimodalRunner,
        Stats,
    )
except ImportError:
    raise RuntimeError(
        "LLM runner is not installed. Please build ExecuTorch from source with EXECUTORCH_BUILD_PYBIND=ON"
    )


__all__ = [
    "GenerationConfig",
    "Image",
    "make_audio_input",
    "make_image_input",
    "make_raw_audio_input",
    "make_text_input",
    "make_token_input",
    "MultimodalInput",
    "MultimodalRunner",
    "Stats",
]
