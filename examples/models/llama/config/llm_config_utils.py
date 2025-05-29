# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from executorch.examples.models.llama.config.llm_config import LlmConfig


def convert_args_to_llm_config(args: argparse.Namespace) -> LlmConfig:
    """
    To support legacy purposes, this function converts CLI args from
    argparse to an LlmConfig, which is used by the LLM export process.
    """
    llm_config = LlmConfig()

    # TODO: conversion code.

    return llm_config
