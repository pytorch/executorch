# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""
Thin wrapper around llama_runner to generate output
"""

import argparse
import subprocess
from argparse import Namespace
from dataclasses import dataclass
from typing import Optional

import torch

EVAL_HEADER = "=====START_EVAL====="
EVAL_FOOTER = "=====END_EVAL====="


@dataclass
class LlamaRunnerResults:
    logits: torch.Tensor
    raw_results: str

    prompt: Optional[str] = None


def run_llama_runner(
    model_path: str,
    tokenizer_path: str,
    prompt: str,
) -> str:
    args = [
        "buck2",
        "run",
        "fbcode//executorch/examples/models/llama2:main",
        "--",
        "--model_path",
        model_path,
        "--tokenizer_path",
        tokenizer_path,
        "--prompt",
        prompt,
        "--eval_mode",
    ]

    # Run the command
    output = subprocess.check_output(args)
    return output.decode()


def parse_results(result: str) -> None:
    """
    Parse the results from the output of the llama_runner
    """
    rows = result.split("\n")

    header = rows.index(EVAL_HEADER) if EVAL_HEADER in rows else None
    footer = rows.index(EVAL_FOOTER) if EVAL_FOOTER in rows else None
    if header is None or footer is None:
        raise ValueError("Could not find eval header or footer in results")
    elif footer - header != 2:
        raise ValueError("Expected 1 row of logits between eval header/footer")

    logit_row = [float(logit) for logit in rows[header + 1].split()]
    logits = torch.tensor(logit_row, dtype=torch.float32).view(1, -1, 32000)

    return LlamaRunnerResults(logits=logits, raw_results=result)


def get_llama_runner_result(
    model_path: str,
    tokenizer_path: str,
    prompt: str,
) -> LlamaRunnerResults:
    """
    Send the prompt to the llama_runner and parse the results
    """
    raw_result = run_llama_runner(
        model_path,
        tokenizer_path,
        prompt,
    )

    parsed_results = parse_results(raw_result)
    parse_results.prompt = prompt
    return parsed_results


def main(args: Namespace) -> None:
    llama_results = get_llama_runner_result(
        args.model_path,
        args.tokenizer_path,
        args.prompt,
    )
    print("Logits Shape: ", llama_results.logits.shape)


def parse_args() -> Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path of Executorch Model")
    parser.add_argument(
        "tokenizer_path", type=str, help="Path of Tokenizer to Feed the Model"
    )
    parser.add_argument(
        "prompt", type=str, default="I like Chocolates", help="Prompt to Feed the Model"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
