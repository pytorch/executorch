#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, NamedTuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from examples.models import MODEL_NAME_TO_MODEL


DEVICE_POOLS_REGEX = re.compile(r"(?P<device_name>[^\+]+)\+(?P<variant>[^\+]+)")
# Device pools for AWS Device Farm. Initially, I choose to distribute models to these pool
# round-robin for simplicity. For public pool, only one per device type is needed because
# AWS will scale the number of devices there for us. However, for private pool, we need to
# manually maintain multiple pools of the same device to evenly distribute models there.
# The pool ARNs are extracted from the output of the following command:
#   aws devicefarm list-device-pools \
#    --arn arn:aws:devicefarm:us-west-2:308535385114:project:02a2cf0f-6d9b-45ee-ba1a-a086587469e6 \
#    --region us-west-2
DEVICE_POOLS = {
    "apple_iphone_15": {
        "public": [
            "arn:aws:devicefarm:us-west-2:308535385114:devicepool:02a2cf0f-6d9b-45ee-ba1a-a086587469e6/3b5acd2e-92e2-4778-b651-7726bafe129d",
        ],
        "ios_18_public": [
            "arn:aws:devicefarm:us-west-2:308535385114:devicepool:02a2cf0f-6d9b-45ee-ba1a-a086587469e6/12c8b15c-8d03-4e07-950d-0a627e7595b4",
        ],
        "private": [
            "arn:aws:devicefarm:us-west-2:308535385114:devicepool:02a2cf0f-6d9b-45ee-ba1a-a086587469e6/55929353-2f28-4ee5-bdff-d1a95f58cb28",
        ],
        "plus_private": [
            "arn:aws:devicefarm:us-west-2:308535385114:devicepool:02a2cf0f-6d9b-45ee-ba1a-a086587469e6/767bfb3e-a00e-4d92-998b-4eafdcf7213b",
        ],
        "pro_private": [
            "arn:aws:devicefarm:us-west-2:308535385114:devicepool:02a2cf0f-6d9b-45ee-ba1a-a086587469e6/1394f34c-2981-4c55-aaa2-246871ac713b",
            "arn:aws:devicefarm:us-west-2:308535385114:devicepool:02a2cf0f-6d9b-45ee-ba1a-a086587469e6/099e8def-4609-4383-8787-76b88e500c1d",
            "arn:aws:devicefarm:us-west-2:308535385114:devicepool:02a2cf0f-6d9b-45ee-ba1a-a086587469e6/d6707270-b009-479e-a83a-7bdb255f9de5",
        ],
    },
    "samsung_galaxy_s22": {
        "public": [
            "arn:aws:devicefarm:us-west-2:308535385114:devicepool:02a2cf0f-6d9b-45ee-ba1a-a086587469e6/e59f866a-30aa-4aa1-87b7-4510e5820dfa",
        ],
        "private": [
            "arn:aws:devicefarm:us-west-2:308535385114:devicepool:02a2cf0f-6d9b-45ee-ba1a-a086587469e6/ea6b049d-1508-4233-9a56-5d9eacbe1078",
            "arn:aws:devicefarm:us-west-2:308535385114:devicepool:02a2cf0f-6d9b-45ee-ba1a-a086587469e6/1fa924a1-5aff-475b-8f4d-f7c6d8de4fe9",
        ],
        "ultra_private": [
            "arn:aws:devicefarm:us-west-2:308535385114:devicepool:02a2cf0f-6d9b-45ee-ba1a-a086587469e6/5f79d72e-e229-4f9c-962f-5d37196fcfe7",
        ],
    },
    "samsung_galaxy_s24": {
        "public": [
            "arn:aws:devicefarm:us-west-2:308535385114:devicepool:02a2cf0f-6d9b-45ee-ba1a-a086587469e6/98f8788c-2e25-4a3c-8bb2-0d1e8897c0db",
        ],
        "ultra_private": [
            "arn:aws:devicefarm:us-west-2:308535385114:devicepool:02a2cf0f-6d9b-45ee-ba1a-a086587469e6/5f79d72e-e229-4f9c-962f-5d37196fcfe7",
        ],
    },
    "google_pixel_8": {
        "pro_public": [
            "arn:aws:devicefarm:us-west-2:308535385114:devicepool:02a2cf0f-6d9b-45ee-ba1a-a086587469e6/d65096ab-900b-4521-be8b-a3619b69236a",
        ],
    },
    "google_pixel_3": {
        "rooted_private": [
            "arn:aws:devicefarm:us-west-2:308535385114:devicepool:02a2cf0f-6d9b-45ee-ba1a-a086587469e6/98d23ca8-ea9e-4fb7-b725-d402017b198d",
        ],
    },
}

# Predefined benchmark configurations
BENCHMARK_CONFIGS = {
    "xplat": [
        "xnnpack_q8",
        "hf_xnnpack_custom_spda_kv_cache_8da4w",
        "et_xnnpack_custom_spda_kv_cache_8da4w",
        "llama3_fb16",
        "llama3_spinquant",
        "llama3_qlora",
    ],
    "android": [
        "qnn_q8",
        # TODO: Add support for llama3 htp
        # "llama3_qnn_htp",
    ],
    "ios": [
        "coreml_fp16",
        "mps",
        "llama3_coreml_ane",
    ],
}


class DisabledConfig(NamedTuple):
    config_name: str
    github_issue: str  # Link to the GitHub issue


# Updated DISABLED_CONFIGS
DISABLED_CONFIGS: Dict[str, List[DisabledConfig]] = {
    "resnet50": [
        DisabledConfig(
            config_name="qnn_q8",
            github_issue="https://github.com/pytorch/executorch/issues/7892",
        ),
    ],
    "w2l": [
        DisabledConfig(
            config_name="qnn_q8",
            github_issue="https://github.com/pytorch/executorch/issues/7634",
        ),
    ],
    "mobilebert": [
        DisabledConfig(
            config_name="mps",
            github_issue="https://github.com/pytorch/executorch/issues/7904",
        ),
        DisabledConfig(
            config_name="qnn_q8",
            github_issue="https://github.com/pytorch/executorch/issues/7946",
        ),
    ],
    "edsr": [
        DisabledConfig(
            config_name="mps",
            github_issue="https://github.com/pytorch/executorch/issues/7905",
        ),
    ],
    "llama": [
        DisabledConfig(
            config_name="mps",
            github_issue="https://github.com/pytorch/executorch/issues/7907",
        ),
    ],
}


def extract_all_configs(data, target_os=None):
    if isinstance(data, dict):
        # If target_os is specified, include "xplat" and the specified branch
        include_branches = {"xplat", target_os} if target_os else data.keys()
        return [
            v
            for key, value in data.items()
            if key in include_branches
            for v in extract_all_configs(value, target_os)
        ]
    elif isinstance(data, list):
        return [v for item in data for v in extract_all_configs(item, target_os)]
    else:
        return [data]


def generate_compatible_configs(model_name: str, target_os=None) -> List[str]:
    """
    Generate a list of compatible benchmark configurations for a given model name and target OS.

    Args:
        model_name (str): The name of the model to generate configurations for.
        target_os (Optional[str]): The target operating system (e.g., 'android', 'ios').

    Returns:
        List[str]: A list of compatible benchmark configurations.

    Raises:
        None

    Example:
        generate_compatible_configs('meta-llama/Llama-3.2-1B', 'ios') -> ['llama3_fb16', 'llama3_coreml_ane']
    """
    configs = []
    if is_valid_huggingface_model_id(model_name):
        configs.append("hf_xnnpack_custom_spda_kv_cache_8da4w")
        if model_name.startswith("meta-llama/"):
            # etLLM recipes for Llama
            repo_name = model_name.split("meta-llama/")[1]
            if "qlora" in repo_name.lower():
                configs = ["llama3_qlora"]
            elif "spinquant" in repo_name.lower():
                configs = ["llama3_spinquant"]
            else:
                configs.extend(["llama3_fb16", "et_xnnpack_custom_spda_kv_cache_8da4w"])
                configs.extend(
                    [
                        config
                        for config in BENCHMARK_CONFIGS.get(target_os, [])
                        if config.startswith("llama")
                    ]
                )
        if model_name.startswith("Qwen/Qwen3"):
            configs.append("et_xnnpack_custom_spda_kv_cache_8da4w")
    elif model_name in MODEL_NAME_TO_MODEL:
        # ExecuTorch in-tree non-GenAI models
        configs.append("xnnpack_q8")
        if target_os != "xplat":
            # Add OS-specific configs
            configs.extend(
                [
                    config
                    for config in BENCHMARK_CONFIGS.get(target_os, [])
                    if not config.startswith("llama")
                ]
            )
    else:
        # Skip unknown models with a warning
        logging.warning(f"Unknown or invalid model name '{model_name}'. Skipping.")

    # Remove disabled configs for the given model
    disabled_configs = DISABLED_CONFIGS.get(model_name, [])
    disabled_config_names = {disabled.config_name for disabled in disabled_configs}
    for disabled in disabled_configs:
        print(
            f"Excluding disabled config: '{disabled.config_name}' for model '{model_name}' on '{target_os}'. Linked GitHub issue: {disabled.github_issue}"
        )
    configs = [config for config in configs if config not in disabled_config_names]
    return configs


def parse_args() -> Any:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    Example:
        parse_args() -> Namespace(models=['mv3', 'meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8'],
                                   os='android',
                                   devices=['samsung_galaxy_s22'])
    """
    from argparse import ArgumentParser

    def comma_separated(value: str):
        """
        Parse a comma-separated string into a list.
        """
        return value.split(",")

    parser = ArgumentParser("Gather all benchmark configs.")
    parser.add_argument(
        "--os",
        type=str,
        choices=["android", "ios"],
        help="The target OS.",
    )
    parser.add_argument(
        "--models",
        type=comma_separated,  # Use the custom parser for comma-separated values
        help=f"Comma-separated model IDs or names. Valid values include {MODEL_NAME_TO_MODEL}.",
    )
    parser.add_argument(
        "--devices",
        type=comma_separated,  # Use the custom parser for comma-separated values
        help=f"Comma-separated device names. Available devices: {list(DEVICE_POOLS.keys())}",
    )
    parser.add_argument(
        "--configs",
        type=comma_separated,  # Use the custom parser for comma-separated values
        help=f"Comma-separated benchmark configs. Available configs: {extract_all_configs(BENCHMARK_CONFIGS)}",
    )

    return parser.parse_args()


def set_output(name: str, val: Any) -> None:
    """
    Set the output value to be used by other GitHub jobs.

    Args:
        name (str): The name of the output variable.
        val (Any): The value to set for the output variable.

    Example:
        set_output("benchmark_configs", {"include": [...]})
    """

    github_output = os.getenv("GITHUB_OUTPUT")
    if not github_output:
        print(f"::set-output name={name}::{val}")
        return

    try:
        with open(github_output, "a") as env:
            env.write(f"{name}={val}\n")
    except (PermissionError, FileNotFoundError):
        # Fall back to printing in case of permission error in unit tests
        print(f"::set-output name={name}::{val}")


def is_valid_huggingface_model_id(model_name: str) -> bool:
    """
    Validate if the model name matches the pattern for HuggingFace model IDs.

    Args:
        model_name (str): The model name to validate.

    Returns:
        bool: True if the model name matches the valid pattern, False otherwise.

    Example:
        is_valid_huggingface_model_id('meta-llama/Llama-3.2-1B') -> True
    """
    pattern = r"^[a-zA-Z0-9-_]+/[a-zA-Z0-9-_.]+$"
    return bool(re.match(pattern, model_name))


def get_benchmark_configs() -> Dict[str, Dict]:  # noqa: C901
    """
    Gather benchmark configurations for a given set of models on the target operating system and devices.
    CHANGE IF this function's return changed:
        extract_model_info() in executorch/.github/scripts/extract_benchmark_results.py IF YOU CHANGE THE RESULT OF THIS FUNCTION.
    Args:
        None

    Returns:
        Dict[str, Dict]: A dictionary containing the benchmark configurations.

    Example:
        get_benchmark_configs() -> {
            "include": [
                {
                    "model": "meta-llama/Llama-3.2-1B",
                    "config": "llama3_qlora",
                    "device_name": "apple_iphone_15",
                    "device_arn": "arn:aws:..."
                },
                {
                    "model": "mv3",
                    "config": "xnnpack_q8",
                    "device_name": "samsung_galaxy_s22",
                    "device_arn": "arn:aws:..."
                },
                ...
            ]
        }
    """
    args = parse_args()
    devices = args.devices
    models = args.models
    target_os = args.os
    target_configs = args.configs

    benchmark_configs = {"include": []}

    for model_name in models:
        configs = []
        configs.extend(generate_compatible_configs(model_name, target_os))
        print(f"Discovered all supported configs for model '{model_name}': {configs}")
        if target_configs is not None:
            for config in target_configs:
                if config not in configs:
                    raise Exception(
                        f"Unsupported config '{config}' for model '{model_name}' on '{target_os}'. Skipped.\n"
                        f"Supported configs are: {configs}"
                    )
            configs = target_configs
            print(f"Using provided configs {configs} for model '{model_name}'")

        # Add configurations for each valid device
        for device in devices:
            # Parse the device name
            m = re.match(DEVICE_POOLS_REGEX, device)
            if not m:
                logging.warning(
                    f"Invalid device name: {device} is not in DEVICE_NAME+VARIANT format. Skipping."
                )
                continue

            device_name = m.group("device_name")
            variant = m.group("variant")

            if device_name not in DEVICE_POOLS:
                logging.warning(f"Unsupported device '{device}'. Skipping.")
                continue

            if variant not in DEVICE_POOLS[device_name]:
                logging.warning(
                    f"Unsupported {device}'s variant '{variant}'. Skipping."
                )
                continue

            device_pool_count = len(DEVICE_POOLS[device_name][variant])
            if not device_pool_count:
                logging.warning(
                    f"No device pool defined for {device}'s variant '{variant}'. Skipping."
                )
                continue

            device_pool_index = 0
            for config in configs:
                if config == "llama3_coreml_ane" and "ios_18" not in variant:
                    variant = "ios_18_public"
                    logging.info(
                        f"Benchmark config '{config}' only works on iOS 18+, auto-upgraded device variant to '{variant}'"
                    )

                record = {
                    "model": model_name,
                    "config": config,
                    "device_name": device_name,
                    "variant": variant,
                    "device_arn": DEVICE_POOLS[device_name][variant][
                        device_pool_index % device_pool_count
                    ],
                }
                benchmark_configs["include"].append(record)

                # Distribute configs to pools of the same device round-robin
                device_pool_index += 1

    set_output("benchmark_configs", json.dumps(benchmark_configs))


if __name__ == "__main__":
    get_benchmark_configs()
