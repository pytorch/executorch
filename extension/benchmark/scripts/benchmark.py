#!/usr/bin/env python3
import json
import logging
import os
from argparse import ArgumentParser
from logging import info
from re import A
from shutil import Error
from typing import Any

import requests


GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
logging.basicConfig(level=logging.INFO)


def parse_args() -> Any:
    parser = ArgumentParser(
        "Run Android and iOS tests on AWS Device Farm via github actions workflow run"
    )

    parser.add_argument(
        "--branch",
        type=str,
        required=True,
        help="what (non-fork) gh branch to use in pytorch/executorch",
    )

    platform = parser.add_mutually_exclusive_group(required=True)
    platform.add_argument(
        "--android",
        action="store_true",
        required=False,
        help="run the test on Android",
    )
    platform.add_argument(
        "--ios",
        action="store_true",
        required=False,
        help="run the test on iOS",
    )

    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help='Comma separated list of Models for benchmarking. Model options: https://github.com/pytorch/executorch/blob/0342babc505bcb90244874e9ed9218d90dd67b87/examples/models/__init__.py#L53 or ok to use HuggingFace model name, e.g. "meta-llama/Llama-3.2-1B"',
    )

    parser.add_argument(
        "--devices",
        type=str,
        required=False,
        default="",
        # TODO update example or add choices once we establish custom device pools
        help="Comma-separated list of specific devices to run the benchmark on. Defaults to device pools for approriate platform. For example, `--devices samsung_galaxy_s22,samsung_galaxy_s24`.",
    )

    parser.add_argument(
        "--benchmark-configs",
        type=str,
        required=False,
        default="",
        help="Comma-separated list of benchmark configs to use. For example, `--benchmark-configs xnnpack_q8,hf_xnnpack_fp32,llama3_fb16` (See https://github.com/pytorch/executorch/blob/main/.ci/scripts/gather_benchmark_configs.py#L29-L47 for options)",
    )

    args, unknown = parser.parse_known_args()
    if len(unknown) > 0:
        info(f"detected unknown flags: {unknown}")
    return args


def run_workflow(platform, branch, models, devices, benchmark_configs):
    dispatch_hook = "/dispatches"
    if platform == "android":
        url = f"https://api.github.com/repos/pytorch/executorch/actions/workflows/android-perf.yml"
    else:
        url = f"https://api.github.com/repos/pytorch/executorch/actions/workflows/apple-perf.yml"

    # see github workflow dispatch for header details https://docs.github.com/en/rest/actions/workflows#create-a-workflow-dispatch-event
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    data = {
        "ref": f"{branch}",
        "inputs": {
            "models": f"{models}",
            "devices": f"{devices}",
            "benchmark_configs": f"{benchmark_configs}",
        },
    }

    resp = requests.post(url + dispatch_hook, headers=headers, data=json.dumps(data))
    if resp.status_code != 204:
        raise Exception(f"Failed to start workflow: {resp.text}")
    else:
        print("Workflow started successfully.")
        if platform == "android":
            print(
                "Find your workflow run here: https://github.com/pytorch/executorch/actions/workflows/android-perf.yml"
            )
        else:
            print(
                "Find your workflow run here: https://github.com/pytorch/executorch/actions/workflows/apple-perf.yml"
            )


def main() -> None:
    args = parse_args()
    platform = None
    if args.android:
        platform = "android"
    elif args.ios:
        platform = "ios"
    if platform:
        resp = run_workflow(
            platform, args.branch, args.models, args.devices, args.benchmark_configs
        )
    else:
        raise Error("No app type specified. Please specify either --android or --ios.")


if __name__ == "__main__":
    main()
