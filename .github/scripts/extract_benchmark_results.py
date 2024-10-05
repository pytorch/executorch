#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import re
import time
import zipfile
from argparse import Action, ArgumentParser, Namespace
from io import BytesIO
from logging import info, warning
from typing import Any, List, Optional
from urllib import error, request


logging.basicConfig(level=logging.INFO)


BENCHMARK_RESULTS_FILENAME = "benchmark_results.json"
ARTIFACTS_FILENAME_REGEX = re.compile(r"(android|ios)-artifacts-(?P<job_id>\d+).json")


class ValidateArtifacts(Action):
    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: Any,
        option_string: Optional[str] = None,
    ) -> None:
        if os.path.isfile(values) and values.endswith(".json"):
            setattr(namespace, self.dest, values)
            return

        parser.error(f"{values} is not a valid JSON file (*.json)")


class ValidateOutputDir(Action):
    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: Any,
        option_string: Optional[str] = None,
    ) -> None:
        if os.path.isdir(values):
            setattr(namespace, self.dest, values)
            return

        parser.error(f"{values} is not a valid directory")


def parse_args() -> Any:
    from argparse import ArgumentParser

    parser = ArgumentParser("extract benchmark results from AWS Device Farm artifacts")
    parser.add_argument(
        "--artifacts",
        type=str,
        required=True,
        action=ValidateArtifacts,
        help="the list of artifacts from AWS in JSON format",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        action=ValidateOutputDir,
        help="the directory to keep the benchmark results",
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="which GitHub repo this workflow run belongs to",
    )
    parser.add_argument(
        "--head-branch",
        type=str,
        required=True,
        help="the head branch that runs",
    )
    parser.add_argument(
        "--workflow-name",
        type=str,
        required=True,
        help="the name of the benchmark workflow",
    )
    parser.add_argument(
        "--workflow-run-id",
        type=int,
        required=True,
        help="the id of the benchmark workflow",
    )
    parser.add_argument(
        "--workflow-run-attempt",
        type=int,
        required=True,
        help="which retry of the workflow this is",
    )

    return parser.parse_args()


def extract_android_benchmark_results(
    job_name: str, artifact_type: str, artifact_s3_url: str
) -> List:
    """
    The benchmark results from Android have already been stored in CUSTOMER_ARTIFACT
    artifact, so we will just need to get it

    Return the list of benchmark results.
    """
    if artifact_type != "CUSTOMER_ARTIFACT":
        return []

    try:
        with request.urlopen(artifact_s3_url) as data:
            with zipfile.ZipFile(BytesIO(data.read())) as customer_artifact:
                for name in customer_artifact.namelist():
                    if BENCHMARK_RESULTS_FILENAME in name:
                        return json.loads(customer_artifact.read(name))

    except error.HTTPError:
        warning(f"Fail to {artifact_type} {artifact_s3_url}")
        return []


def extract_job_id(artifacts_filename: str) -> int:
    """
    Extract the job id from the artifacts filename
    """
    m = ARTIFACTS_FILENAME_REGEX.match(os.path.basename(artifacts_filename))
    if not m:
        return 0
    return int(m.group("job_id"))


def transform(
    app_type: str,
    benchmark_results: List,
    repo: str,
    head_branch: str,
    workflow_name: str,
    workflow_run_id: int,
    workflow_run_attempt: int,
    job_name: str,
    job_id: int,
) -> List:
    """
    Transform the benchmark results into the format writable into the benchmark database
    """
    # Overwrite the device name here with the job name as it has more information about
    # the device, i.e. Samsung Galaxy S22 5G instead of just Samsung
    for r in benchmark_results:
        r["deviceInfo"]["device"] = job_name

    # TODO (huydhn): This is the current schema of the database oss_ci_benchmark_v2,
    # and I'm trying to fit ET benchmark results into it, which is kind of awkward.
    # However, the schema is going to be updated soon
    return [
        {
            # GH-info to identify where the benchmark is run
            "repo": repo,
            "head_branch": head_branch,
            "workflow_id": workflow_run_id,
            "run_attempt": workflow_run_attempt,
            "job_id": job_id,
            # The model
            "name": f"{r['benchmarkModel']['name']} {r['benchmarkModel'].get('backend', '')}".strip(),
            "dtype": (
                r["benchmarkModel"]["quantization"]
                if r["benchmarkModel"]["quantization"]
                else "unknown"
            ),
            # The metric value
            "metric": r["metric"],
            "actual": r["actualValue"],
            "target": r["targetValue"],
            # The device
            "device": r["deviceInfo"]["device"],
            "arch": r["deviceInfo"].get("os", ""),
            # Not used here, just set it to something unique here
            "filename": workflow_name,
            "test_name": app_type,
            "runner": job_name,
        }
        for r in benchmark_results
    ]


def main() -> None:
    args = parse_args()

    # Across all devices
    all_benchmark_results = []

    with open(args.artifacts) as f:
        for artifact in json.load(f):
            app_type = artifact.get("app_type", "")
            # We expect this to be set to either ANDROID_APP or IOS_APP
            if not app_type or app_type not in ["ANDROID_APP", "IOS_APP"]:
                info(
                    f"App type {app_type} is not recognized in artifact {json.dumps(artifact)}"
                )
                continue

            job_name = artifact["job_name"]
            artifact_type = artifact["type"]
            artifact_s3_url = artifact["s3_url"]

            if app_type == "ANDROID_APP":
                benchmark_results = extract_android_benchmark_results(
                    job_name, artifact_type, artifact_s3_url
                )
                if benchmark_results:
                    benchmark_results = transform(
                        app_type,
                        benchmark_results,
                        args.repo,
                        args.head_branch,
                        args.workflow_name,
                        args.workflow_run_id,
                        args.workflow_run_attempt,
                        job_name,
                        extract_job_id(args.artifacts),
                    )
                    all_benchmark_results.extend(benchmark_results)

            if app_type == "IOS_APP":
                # TODO (huydhn): Implement the logic for iOS next
                pass

    if all_benchmark_results:
        output_file = os.path.basename(args.artifacts)
        with open(f"{args.output_dir}/{output_file}", "w") as f:
            json.dump(all_benchmark_results, f)


if __name__ == "__main__":
    main()
