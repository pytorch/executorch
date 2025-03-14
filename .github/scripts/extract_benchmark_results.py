#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import glob
import json
import logging
import os
import re
import zipfile
from argparse import Action, ArgumentParser, Namespace
from io import BytesIO
from logging import info, warning
from typing import Any, DefaultDict, Dict, List, Optional
from urllib import error, request


logging.basicConfig(level=logging.INFO)


BENCHMARK_RESULTS_FILENAME = "benchmark_results.json"
ARTIFACTS_FILENAME_REGEX = re.compile(r"(android|ios)-artifacts-(?P<job_id>\d+).json")
BENCHMARK_CONFIG_REGEX = re.compile(r"The benchmark config is (?P<benchmark_config>.+)")

# iOS-related regexes and variables
IOS_TEST_SPEC_REGEX = re.compile(
    r"Test Case\s+'-\[(?P<test_class>\w+)\s+(?P<test_name>[\w\+]+)\]'\s+measured\s+\[(?P<metric>.+)\]\s+average:\s+(?P<value>[\d\.]+),"
)
IOS_TEST_NAME_REGEX = re.compile(
    r"test_(?P<method>forward|load|generate)_(?P<model_name>[\w\+]+)_pte.*iOS_(?P<ios_ver>\w+)_iPhone(?P<iphone_ver>\w+)"
)
# The backend name could contain +, i.e. tinyllama_xnnpack+custom+qe_fp32
IOS_MODEL_NAME_REGEX = re.compile(
    r"(?P<model>[^_]+)_(?P<backend>[\w\+]+)_(?P<dtype>\w+)"
)


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


class ValidateDir(Action):
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
        action=ValidateDir,
        help="the directory to keep the benchmark results",
    )
    parser.add_argument(
        "--benchmark-configs",
        type=str,
        required=True,
        action=ValidateDir,
        help="the directory to keep the benchmark configs",
    )

    parser.add_argument(
        "--app",
        type=str,
        required=True,
        choices=["android", "ios"],
        help="the type of app, ios or android, this is mainly used when a failed job happens to generate default record",
    )

    return parser.parse_args()


def extract_android_benchmark_results(artifact_type: str, artifact_s3_url: str) -> List:
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
    except json.decoder.JSONDecodeError:
        # This is to handle the case where there is no benchmark results
        warning(f"Fail to load the benchmark results from {artifact_s3_url}")
        return []
    return []


def initialize_ios_metadata(test_name: str) -> Dict[str, Any]:
    """
    Extract the benchmark metadata from the test name, for example:
        test_forward_llama2_pte_iOS_17_2_1_iPhone15_4
        test_load_resnet50_xnnpack_q8_pte_iOS_17_2_1_iPhone15_4
    """
    m = IOS_TEST_NAME_REGEX.match(test_name)
    if not m:
        return {}

    method = m.group("method")
    model_name = m.group("model_name")
    ios_ver = m.group("ios_ver").replace("_", ".")
    iphone_ver = m.group("iphone_ver").replace("_", ".")

    # The default backend and quantization dtype if the script couldn't extract
    # them from the model name
    backend = ""
    quantization = "unknown"

    m = IOS_MODEL_NAME_REGEX.match(model_name)
    if m:
        backend = m.group("backend")
        quantization = m.group("dtype")
        model_name = m.group("model")

    return {
        "benchmarkModel": {
            "backend": backend,
            "quantization": quantization,
            "name": model_name,
        },
        "deviceInfo": {
            "arch": f"iPhone {iphone_ver}",
            "device": f"iPhone {iphone_ver}",
            "os": f"iOS {ios_ver}",
            "availMem": 0,
            "totalMem": 0,
        },
        "method": method,
        # These fields will be populated later by extract_ios_metric
        "metric": "",
        "actualValue": 0,
        "targetValue": 0,
    }


def extract_ios_metric(
    benchmark_result: Dict[str, Any],
    test_name: str,
    metric_name: str,
    metric_value: float,
) -> Dict[str, Any]:
    """
    Map the metric name from iOS xcresult to the benchmark result
    """
    method = benchmark_result.get("method", "")
    if not method:
        return benchmark_result

    # NB: This looks brittle, but unless we can return iOS benchmark results in JSON
    # format by the test, the mapping is needed to match with Android test
    if method == "load":
        if metric_name == "Clock Monotonic Time, s":
            benchmark_result["metric"] = "model_load_time(ms)"
            benchmark_result["actualValue"] = metric_value * 1000

        elif metric_name == "Memory Peak Physical, kB":
            # NB: Showing the value in mB is friendlier IMO
            benchmark_result["metric"] = "peak_load_mem_usage(mb)"
            benchmark_result["actualValue"] = metric_value / 1024

    elif method == "forward":
        if metric_name == "Clock Monotonic Time, s":
            benchmark_result["metric"] = "avg_inference_latency(ms)"
            benchmark_result["actualValue"] = metric_value * 1000

        elif metric_name == "Memory Peak Physical, kB":
            # NB: Showing the value in mB is friendlier IMO
            benchmark_result["metric"] = "peak_inference_mem_usage(mb)"
            benchmark_result["actualValue"] = metric_value / 1024

    elif method == "generate":
        if metric_name == "Clock Monotonic Time, s":
            benchmark_result["metric"] = "generate_time(ms)"
            benchmark_result["actualValue"] = metric_value * 1000

        elif metric_name == "Tokens Per Second, t/s":
            benchmark_result["metric"] = "token_per_sec"
            benchmark_result["actualValue"] = metric_value

    return benchmark_result


def extract_ios_benchmark_results(artifact_type: str, artifact_s3_url: str) -> List:
    """
    The benchmark results from iOS are currently from xcresult, which could either
    be parsed from CUSTOMER_ARTIFACT or get from the test spec output. The latter
    is probably easier to process
    """
    if artifact_type != "TESTSPEC_OUTPUT":
        return []

    try:
        benchmark_results = []

        with request.urlopen(artifact_s3_url) as data:
            current_test_name = ""
            current_metric_name = ""
            current_record = {}

            for line in data.read().decode("utf8").splitlines():
                s = IOS_TEST_SPEC_REGEX.search(line)
                if not s:
                    continue

                test_name = s.group("test_name")
                metric_name = s.group("metric")
                metric_value = float(s.group("value"))

                if test_name != current_test_name or metric_name != current_metric_name:
                    if current_record and current_record.get("metric", ""):
                        # Save the benchmark result in the same format used by Android
                        benchmark_results.append(current_record.copy())

                    current_test_name = test_name
                    current_metric_name = metric_name
                    current_record = initialize_ios_metadata(current_test_name)

                current_record = extract_ios_metric(
                    current_record, test_name, metric_name, metric_value
                )

            if current_record and current_record.get("metric", ""):
                benchmark_results.append(current_record.copy())

        return benchmark_results

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


def read_all_benchmark_configs() -> Dict[str, Dict[str, str]]:
    """
    Read all the benchmark configs that we can find
    """
    benchmark_configs = {}

    for file in glob.glob(f"{benchmark_configs}/*.json"):
        filename = os.path.basename(file)
        with open(file) as f:
            try:
                benchmark_configs[filename] = json.load(f)
            except json.JSONDecodeError as e:
                warning(f"Fail to load benchmark config {file}: {e}")

    return benchmark_configs


def read_benchmark_config(
    artifact_s3_url: str, benchmark_configs_dir: str
) -> Dict[str, str]:
    """
    Get the correct benchmark config for this benchmark run
    """
    try:
        with request.urlopen(artifact_s3_url) as data:
            for line in data.read().decode("utf8").splitlines():
                m = BENCHMARK_CONFIG_REGEX.match(line)
                if not m:
                    continue

                benchmark_config = m.group("benchmark_config")
                filename = os.path.join(
                    benchmark_configs_dir, f"{benchmark_config}.json"
                )

                if not os.path.exists(filename):
                    warning(f"There is no benchmark config {filename}")
                    continue

                with open(filename) as f:
                    try:
                        return json.load(f)
                    except json.JSONDecodeError as e:
                        warning(f"Fail to load benchmark config {filename}: {e}")
    except error.HTTPError:
        warning(f"Fail to read the test spec output at {artifact_s3_url}")

    return {}


def transform(
    app_type: str,
    benchmark_results: List,
    benchmark_config: Dict[str, str],
    job_name: str,
) -> List:
    """
    Transform the benchmark results into the format writable into the benchmark database
    """
    # Overwrite the device name here with the job name as it has more information about
    # the device, i.e. Samsung Galaxy S22 5G instead of just Samsung
    for r in benchmark_results:
        r["deviceInfo"]["device"] = job_name

    # From https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database
    return [
        {
            "benchmark": {
                "name": "ExecuTorch",
                "mode": "inference",
                "extra_info": {
                    "app_type": app_type,
                    # Just keep a copy of the benchmark config here
                    "benchmark_config": json.dumps(benchmark_config),
                },
            },
            "model": {
                "name": benchmark_config.get("model", r["benchmarkModel"]["name"]),
                "type": "OSS model",
                "backend": benchmark_config.get(
                    "config", r["benchmarkModel"].get("backend", "")
                ),
            },
            "metric": {
                "name": r["metric"],
                "benchmark_values": [r["actualValue"]],
                "target_value": r["targetValue"],
                "extra_info": {
                    "method": r.get("method", ""),
                },
            },
            "runners": [
                {
                    "name": r["deviceInfo"]["device"],
                    "type": r["deviceInfo"]["os"],
                    "avail_mem_in_gb": r["deviceInfo"].get("availMem", ""),
                    "total_mem_in_gb": r["deviceInfo"].get("totalMem", ""),
                }
            ],
        }
        for r in benchmark_results
    ]


def extract_model_info(git_job_name: str) -> Optional[Dict[str, str]]:
    """
    Get model infomation form git_job_name, for example:
        benchmark-on-device (ic4, qnn_q8, samsung_galaxy_s24, arn:aws:devicefarm:us-west-2:308535385114:d... / mobile-job (android)
        benchmark-on-device (llama, xnnpack_q8, apple_iphone_15, arn:aws:devicefarm:us-west-2:30853538511... / mobile-job (ios)
    """
    # Extract content inside the first parentheses,

    pattern = r"benchmark-on-device \((.+)"
    match = re.search(pattern, git_job_name)
    if not match:
        warning(
            f"pattern not found from git_job_name {git_job_name}, cannot extract correct names"
        )
        return None

    extracted_content = match.group(1)  # Get content after the opening parenthesis
    items = extracted_content.split(",")
    if len(items) < 3:
        warning(
            f"expect at least 3 items extrac from git_job_name {git_job_name}, but got {items}"
        )
        return None

    return {
        "model_name": items[0].strip(),
        "model_backend": items[1].strip(),
        "device_pool_name": items[2].strip(),
    }


def transform_failure_record(
    app_type: str,
    level: str,
    model_name: str,
    model_backend: str,
    device_name: str,
    device_os: str,
    result: str,
    report: Any = {},
) -> Any:
    """
    Transform the benchmark results into the format writable into the benchmark database for job failures
    """
    # From https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database
    return {
        "benchmark": {
            "name": "ExecuTorch",
            "mode": "inference",
            "extra_info": {
                "app_type": app_type,
                "job_conclusion": result,
                "failure_level": level,
                "job_report": json.dumps(report),
            },
        },
        "model": {
            "name": model_name,
            "type": "OSS model",
            "backend": model_backend,
        },
        "metric": {
            "name": "FAILURE_REPORT",
            "benchmark_values": 0,
            "target_value": 0,
            "extra_info": {
                "method": "",
            },
        },
        "runners": [
            {
                "name": device_name,
                "type": device_os,
                "avail_mem_in_gb": "",
                "total_mem_in_gb": "",
            }
        ],
    }


def to_job_report_map(job_reports) -> Dict[str, Any]:
    return {job_report["arn"]: job_report for job_report in job_reports}


def group_by_arn(artifacts: List) -> Dict[str, List]:
    """
    Group the artifacts by the job ARN
    """
    arn_to_artifacts = DefaultDict(list)
    for artifact in artifacts:
        job_arn = artifact.get("job_arn", "")
        app_type = artifact.get("app_type", "")
        if not app_type or app_type not in ["ANDROID_APP", "IOS_APP"]:
            info(
                f"App type {app_type} is not recognized in artifact {json.dumps(artifact)}"
            )
            continue
        if not job_arn:
            info(f"missing job_arn in artifact {json.dumps(artifact)}")
            continue
        arn_to_artifacts[job_arn].append(artifact)
    return arn_to_artifacts


# get the benchmark config from TestSpec file if any exist
def get_benchmark_config(
    artifacts: List[Dict[str, Any]], benchmark_configs: str
) -> Dict[str, str]:
    result = next(
        (artifact for artifact in artifacts if artifact["type"] == "TESTSPEC_OUTPUT"),
        None,
    )
    if not result:
        return {}
    artifact_s3_url = result["s3_url"]
    return read_benchmark_config(artifact_s3_url, benchmark_configs)


def extractBenchmarkResultFromArtifact(
    artifact: Dict[str, Any],
    benchmark_config: Dict[str, str],
) -> List[Any]:
    job_name = artifact.get("job_name", "")
    artifact_type = artifact.get("type", "")
    artifact_s3_url = artifact.get("s3_url", "")
    app_type = artifact.get("app_type", "")

    info(
        f"Processing {app_type} artifact: {job_name} {artifact_type} {artifact_s3_url}"
    )
    benchmark_results = []
    if app_type == "ANDROID_APP":
        benchmark_results = extract_android_benchmark_results(
            artifact_type, artifact_s3_url
        )
    if app_type == "IOS_APP":
        benchmark_results = extract_ios_benchmark_results(
            artifact_type, artifact_s3_url
        )
    if not benchmark_results:
        return []
    return transform(app_type, benchmark_results, benchmark_config, job_name)


def getAppType(type: str):
    match type:
        case "ios":
            return "IOS_APP"
        case "android":
            return "ANDROID_APP"
    warning(
        f"unknown device type detected: {type}, currently we only support ios and android"
    )
    return "UNKNOWN"


def getDeviceOsType(type: str):
    match type:
        case "ios":
            return "iOS"
        case "android":
            return "Android"
    return "UNKNOWN"


def generateGitJobLevelFailureRecord(git_job_name: str, app: str) -> Any:
    """
    generates benchmark record for GIT_JOB level failure, this is mainly used as placeholder in UI to indicate job failures.
    """
    level = "GIT_JOB"
    app_type = getAppType(app)
    device_prefix = getDeviceOsType(app)

    model_infos = extract_model_info(git_job_name)
    model_name = "UNKNOWN"
    model_backend = "UNKNOWN"
    device_pool_name = "UNKNOWN"

    if model_infos:
        model_name = model_infos["model_name"]
        model_backend = model_infos["model_backend"]
        device_pool_name = model_infos["device_pool_name"]
    return transform_failure_record(
        app_type,
        level,
        model_name,
        model_backend,
        device_pool_name,
        device_prefix,
        "FAILURE",
    )


def generateDeviceLevelFailureRecord(
    git_job_name: str, job_report: Any, app: str
) -> Any:
    """
    generates benchmark record for DEVICE_JOB level failure, this is mainly used as placeholder in UI to indicate job failures.
    """
    level = "DEVICE_JOB"
    model_infos = extract_model_info(git_job_name)
    model_name = "UNKNOWN"
    model_backend = "UNKNOWN"
    osPrefix = getDeviceOsType(app)
    job_report_os = job_report["os"]

    # make sure the device os name has prefix iOS and Android
    device_os = job_report_os
    if not job_report_os.startswith(osPrefix):
        device_os = f"{osPrefix} {job_report_os}"

    if model_infos:
        model_name = model_infos["model_name"]
        model_backend = model_infos["model_backend"]
    return transform_failure_record(
        job_report["app_type"],
        level,
        model_name,
        model_backend,
        job_report["name"],
        device_os,
        job_report["result"],
        job_report,
    )


def process_benchmark_results(content: Any, app: str, benchmark_configs: str):
    """
    main code to run to extract benchmark results from artifacts.
    Job can be failed at two levels: GIT_JOB and DEVICE_JOB. If any job fails, generate failure benchmark record.
    """
    artifacts = content.get("artifacts")
    git_job_name = content["git_job_name"]

    # this indicated that the git job fails, generate a failure record
    if not artifacts:
        info(f"job failed at GIT_JOB level with git job name {git_job_name}")
        return [generateGitJobLevelFailureRecord(git_job_name, app)]

    arn_to_artifacts = group_by_arn(artifacts)
    job_reports = content["job_reports"]
    arn_to_job_report = to_job_report_map(job_reports)

    all_benchmark_results = []

    # process mobile job's benchmark results. Each job represent one device+os in device pool
    for job_arn, job_artifacts in arn_to_artifacts.items():
        job_report = arn_to_job_report.get(job_arn)

        if not job_report:
            info(
                f"job arn {job_arn} is not recognized in job_reports list {json.dumps(job_reports)}, skip the process"
            )
            continue

        result = job_report.get("result", "")
        if result != "PASSED":
            arn = job_report["arn"]
            info(f"job {arn} failed at DEVICE_JOB level with result {result}")
            # device test failed, generate a failure record instead
            all_benchmark_results.append(
                generateDeviceLevelFailureRecord(git_job_name, job_report, app)
            )
        else:
            benchmark_config = get_benchmark_config(job_artifacts, benchmark_configs)
            for job_artifact in job_artifacts:
                # generate result for each schema
                results = extractBenchmarkResultFromArtifact(
                    job_artifact, benchmark_config
                )
                all_benchmark_results.extend(results)
    return all_benchmark_results


def main() -> None:
    args = parse_args()
    with open(args.artifacts) as f:
        content = json.load(f)
        all_benchmark_results = process_benchmark_results(
            content, args.app, args.benchmark_configs
        )
    # add v3 in case we have higher version of schema
    output_dir = os.path.join(args.output_dir, "v3")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.basename(args.artifacts)
    with open(f"{output_dir}/{output_file}", "w") as f:
        json.dump(all_benchmark_results, f)


if __name__ == "__main__":
    main()
