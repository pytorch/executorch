#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import unittest
from re import M
from typing import Any, Dict
from unittest import mock
from unittest.mock import MagicMock

from extract_benchmark_results import (
    extract_android_benchmark_results,
    extract_ios_benchmark_results,
    process_benchmark_results,
)


def get_mock_happy_flow_content(app_type: str = "IOS_APP"):
    return {
        "git_job_name": "benchmark-on-device (ic4, mps, apple_iphone_15, arn:aws:devicefarm:us-west-2:308535385114:devicep... / mobile-job (ios)",
        "artifacts": [
            {
                "arn": "1",
                "name": "Syslog",
                "type": "DEVICE_LOG",
                "extension": "syslog",
                "url": "https://job_arn_1_device_log",  # @lint-ignore
                "s3_url": "https://job_arn_1/test-workflow1/1/syslog.syslog",  # @lint-ignore
                "app_type": app_type,
                "job_name": "job_arn_1_name",
                "os": "14",
                "job_arn": "job_arn_1",
                "job_conclusion": "PASSED",
            },
            {
                "arn": "2",
                "name": "Test spec output",
                "type": "TESTSPEC_OUTPUT",
                "extension": "txt",
                "url": "job_arn_1_test_spec_output",
                "s3_url": "job_arn_1_test_spec_output",
                "app_type": app_type,
                "job_name": "job_arn_1_device_name",
                "os": "14",
                "job_arn": "job_arn_1",
                "job_conclusion": "PASSED",
            },
            {
                "arn": "3",
                "name": "Customer Artifacts",
                "type": "CUSTOMER_ARTIFACT",
                "extension": "zip",
                "url": "https://job_arn_1_customer_artifact",  # @lint-ignore
                "s3_url": "https://job_arn_1_customer_artifact1",  # @lint-ignore
                "app_type": app_type,
                "job_name": "job_arn_1_device_name",
                "os": "14",
                "job_arn": "job_arn_1",
                "job_conclusion": "PASSED",
            },
            {
                "arn": "5",
                "name": "Syslog",
                "type": "DEVICE_LOG",
                "extension": "syslog",
                "url": "https://job_arn_1_device_log",  # @lint-ignore
                "s3_url": "https://job_arn_1/test-workflow1/1/syslog.syslog",  # @lint-ignore
                "app_type": app_type,
                "job_name": "job_arn_2_name",
                "os": "14",
                "job_arn": "job_arn_2",
                "job_conclusion": "PASSED",
            },
            {
                "arn": "6",
                "name": "Test spec output",
                "type": "TESTSPEC_OUTPUT",
                "extension": "txt",
                "url": "job_arn_2_test_spec_output",
                "s3_url": "job_arn_2_test_spec_output",
                "app_type": app_type,
                "job_name": "job_arn_2_name",
                "os": "14",
                "job_arn": "job_arn_2",
                "job_conclusion": "PASSED",
            },
            {
                "arn": "7",
                "name": "Customer Artifacts",
                "type": "CUSTOMER_ARTIFACT",
                "extension": "zip",
                "url": "https://job_arn_1_customer_artifact",  # @lint-ignore
                "s3_url": "https://job_arn_1_customer_artifact1",  # @lint-ignore
                "app_type": app_type,
                "job_name": "job_arn_2_name",
                "os": "14",
                "job_arn": "job_arn_2",
                "job_conclusion": "PASSED",
            },
        ],
        "run_report": {
            "name": "mobile-job-ios-1",
            "arn": "run_arn_1",
            "report_type": "run",
            "status": "COMPLETED",
            "result": "PASSED",
            "app_type": app_type,
            "infos": {},
            "parent_arn": "",
        },
        "job_reports": [
            {
                "name": "job_arn_1_report_device_name",
                "arn": "job_arn_1",
                "report_type": "job",
                "status": "COMPLETED",
                "result": "PASSED",
                "app_type": app_type,
                "infos": {},
                "parent_arn": "run_arn_1",
                "os": "14",
            },
            {
                "name": "job_arn_2_name_report",
                "arn": "job_arn_2",
                "report_type": "job",
                "status": "COMPLETED",
                "result": "PASSED",
                "app_type": app_type,
                "infos": {},
                "parent_arn": "run_arn_1",
                "os": "14",
            },
        ],
    }


def mockExtractBenchmarkResults(artifact_type, artifact_s3_url):
    if artifact_type != "TESTSPEC_OUTPUT":
        return []
    if artifact_s3_url == "job_arn_1_test_spec_output":
        return [get_mock_extract_result()[0]]
    return [get_mock_extract_result()[1]]


class Test(unittest.TestCase):
    @mock.patch("extract_benchmark_results.extract_ios_benchmark_results")
    @mock.patch("extract_benchmark_results.read_benchmark_config")
    def test_process_benchmark_results_when_ios_succuess_then_returnBenchmarkResults(
        self, read_benchmark_config_mock, extract_ios_mock
    ):
        # setup mocks
        content = get_mock_happy_flow_content()
        extract_ios_mock.side_effect = (
            lambda artifact_type, artifact_s3_url: mockExtractBenchmarkResults(
                artifact_type, artifact_s3_url
            )
        )
        read_benchmark_config_mock.return_value = {}

        # execute
        result = process_benchmark_results(content, "ios", "benchmark_configs")

        # assert
        self.assertGreaterEqual(len(result), 2)
        self.assertNotEqual(result[0]["metric"]["name"], "FAILURE_REPORT")
        self.assertNotEqual(result[1]["metric"]["name"], "FAILURE_REPORT")

    @mock.patch("extract_benchmark_results.extract_android_benchmark_results")
    @mock.patch("extract_benchmark_results.read_benchmark_config")
    def test_process_benchmark_results_when_android_succuess_then_returnBenchmarkResults(
        self, read_benchmark_config_mock, extract_android_mock
    ):
        # setup mocks
        content = get_mock_happy_flow_content("ANDROID_APP")
        extract_android_mock.side_effect = (
            lambda artifact_type, artifact_s3_url: mockExtractBenchmarkResults(
                artifact_type, artifact_s3_url
            )
        )
        read_benchmark_config_mock.return_value = {}

        # execute
        result = process_benchmark_results(content, "android", "benchmark_configs")
        self.assertGreaterEqual(len(result), 2)

    def test_process_benchmark_results_when_ANDROID_git_job_fails_then_returnBenchmarkRecordWithFailure(
        self,
    ):
        # setup mocks
        # mimic artifact when job is failed.
        content = {
            "git_job_name": "benchmark-on-device (ic4, qnn_q8, samsung_galaxy_s22, arn:aws:devicefarm:us-west-2:308535385114:d... / mobile-job (android)"
        }

        # execute
        result = process_benchmark_results(content, "android", "benchmark_configs")

        # assert
        self.assertGreaterEqual(len(result), 1)

        self.assertEqual(
            result[0]["model"],
            {
                "name": "ic4",
                "type": "OSS model",
                "backend": "qnn_q8",
            },
        )
        self.assertEqual(
            result[0]["benchmark"],
            {
                "name": "ExecuTorch",
                "mode": "inference",
                "extra_info": {
                    "app_type": "ANDROID_APP",
                    "job_conclusion": "FAILURE",
                    "failure_type": "GIT_JOB",
                    "job_report": "{}",
                },
            },
        )

        self.assertEqual(result[0]["runners"][0]["name"], "samsung_galaxy_s22")
        self.assertEqual(result[0]["runners"][0]["type"], "Android")
        self.assertEqual(result[0]["metric"]["name"], "FAILURE_REPORT")

    def test_process_benchmark_results_when_IOS_git_job_fails_then_returnBenchmarkRecordWithFailure(
        self,
    ):
        # setup mocks
        # mimic artifact when job is failed.
        content = {
            "git_job_name": "benchmark-on-device (ic4, mps, apple_iphone_15, arn:aws:devicefarm:us-west-2:308535385114:devicep... / mobile-job (ios)"
        }

        # execute
        result = process_benchmark_results(content, "ios", "benchmark_configs")

        # assert
        self.assertGreaterEqual(len(result), 1)

        self.assertEqual(
            result[0]["model"],
            {
                "name": "ic4",
                "type": "OSS model",
                "backend": "mps",
            },
        )
        self.assertEqual(
            result[0]["benchmark"],
            {
                "name": "ExecuTorch",
                "mode": "inference",
                "extra_info": {
                    "app_type": "IOS_APP",
                    "job_conclusion": "FAILURE",
                    "failure_type": "GIT_JOB",
                    "job_report": "{}",
                },
            },
        )
        self.assertEqual(result[0]["runners"][0]["name"], "apple_iphone_15")
        self.assertEqual(result[0]["runners"][0]["type"], "iOS")
        self.assertEqual(result[0]["metric"]["name"], "FAILURE_REPORT")

    @mock.patch("extract_benchmark_results.extract_ios_benchmark_results")
    @mock.patch("extract_benchmark_results.read_benchmark_config")
    def test_process_benchmark_results_when_one_IOS_mobile_job_fails_then_returnBenchmarkRecordWithFailure(
        self, read_benchmark_config_mock, extract_ios_mock
    ):
        # setup mocks
        content = get_mock_happy_flow_content()
        content["job_reports"][0]["result"] = "FAILED"

        extract_ios_mock.side_effect = (
            lambda artifact_type, artifact_s3_url: mockExtractBenchmarkResults(
                artifact_type, artifact_s3_url
            )
        )
        read_benchmark_config_mock.return_value = {}

        # execute
        result = process_benchmark_results(content, "ios", "benchmark_configs")

        # assert
        self.assertGreaterEqual(len(result), 2)
        self.assertEqual(
            result[0]["model"],
            {
                "name": "ic4",
                "type": "OSS model",
                "backend": "mps",
            },
        )
        self.assertEqual(result[0]["metric"]["name"], "FAILURE_REPORT")

        self.assertNotEqual(result[1]["metric"]["name"], "FAILURE_REPORT")

    @mock.patch("extract_benchmark_results.extract_ios_benchmark_results")
    @mock.patch("extract_benchmark_results.read_benchmark_config")
    def test_process_benchmark_results_when_one_mobile_job_fails_with_invalid_app_type_then_throw_errors(
        self, read_benchmark_config_mock, extract_ios_mock
    ):
        # setup mocks
        content = get_mock_happy_flow_content()
        content["job_reports"][0]["result"] = "FAILED"

        extract_ios_mock.side_effect = (
            lambda artifact_type, artifact_s3_url: mockExtractBenchmarkResults(
                artifact_type, artifact_s3_url
            )
        )
        read_benchmark_config_mock.return_value = {}

        # execute
        with self.assertRaises(ValueError) as context:
            _ = process_benchmark_results(content, "random", "benchmark_configs")

        # assert
        self.assertTrue(
            "unknown device type detected: random" in str(context.exception)
        )
        read_benchmark_config_mock.assert_not_called()
        extract_ios_mock.assert_not_called()

    def test_process_benchmark_results_when_git_job_fails_with_invalid_git_job_name_then_throw_errors(
        self,
    ):
        # setup mocks
        # mimic artifact when job is failed.
        content = {
            "git_job_name": "benchmark-on (ic4, qnn_q8, samsung_galaxy_s22, arn:aws:devicefarm:us-west-2:308535385114:d... / mobile-job (android)"
        }

        # execute
        with self.assertRaises(ValueError) as context:
            _ = process_benchmark_results(content, "ios", "benchmark_configs")

        # assert
        print("exception yang:", str(context.exception))
        self.assertTrue(
            "regex pattern not found from git_job_name" in str(context.exception)
        )


def get_mock_extract_result():
    return [
        {
            "benchmarkModel": {
                "backend": "q1",
                "quantization": 0,
                "name": "ic4",
            },
            "deviceInfo": {
                "arch": "extract arch",
                "device": "extract device",
                "os": "extract os",
                "availMem": 0,
                "totalMem": 0,
            },
            "method": "",
            "metric": "metric1",
            "actualValue": 100,
            "targetValue": 100,
        },
        {
            "benchmarkModel": {
                "backend": "q2",
                "quantization": 0,
                "name": "ic4",
            },
            "deviceInfo": {
                "arch": "extract arch",
                "device": "extract device",
                "os": "extract os",
                "availMem": 0,
                "totalMem": 0,
            },
            "method": "",
            "metric": "metric2",
            "actualValue": 200,
            "targetValue": 200,
        },
    ]


if __name__ == "__main__":
    unittest.main()
