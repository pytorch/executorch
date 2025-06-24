import importlib.util
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd


class TestBenchmarkAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script_path = os.path.join(
            ".ci", "scripts", "benchmark_tooling", "get_benchmark_analysis_data.py"
        )
        spec = importlib.util.spec_from_file_location(
            "get_benchmark_analysis_data", script_path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module  # Register before execution
        spec.loader.exec_module(module)
        cls.module = module

    """Test the validate_iso8601_no_ms function."""

    def test_valid_iso8601(self):
        """Test with valid ISO8601 format."""
        valid_date = "2025-06-01T00:00:00"
        result = self.module.validate_iso8601_no_ms(valid_date)
        self.assertEqual(result, valid_date)

    def test_invalid_iso8601(self):
        """Test with invalid ISO8601 format."""
        invalid_dates = [
            "2025-06-01",  # Missing time
            "2025-06-01 00:00:00",  # Space instead of T
            "2025-06-01T00:00:00.000",  # With milliseconds
            "not-a-date",  # Not a date at all
        ]
        for invalid_date in invalid_dates:
            with self.subTest(invalid_date=invalid_date):
                with self.assertRaises(self.module.argparse.ArgumentTypeError):
                    self.module.validate_iso8601_no_ms(invalid_date)

    def test_output_type_values(self):
        """Test that OutputType has the expected values."""
        self.assertEqual(self.module.OutputType.EXCEL.value, "excel")
        self.assertEqual(self.module.OutputType.PRINT.value, "print")
        self.assertEqual(self.module.OutputType.CSV.value, "csv")
        self.assertEqual(self.module.OutputType.JSON.value, "json")
        self.assertEqual(self.module.OutputType.DF.value, "df")

    def setUp(self):
        """Set up test fixtures."""
        self.maxDiff = None

        self.fetcher = self.module.ExecutorchBenchmarkFetcher(
            env="prod", disable_logging=True
        )

        # Sample data for testing
        self.sample_data_1 = [
            {
                "groupInfo": {
                    "model": "llama3",
                    "backend": "qlora",
                    "device": "Iphone 15 pro max (private)",
                    "arch": "ios_17",
                },
                "rows": [
                    {
                        "workflow_id": 1,
                        "job_id": 1,
                        "metadata_info.timestamp": "2025-06-15T15:00:00Z",
                        "metric_1": 2.0,
                    },
                    {
                        "workflow_id": 2,
                        "job_id": 2,
                        "metadata_info.timestamp": "2025-06-15T14:00:00Z",
                        "metric_1": 3.0,
                    },
                ],
            },
            {
                "groupInfo": {
                    "model": "mv3",
                    "backend": "xnnpack_q8",
                    "device": "s22_5g",
                    "arch": "android_13",
                },
                "rows": [
                    {
                        "workflow_id": 3,
                        "job_id": 3,
                        "metadata_info.timestamp": "2025-06-15T17:00:00Z",
                        "metric_1": 2.0,
                    },
                    {
                        "workflow_id": 4,
                        "job_id": 5,
                        "metadata_info.timestamp": "2025-06-15T14:00:00Z",
                        "metric_1": 3.0,
                    },
                ],
            },
        ]

        self.sample_data_2 = [
            {
                "groupInfo": {
                    "model": "llama3",
                    "backend": "qlora",
                    "device": "Iphone 15 pro max (private)",
                    "arch": "ios_17.4.3",
                },
                "rows": [
                    {
                        "workflow_id": 1,
                        "job_id": 1,
                        "metadata_info.timestamp": "2025-06-15T15:00:00Z",
                        "metric_1": 2.0,
                    },
                    {
                        "workflow_id": 2,
                        "job_id": 2,
                        "metadata_info.timestamp": "2025-06-15T14:00:00Z",
                        "metric_1": 3.0,
                    },
                ],
            },
            {
                "groupInfo": {
                    "model": "llama3",
                    "backend": "qlora",
                    "device": "Iphone 15 pro max",
                    "arch": "ios_17.4.3",
                },
                "rows": [
                    {
                        "workflow_id": 6,
                        "job_id": 6,
                        "metadata_info.timestamp": "2025-06-15T17:00:00Z",
                        "metric_1": 1.0,
                    },
                    {
                        "workflow_id": 8,
                        "job_id": 8,
                        "metadata_info.timestamp": "2025-06-15T14:00:00Z",
                        "metric_1": 1.0,
                    },
                ],
            },
            {
                "groupInfo": {
                    "model": "mv3",
                    "backend": "xnnpack_q8",
                    "device": "s22_5g",
                    "arch": "android_13",
                },
                "rows": [
                    {
                        "workflow_id": 3,
                        "job_id": 3,
                        "metadata_info.timestamp": "2025-06-15T17:00:00Z",
                        "metric_1": 2.0,
                    },
                    {
                        "workflow_id": 4,
                        "job_id": 5,
                        "metadata_info.timestamp": "2025-06-15T14:00:00Z",
                        "metric_1": 3.0,
                    },
                ],
            },
        ]

    def test_init(self):
        """Test initialization of ExecutorchBenchmarkFetcher."""
        self.assertEqual(self.fetcher.env, "prod")
        self.assertEqual(self.fetcher.base_url, "https://hud.pytorch.org")
        self.assertEqual(
            self.fetcher.query_group_table_by_fields,
            ["model", "backend", "device", "arch"],
        )
        self.assertEqual(
            self.fetcher.query_group_row_by_fields,
            ["workflow_id", "job_id", "metadata_info.timestamp"],
        )
        self.assertTrue(self.fetcher.disable_logging)
        self.assertEqual(self.fetcher.matching_groups, {})

    def test_get_base_url(self):
        """Test _get_base_url method."""
        self.assertEqual(self.fetcher._get_base_url(), "https://hud.pytorch.org")

        # Test with local environment
        local_fetcher = self.module.ExecutorchBenchmarkFetcher(env="local")
        self.assertEqual(local_fetcher._get_base_url(), "http://localhost:3000")

    def test_normalize_string(self):
        """Test normalize_string method."""
        test_cases = [
            ("Test String", "test_string"),
            ("test_string", "test_string"),
            ("test string", "test_string"),
            ("test--string", "test_string"),
            ("test  (private)", "test"),
            ("test@#$%^&*", "test_"),
        ]

        for input_str, expected in test_cases:
            with self.subTest(input_str=input_str):
                result = self.fetcher.normalize_string(input_str)
                self.assertEqual(result, expected)

    @patch("requests.get")
    def test_fetch_execu_torch_data_success(self, mock_get):
        """Test _fetch_execu_torch_data method with successful response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.sample_data_1
        mock_get.return_value = mock_response

        result = self.fetcher._fetch_execu_torch_data(
            "2025-06-01T00:00:00", "2025-06-02T00:00:00"
        )

        self.assertEqual(result, self.sample_data_1)
        mock_get.assert_called_once()

    @patch("requests.get")
    def test_fetch_execu_torch_data_failure(self, mock_get):
        """Test _fetch_execu_torch_data method with failed response."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_get.return_value = mock_response

        result = self.fetcher._fetch_execu_torch_data(
            "2025-06-01T00:00:00", "2025-06-02T00:00:00"
        )

        self.assertIsNone(result)
        mock_get.assert_called_once()

    def test_filter_out_failure_only(self):
        """Test _filter_out_failure_only method."""
        test_data = [
            {
                "rows": [
                    {
                        "workflow_id": 1,
                        "job_id": 2,
                        "metadata_info.timestamp": 3,
                        "FAILURE_REPORT": "0",
                    },
                    {
                        "workflow_id": 4,
                        "job_id": 5,
                        "metadata_info.timestamp": 6,
                        "metric": 7.0,
                    },
                ]
            },
            {
                "rows": [
                    {
                        "workflow_id": 8,
                        "job_id": 9,
                        "metadata_info.timestamp": 10,
                        "metric": 11.0,
                    },
                ]
            },
            {
                "rows": [
                    {
                        "workflow_id": 10,
                        "job_id": 12,
                        "metadata_info.timestamp": 3,
                        "FAILURE_REPORT": "0",
                    },
                    {
                        "workflow_id": 21,
                        "job_id": 15,
                        "metadata_info.timestamp": 6,
                        "FAILURE_REPORT": "0",
                    },
                ]
            },
        ]

        expected = [
            {
                "rows": [
                    {
                        "workflow_id": 4,
                        "job_id": 5,
                        "metadata_info.timestamp": 6,
                        "metric": 7.0,
                    },
                ]
            },
            {
                "rows": [
                    {
                        "workflow_id": 8,
                        "job_id": 9,
                        "metadata_info.timestamp": 10,
                        "metric": 11.0,
                    },
                ]
            },
        ]

        result = self.fetcher._filter_out_failure_only(test_data)
        self.assertEqual(result, expected)

    def test_filter_public_result(self):
        """Test _filter_public_result method."""
        private_list = [
            {"table_name": "model1_backend1"},
            {"table_name": "model2_backend2"},
        ]

        public_list = [
            {"table_name": "model1_backend1"},
            {"table_name": "model3_backend3"},
        ]

        expected = [{"table_name": "model1_backend1"}]

        result = self.fetcher._filter_public_result(private_list, public_list)
        self.assertEqual(result, expected)

    @patch(
        "get_benchmark_analysis_data.ExecutorchBenchmarkFetcher._fetch_execu_torch_data"
    )
    def test_filter_private_results(self, mock_fetch):
        """Test filter_private_results method with various filter combinations."""
        # Create test data
        test_data = [
            {
                "groupInfo": {
                    "model": "mv3",
                    "backend": "coreml_fp16",
                    "device": "Apple iPhone 15 Pro (private)",
                    "arch": "iOS 18.0",
                    "total_rows": 10,
                    "aws_type": "private",
                },
                "rows": [{"metric_1": 1.0}],
            },
            {
                "groupInfo": {
                    "model": "mv3",
                    "backend": "test_backend",
                    "device": "Apple iPhone 15 Pro (private)",
                    "arch": "iOS 14.1.0",
                    "total_rows": 10,
                    "aws_type": "private",
                },
                "rows": [{"metric_1": 1.0}],
            },
            {
                "groupInfo": {
                    "model": "mv3",
                    "backend": "xnnpack_q8",
                    "device": "Samsung Galaxy S22 Ultra 5G (private)",
                    "arch": "Android 14",
                    "total_rows": 10,
                    "aws_type": "private",
                },
                "rows": [{"metric_1": 2.0}],
            },
            {
                "groupInfo": {
                    "model": "mv3",
                    "backend": "xnnpack_q8",
                    "device": "Samsung Galaxy S22 Ultra 5G (private)",
                    "arch": "Android 13",
                    "total_rows": 10,
                    "aws_type": "private",
                },
                "rows": [{"metric_1": 2.0}],
            },
            {
                "groupInfo": {
                    "model": "meta-llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8",
                    "backend": "llama3_spinquant",
                    "device": "Apple iPhone 15",
                    "arch": "iOS 18.0",
                    "total_rows": 19,
                    "aws_type": "public",
                },
                "rows": [{"metric_1": 2.0}],
            },
            {
                "groupInfo": {
                    "model": "mv3",
                    "backend": "coreml_fp16",
                    "device": "Apple iPhone 15 Pro Max",
                    "arch": "iOS 17.0",
                    "total_rows": 10,
                    "aws_type": "public",
                },
                "rows": [{"metric_1": 2.0}],
            },
            {
                "groupInfo": {
                    "model": "meta-llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8",
                    "backend": "test",
                    "device": "Samsung Galaxy S22 Ultra 5G",
                    "arch": "Android 14",
                    "total_rows": 10,
                    "aws_type": "public",
                },
                "rows": [{"metric_1": 2.0}],
            },
        ]

        mock_fetch.return_value = test_data
        self.fetcher.run("2025-06-01T00:00:00", "2025-06-02T00:00:00")

        # Test with no filters
        empty_filters = self.module.BenchmarkFilters(
            models=None, backends=None, devicePoolNames=None
        )

        result = self.fetcher.filter_private_results(test_data, empty_filters)
        self.assertEqual(result, test_data)

        # Test with model filter
        model_filters = self.module.BenchmarkFilters(
            models=["meta-llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8"],
            backends=None,
            devicePoolNames=None,
        )
        result = self.fetcher.filter_private_results(test_data, model_filters)
        self.assertEqual(len(result), 2)
        self.assertTrue(
            all(
                item["groupInfo"]["model"]
                == "meta-llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8"
                for item in result
            )
        )

        # Test with backend filter
        backend_filters = self.module.BenchmarkFilters(
            models=None, backends=["coreml_fp16", "test"], devicePoolNames=None
        )
        result = self.fetcher.filter_private_results(test_data, backend_filters)
        self.assertEqual(len(result), 3)
        self.assertTrue(
            all(
                item["groupInfo"]["backend"] in ["coreml_fp16", "test"]
                for item in result
            )
        )

        # Test with device filter
        device_filters = self.module.BenchmarkFilters(
            models=None, backends=None, devicePoolNames=["samsung_s22_private"]
        )
        result = self.fetcher.filter_private_results(test_data, device_filters)
        self.assertEqual(len(result), 2)
        self.assertTrue(
            all(
                "Samsung Galaxy S22 Ultra 5G (private)" in item["groupInfo"]["device"]
                for item in result
            )
        )

        # Test with combined filters (And logic fails)
        combined_filters = self.module.BenchmarkFilters(
            models=["meta-llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8"],
            backends=["xnnpack_q8"],
            devicePoolNames=None,
        )
        result = self.fetcher.filter_private_results(test_data, combined_filters)
        self.assertEqual(len(result), 0)

        # Test with combined filters (And logic success)
        combined_filters = self.module.BenchmarkFilters(
            models=["mv3"],
            backends=None,
            devicePoolNames=["apple_iphone_15_private"],
        )
        result = self.fetcher.filter_private_results(test_data, combined_filters)
        self.assertEqual(len(result), 2)

    @patch(
        "get_benchmark_analysis_data.ExecutorchBenchmarkFetcher._fetch_execu_torch_data"
    )
    def test_run_without_public_match(self, mock_fetch):
        """Test run method."""
        # Setup mocks
        mock_fetch.return_value = self.sample_data_1
        # Run the method
        self.fetcher.run("2025-06-01T00:00:00", "2025-06-02T00:00:00")
        result = self.fetcher.get_result()

        # Verify results
        self.assertEqual(result, {"private": [self.sample_data_1[0]], "public": []})
        self.assertEqual(len(self.fetcher.matching_groups), 2)
        self.assertIn("private", self.fetcher.matching_groups)
        self.assertIn("public", self.fetcher.matching_groups)

        # Verify mocks were called
        mock_fetch.assert_called_once_with("2025-06-01T00:00:00", "2025-06-02T00:00:00")

    @patch(
        "get_benchmark_analysis_data.ExecutorchBenchmarkFetcher._fetch_execu_torch_data"
    )
    def test_run_with_public_match(self, mock_fetch):
        """Test run method."""
        # Setup mocks
        mock_fetch.return_value = self.sample_data_2

        # Run the method
        self.fetcher.run("2025-06-01T00:00:00", "2025-06-02T00:00:00")
        result = self.fetcher.get_result()

        # Verify results
        self.assertEqual(
            result,
            {"private": [self.sample_data_2[0]], "public": [self.sample_data_2[1]]},
        )
        self.assertEqual(len(self.fetcher.matching_groups), 2)
        self.assertIn("private", self.fetcher.matching_groups)
        self.assertIn("public", self.fetcher.matching_groups)
        # Verify mocks were called
        mock_fetch.assert_called_once_with("2025-06-01T00:00:00", "2025-06-02T00:00:00")

    @patch(
        "get_benchmark_analysis_data.ExecutorchBenchmarkFetcher._fetch_execu_torch_data"
    )
    def test_run_with_failure_report(self, mock_fetch):
        """Test run method."""
        # Setup mocks
        mock_data = [
            {
                "groupInfo": {
                    "model": "llama3",
                    "backend": "qlora",
                    "device": "Iphone 15 pro max (private)",
                    "arch": "ios_17.4.3",
                },
                "rows": [
                    {
                        "workflow_id": 1,
                        "job_id": 2,
                        "metadata_info.timestamp": 3,
                        "FAILURE_REPORT": "0",
                    },
                    {
                        "workflow_id": 4,
                        "job_id": 5,
                        "metadata_info.timestamp": 6,
                        "metric": 7.0,
                    },
                ],
            },
            {
                "groupInfo": {
                    "model": "llama3",
                    "backend": "qlora",
                    "device": "Iphone 15 pro max",
                    "arch": "ios_17.4.3",
                },
                "rows": [
                    {
                        "workflow_id": 1,
                        "job_id": 2,
                        "metadata_info.timestamp": 3,
                        "FAILURE_REPORT": "0",
                    },
                    {
                        "workflow_id": 1,
                        "job_id": 2,
                        "metadata_info.timestamp": 3,
                        "FAILURE_REPORT": "0",
                    },
                ],
            },
        ]

        expected_private = {
            "groupInfo": {
                "model": "llama3",
                "backend": "qlora",
                "device": "Iphone 15 pro max (private)",
                "arch": "ios_17.4.3",
                "aws_type": "private",
            },
            "rows": [
                {
                    "workflow_id": 4,
                    "job_id": 5,
                    "metadata_info.timestamp": 6,
                    "metric": 7.0,
                },
            ],
            "table_name": "llama3-qlora-iphone_15_pro_max-ios_17.4.3",
        }
        mock_fetch.return_value = mock_data
        # Run the method
        self.fetcher.run("2025-06-01T00:00:00", "2025-06-02T00:00:00")
        result = self.fetcher.get_result()
        # Verify results
        self.assertEqual(result.get("private", []), [expected_private])
        self.assertEqual(len(self.fetcher.matching_groups), 2)
        self.assertIn("private", self.fetcher.matching_groups)
        self.assertIn("public", self.fetcher.matching_groups)
        # Verify mocks were called
        mock_fetch.assert_called_once_with("2025-06-01T00:00:00", "2025-06-02T00:00:00")

    @patch(
        "get_benchmark_analysis_data.ExecutorchBenchmarkFetcher._fetch_execu_torch_data"
    )
    def test_run_no_data(self, mock_fetch):
        """Test run method when no data is fetched."""
        mock_fetch.return_value = None

        result = self.fetcher.run("2025-06-01T00:00:00", "2025-06-02T00:00:00")

        self.assertIsNone(result)
        self.assertEqual(self.fetcher.matching_groups, {})
        mock_fetch.assert_called_once_with("2025-06-01T00:00:00", "2025-06-02T00:00:00")

    @patch(
        "get_benchmark_analysis_data.ExecutorchBenchmarkFetcher._fetch_execu_torch_data"
    )
    def test_run_with_filters(self, mock_fetch):
        """Test run method with filters."""
        # Setup mock data
        mock_data = [
            {
                "groupInfo": {
                    "model": "llama3",
                    "backend": "qlora",
                    "device": "Iphone 15 pro max (private)",
                    "arch": "ios_17",
                },
                "rows": [{"metric_1": 1.0}],
            },
            {
                "groupInfo": {
                    "model": "mv3",
                    "backend": "xnnpack_q8",
                    "device": "s22_5g (private)",
                    "arch": "android_13",
                },
                "rows": [{"metric_1": 2.0}],
            },
            {
                "groupInfo": {
                    "model": "mv3",
                    "backend": "xnnpack_q8",
                    "device": "s22_5g",
                    "arch": "android_13",
                },
                "rows": [{"metric_1": 3.0}],
            },
        ]
        mock_fetch.return_value = mock_data

        # Create filters for llama3 model only
        filters = self.module.BenchmarkFilters(
            models=["llama3"], backends=None, devicePoolNames=None
        )
        # Run the method with filters
        self.fetcher.run("2025-06-01T00:00:00", "2025-06-02T00:00:00", filters)
        result = self.fetcher.get_result()
        print("result1", result)

        # Verify results - should only have llama3 in private results
        self.assertEqual(len(result["private"]), 1)
        self.assertEqual(result["private"][0]["groupInfo"]["model"], "llama3")

        # Public results should be empty since there's no matching table_name
        self.assertEqual(result["public"], [])

        # Test with backend filter
        filters = self.module.BenchmarkFilters(
            models=None, backends=["xnnpack_q8"], devicePoolNames=None
        )
        self.fetcher.run("2025-06-01T00:00:00", "2025-06-02T00:00:00", filters)
        result = self.fetcher.get_result()

        print("result", result)

        # Verify results - should only have xnnpack_q8 in private results
        self.assertEqual(len(result["private"]), 1)
        self.assertEqual(result["private"][0]["groupInfo"]["backend"], "xnnpack_q8")

        # Public results should have the matching xnnpack_q8 entry
        self.assertEqual(len(result["public"]), 1)
        self.assertEqual(result["public"][0]["groupInfo"]["backend"], "xnnpack_q8")

    def test_to_dict(self):
        """Test to_dict method."""
        # Setup test data
        self.fetcher.matching_groups = {
            "private": self.module.MatchingGroupResult(
                category="private", data=[{"key": "private_value"}]
            ),
            "public": self.module.MatchingGroupResult(
                category="public", data=[{"key": "public_value"}]
            ),
        }

        expected = {
            "private": [{"key": "private_value"}],
            "public": [{"key": "public_value"}],
        }

        result = self.fetcher.to_dict()
        self.assertEqual(result, expected)

    def test_to_df(self):
        """Test to_df method."""
        # Setup test data
        self.fetcher.matching_groups = {
            "private": self.module.MatchingGroupResult(
                category="private",
                data=[{"groupInfo": {"model": "llama3"}, "rows": [{"metric1": 1.0}]}],
            ),
        }

        result = self.fetcher.to_df()

        self.assertIn("private", result)
        self.assertEqual(len(result["private"]), 1)
        self.assertIn("groupInfo", result["private"][0])
        self.assertIn("df", result["private"][0])
        self.assertIsInstance(result["private"][0]["df"], pd.DataFrame)
        self.assertEqual(result["private"][0]["groupInfo"], {"model": "llama3"})

    @patch("os.makedirs")
    @patch("json.dump")
    @patch("builtins.open", new_callable=mock_open)
    def test_to_json(self, mock_file, mock_json_dump, mock_makedirs):
        """Test to_json method."""
        # Setup test data
        self.fetcher.matching_groups = {
            "private": self.module.MatchingGroupResult(
                category="private", data=[{"key": "value"}]
            ),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.fetcher.to_json(temp_dir)

            # Check that the file path is returned
            self.assertEqual(result, os.path.join(temp_dir, "benchmark_results.json"))

            # Check that the file was opened for writing
            mock_file.assert_called_once_with(
                os.path.join(temp_dir, "benchmark_results.json"), "w"
            )

            # Check that json.dump was called with the expected data
            mock_json_dump.assert_called_once()
            args, _ = mock_json_dump.call_args
            self.assertEqual(args[0], {"private": [{"key": "value"}]})

    @patch("pandas.DataFrame.to_excel")
    @patch("pandas.ExcelWriter")
    @patch("os.makedirs")
    def test_to_excel(self, mock_makedirs, mock_excel_writer, mock_to_excel):
        """Test to_excel method."""
        # Setup test data
        self.fetcher.matching_groups = {
            "private": self.module.MatchingGroupResult(
                category="private",
                data=[
                    {
                        "groupInfo": {"model": "llama3"},
                        "rows": [{"metric1": 1.0}],
                        "table_name": "llama3_table",
                    }
                ],
            ),
        }

        # Mock the context manager for ExcelWriter
        mock_writer = MagicMock()
        mock_excel_writer.return_value.__enter__.return_value = mock_writer
        mock_writer.book = MagicMock()
        mock_writer.book.add_worksheet.return_value = MagicMock()
        mock_writer.sheets = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            self.fetcher.to_excel(temp_dir)

            # Check that ExcelWriter was called with the expected path
            mock_excel_writer.assert_called_once_with(
                os.path.join(temp_dir, "private.xlsx"), engine="xlsxwriter"
            )

            # Check that to_excel was called
            mock_to_excel.assert_called_once()

    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pandas.DataFrame.to_csv")
    def test_to_csv(self, mock_to_csv, mock_file, mock_makedirs):
        """Test to_csv method."""
        # Setup test data
        self.fetcher.matching_groups = {
            "private": self.module.MatchingGroupResult(
                category="private",
                data=[{"groupInfo": {"model": "llama3"}, "rows": [{"metric1": 1.0}]}],
            ),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            self.fetcher.to_csv(temp_dir)

            # Check that the directory was created
            mock_makedirs.assert_called()

            # Check that the file was opened for writing
            mock_file.assert_called_once()

            # Check that to_csv was called
            mock_to_csv.assert_called_once()

    def test_to_output_type(self):
        """Test _to_output_type method."""
        # Test with string values
        self.assertEqual(
            self.fetcher._to_output_type("excel"), self.module.OutputType.EXCEL
        )
        self.assertEqual(
            self.fetcher._to_output_type("print"), self.module.OutputType.PRINT
        )
        self.assertEqual(
            self.fetcher._to_output_type("csv"), self.module.OutputType.CSV
        )
        self.assertEqual(
            self.fetcher._to_output_type("json"), self.module.OutputType.JSON
        )
        self.assertEqual(self.fetcher._to_output_type("df"), self.module.OutputType.DF)

        # Test with enum values
        self.assertEqual(
            self.fetcher._to_output_type(self.module.OutputType.EXCEL),
            self.module.OutputType.EXCEL,
        )

        # Test with invalid values
        self.assertEqual(
            self.fetcher._to_output_type("invalid"), self.module.OutputType.JSON
        )
        self.assertEqual(self.fetcher._to_output_type(123), self.module.OutputType.JSON)

    @patch("get_benchmark_analysis_data.ExecutorchBenchmarkFetcher.to_json")
    @patch("get_benchmark_analysis_data.ExecutorchBenchmarkFetcher.to_df")
    @patch("get_benchmark_analysis_data.ExecutorchBenchmarkFetcher.to_excel")
    @patch("get_benchmark_analysis_data.ExecutorchBenchmarkFetcher.to_csv")
    def test_output_data(self, mock_to_csv, mock_to_excel, mock_to_df, mock_to_json):
        """Test output_data method."""
        # Setup test data
        self.fetcher.matching_groups = {
            "private": self.module.MatchingGroupResult(
                category="private", data=[{"key": "value"}]
            ),
        }

        # Test PRINT output
        result = self.fetcher.output_data(self.module.OutputType.PRINT)
        self.assertEqual(result, {"private": [{"key": "value"}]})

        # Test JSON output
        mock_to_json.return_value = "/path/to/file.json"
        result = self.fetcher.output_data(self.module.OutputType.JSON)
        self.assertEqual(result, {"private": [{"key": "value"}]})
        mock_to_json.assert_called_once_with(".")

        # Test DF output
        mock_to_df.return_value = {"private": [{"df": "value"}]}
        result = self.fetcher.output_data(self.module.OutputType.DF)
        self.assertEqual(result, {"private": [{"df": "value"}]})
        mock_to_df.assert_called_once()

        # Test EXCEL output
        result = self.fetcher.output_data(self.module.OutputType.EXCEL)
        self.assertEqual(result, {"private": [{"key": "value"}]})
        mock_to_excel.assert_called_once_with(".")

        # Test CSV output
        result = self.fetcher.output_data(self.module.OutputType.CSV)
        self.assertEqual(result, {"private": [{"key": "value"}]})
        mock_to_csv.assert_called_once_with(".")


if __name__ == "__main__":
    unittest.main()
