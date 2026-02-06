"""
ExecutorchBenchmark Analysis Data Retrieval

This module provides tools for fetching, processing, and analyzing benchmark data
from the HUD Open API for ExecutorchBenchmark. It supports filtering data by (private) device pool names,
backends, and models, exporting results in various formats (JSON, DataFrame, Excel, CSV),
and customizing data retrieval parameters.
"""

import argparse
import json
import logging
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import requests
from yaspin import yaspin

logging.basicConfig(level=logging.INFO)

# add here just for the records
VALID_PRIVATE_DEVICE_POOLS_MAPPINGS = {
    "apple_iphone_15_private": [
        ("Apple iPhone 15 Pro (private)", "iOS 18.4.1"),
        ("Apple iPhone 15 (private)", "iOS 18.0"),
        ("Apple iPhone 15 Plus (private)", "iOS 17.4.1"),
    ],
    "samsung_s22_private": [
        ("Samsung Galaxy S22 Ultra 5G (private)", "Android 14"),
        ("Samsung Galaxy S22 5G (private)", "Android 13"),
    ],
}

VALID_PRIVATE_DEVICE_POOLS_NAMES = list(VALID_PRIVATE_DEVICE_POOLS_MAPPINGS.keys())


class OutputType(Enum):
    """
    Enumeration of supported output formats for benchmark data.

    Values:
        EXCEL: Export data to Excel spreadsheets
        PRINT: Print data to console (default)
        CSV: Export data to CSV files
        JSON: Export data to JSON files
        DF: Return data as pandas DataFrames
    """

    EXCEL = "excel"
    PRINT = "print"
    CSV = "csv"
    JSON = "json"
    DF = "df"


@dataclass
class BenchmarkQueryGroupDataParams:
    """
    Parameters for querying benchmark data from HUD API.

    Attributes:
        repo: Repository name (e.g., "pytorch/executorch")
        benchmark_name: Name of the benchmark (e.g., "ExecuTorch")
        start_time: ISO8601 formatted start time
        end_time: ISO8601 formatted end time
        group_table_by_fields: Fields to group tables by
        group_row_by_fields: Fields to group rows by
    """

    repo: str
    benchmark_name: str
    start_time: str
    end_time: str
    group_table_by_fields: list
    group_row_by_fields: list


@dataclass
class MatchingGroupResult:
    """
    Container for benchmark results grouped by category.

    Attributes:
        category: Category name (e.g., 'private', 'public')
        data: List of benchmark data for this category
    """

    category: str
    data: list


@dataclass
class BenchmarkFilters:
    models: list
    backends: list
    devicePoolNames: list


BASE_URLS = {
    "local": "http://localhost:3000",
    "prod": "https://hud.pytorch.org",  # @lint-ignore
}


def validate_iso8601_no_ms(value: str):
    """
    Validate that a string is in ISO8601 format without milliseconds.
    Args:
        value: String to validate (format: YYYY-MM-DDTHH:MM:SS)
    """
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S").strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid datetime format for '{value}'. Expected: YYYY-MM-DDTHH:MM:SS"
        )


class ExecutorchBenchmarkFetcher:
    """
    Fetch and process benchmark data from HUD API for ExecutorchBenchmark.

    This class provides methods to:
    1. Fetch all benchmark data for a specified time range
    2. Get all private device info within the time range
    3. Filter the private device data if filter is provided
    4. Then use the filtered private device data to find matched the public device data using [model, backend, device, arch]
    3. Export results in various formats (JSON, DataFrame, Excel, CSV)

    Usage:
        fetcher = ExecutorchBenchmarkFetcher()
        fetcher.run(start_time, end_time)
        fetcher.output_data(OutputType.EXCEL, output_dir="./results")
    """

    def __init__(
        self,
        env: str = "prod",
        disable_logging: bool = False,
        group_table_fields=None,
        group_row_fields=None,
    ):
        """
        Initialize the ExecutorchBenchmarkFetcher.

        Args:
            env: Environment to use ('local' or 'prod')
            disable_logging: Whether to suppress log output
            group_table_fields: Custom fields to group tables by (defaults to device, backend, arch, model)
            group_row_fields: Custom fields to group rows by (defaults to workflow_id, job_id, granularity_bucket)
        """
        self.env = env
        self.base_url = self._get_base_url()
        self.query_group_table_by_fields = (
            group_table_fields
            if group_table_fields
            else ["model", "backend", "device", "arch"]
        )
        self.query_group_row_by_fields = (
            group_row_fields
            if group_row_fields
            else ["workflow_id", "job_id", "metadata_info.timestamp"]
        )
        self.data = None
        self.disable_logging = disable_logging
        self.matching_groups: Dict[str, MatchingGroupResult] = {}

    def run(
        self,
        start_time: str,
        end_time: str,
        filters: Optional[BenchmarkFilters] = None,
    ) -> None:
        # reset group & raw data for new run
        self.matching_groups = {}
        self.data = None

        data = self._fetch_execu_torch_data(start_time, end_time)
        if data is None:
            logging.warning("no data fetched from the HUD API")
            return None
        self._proces_raw_data(data)
        self._process_private_public_data(filters)

    def _filter_out_failure_only(
        self, data_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Clean data by removing rows that only contain FAILURE_REPORT metrics.

        Args:
            data_list: List of benchmark data dictionaries

        Returns:
            Filtered list with rows containing only FAILURE_REPORT removed
        """
        ONLY = {"workflow_id", "metadata_info.timestamp", "job_id", "FAILURE_REPORT"}
        for item in data_list:
            filtered_rows = [
                row
                for row in item.get("rows", [])
                # Keep row only if it has additional fields beyond ONLY
                if not set(row.keys()).issubset(ONLY)
            ]
            item["rows"] = filtered_rows
        return [item for item in data_list if item.get("rows")]

    def _filter_public_result(self, private_list, all_public):
        # find intersection betwen private and public tables.
        common = list(
            set([item["table_name"] for item in private_list])
            & set([item["table_name"] for item in all_public])
        )

        if not self.disable_logging:
            logging.info(
                f"Found {len(common)} table names existed in both private and public, use it to filter public tables:"
            )
            logging.info(json.dumps(common, indent=1))
        filtered_public = [item for item in all_public if item["table_name"] in common]
        return filtered_public

    def get_result(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get a deep copy of the benchmark results.

        Returns:
            Dictionary containing benchmark results grouped by category
        """
        return deepcopy(self.to_dict())

    def to_excel(self, output_dir: str = ".") -> None:
        """
        Export benchmark results to Excel files.
        Creates two Excel files:
        - res_private.xlsx: Results for private devices
        - res_public.xlsx: Results for public devices
        Each file contains multiple sheets, one per benchmark configuration for private and public.
        Args:
            output_dir: Directory to save Excel files
        """
        for item in self.matching_groups.values():
            self._write_multi_sheet_excel(item.data, output_dir, item.category)

    def _write_multi_sheet_excel(self, data_list, output_dir, file_name):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
        else:
            logging.info(f"Using existing output directory: {output_dir}")
        file = os.path.join(output_dir, f"{file_name}.xlsx")
        with pd.ExcelWriter(file, engine="xlsxwriter") as writer:
            workbook = writer.book
            for idx, entry in enumerate(data_list):
                sheet_name = f"table{idx+1}"
                df = pd.DataFrame(entry.get("rows", []))

                # Encode metadata as compact JSON string
                meta = entry.get("groupInfo", {})
                json_str = json.dumps(meta, separators=(",", ":"))

                worksheet = workbook.add_worksheet(sheet_name)
                writer.sheets[sheet_name] = worksheet

                # Write JSON into A1
                worksheet.write_string(0, 0, json_str)

                logging.info(
                    f"Wrting excel sheet to file {file} with sheet name {sheet_name} for {entry['table_name']}"
                )
                # Write DataFrame starting at row 2 (index 1)
                df.to_excel(writer, sheet_name=sheet_name, startrow=1, index=False)

    def output_data(
        self, output_type: OutputType = OutputType.PRINT, output_dir: str = "."
    ) -> Any:
        """
        Generate output in the specified format.

        Supports multiple output formats:
        - PRINT: Print results to console
        - JSON: Export to JSON files
        - DF: Return as pandas DataFrames
        - EXCEL: Export to Excel files
        - CSV: Export to CSV files

        Args:
            output_type: Format to output the data in
            output_dir: Directory to save output files (for file-based formats)

        Returns:
            Benchmark results in the specified format
        """
        logging.info(
            f"Generating output with type {output_type}: {[self.matching_groups.keys()]}"
        )

        o_type = self._to_output_type(output_type)
        if o_type == OutputType.PRINT:
            logging.info("\n ========= Generate print output ========= \n")
            logging.info(json.dumps(self.get_result(), indent=2))
        elif o_type == OutputType.JSON:
            logging.info("\n ========= Generate json output ========= \n")
            file_path = self.to_json(output_dir)
            logging.info(f"success, please check {file_path}")
        elif o_type == OutputType.DF:
            logging.info("\n ========= Generate dataframe output ========= \n")
            res = self.to_df()
            logging.info(res)
            return res
        elif o_type == OutputType.EXCEL:
            logging.info("\n ========= Generate excel output ========= \n")
            self.to_excel(output_dir)
        elif o_type == OutputType.CSV:
            logging.info("\n ========= Generate csv output ========= \n")
            self.to_csv(output_dir)
        return self.get_result()

    def _to_output_type(self, output_type: Any) -> OutputType:
        if isinstance(output_type, str):
            try:
                return OutputType(output_type.lower())
            except ValueError:
                logging.warning(
                    f"Invalid output type string: {output_type}. Defaulting to PRINT"
                )
                return OutputType.JSON
        elif isinstance(output_type, OutputType):
            return output_type
        logging.warning(f"Invalid output type: {output_type}. Defaulting to JSON")
        return OutputType.JSON

    def to_json(self, output_dir: str = ".") -> Any:
        """
        Export benchmark results to a JSON file.

        Args:
            output_dir: Directory to save the JSON file

        Returns:
            Path to the generated JSON file
        """
        data = self.get_result()
        return self.generate_json_file(data, "benchmark_results", output_dir)

    def generate_json_file(self, data, file_name, output_dir: str = "."):
        """
        Generate a JSON file from the provided data.

        Args:
            data: Data to write to the JSON file
            file_name: Name for the JSON file (without extension)
            output_dir: Directory to save the JSON file

        Returns:
            Path to the generated JSON file
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
        else:
            logging.info(f"Using existing output directory: {output_dir}")
        path = os.path.join(output_dir, file_name + ".json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Convert benchmark results to a dictionary.

        Returns:
            Dictionary with categories as keys and benchmark data as values
        """
        result = {}
        for item in self.matching_groups.values():
            result[item.category] = item.data
        return result

    def to_df(self) -> Dict[str, List[Dict[str, Union[Dict[str, Any], pd.DataFrame]]]]:
        """
        Convert benchmark results to pandas DataFrames.

        Creates a dictionary with categories as keys and lists of DataFrames as values.
        Each DataFrame represents one benchmark configuration.

        Returns:
            Dictionary mapping categories ['private','public'] to lists of DataFrames "df" with metadata 'groupInfo'.

        """
        result = {}
        for item in self.matching_groups.values():
            result[item.category] = [
                {
                    "groupInfo": item.get("groupInfo", {}),
                    "df": pd.DataFrame(item.get("rows", [])),
                }
                for item in item.data
            ]
        return result

    def to_csv(self, output_dir: str = ".") -> None:
        """
        Export benchmark results to CSV files.

        Creates two CSV folders and one json file:
        - private/: Results for private devices
        - public/: Results for public devices
        - benchmark_name_mappings.json: json dict which maps the generated csv file_name to

        Each file contains multiple CSV files, one per benchmark configuration for private and public.

        Args:
            output_dir: Directory to save CSV files
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
        else:
            logging.info(f"Using existing output directory: {output_dir}")

        for item in self.matching_groups.values():
            path = os.path.join(output_dir, item.category)
            self._write_multiple_csv_files(item.data, path)

    def _write_multiple_csv_files(
        self, data_list: List[Dict[str, Any]], output_dir: str, prefix: str = ""
    ) -> None:
        """
        Write multiple benchmark results to CSV files.

        Creates a CSV file for each benchmark configuration, with metadata
        as a JSON string in the first row and data in subsequent rows.

        Args:
            data_list: List of benchmark result dictionaries
            output_dir: Directory to save CSV files
            prefix: Optional prefix for CSV filenames
        """
        os.makedirs(output_dir, exist_ok=True)
        for idx, entry in enumerate(data_list):
            filename = f"{prefix}_table{idx+1}.csv" if prefix else f"table{idx+1}.csv"
            file_path = os.path.join(output_dir, filename)

            # Prepare DataFrame
            df = pd.DataFrame(entry.get("rows", []))

            # Prepare metadata JSON (e.g. groupInfo)
            meta = entry.get("groupInfo", {})
            json_str = json.dumps(meta, separators=(",", ":"))

            logging.info(f"Wrting csv file to {file_path}")

            # Write metadata and data
            with open(file_path, "w", encoding="utf-8", newline="") as f:
                f.write(json_str + "\n")  # First row: JSON metadata
                df.to_csv(f, index=False)  # Remaining rows: DataFrame rows

    def _get_base_url(self) -> str:
        """
        Get the base URL for API requests based on environment.

        Returns:
            Base URL string for the configured environment
        """
        return BASE_URLS[self.env]

    def get_all_private_devices(self) -> Tuple[List[Any], List[Any]]:
        """
        Print all devices found in the data.
        Separates results by category and displays counts.
        This is useful for debugging and understanding what data is available.
        """
        if not self.data:
            logging.info("No data found, please call get_data() first")
            return ([], [])

        all_private = {
            (group.get("device", ""), group.get("arch", ""))
            for item in self.data
            if (group := item.get("groupInfo", {})).get("aws_type") == "private"
        }
        iphone_set = {pair for pair in all_private if "iphone" in pair[0].lower()}
        samsung_set = {pair for pair in all_private if "samsung" in pair[0].lower()}

        # logging
        logging.info(
            f"Found private {len(iphone_set)} iphone devices: {list(iphone_set)}"
        )
        logging.info(
            f"Found private {len(samsung_set)} samsung devices: {list(samsung_set)}"
        )
        return (list(iphone_set), list(samsung_set))

    def _generate_table_name(
        self, group_info: Dict[str, Any], fields: List[str]
    ) -> str:
        """
        Generate a table name from group info fields.

        Creates a normalized string by joining specified fields from group info.

        Args:
            group_info: Dictionary containing group information
            fields: List of field names to include in the table name

        Returns:
            Normalized table name string
        """
        name = "-".join(
            self.normalize_string(group_info[k])
            for k in fields
            if k in group_info and group_info[k]
        )

        return name

    def _proces_raw_data(self, input_data: List[Dict[str, Any]]):
        """
        Process raw benchmark data.
        """
        logging.info(f"fetched {len(input_data)} data from HUD")
        data = self._clean_data(input_data)

        for item in data:
            org_group = item.get("groupInfo", {})
            if org_group.get("device", "").find("private") != -1:
                item["groupInfo"]["aws_type"] = "private"
            else:
                item["groupInfo"]["aws_type"] = "public"
            # Add full name joined by the group key fields
            item["table_name"] = self._generate_table_name(
                org_group, self.query_group_table_by_fields
            )
        self.data = deepcopy(data)

    def _process_private_public_data(self, filters: Optional[BenchmarkFilters]):
        """
        Process raw benchmark data.
        """
        if not self.data:
            logging.info("No data found, please call get_data() first")
            return

        #
        private_list = sorted(
            (
                item
                for item in self.data
                if item.get("groupInfo", {}).get("aws_type") == "private"
            ),
            key=lambda x: x["table_name"],
        )

        if filters:
            logging.info(f"Found {len(private_list)} private tables before filtering")
            private_list = self.filter_private_results(private_list, filters)
        else:
            logging.info("filters is None, using all private results")

        all_public = sorted(
            (
                item
                for item in self.data
                if item.get("groupInfo", {}).get("aws_type") == "public"
            ),
            key=lambda x: x["table_name"],
        )
        public_list = self._filter_public_result(private_list, all_public)

        logging.info(
            f"Found {len(private_list)} private tables, {[item['table_name'] for item in private_list]}"
        )
        logging.info(
            f"Found assoicated {len(public_list)} public tables, {json.dumps([item['table_name'] for item in public_list],indent=2)}"
        )

        self.matching_groups["private"] = MatchingGroupResult(
            category="private", data=private_list
        )
        self.matching_groups["public"] = MatchingGroupResult(
            category="public", data=public_list
        )

    def _clean_data(self, data_list):
        # filter data with arch equal exactly "",ios and android, this normally
        # indicates it's job-level falure indicator
        removed_gen_arch = [
            item
            for item in data_list
            if (arch := item.get("groupInfo", {}).get("arch")) is not None
            and arch.lower() not in ("ios", "android")
        ]
        data = self._filter_out_failure_only(removed_gen_arch)
        return data

    def _fetch_execu_torch_data(
        self, start_time: str, end_time: str
    ) -> Optional[List[Dict[str, Any]]]:
        url = f"{self.base_url}/api/benchmark/group_data"
        params_object = BenchmarkQueryGroupDataParams(
            repo="pytorch/executorch",
            benchmark_name="ExecuTorch",
            start_time=start_time,
            end_time=end_time,
            group_table_by_fields=self.query_group_table_by_fields,
            group_row_by_fields=self.query_group_row_by_fields,
        )
        params = {k: v for k, v in params_object.__dict__.items() if v is not None}
        with yaspin(text="Waiting for response", color="cyan") as spinner:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                spinner.ok("V")
                return response.json()
            else:
                logging.info(f"Failed to fetch benchmark data ({response.status_code})")
                logging.info(response.text)
                spinner.fail("x")
                return None

    def normalize_string(self, s: str) -> str:
        s = s.lower().strip()
        s = s.replace("+", "plus")
        s = s.replace("-", "_")
        s = s.replace(" ", "_")
        s = re.sub(r"[^\w\-\.\(\)]", "_", s)
        s = re.sub(r"_{2,}", "_", s)
        s = s.replace("_(", "(").replace("(_", "(")
        s = s.replace(")_", ")").replace("_)", ")")
        s = s.replace("(private)", "")
        return s

    def filter_private_results(
        self, all_privates: List[Dict[str, Any]], filters: BenchmarkFilters
    ):
        """
        dynamically filter private device data based on filters, if any.
        fetch all private devices within the time range, and then filter based on filter parameters
        such as device_pool, backends, and models.
        """
        private_devices = self.get_all_private_devices()

        device_pool = filters.devicePoolNames or set()
        backends = filters.backends or set()
        models = filters.models or set()

        if not backends and not device_pool and not models:
            logging.info("No filters provided, using all private results")
            return all_privates

        device_ios_match = set()
        # hardcoded since we only have 2 device pools, each for iphone and samsung
        if "apple_iphone_15_private" in device_pool:
            device_ios_match.update(
                private_devices[0]
            )  # assumed to be list of (device, arch)
        if "samsung_s22_private" in device_pool:
            device_ios_match.update(private_devices[1])
        logging.info(
            f"Applying filter: backends={backends}, devices={device_pool}, models={models}, pair_filter={bool(device_ios_match)}"
        )
        results = []
        for item in all_privates:
            info = item.get("groupInfo", {})
            if backends and info.get("backend") not in backends:
                continue

            if device_ios_match:
                # must match both device and arch in a record, otherwise skip
                pair = (info.get("device", ""), info.get("arch", ""))
                if pair not in device_ios_match:
                    continue
            if models and info.get("model", "") not in models:
                continue
            results.append(item)

        logging.info(
            f"Filtered from private data {len(all_privates)} → {len(results)} results"
        )
        if not results:
            logging.info("No results matched the filters. Something is wrong.")
        return results


def argparsers():
    parser = argparse.ArgumentParser(description="Benchmark Analysis Runner")

    # Required common args
    parser.add_argument(
        "--startTime",
        type=validate_iso8601_no_ms,
        required=True,
        help="Start time, ISO format (e.g. 2025-06-01T00:00:00)",
    )
    parser.add_argument(
        "--endTime",
        type=validate_iso8601_no_ms,
        required=True,
        help="End time, ISO format (e.g. 2025-06-06T00:00:00)",
    )
    parser.add_argument(
        "--env", choices=["local", "prod"], default="prod", help="Environment"
    )

    parser.add_argument(
        "--no-silent",
        action="store_false",
        dest="silent",
        default=True,
        help="Allow output (disable silent mode)",
    )

    # Options for generate_data
    parser.add_argument(
        "--outputType",
        choices=["json", "df", "csv", "print", "excel"],
        default="print",
        help="Output format (only for generate_data)",
    )

    parser.add_argument(
        "--outputDir", default=".", help="Output directory, default is ."
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        help="Filter results by one or more backend full name(e.g. --backends qlora mv3) (OR logic within backends scope, AND logic with other filter type)",
    )
    parser.add_argument(
        "--device-pools",
        nargs="+",  # allow one or more values
        choices=VALID_PRIVATE_DEVICE_POOLS_NAMES,
        help="List of devices to include [apple_iphone_15_private, samsung_s22_private, you can include both] (OR logic within private-device-pools scope, AND logic with other filter type)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Filter by one or more models (e.g. --backend 'meta-llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8' 'mv3') (OR logic withn models scope, AND logic with other filter type)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = argparsers()
    fetcher = ExecutorchBenchmarkFetcher(args.env, args.silent)
    result = fetcher.run(
        args.startTime,
        args.endTime,
        filters=BenchmarkFilters(
            models=args.models,
            backends=args.backends,
            devicePoolNames=args.device_pools,
        ),
    )
    fetcher.output_data(args.outputType, args.outputDir)
