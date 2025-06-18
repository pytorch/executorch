"""
ExecutorchBenchmark Analysis Data Retrieval

This module provides tools for fetching, processing, and analyzing benchmark data
from the HUD Open API for ExecutorchBenchmark. It supports filtering data by device
types (private and public), exporting results in various formats (JSON, DataFrame, Excel, CSV),
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
from typing import Any, Dict, List

import pandas as pd
import requests
from yaspin import yaspin

logging.basicConfig(level=logging.INFO)


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
        category: Category name (e.g., "private", "public")
        data: List of benchmark data for this category
    """

    category: str
    data: list


@dataclass
class BenchmarkFilters:
    models: list
    backends: list
    devices: list


BASE_URLS = {
    "local": "http://localhost:3000",
    "prod": "https://hud.pytorch.org",
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
    1. Fetch benchmark data for a specified time range
    2. Process and categorize data into private and public device results
    3. Export results in various formats (JSON, DataFrame, Excel, CSV)

    Usage:
        fetcher = ExecutorchBenchmarkFetcher()
        fetcher.run(start_time, end_time)
        fetcher.output_data(OutputType.EXCEL, output_dir="./results")
    """

    def __init__(
        self,
        env="prod",
        disable_logging=False,
        group_table_fields=None,
        group_row_fields=None,
    ):
        """
        Initialize the ExecutorchBenchmarkFetcher.

        Args:
            env: Environment to use ("local" or "prod")
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
            else ["workflow_id", "job_id", "granularity_bucket"]
        )
        self.data = None
        self.disable_logging = disable_logging
        self.matching_groups: Dict[str, MatchingGroupResult] = {}

    def add_abbreviations(self, abbreviations: Dict[str, str]):
        self.abbreviations = abbreviations

    def run(
        self,
        start_time: str,
        end_time: str,
        filters: BenchmarkFilters,
    ) -> None:
        # reset group & raw data for new run
        self.matching_groups = {}
        self.data = None

        data = self._fetch_execu_torch_data(start_time, end_time)
        if data is None:
            logging.warning("no data fetched from the HUD API")
            return None
        res = self._process(data, filters)
        self.data = res.get("data", [])
        private_list = res.get("private", [])
        public_list = self._filter_public_result(private_list, res["public"])

        self.matching_groups["private"] = MatchingGroupResult(
            category="private", data=private_list
        )
        self.matching_groups["public"] = MatchingGroupResult(
            category="public", data=public_list
        )

    def _filter_out_failure_only(
        self, data_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        clean FAILURE_REPORT only metrics
        """
        ONLY = {"workflow_id", "granularity_bucket", "job_id", "FAILURE_REPORT"}
        for item in data_list:
            filtered_rows = [
                row
                for row in item.get("rows", [])
                # Keep row only if it has additional fields beyond ONLY
                if not set(row.keys()).issubset(ONLY)
            ]
            item["rows"] = filtered_rows
        return [item for item in data_list if item.get("rows")]

    def _filter_public_result(self, private_list, public_list):
        """
        Filter public device results to match private device configurations.

        Finds the intersection of table names between private and public results
        to ensure comparable data sets.

        Args:
            private_list: List of benchmark results for private devices
            public_list: List of benchmark results for public devices

        Returns:
            Filtered list of public device results that match private device configurations
        """
        # find intersection betwen private and public tables.
        common = list(
            set([item["table_name"] for item in private_list])
            & set([item["table_name"] for item in public_list])
        )

        if not self.disable_logging:
            logging.info(
                f"Found {len(common)} table names existed in both private and public, use it to filter public tables:"
            )
            logging.info(json.dumps(common, indent=1))
        filtered_public = [item for item in public_list if item["table_name"] in common]
        return filtered_public

    def get_result(self):
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
                    f"Wrting excel sheet to file {file} with sheet name {sheet_name} for {entry["table_name"]}"
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
        logging.info(f"Generating output with type: {[self.matching_groups.keys()]}")
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

    def to_dict(self) -> Any:
        """
        Convert benchmark results to a dictionary.

        Returns:
            Dictionary with categories as keys and benchmark data as values
        """
        result = {}
        for item in self.matching_groups.values():
            result[item.category] = item.data
        return result

    def to_df(self) -> Any:
        """
        Convert benchmark results to pandas DataFrames.

        Creates a dictionary with categories as keys and lists of DataFrames as values.
        Each DataFrame represents one benchmark configuration.

        Returns:
            Dictionary mapping categories to lists of DataFrames with metadata
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

    def _write_multiple_csv_files(self, data_list, output_dir, prefix=""):
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
        base_urls = {
            "local": "http://localhost:3000",
            "prod": "https://hud.pytorch.org",
        }
        return base_urls[self.env]

    def print_all_groups_info(self) -> None:
        """
        Print all benchmark table group info found in the data.
        Separates results by category and displays counts.
        This is useful for debugging and understanding what data is available.
        """
        if not self.data or not self.matching_groups:
            logging.info("No data found, please call get_data() first")
            return
        logging.info(
            "=========== Full list of table info from HUD API =============\n"
            " please use values in field `info` for filtering, "
            "while `groupInfo` holds the original benchmark metadata"
        )
        names = []
        for item in self.data:
            names.append(
                {
                    "table_name": item.get("table_name", ""),
                    "groupInfo": item.get("groupInfo", {}),
                    "info": item.get("info", {}),
                    "counts": len(item.get("rows", [])),
                }
            )
        for name in names:
            logging.info(json.dumps(name, indent=2))

    def _generate_table_name(self, group_info: dict, fields: list[str]) -> str:
        name = "_".join(
            self.normalize_string(group_info[k])
            for k in fields
            if k in group_info and group_info[k]
        )
        if "(private)" in name:
            name = name.replace("(private)", "")
        return name

    def simplify_ios(self, s: str) -> str:
        return s.split(".")[0]

    def _generate_matching_name(self, group_info: dict, fields: list[str]) -> str:
        info = deepcopy(group_info)
        name = "_".join(
            self.normalize_string(info[k]) for k in fields if k in info and info[k]
        )
        if "(private)" in name:
            name = name.replace("(private)", "")
            # name = name +'(private)'
        return name

    def _process(self, input_data: List[Dict[str, Any]], filters: BenchmarkFilters):
        """
        Process raw benchmark data.

        This method:
        1. clean the data that generated by FAILURE_REPORT,
        2. Creates table_name from info
        3. Determines aws_type (public/private) based on info.device
        4. Sorts results by table_name
        Args:
            input_data: Raw benchmark data from API
        Returns:
            Processed benchmark data
        """
        # filter data with arch equal exactly "",ios and android, this normally indicates it's job-level falure indicator
        logging.info(f"fetched {len(input_data)} data from HUD")
        data = self._clean_data(input_data)
        private = []
        public = []

        for item in data:
            # normalized string values groupInfo to info
            item["info"] = {
                k: self.normalize_string(v)
                for k, v in item.get("groupInfo", {}).items()
                if v is not None and isinstance(v, str)
            }
            group = item.get("info", {})
            # Add full name joined by the group key fields
            item["table_name"] = self._generate_table_name(
                group, self.query_group_table_by_fields
            )

            # Mark aws_type: private or public
            if group.get("device", "").find("private") != -1:
                item["info"]["aws_type"] = "private"
            else:
                item["info"]["aws_type"] = "public"
                public.append(item)
        raw_data = deepcopy(data)

        # applies customized filters if any
        data = self.filter_results(data, filters)
        # generate private and public results
        private = sorted(
            (
                item
                for item in data
                if item.get("info", {}).get("aws_type") == "private"
            ),
            key=lambda x: x["table_name"],
        )
        public = sorted(
            (item for item in data if item.get("info", {}).get("aws_type") == "public"),
            key=lambda x: x["table_name"],
        )
        logging.info(
            f"fetched clean data {len(data)}, private:{len(private)}, public:{len(public)}"
        )
        return {"data": raw_data, "private": private, "public": public}

    def _clean_data(self, data_list):
        removed_gen_arch = [
            item
            for item in data_list
            if (arch := item.get("groupInfo", {}).get("arch")) is not None
            and arch.lower() not in ("ios", "android")
        ]

        data = self._filter_out_failure_only(removed_gen_arch)
        return data

    def _fetch_execu_torch_data(self, start_time, end_time):
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
        s = s.replace("+","plus")
        s = s.replace("_", "-")
        s = s.replace(" ", "-")
        s = re.sub(r"[^\w\-\.\(\)]", "-", s)
        s = re.sub(r"-{2,}", "-", s)
        s = s.replace("-(", "(").replace("(-", "(")
        s = s.replace(")-", ")").replace("-)", ")")
        return s

    def filter_results(self, data: List, filters: BenchmarkFilters):
        backends = filters.backends
        devices = filters.devices
        models = filters.models

        if not backends and not devices and not models:
            return data
        logging.info(
            f"applies OR filter: backends {backends},  devices:{devices},models:{models} "
        )
        pre_len = len(data)
        results = []
        for item in data:
            info = item.get("info", {})
            if backends and info.get("backend") not in backends:
                continue
            if devices and not any(dev in info.get("device", "") for dev in devices):
                continue
            if models and info.get("model", "") not in models:
                continue
            results.append(item)
        after_len = len(results)
        logging.info(f"applied customized filter before: {pre_len}, after: {after_len}")
        if after_len == 0:
            logging.info(
                "it seems like there is no result matches the filter values"
                ", please run script --no-silent again, and search for values in field"
                " 'info' for right format"
            )
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
        help="Filter results by one or more backend full name(e.g. --backend qlora mv3) (OR logic)",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        help="Filter results by device names (e.g. --devices samsung-galaxy-s22-5g)(OR logic)",
    )
    parser.add_argument("--models", nargs="+", help="Filter by models (OR logic)")
    return parser.parse_args()


if __name__ == "__main__":
    args = argparsers()
    fetcher = ExecutorchBenchmarkFetcher(args.env, args.silent)
    result = fetcher.run(
        args.startTime,
        args.endTime,
        filters=BenchmarkFilters(
            models=args.models, backends=args.backends, devices=args.devices
        ),
    )
    if not args.silent:
        fetcher.print_all_groups_info()
    fetcher.output_data(args.outputType, args.outputDir)
