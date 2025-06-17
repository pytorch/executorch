import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO)

# Default private_device_matching_list
private_device_matching_list = [
    ["llama3", "qlora", "s22_5g", "android_13"],
    ["llama3", "spinq", "s22_5g", "android_13"],
    ["mv3", "qnn", "s22_5g", "android_13"],
    ["mv3", "xnnpack_q8", "s22_5g", "android_13"],
    ["llama3", "qlora", "s22_ultra_5g", "android_14"],
    ["llama3", "spinq", "s22_ultra_5g", "android_14"],
    ["mv3", "qnn", "s22_ultra_5g", "android_14"],
    ["mv3", "xnnpack_q8", "s22_ultra_5g", "android_14"],
    ["mv3", "xnnpack_q8", "pixel3_rooted", "android"],
    ["llama3", "qlora", "iphone_15_pro_max", "ios_17"],
    ["llama3", "spinq", "iphone_15_pro_max", "ios_17"],
    ["mv3", "xnnpack_q8", "iphone_15_pro_max", "ios_17"],
    ["mv3", "coreml", "iphone_15_pro_max", "ios_17"],
    ["mv3", "mps", "iphone_15_pro_max", "ios_17"],
    ["llama3", "qlora", "iphone_15", "ios_18.0"],
    ["llama3", "spinq", "iphone_15", "ios_18.0"],
    ["mv3", "xnnpack_q8", "iphone_15", "ios_18.0"],
    ["mv3", "coreml", "iphone_15", "ios_18.0"],
    ["mv3", "mps", "iphone_15", "ios_18.0"],
]

# Default public_device_matching_list
public_device_matching_list = [
    ["llama3", "qlora", "s22_5g", "android_13"],
    ["llama3", "spinq", "s22_5g", "android_13"],
    ["mv3", "qnn", "s22_5g", "android_13"],
    ["mv3", "xnnpack_q8", "s22_5g", "android_13"],
    ["llama3", "spinq", "s22_5g", "android_12"],
    ["llama3", "qlora", "s22_ultra_5g", "android"],
    ["llama3", "spinq", "s22_ultra_5g", "android_12"],
    ["mv3", "xnnpack_q8", "s22_ultra_5g", "android_12"],
    ["mv3", "qnn", "s22_ultra_5g", "android_12"],
    ["llama3", "qlora", "iphone_15_pro_max", "ios_17"],
    ["llama3", "spinq", "iphone_15_pro_max", "ios_17"],
    ["mv3", "xnnpack_q8", "iphone_15_pro_max", "ios_17"],
    ["mv3", "coreml", "iphone_15_pro_max", "ios_17"],
    ["mv3", "mps", "iphone_15_pro_max", "ios_17"],
    ["llama3", "qlora", "iphone_15", "ios_18.0"],
    ["llama3", "spinq", "iphone_15", "ios_18.0"],
    ["mv3", "xnnpack_q8", "iphone_15", "ios_18.0"],
    ["mv3", "coreml", "iphone_15", "ios_18.0"],
    ["mv3", "mps", "iphone_15", "ios_18.0"],
]


# The abbreviations used to generate the short name for the benchmark result table
# this is used to avoid the long table name issue when generating excel file (<=31 characters)
ABBREVIATIONS = {
    "samsung": "smg",
    "galaxy": "gx",
    "ultra": "ul",
    "android": "and",
    "iphone": "ip",
    "xnnpackq8": "xnnq8",
}
def abbreviate(s):
    for full, abbr in ABBREVIATIONS.items():
        s = s.replace(full, abbr)
    return s


@dataclass
class BenchmarkQueryGroupDataParams:
    repo: str
    benchmark_name: str
    start_time: str
    end_time: str
    group_table_by_fields: list
    group_row_by_fields: list


BASE_URLS = {
    "local": "http://localhost:3000",
    "prod": "https://hud.pytorch.org",
}


def validate_iso8601_no_ms(value):
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S").strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid datetime format for '{value}'. Expected: YYYY-MM-DDTHH:MM:SS"
        )


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", choices=["local", "prod"], default="prod", help="Choose environment"
    )
    parser.add_argument(
        "--startTime",
        type=validate_iso8601_no_ms,
        required=True,
        help="Start time in ISO format (e.g. 2025-06-01T00:00:00)",
    )
    parser.add_argument(
        "--endTime",
        type=validate_iso8601_no_ms,
        required=True,
        help="End time in ISO format (e.g. 2025-06-06T00:00:00)",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="disable all loggings",
    )
    parser.add_argument(
        "--outputType",
        choices=["json", "excel", "df"],
        default="print",
        help="Choose output type for your run",
    )
    parser.add_argument(
        "--outputDir",
        default=".",
        help="Only used when output-type is excel, default to current directory",
    )

    return parser.parse_args()


class ExecutorchBenchmarkFetcher:
    """
    Fetch and process benchmark data from HUD API for ExecutorchBenchmark.

    Usage:
        fetcher = ExecutorchBenchmarkFetcher()
        fetcher.run(start_time, end_time, private_device_matching_list, public_device_matching_list)
        # Convert results to DataFrames
        private_dfs, public_dfs = fetcher.to_df()
        # Export results to Excel files
        fetcher.to_excel(output_dir=".")
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
            else ["device", "backend", "arch", "model"]
        )
        self.query_group_row_by_fields = (
            group_row_fields
            if group_row_fields
            else ["workflow_id", "job_id", "granularity_bucket"]
        )
        self.data = None
        self.disable_logging = disable_logging
        self.results_private = []
        self.results_public = []

    def run(
        self,
        start_time: str,
        end_time: str,
        privateDeviceMatchings: List[List[str]],
        publicDeviceMatchings: List[List[str]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Execute the benchmark data fetching and processing workflow.

        This method orchestrates the entire process:
        1. Fetches raw data from the HUD API for the specified time range
        2. Processes and normalizes the data
        3. Filters results based on device matching criteria for both private and public devices

        Args:
            start_time: ISO8601 formatted start time (YYYY-MM-DDTHH:MM:SS)
            end_time: ISO8601 formatted end time (YYYY-MM-DDTHH:MM:SS)
            privateDeviceMatchings: List of keyword lists for matching private devices
            publicDeviceMatchings: List of keyword lists for matching public devices
        Returns:
            Tuple containing (private_device_results, public_device_results)
        """
        self.data = self._fetch_data(start_time, end_time)
        if not self.disable_logging:
            self.print_all_names()

        if not self.disable_logging:
            logging.info(
                f"\n ========= Search tables specific for matching keywords ========= \n"
            )
        self.results_private = self.find_target_tables(privateDeviceMatchings, True)
        self.results_public = self.find_target_tables(publicDeviceMatchings, False)

        logging.info(
            f"Found {len(self.results_private)} private device benchmark results, and {len(self.results_public)} public device benchmark results"
        )
        return (self.results_private, self.results_public)

    def to_df(self) -> Tuple[Any, Any]:
        """
        Convert benchmark results to pandas DataFrames.

        Transforms the raw benchmark results into DataFrames for easier analysis
        and manipulation.

        Returns:
            Tuple containing (private_device_dataframes, public_device_dataframes)
            Each item is a list of dictionaries with 'groupInfo' and 'df' keys
        """
        private_dfs = [
            {"groupInfo": item["groupInfo"], "df": pd.DataFrame(item["rows"])}
            for item in self.results_private
        ]
        public_dfs = [
            {"groupInfo": item["groupInfo"], "df": pd.DataFrame(item["rows"])}
            for item in self.results_public
        ]
        return (private_dfs, public_dfs)

    def to_excel(self, output_dir: str = ".") -> None:
        """
        Export benchmark results to Excel files.

        Creates two Excel files:
        - res_private.xlsx: Results for private devices
        - res_public.xlsx: Results for public devices

        Each file contains multiple sheets, one per benchmark configuration.

        Args:
            output_dir: Directory to save Excel files
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
        else:
            logging.info(f"Using existing output directory: {output_dir}")
        private_path = os.path.join(output_dir, "res_private.xlsx")
        public_path = os.path.join(output_dir, "res_public.xlsx")
        self._write_multi_sheet_excel(self.results_private, private_path)
        self._write_multi_sheet_excel(self.results_public, public_path)

    def _write_multi_sheet_excel(
        self, data_list: List[Dict[str, Any]], output_path: str
    ) -> None:
        """
        Write multiple benchmark results to sheets in an Excel file.

        Creates an Excel file with multiple sheets, one for each benchmark configuration.
        Handles sheet name length limitations and truncates names if necessary.

        Args:
            data_list: List of benchmark result dictionaries
            output_path: Path to save the Excel file
        """
        logging.info(
            f"\n ========= Generate excel file with multiple sheets for {output_path}========= \n"
        )
        with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
            for idx, entry in enumerate(data_list):
                sheet_name = entry.get("short_name", f"sheet{idx+1}")
                logging.info(f"sheet_name: {sheet_name} with length: {len(sheet_name)}")
                if len(sheet_name) > 31:
                    logging.warning(
                        f"sheet name {sheet_name} is too long, truncating to 31 characters with prefix idx"
                    )
                    sheet_name += f"{idx+1}_{sheet_name}"
                    sheet_name = sheet_name[:31]
                rows = entry.get("rows", [])

                logging.info(f"Writing {sheet_name} to excel: {output_path}")
                df = pd.DataFrame(rows)
                df.to_excel(writer, sheet_name=sheet_name or "Sheet", index=False)

    def _fetch_data(
        self, start_time: str, end_time: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch and process benchmark data for the specified time range.

        Args:
            start_time: ISO8601 formatted start time
            end_time: ISO8601 formatted end time

        Returns:
            Processed benchmark data or None if fetch failed
        """
        data = self._fetch_execu_torch_data(start_time, end_time)
        if data is None:
            return None
        self.data = self._process(data)
        return self.data

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

    def print_all_names(self) -> None:
        """
        Print all benchmark table names found in the data.

        Separates results by device type (public/private) and displays counts.
        This is useful for debugging and understanding what data is available.
        """
        if not self.data:
            return
        logging.info("peeking table result:")
        logging.info(json.dumps(self.data[0], indent=2))
        public_ones = [
            item["table_name"]
            for item in self.data
            if item["groupInfo"]["aws_type"] == "public"
        ]
        private_ones = [
            item["table_name"]
            for item in self.data
            if item["groupInfo"]["aws_type"] == "private"
        ]
        # Print all found benchmark table names
        logging.info(
            f"\n============List all benchmark result table names (Public and Private) below =================\n"
        )
        logging.info(
            f"\n============ public device benchmark results({len(public_ones)})=================\n"
        )
        for name in public_ones:
            logging.info(name)
        logging.info(
            f"\n======= private device benchmark results({len(private_ones)})=======\n"
        )
        for name in private_ones:
            logging.info(name)

    def _generate_table_name(self, group_info:dict, fields: list[str]) -> str:
        name = "|".join(group_info[k] for k in fields if k in group_info and group_info[k])
        return self.normalize_string(name)

    def _process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process raw benchmark data.

        This method:
        1. Normalizes string values in groupInfo
        2. Creates table_name from group info components
        3. Determines aws_type (public/private) based on device name
        4. Sorts results by table_name

        Args:
            data: Raw benchmark data from API

        Returns:
            Processed benchmark data
        """
        for item in data:
            # normalized string values in groupInfo
            item["groupInfo"] = {
                k: self.normalize_string(v)
                for k, v in item.get("groupInfo", {}).items()
                if v is not None and isinstance(v, str)
            }
            group = item.get("groupInfo", {})
            name = self._generate_table_name(group, self.query_group_table_by_fields)

            # Add full name joined by the group key fields
            item["table_name"] = name

            # Mark aws_type: private or public
            if group.get("device", "").find("private") != -1:
                item["groupInfo"]["aws_type"] = "private"
            else:
                item["groupInfo"]["aws_type"] = "public"
                
        data.sort(key=lambda x: x["table_name"])
        logging.info(f"fetched {len(data)} table views")
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
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            logging.info(f"Failed to fetch benchmark data ({response.status_code})")
            logging.info(response.text)
            return None

    def generate_short_name(self, words, size):
        shortKeys = [k.replace("_", "") for k in words]
        s = "_".join(shortKeys)
        if size > 0:
            logging.warning(
                f"we found more than one table matches the keywords, adding size to distinguish: {s}"
            )
            s += s + "_" + str(size)
        for full, abbr in ABBREVIATIONS.items():
            s = s.replace(full, abbr)
        return s

    def find_target_tables(self, keywords, is_private) -> List[Any]:
        if not self.data:
            logging.info("No data found, please call get_data() first")
            return []
        matchings = []
        results = {}
        for keyword_list in keywords:
            norm_keywords = [kw.lower().replace(" ", "_") for kw in keyword_list]
            match = []
            for item in self.data:
                key = item.get("table_name", "")
                if not key:
                    continue
                if all(kw in key for kw in norm_keywords):
                    is_item_private = (
                        item.get("groupInfo", {}).get("aws_type", "") == "private"
                    )
                    if is_private is not is_item_private:
                        continue # skip if not matching private/public device

                    # generate short name for each table data
                    item["short_name"] = self.generate_short_name(
                        norm_keywords, len(match)
                    )
                    match.append(key)
                    results[key] = item
            matchings.append((keyword_list, match))
        if not self.disable_logging:
            logging.info(
                f"\n============ MATCHING Found results: {len(results)}=========\n"
            )
            for keywords, match in matchings:
                logging.info(f"Keywords: {keywords}: matchings: {match}")
        return list(results.values())

    def normalize_string(self, s, replace="_"):
        return s.lower().replace(" ", replace)


if __name__ == "__main__":
    args = argparser()
    fetcher = ExecutorchBenchmarkFetcher(args.env, args.silent)
    private, public = fetcher.run(
        args.startTime,
        args.endTime,
        private_device_matching_list,
        public_device_matching_list,
    )

    if args.outputType == "df":
        private, public = fetcher.to_df()
        logging.info(
            f"=====================Printing private device benchmark results in dataframe====================="
        )
        for item in private:
            logging.info(item["groupInfo"])
            logging.info(item["df"])
            logging.info("\n")
        logging.info(
            f"======================Printing public device benchmark results in dataframe====================="
        )
        for item in public:
            logging.info(item["groupInfo"])
            logging.info(item["df"])
            logging.info("\n")
    elif args.outputType == "excel":
        logging.info(
            f"Writing  benchmark results to excel file: {args.outputDir}/res_private.xlsx"
        )
        fetcher.to_excel(args.outputDir)
    else:
        logging.info(
            f"======================Printing private device benchmark results in json format======================"
        )
        print(json.dumps(private, indent=2))
        logging.info(
            f"======================Printing public device benchmark results in json format======================"
        )
        print(json.dumps(public, indent=2))
