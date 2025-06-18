import argparse
from copy import deepcopy
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from re import A, L
from typing import Any, Dict, List, Optional, Tuple
import re

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO)

# Default private_device_matching_list
DEFAULT_PRIVATE_MATCHING_LIST = [
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
DEFAULT_PUBLIC_MATCHING_LIST = [
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
# this is used to avoid the long table name issue when generating csv file (<=100 characters)
DEFAULT_ABBREVIATIONS = {
    "samsung": "",
    "galaxy": "",
    "5g": "",
    "private":"",
    "xnnpackq8": "xnnq8",
    "iphone15promax": "iphone15max",
    "meta-llama/llama-3.2-1b": "llama3.2",
}

from enum import Enum

class OutputType(Enum):
    PRINT = "print"
    CSV = "csv"
    JSON = "json"
    DF = "df"

@dataclass
class BenchmarkQueryGroupDataParams:
    repo: str
    benchmark_name: str
    start_time: str
    end_time: str
    group_table_by_fields: list
    group_row_by_fields: list

@dataclass
class MatchingGroupResult:
    category: str
    keywords: list
    data: list


@dataclass
class MatchingGroupInput:
    category: str
    keywords: list
    conditions: list

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

def parse_filter_group(value: str) -> dict:
    include = []
    exclude = []
    parts = value.split(";")
    for part in parts:
        if part.startswith("include="):
            include = part[len("include="):].split(",")
        elif part.startswith("exclude="):
            exclude = part[len("exclude="):].split(",")

    return {"include": include, "exclude": exclude}

class ExecutorchBenchmarkFetcher:
    """
    Fetch and process benchmark data from HUD API for ExecutorchBenchmark.

    Usage:
        fetcher = ExecutorchBenchmarkFetcher()
        fetcher.run(start_time, end_time, private_device_matching_list, public_device_matching_list)
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
            else ["model", "backend","device","arch"]
        )
        self.query_group_row_by_fields = (
            group_row_fields
            if group_row_fields
            else ["workflow_id", "job_id", "granularity_bucket"]
        )
        self.data = None
        self.disable_logging = disable_logging
        self.abbreviations = DEFAULT_ABBREVIATIONS
        self.matching_groups: Dict[str, MatchingGroupResult]= {}
        self.origin_mappings: Dict[str, Dict[str,Any]] = {}

    def add_abbreviations(self, abbreviations: Dict[str, str]):
        self.abbreviations = abbreviations

    def generate_matching_list(
        self,
        start_time: str,
        end_time: str,
        filter_groups: List[dict],
        category: str = "unknown",
        output_type: OutputType = OutputType.PRINT,
        output_dir: str = "."
    ):
        filter_groups = filter_groups or [{"include": [], "exclude": []}]
        o_type = self._to_output_type(output_type)
        logging.info(f"filter_groups applied {filter_groups} with output_type {o_type}")
        data = self._fetch_data(start_time, end_time)
        if data is None:
            logging.info("No data found")
            return []
        results = []
        seen = set()

        for item in data:
            name = item["table_name"]
            group_info = item["info"]
            matched = False
            for group in filter_groups:
                include = group.get("include", [])
                exclude = group.get("exclude", [])
                if include and not all(kw.lower() in name for kw in include):
                    continue
                if exclude and any(kw.lower() in name for kw in exclude):
                    continue
                matched = True
                break  # matched one group, no need to evaluate more
            if matched:
                key = tuple(group_info.get(k, "") for k in self.query_group_table_by_fields)
                if key not in seen:
                    results.append([
                        group_info[k] for k in self.query_group_table_by_fields
                        if k in group_info and group_info[k]
                    ])
                    seen.add(key)
        if o_type == OutputType.JSON:
            self.generate_json_file(results, category, output_dir)
        else:
            logging.info("Print result")
            logging.info(json.dumps(results, indent=2))

        logging.info(f"generated {len(results)} matching list items")
        return results


    def run(
        self,
        start_time: str,
        end_time: str,
        matching_inputs: List[MatchingGroupInput] = [],
    ) -> Any:
        """
        Execute the benchmark data fetching and processing workflow.
        """
        self.data = self._fetch_data(start_time, end_time)
        # reset everything for generate the new output
        self.matching_groups = {}
        for matching_input in matching_inputs:
            category = matching_input.category
            keywords = matching_input.keywords
            result = self.find_target_tables(keywords, matching_input.conditions)
            self.matching_groups[category] = MatchingGroupResult(category, keywords, result)
        return self.data

    def get_result(self):
        return deepcopy(self.to_dict())

    def output_data(
        self,
        output_type: OutputType = OutputType.PRINT,
        output_dir: str = ".") -> Any:

        logging.info(f"Generating output with type: {[category for category in self.matching_groups.keys()]}")

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
        elif o_type == OutputType.CSV:
            logging.info("\n ========= Generate csv output ========= \n")
            self.to_csv(output_dir)
        return self.get_result()

    def _to_output_type(self, output_type: Any) -> OutputType:
        if isinstance(output_type, str):
            try:
                return OutputType(output_type.lower())
            except ValueError:
                logging.warning(f"Invalid output type string: {output_type}. Defaulting to PRINT")
                return OutputType.JSON
        elif isinstance(output_type, OutputType):
            return output_type
        logging.warning(f"Invalid output type: {output_type}. Defaulting to JSON")
        return OutputType.JSON

    def to_json(self,output_dir: str = ".") -> Any:
        data = self.get_result()
        return self.generate_json_file(data, "benchmark_results", output_dir)

    def generate_json_file(self,data,file_name, output_dir: str = "."):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
        else:
            logging.info(f"Using existing output directory: {output_dir}")
        path = os.path.join(output_dir, file_name + ".json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def to_dict(self)-> Any:
        result = {}
        for item in self.matching_groups.values():
            result[item.category] = item.data
        return result

    def to_df(self) -> Any:
        """
        Convert benchmark results to pandas DataFrames.

        Transforms the raw benchmark results into DataFrames for easier analysis
        and manipulation.

        Returns:
            Tuple containing (private_device_dataframes, public_device_dataframes)
            Each item is a list of dictionaries with 'info' and 'df' keys
        """
        result = {}
        for item in self.matching_groups.values():
            result[item.category] =[
            {"info": item["info"], "df": pd.DataFrame(item["rows"])}
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
        self, data_list: List[Dict[str, Any]], output_dir: str, file_prefix=""
    ) -> None:
        """
        Write multiple benchmark results to separate CSV files.
        Each entry in `data_list` becomes its own CSV file.
        Args:
            data_list: List of benchmark result dictionaries
            output_dir: Directory to save the CSV files
        """
        os.makedirs(output_dir, exist_ok=True)
        logging.info(
            f"\n ========= Generating multiple CSV files in {output_dir} ========= \n"
        )
        for _, entry in enumerate(data_list):
            table_name = entry.get("table_name","")
            if not table_name:
                continue
            file_name = self.generate_short_name(table_name)
            if file_prefix:
                file_name = file_prefix+ '_' + file_name
            file_path = os.path.join(output_dir, f"{file_name}.csv")
            rows = entry.get("rows", [])

            if len(file_name) > 120:
                logging.warning(
                    f"File path '{file_path}' is too long, this may cause csv failure"
            )
            logging.info(f"\noriginal table name:{table_name}")
            logging.info(f"Writing CSV to path(len({len(file_path)})): '{file_path}' with {len(rows)} rows")
            df = pd.DataFrame(rows)
            df.to_csv(file_path, index=False)

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

    def print_all_groups_info(self) -> None:
        """
        Print all benchmark table group info found in the data.
        Separates results by category and displays counts.
        This is useful for debugging and understanding what data is available.
        """

        if not self.data or not self.matching_groups:
            logging.info("No data found, please call get_data() first")
            return

        logging.info("peeking table result:")
        logging.info(json.dumps(self.data[0], indent=2))

        for item in self.matching_groups.values():
            logging.info(f" all {item.category} benchmark results")
            names = []
            for row in item.data:
                names.append({
                    "table_name": row["table_name"],
                    "info": row["info"],
                    "counts": len(row["rows"])
                })
            logging.info(
            f"\n============ {item.category} benchmark results({len(names)})=================\n"
            )
            for name in names:
                logging.info(json.dumps(name, indent=2))

    def _generate_table_name(self, group_info: dict, fields: list[str]) -> str:
        name = "_".join(
            self.normalize_string(group_info[k]) for k in fields if k in group_info and group_info[k]
        )
        return name

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
            # normalized string values groupInfo to info
            item["info"] = {
                k: self.normalize_string(v)
                for k, v in item.get("groupInfo", {}).items()
                if v is not None and isinstance(v, str)
            }
            group = item.get("info", {})
            name = self._generate_table_name(group, self.query_group_table_by_fields)
            # Add full name joined by the group key fields
            item["table_name"] = name

            # Mark aws_type: private or public
            if group.get("device", "").find("private") != -1:
                item["info"]["aws_type"] = "private"
            else:
                item["info"]["aws_type"] = "public"
        data.sort(key=lambda x: x["table_name"])
        logging.info(f"fetched {len(data)} table views from HUD ")
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

    def generate_short_name(self, name):
        s = name
        for full, abbr in self.abbreviations.items():
            s = s.replace(full, abbr)
        s = re.sub(r"-{2,}", "-", s)
        return s

    def _match_filter(self, item: dict, filter_str: str) -> bool:
        """Evaluate whether `item` satisfies a dot-notated filter like 'group_info.aws_type=private'."""
        try:
            key_path, expected = filter_str.split("=", 1)
            keys = key_path.strip().split(".")
            current = item
            for k in keys:
                current = current.get(k, {})
            return current == expected
        except Exception as e:
            logging.info(f"Failed to evaluate filter '{filter_str}': {e}")
        return False

    def find_target_tables(self, keywords, conditions) -> List[Any]:
        if not self.data:
            logging.info("No data found, please call get_data() first")
            return []
        matchings = []
        results = {}
        for keyword_list in keywords:
            norm_keywords = [self.normalize_string(kw) for kw in keyword_list]
            match = []
            for item in self.data:
                table_name = item.get("table_name", "")
                if not table_name:
                    continue
                if all(kw in table_name for kw in norm_keywords):
                    condition_meets = True
                    # for condition checks, any condition check fails with causes the item to be skipped to add to the category result
                    for condition in conditions:
                        condition_meets &= self._match_filter(item, condition)
                    if not condition_meets:
                        continue
                    match.append(table_name)
                    results[table_name] = item
            matchings.append((norm_keywords, match))
        if not self.disable_logging:
            logging.info(
                f"\n============ MATCHING Found results: {len(results)}=========\n"
            )
            for keywords, match in matchings:
                logging.info(f"Keywords: {keywords} {len(match)} matchings: {match}")
        return list(results.values())

    def normalize_string(self, s: str) -> str:
        s = s.lower().strip()
        s = s.replace("(", "").replace(")", "")
        s = s.replace("_", "-")
        s = s.replace(" ", "-")
        s = re.sub(r"[^\w\-.]", "-", s)
        s = re.sub(r"-{2,}", "-", s)
        return s

def argparsers():
    parser = argparse.ArgumentParser(description="Multi-task runner")

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--startTime",
        type=validate_iso8601_no_ms,
        required=True,
        help="Start time in ISO format (e.g. 2025-06-01T00:00:00)"
    )
    common_parser.add_argument(
        "--endTime",
        type=validate_iso8601_no_ms,
        required=True,
        help="End time in ISO format (e.g. 2025-06-06T00:00:00)"
    )
    common_parser.add_argument(
        "--env", choices=["local", "prod"], default="prod", help="Choose environment"
    )
    common_parser.add_argument(
    "--silent",
    action="store_true",
    help="Disable all logging"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    # generate_data
    generate_data = subparsers.add_parser(
    "generate_data", parents=[common_parser], help="generate data from HUD API"
    )
    generate_data.add_argument("--outputType", choices=["json", "df", "csv", "print"], default="print")
    generate_data.add_argument("--outputDir", default=".")
    generate_data.add_argument("--includePrivate", default=True)
    generate_data.add_argument("--includePublic", default=True)
    generate_data.add_argument("--private-matching-json-path",default=None)
    generate_data.add_argument("--public-matching-json-path",default=None)

    # fetch_list
    fetch_list = subparsers.add_parser(
    "get_matching_list", parents=[common_parser], help="Run fetch_matching_list")
    fetch_list.add_argument("--filters", nargs="*", default=[])
    fetch_list.add_argument("--excludeFilters", nargs="*", default=[])
    fetch_list.add_argument("--category", required=True,help="Run fetch_matching_list to filter designed_list")
    fetch_list.add_argument("--outputDir", default=".")
    fetch_list.add_argument("--outputType", choices=["json", "print"], default="print")
    fetch_list.add_argument("--filter",type=parse_filter_group,action="append", default=[],help="Filter group, e.g. 'include=iphone,metal;exclude=simulator'"
    )

    return parser.parse_args()

def get_matching_list(args):
    default_matching_inputs = []
    if args.includePrivate:
        private_list = DEFAULT_PRIVATE_MATCHING_LIST
        if args.private_matching_json_path:
            with open(args.private_matching_json_path, "r") as f:
                private_list = json.load(f)
        default_matching_inputs.append( MatchingGroupInput(
            category="private",
            keywords=private_list,
            conditions=[
                'info.aws_type=private',
            ]),)
    if args.includePublic:
        public_list = DEFAULT_PUBLIC_MATCHING_LIST
        if args.public_matching_json_path:
            with open(args.public_matching_json_path, "r") as f:
                public_list = json.load(f)
        default_matching_inputs.append(MatchingGroupInput(
            category="public",
            keywords= public_list,
            conditions=[
                'info.aws_type=public',
            ]))
    return default_matching_inputs

if __name__ == "__main__":
    args = argparsers()
    fetcher = ExecutorchBenchmarkFetcher(args.env, args.silent)
    if args.command == "generate_data":
        default_matching_inputs = get_matching_list(args)
        result = fetcher.run(
            args.startTime,
            args.endTime,
            default_matching_inputs,
        )
        fetcher.output_data(args.outputType, args.outputDir)

    elif args.command == "get_matching_list":
        res = fetcher.generate_matching_list(
            args.startTime,
            args.endTime,
            args.filter,
            args.category,
            args.outputType,
            args.outputDir
        )
