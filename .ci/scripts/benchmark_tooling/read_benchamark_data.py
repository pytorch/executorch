import argparse
import json
from typing import Tuple, List, Any
import requests
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import logging
import os

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
    Fetch benchmark data from HUD
    Usage:
        fetcher = ExecutorchBenchmarkFetcher()
        fetcher.get_data(start_time, end_time)

        fetcher.toDataFrame() -> return a list of dataframes, one for private devices, one for public devices
        fetcher.toExcelSheet(output_dir=".") -> write to excel files, one for private devices, one for public devices
    """

    def __init__(self, env="prod", disable_logging=False):
        self.env = env
        self.base_url = self._get_base_url()
        self.query_group_table_by_fields = ["device", "backend", "arch", "model"]
        self.query_group_row_by_fields = ["workflow_id", "job_id", "granularity_bucket"]
        self.data = None
        self.disable_logging = disable_logging
        self.results_private = []
        self.results_public = []

    def run(
        self, start_time, end_time, privateDeviceMatchings, publicDeviceMatchings
    ) -> Tuple[List[Any], List[Any]]:

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

    def toDataFrame(self):
        private_dfs = [
            {"groupInfo": item["groupInfo"], "df": pd.DataFrame(item["rows"])}
            for item in self.results_private
        ]
        public_dfs = [
            {"groupInfo": item["groupInfo"], "df": pd.DataFrame(item["rows"])}
            for item in self.results_public
        ]
        return (private_dfs, public_dfs)

    def toExcelSheet(self, output_dir="."):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
        else:
            logging.info(f"Using existing output directory: {output_dir}")
        private_path = os.path.join(output_dir, "res_private.xlsx")
        public_path = os.path.join(output_dir, "res_public.xlsx")
        self._write_multi_sheet_excel(self.results_private, private_path)
        self._write_multi_sheet_excel(self.results_public, public_path)

    def _write_multi_sheet_excel(self, data_list, output_path):
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

    def _fetch_data(self, start_time, end_time):
        data = self._fetch_execu_torch_data(start_time, end_time)
        if data is None:
            return None
        self.data = self._process(data)
        return self.data

    def _get_base_url(self):
        base_urls = {
            "local": "http://localhost:3000",
            "prod": "https://hud.pytorch.org",
        }
        return base_urls[self.env]

    def print_all_names(self):
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

    def _process(self, data):
        for item in data:
            group = item.get("groupInfo", {})
            item["groupInfo"] = {
                k: self.normalize_string(v)
                for k, v in group.items()
                if v is not None and isinstance(v, str)
            }
            name = (
                f"{group['model']}|{group['backend']}|{group['device']}|{group['arch']}"
            )
            name = self.normalize_string(name)
            item["table_name"] = name
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
                        continue
                    item["short_name"] = self.generate_short_name(
                        norm_keywords, len(match)
                    )
                    match.append(key)
                    # avoid duplicates
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
        private, public = fetcher.toDataFrame()
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
        fetcher.toExcelSheet(args.outputDir)
    else:
        logging.info(
            f"======================Printing private device benchmark results in json format======================"
        )
        print(json.dumps(private, indent=2))
        logging.info(
            f"======================Printing public device benchmark results in json format======================"
        )
        print(json.dumps(public, indent=2))
