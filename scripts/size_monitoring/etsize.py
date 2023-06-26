import argparse
import getpass
import hashlib
import re
import subprocess
import tempfile

from datetime import datetime

import executorch.scripts.size_monitoring.build_utils as utils

from executorch.scripts.size_monitoring.clang_linker_map_parse import (
    clang_parse_linker_map,
)

from executorch.scripts.size_monitoring.gcc_linker_map_parse import gcc_parse_linker_map
from libfb.py.fburl import FBUrlError, get_fburl, resolve_fburl
from libfb.py.scuba_url import ScubaDrillstate, ScubaURL
from rfe.scubadata.scubadata_py3 import Sample, ScubaData

PORTABLE_OPS_PATH = "executorch/kernels/portable/cpu/"

# Buck arguments used for all platforms.
COMMON_BUCK_ARGS = [
    # Disable program verification, which adds ~30kB to the binary size.
    "-c",
    "executorch.enable_program_verification=false",
    "-c",
    "cxx.linker_map_enabled=true",
]

# Maps platform names to buck commandline args for building the test target.
BUCK_COMMANDS_DICT = {
    "x86": [
        [
            "buck2",
            "build",
            "@arvr/mode/linux/opt-stripped",
            "-c",
            "cxx.extra_cxxflags=-fno-exceptions -fno-rtti",
        ]
        + COMMON_BUCK_ARGS,
    ],
    "aarch64": [
        [
            "buck2",
            "build",
            "@arvr/mode/android/linux/opt-stripped",
            "-c",
            "ndk.custom_libcxx=false",
            "-c",
            "user.extra_cxxflags=-fno-exceptions -fno-rtti",
        ]
        + COMMON_BUCK_ARGS,
    ],
    "xtensa": [
        [
            "buck2",
            "build",
            "@//arvr/projects/jarvis/mode/Harmony_HiFi4_Opus_Tie_5/opt",
        ]
        + COMMON_BUCK_ARGS,
    ],
}

size_test_target = ["xplat/executorch/test:size_test_static[linker-map]", "--out"]
size_test_all_ops_target = [
    "xplat/executorch/test:size_test_all_ops_static[linker-map]",
    "--out",
]


def generate_scuba_fburl(scuba_url):
    scuba_url = str(scuba_url)
    key = hashlib.md5(scuba_url.encode()).hexdigest()[:20]
    try:
        fb_url = "https://fburl.com/{}".format(key)
        resolve_fburl(fb_url)
        return fb_url
    except FBUrlError:
        return get_fburl(scuba_url, string_key=key)


def extract_op_name(file_name, gcc_or_clang):
    if gcc_or_clang == "gcc":
        matches = re.findall(r"\((.*)\)", file_name)
        assert len(matches) != 0, "Failed to find op name in filename"
        if "op_" not in matches[0]:
            return None
        return matches[0].replace(".cpp.o", "")
    else:
        split_str = file_name.split("/")
        if "op_" not in split_str[-1]:
            return None
        matches = re.findall(r"\((.*)\)", split_str[-1])
        if len(matches) == 0:
            op_name = split_str[-1]
        else:
            op_name = matches[0]
        return op_name.replace(".cpp.o", "")


def get_current_time():
    currentDateAndTime = datetime.now()
    return currentDateAndTime.strftime("%m/%d/%Y, %H:%M:%S")


def get_commit_hash() -> str:
    commit_id = subprocess.check_output(["hg", "id"]).decode("utf-8").strip()
    return commit_id


def upload_to_scuba(sizes, platform_name, no_fburl, gcc_or_clang, ops=False):
    scuba_table = "etsize"
    run_id = get_current_time() + " " + get_commit_hash()
    samples = []
    for file_name, section_sizes_dict in sizes.items():
        for section_name, section_size in section_sizes_dict.items():
            sample = Sample()
            sample.setTimeColumnNow()
            sample.addNormalValue("run_id", run_id)
            sample.addNormalValue("unixname", getpass.getuser())
            sample.addNormalValue("platform_name", platform_name)
            sample.addNormalValue("file_name", file_name)
            if PORTABLE_OPS_PATH in file_name:
                op_name = extract_op_name(file_name, gcc_or_clang)
                if op_name:
                    sample.addNormalValue("op_name", op_name)
            sample.addNormalValue("section_name", section_name)
            sample.addIntValue("section_size", section_size)
            sample.addIntValue("ops", ops)
            samples.append(sample)

    scuba_data = ScubaData(scuba_table)
    scuba_data.add_samples(samples)
    if not ops:
        executorch_core_query = (
            ScubaDrillstate()
            .setStartTime("-3 hours")
            .setEndTime("now")
            .setGroupBy(["section_name"])
            .setMetric("sum")
            .addEqConstraint("run_id", run_id)
            .addContainsConstraint(
                "file_name",
                ["executorch", "cxx", "g++", "gcc", "libc", "libgcc", "<internal>"],
            )
            .addContainsConstraint("section_name", ["text", "bss", "data"])
            .addNotContainsConstraint(
                "file_name", ["portable", "all_ops", "demangle", "unwind"]
            )
            .setCols([])
            .setOrderColumn("section_size")
        )
    else:
        portable_kernels_query = (
            ScubaDrillstate()
            .setStartTime("-3 hours")
            .setEndTime("now")
            .setGroupBy(["op_name"])
            .setMetric("sum")
            .addConstraint("op_name", "not_empty_string", None)
            .addEqConstraint("run_id", run_id)
            .addContainsConstraint("section_name", ["text", "bss", "data"])
            .setCols([])
            .setCompareTime(None)
            .setOrderColumn("section_size")
        )
    if not no_fburl:
        if not ops:
            core_scuba_url = generate_scuba_fburl(
                ScubaURL(scuba_table, executorch_core_query, view="Pie")
            )
            print(f"Link to ExecuTorch core library size chart => {core_scuba_url}")
        else:
            portable_scuba_url = generate_scuba_fburl(
                ScubaURL(scuba_table, portable_kernels_query, view="bar_chart_client")
            )
            print(f"Link to portable kernels size chart {portable_scuba_url}")


def build_and_generate_map_file(platform_name, ops=False):
    buck_cmd = BUCK_COMMANDS_DICT[platform_name]
    if not ops:
        buck_cmd = buck_cmd[0] + size_test_target
    else:
        buck_cmd = buck_cmd[0] + size_test_all_ops_target
    map_file_path = tempfile.mktemp()
    with utils.change_directory(utils.get_hg_root()):
        utils.execute_cmd(buck_cmd + [map_file_path])
    return map_file_path


def parse_and_log_to_scuba(map_file_path, ops=False):
    # Figure out whether the linker map file that was passed in was generated by a
    # gcc or clang based compiler.
    with open(map_file_path) as f:
        if f.readline().strip().split()[0] == "VMA":
            gcc_or_clang = "clang"
        else:
            gcc_or_clang = "gcc"
    if gcc_or_clang == "gcc":
        sizes = gcc_parse_linker_map(map_file_path)
    else:
        sizes = clang_parse_linker_map(map_file_path)
    if not args.no_scuba_logging:
        upload_to_scuba(sizes, args.platform_name, args.no_fburl, gcc_or_clang, ops)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_file", help="map.txt generated by `-Wl,-Map=map.txt`")
    parser.add_argument(
        "--platform_name",
        help="Target platform. Valid values are x86, aarch64 (ARM 64-bit) and xtensa",
        choices=["x86", "aarch64", "xtensa"],
    )
    parser.add_argument(
        "--no_scuba_logging", help="Disables logging data to scuba", action="store_true"
    )
    parser.add_argument(
        "--no_fburl",
        help="Log to scuba but don't generate fburl's for dashboard view",
        action="store_true",
    )
    args = parser.parse_args()

    if args.map_file is None:
        map_file_path = build_and_generate_map_file(args.platform_name)
        parse_and_log_to_scuba(map_file_path)
        map_file_path_ops = build_and_generate_map_file(args.platform_name, ops=True)
        parse_and_log_to_scuba(map_file_path_ops, ops=True)
    else:
        map_file_path = args.map_file
        parse_and_log_to_scuba(map_file_path)
