import argparse
import json
import sys

from executorch.exir.serialize import deserialize_from_flatbuffer

from executorch.profiler.parse_profiler_results import (
    deserialize_profile_results,
    fetch_frame_list_short_repr,
    mem_profile_table,
    profile_aggregate_framework_tax,
    profile_framework_tax_table,
    profile_table,
    ProfileData,
)

from executorch.profiler.profiler_results_scuba import upload_to_scuba


def gen_chrome_traceevents_json(
    profile_results_path: str, model_ff_path: str, json_out_path: str
):
    with open(profile_results_path, "rb") as prof_res_file:
        prof_res_buf = prof_res_file.read()
        model_ff_buf = None
        if model_ff_path is not None:
            with open(model_ff_path, "rb") as model_ff_file:
                model_ff_buf = model_ff_file.read()

    prof_data, _ = deserialize_profile_results(prof_res_buf)
    prof_table = profile_table(prof_data, model_ff_buf)

    print(prof_table)

    root = {}
    trace_events = []
    root["traceEvents"] = trace_events

    for block_name, prof_data_list in prof_data.items():
        for d in prof_data_list:
            stacktrace_short = None
            if model_ff_buf is not None:
                stacktrace_short = fetch_frame_list_short_repr(
                    deserialize_from_flatbuffer(model_ff_buf),
                    0,
                    d.chain_idx,
                    d.instruction_idx,
                )
            name = d.name
            if stacktrace_short is not None:
                name += " [" + stacktrace_short + "]"
            for ts_ms, duration_ms in zip(d.ts, d.duration):
                e = {}

                MS_TO_US_SCALE = 1000
                ts_us = ts_ms * MS_TO_US_SCALE
                duration_us = duration_ms * MS_TO_US_SCALE
                e["name"] = block_name + ":" + name
                e["cat"] = "cpu_op"
                e["ph"] = "X"
                e["ts"] = int(ts_us)
                e["dur"] = int(duration_us)
                e["pid"] = 0
                e["tid"] = 0
                trace_events.append(e)

    json_content: str = json.dumps(root, indent=2)

    with open(json_out_path, "wb") as json_file:
        json_file.write(json_content.encode("ascii"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prof_results_bin", help="profiling results binary file")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="path to executorch flatbuffer model",
    )
    parser.add_argument("--chrome_json_path", type=str, default=None)
    args = parser.parse_args()

    if args.chrome_json_path is not None:
        gen_chrome_traceevents_json(
            args.prof_results_bin, args.model_path, args.chrome_json_path
        )
        return 0

    with open(args.prof_results_bin, "rb") as prof_results_file:
        out_bytes = prof_results_file.read()

    model_bytes = None
    if args.model_path is not None:
        with open(args.model_path, "rb") as model_file:
            model_bytes = model_file.read()

    prof_data, mem_allocations = deserialize_profile_results(out_bytes)
    framework_tax_data = profile_aggregate_framework_tax(prof_data)

    prof_tables = profile_table(prof_data, model_bytes)
    for table in prof_tables:
        print(table)

    prof_tables_agg = profile_framework_tax_table(framework_tax_data)
    for table in prof_tables_agg:
        print(table)

    mem_prof_tables = mem_profile_table(mem_allocations)
    for table in mem_prof_tables:
        print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
