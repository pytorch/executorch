# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys

from executorch.profiler.parse_profiler_results import (
    deserialize_profile_results,
    mem_profile_table,
    profile_aggregate_framework_tax,
    profile_framework_tax_table,
    profile_table,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prof_results_bin", help="profiling results binary file")
    args = parser.parse_args()

    with open(args.prof_results_bin, "rb") as prof_results_file:
        out_bytes = prof_results_file.read()

    prof_data, mem_allocations = deserialize_profile_results(out_bytes)
    framework_tax_data = profile_aggregate_framework_tax(prof_data)

    prof_tables = profile_table(prof_data)
    for table in prof_tables:
        print(table)

    prof_tables_agg = profile_framework_tax_table(framework_tax_data)
    for table in prof_tables_agg:
        print(table)

    mem_prof_tables = mem_profile_table(mem_allocations)
    for table in mem_prof_tables:
        print(table)
    return 0


def invoke_main() -> None:
    sys.exit(main())


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
