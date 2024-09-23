# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import argparse

from executorch.devtools import Inspector
from executorch.devtools.inspector import compare_results, TimeScale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--etdump_path",
        required=True,
        help="Provide an ETDump file path.",
    )
    parser.add_argument(
        "--source_time_scale",
        type=str,
        choices=[ts.value for ts in TimeScale],
        help="Enter the source time scale (ns, us, ms, s, cycles)",
        default=TimeScale.NS.value,
    )
    parser.add_argument(
        "--target_time_scale",
        type=str,
        choices=[ts.value for ts in TimeScale],
        help="Enter the target time scale (ns, us, ms, s, cycles)",
        default=TimeScale.MS.value,
    )
    parser.add_argument(
        "--etrecord_path",
        required=False,
        help="Provide an optional ETRecord file path.",
    )
    parser.add_argument(
        "--debug_buffer_path",
        required=False,
        help="Provide an optional buffer file path.",
    )
    parser.add_argument("--compare_results", action="store_true")

    args = parser.parse_args()

    inspector = Inspector(
        etdump_path=args.etdump_path,
        etrecord=args.etrecord_path,
        debug_buffer_path=args.debug_buffer_path,
        source_time_scale=TimeScale(args.source_time_scale),
        target_time_scale=TimeScale(args.target_time_scale),
    )
    inspector.print_data_tabular()
    if args.compare_results:
        for event_block in inspector.event_blocks:
            if event_block.name == "Execute":
                compare_results(
                    reference_output=event_block.reference_output,
                    run_output=event_block.run_output,
                    plot=True,
                )


if __name__ == "__main__":
    main()  # pragma: no cover
