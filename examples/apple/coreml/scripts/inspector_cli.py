# Copyright © 2024 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.

import argparse

import sys

from pathlib import Path

from executorch.devtools import Inspector
from executorch.devtools.inspector import compare_results


def get_root_dir_path() -> Path:
    return Path().resolve().parent.parent.parent.parent


sys.path.append(str((get_root_dir_path() / "examples").resolve()))

from inspector_utils import convert_coreml_delegate_time, parse_coreml_delegate_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--etdump_path",
        required=True,
        help="Provide an ETDump file path.",
    )
    parser.add_argument(
        "--etrecord_path",
        required=False,
        help="Provide an optional ETRecord file path.",
    )
    parser.add_argument(
        "--debug_buffer_path",
        required=False,
        help="Provide an optional debug buffer file path.",
    )
    parser.add_argument("--compare_results", action="store_true")

    args = parser.parse_args()

    inspector = Inspector(
        etdump_path=args.etdump_path,
        etrecord=args.etrecord_path,
        debug_buffer_path=args.debug_buffer_path,
        delegate_metadata_parser=parse_coreml_delegate_metadata,
        delegate_time_scale_converter=convert_coreml_delegate_time,
    )
    inspector.print_data_tabular(include_delegate_debug_data=True)
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
