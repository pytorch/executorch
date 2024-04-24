# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json

from typing import Any, Dict, Final, List, Tuple, Union

from executorch.sdk import Inspector
from executorch.sdk.inspector._inspector_utils import compare_results

COREML_METADATA_KEYS: Final[List[Tuple[str, str]]] = [
    ("operatorName", "coreml_operator"),
    ("estimatedCost", "coreml_estimated_cost"),
    ("preferredComputeUnit", "coreml_preferred_device"),
    ("supportedComputeUnits", "coreml_supported_devices"),
]


def parse_coreml_delegate_metadata(delegate_metadatas: List[str]) -> Dict[str, Any]:
    try:
        coreml_metadata: Dict[str, Any] = json.loads(delegate_metadatas[0])
        result: Dict[str, str] = {}
        for col_key, col_name in COREML_METADATA_KEYS:
            value = coreml_metadata.get(col_key, None)
            if value is not None:
                result[col_name] = value
        return result

    except ValueError:
        return {}


def convert_coreml_delegate_time(
    event_name: Union[str, int], input_time: Union[int, float]
) -> Union[int, float]:
    return input_time / (1000 * 1000)


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
        help="Provide an optional buffer file path.",
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
