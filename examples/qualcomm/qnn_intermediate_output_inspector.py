# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from executorch.devtools import Inspector


def main(args):
    # Create an Inspector instance with etdump and the debug buffer.
    inspector = Inspector(
        etdump_path=args.etdump_path,
        etrecord=args.etrecord_path,
        debug_buffer_path=args.debug_buffer_path,
    )

    # Accessing intermediate outputs from each event (an event here is essentially an instruction that executed in the runtime).
    for event_block in inspector.event_blocks:
        if event_block.name == "Execute":
            for event in event_block.events:
                # If user enables profiling and dump intermediate outputs the same time, we need to skip the profiling event
                if event.perf_data is not None and event.is_delegated_op:
                    continue
                print("Event Name: ", event.name)
                print(event.debug_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--etdump_path",
        required=True,
        help="Provide an ETDump file path. File extension should be .etdp",
    )
    parser.add_argument(
        "--etrecord_path",
        required=False,
        default=None,
        help="Provide an optional ETRecord file path. File extension should be .bin",
    )
    parser.add_argument(
        "--debug_buffer_path",
        required=False,
        default=None,
        help="Provide an optional debug buffer file path. File extension should be .bin",
    )
    args = parser.parse_args()

    main(args)
