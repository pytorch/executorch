# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Print profiling table for the NXP Neutron NPU model

from typing import Any, Union

from executorch.backends.nxp.tests.profiling_utils import (
    get_neutron_compiler_version,
    get_neutron_driver_version,
    get_neutron_kernel_kinds,
)

from executorch.devtools import Inspector

# Global mapping of Neutron kernel IDs to names used by the delegate metadata parser.
kernel_kinds = {}


def parse_delegate_metadata(
    delegate_metadatas: list[bytes],
) -> Union[list[str], dict[str, Any]]:
    """Metadata parser for Neutron Backend metadata.

    The parser is a callable that deserializes the data and returns neutron kernel number.
    The deserialized data is then added back to the corresponding event in the event block for user consumption.
    """

    metadata_list = []
    for metadata_bytes in delegate_metadatas:
        if len(metadata_bytes) == 1:
            function_code = metadata_bytes[0]
            if function_code == 0:
                metadata_list.append("Profiling dump")
            else:
                metadata_list.append(
                    kernel_kinds.get(
                        function_code, "Neutron kernel " + str(function_code)
                    )
                )
        elif len(metadata_bytes) == 2:
            metadata_list.append("Profiling dump")
        else:
            metadata_list.append("Invalid metadata size")
    return metadata_list


if __name__ == "__main__":

    try:
        etrecord_path = "etrecord/etrecord.bin"
        etdump_path = "etdump/trace.etdump"

        driver_version = get_neutron_driver_version(etdump_path)
        compiler_version = get_neutron_compiler_version()
        if driver_version and driver_version == compiler_version:
            kernel_kinds = get_neutron_kernel_kinds()

        inspector = Inspector(
            etdump_path=etdump_path,
            etrecord=etrecord_path,
            delegate_metadata_parser=parse_delegate_metadata,
        )

        # Access raw event data and filter quantized_decomposed nodes
        for event_block in inspector.event_blocks:
            for event in event_block.events:
                if hasattr(event, "op_types") and isinstance(event.op_types, list):
                    # Filter out quantized_decomposed ops from the actual list
                    filtered = [
                        op for op in event.op_types if "quantized_decomposed" not in op
                    ]
                    event.op_types = filtered if filtered else event.op_types

        inspector.print_data_tabular(include_delegate_debug_data=True)
    except Exception as e:
        print(f"Error during inspection: {type(e).__name__}: {e}")
