# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

print("Parsing etdump with PMU counter metadata.")

import logging
from pprint import pprint

# Avoid printing some warnings.
logging.disable(logging.WARNING)
logging.captureWarnings(True)

from executorch.backends.arm.ethosu.parse_delegate_metadata import (  # noqa: E402
    parse_ethosu_delegate_pmu_metadata,
    parse_pmu_profile_event,
)
from executorch.devtools.inspector import Inspector, TimeScale  # noqa: E402

inspector = Inspector(
    etdump_path="etdump.bin",
    source_time_scale=TimeScale.CYCLES,
    target_time_scale=TimeScale.CYCLES,
    delegate_metadata_parser=parse_ethosu_delegate_pmu_metadata,
)
for block in inspector.event_blocks:
    for event in block.events:
        if event.name == "ethosu.pmu.counters":
            # There are two ways to decode the same PMU measurements.
            # Inspector's registered parser returns dictionaries of samples.
            print("\nOption 1 for parsing metadata:")
            pprint(event.delegate_debug_metadatas)

            # Parse each raw sample directly into an EthosUPmuProfileEvent
            # dataclass instead. This also works without a registered parser.
            print("\nOption 2 for parsing metadata:")
            for metadata in event.raw_delegate_debug_metadatas:
                if isinstance(metadata, (bytes, bytearray)):
                    record = parse_pmu_profile_event(bytes(metadata))
                    pprint(record)
