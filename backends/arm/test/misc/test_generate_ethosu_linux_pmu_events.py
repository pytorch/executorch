# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from executorch.backends.arm.scripts.generate_ethosu_linux_pmu_events import (
    main,
    parse_events,
)

_HEADER = """
#define PMU_EVENT_MAX_IDX 3
static uint32_t pmu_event_lookup[PMU_EVENT_MAX_IDX] = {
    PMU_EVENT_NO_EVENT,
    PMU_EVENT_NPU_IDLE, /* hardware encoding is 32, configuration index is 1 */
    PMU_EVENT_CYCLE,
};
"""


class TestGenerateEthosULinuxPmuEvents(unittest.TestCase):
    """Validate translation generation and incompatible driver headers."""

    def test_preserves_linux_configuration_order(self) -> None:
        """Read lookup indices independently of hardware encodings."""
        self.assertEqual(parse_events(_HEADER), ["NO_EVENT", "NPU_IDLE", "CYCLE"])

    def test_rejects_incompatible_tables(self) -> None:
        """Fail instead of generating a partial or ambiguous translation."""
        for header in (
            "",
            _HEADER.replace("IDX 3", "IDX 4"),
            _HEADER.replace("PMU_EVENT_CYCLE", "PMU_EVENT_NPU_IDLE"),
            _HEADER.replace("PMU_EVENT_NO_EVENT", "PMU_EVENT_MAC_ACTIVE"),
            _HEADER.replace("PMU_EVENT_CYCLE", "17"),
            _HEADER.replace("PMU_EVENT_CYCLE", "[2] = PMU_EVENT_CYCLE"),
        ):
            with self.subTest(header=header), self.assertRaises(ValueError):
                parse_events(header)

    def test_generates_symbolic_references_for_both_products(self) -> None:
        """Generate compile-checked symbols using the selected driver names."""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "interface.h"
            source.write_text(_HEADER)
            output = Path(directory) / "generated" / "events.h"
            with patch(
                "sys.argv",
                [
                    "generate_ethosu_linux_pmu_events",
                    "--u65-header",
                    str(source),
                    "--u85-header",
                    str(source),
                    "--output",
                    str(output),
                ],
            ):
                main()
            generated = output.read_text()
            self.assertIn("kLinuxU65PmuEvents", generated)
            self.assertIn("kLinuxU85PmuEvents", generated)
            self.assertEqual(generated.count("ETHOSU_PMU_NPU_IDLE,"), 2)
            self.assertLess(
                generated.index("ETHOSU_PMU_NPU_IDLE,"),
                generated.index("ETHOSU_PMU_CYCLE,"),
            )
