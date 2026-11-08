# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import struct
import unittest

from executorch.backends.arm.ethosu.parse_delegate_metadata import (
    _COUNTER,
    _HEADER,
    _MAGIC_PREFIX,
    EthosUPlatform,
    EthosUPmuEvent,
    EthosUProduct,
    parse_ethosu_delegate_pmu_metadata,
    parse_pmu_profile_event,
)


def _metadata(
    *counters: tuple[int, int, int],
    flags: int = 2,
    product: int = 2,
    platform: int = 0,
    macs_per_cycle: int = 256,
) -> bytes:
    header = _HEADER.pack(
        _MAGIC_PREFIX + b"1",
        product,
        macs_per_cycle,
        platform,
        flags,
        0x50000000,
        123456,
        len(counters),
    )
    return header + b"".join(_COUNTER.pack(*counter) for counter in counters)


class TestEthosUDelegateMetadata(unittest.TestCase):
    """Check Ethos-U PMU metadata decoding and Inspector adaptation."""

    def test_decodes_product(self) -> None:
        """Map each core-driver product ID to its enum value."""
        for product in EthosUProduct:
            with self.subTest(product=product):
                self.assertIs(
                    parse_pmu_profile_event(_metadata(product=product.value)).product,
                    product,
                )

    def test_decodes_profile_event(self) -> None:
        """Decode counters into their symbolic event types."""
        metadata = _metadata((97, 67, 1), (178, 11, 0))

        record = parse_pmu_profile_event(metadata)

        self.assertIs(record.product, EthosUProduct.U85)
        self.assertIs(record.platform, EthosUPlatform.CoreDriver)
        self.assertEqual(record.macs_per_cycle, 256)
        self.assertTrue(record.success)
        self.assertEqual(record.register_base, 0x50000000)
        self.assertEqual(record.cycle_count, 123456)
        self.assertIsInstance(record.cycle_count, float)
        self.assertFalse(record.cycle_overflow)
        self.assertEqual(len(record.counters), 2)
        self.assertEqual(
            record.counters[0].event_type, EthosUPmuEvent.ETHOSU_PMU_MAC_ACTIVE
        )
        self.assertTrue(math.isnan(record.counters[0].count))
        self.assertTrue(record.counters[0].overflow)
        self.assertEqual(
            record.counters[1].event_type, EthosUPmuEvent.ETHOSU_PMU_WD_ACTIVE
        )
        self.assertFalse(record.counters[1].overflow)
        self.assertEqual(record.counters[1].count, 11)
        self.assertIsInstance(record.counters[1].count, float)

    def test_cycle_overflow_returns_nan(self) -> None:
        """Use NaN when the cycle counter overflows."""
        record = parse_pmu_profile_event(_metadata(flags=3))

        self.assertTrue(record.cycle_overflow)
        self.assertTrue(math.isnan(record.cycle_count))

    def test_inspector_adapter_preserves_sample_positions(self) -> None:
        """Keep sample indices when unrelated or empty entries occur."""
        metadata = _metadata((112, 45, 0))

        parsed = parse_ethosu_delegate_pmu_metadata([b"", b"other", metadata])

        self.assertEqual(parsed["ethosu_pmu"][:2], [None, None])
        sample = parsed["ethosu_pmu"][2]
        self.assertIs(sample["product"], EthosUProduct.U85)
        self.assertIs(sample["platform"], EthosUPlatform.CoreDriver)
        self.assertEqual(sample["macs_per_cycle"], 256)
        self.assertEqual(sample["cycle_count"], 123456)
        self.assertEqual(len(sample["counters"]), 1)
        counter = sample["counters"][0]
        self.assertIs(counter["event_type"], EthosUPmuEvent.ETHOSU_PMU_NPU_IDLE)
        self.assertEqual(counter["event_id"], 112)
        self.assertEqual(counter["count"], 45)
        self.assertFalse(counter["overflow"])
        self.assertEqual(parse_ethosu_delegate_pmu_metadata([b"other", ""]), {})

    def test_rejects_invalid_metadata(self) -> None:
        """Reject malformed Ethos-U records and unsupported event IDs."""
        valid = _metadata((97, 67, 1))
        invalid = {
            "short header": (valid[:39], "shorter than its header"),
            "wrong magic": (b"XYZ1" + valid[4:], "magic"),
            "unsupported version": (
                b"EUC2" + valid[4:],
                r"supported versions: \{1\}",
            ),
            "invalid version": (b"EUCx" + valid[4:], "version"),
            "extra byte": (valid + b"\x00", "expected"),
            "too many counters": (
                valid[:36] + struct.pack("<I", 9) + valid[40:],
                "Too many",
            ),
            "unknown event": (
                valid[:40] + _COUNTER.pack(999, 67, 1),
                "Unknown",
            ),
            "unknown product": (
                valid[:4] + struct.pack("<I", 3) + valid[8:],
                "Unknown Ethos-U product ID",
            ),
            "invalid overflow": (
                valid[:40] + _COUNTER.pack(97, 67, 2),
                "overflow",
            ),
            "unknown platform": (
                valid[:12] + struct.pack("<I", 1) + valid[16:],
                "Unknown Ethos-U platform ID",
            ),
            "unknown flags": (
                valid[:16] + struct.pack("<I", 4) + valid[20:],
                "flags",
            ),
        }
        for name, (metadata, message) in invalid.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                parse_pmu_profile_event(metadata)

        with self.assertRaisesRegex(ValueError, "version"):
            parse_ethosu_delegate_pmu_metadata([b"EUC2" + valid[4:]])
