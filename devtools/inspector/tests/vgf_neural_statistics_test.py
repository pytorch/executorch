# Copyright 2026 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import base64
import json
import unittest
from typing import Any, Dict

from executorch.devtools.inspector import Event, EventBlock, Inspector
from executorch.devtools.inspector.vgf_neural_statistics import (
    parse_vgf_neural_statistics_delegate_metadata,
    parse_vgf_neural_statistics_metadata,
    SCHEMA,
    SCHEMA_VERSION,
)


def _metadata_payload(data_available: bool = True) -> Dict[str, Any]:
    """Build a fake VGF neural statistics metadata payload."""
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "backend": "VgfBackend",
        "api": "VK_ARM_data_graph",
        "event_name": "VGF_NEURAL_STATISTICS",
        "api_available": True,
        "data_available": data_available,
        "available": data_available,
        "reason": "" if data_available else "Statistics data unavailable",
        "segments": [
            {
                "segment_id": 0,
                "is_data_graph_pipeline": True,
                "statistics_bind_point_available": data_available,
                "statistics_memory_host_visible": data_available,
                "statistics_memory_host_coherent": data_available,
                "statistics_bind_point_reason": "",
                "debug_database": {
                    "available": data_available,
                    "is_text": False,
                    "vulkan_result": 0,
                    "size": 3 if data_available else 0,
                    "encoding": "base64",
                    "reason": "",
                    "data": (
                        base64.b64encode(b"\x01\x02\x03").decode("ascii")
                        if data_available
                        else ""
                    ),
                },
                "statistics_info": {
                    "available": data_available,
                    "is_text": True,
                    "vulkan_result": 0,
                    "size": 4 if data_available else 0,
                    "encoding": "base64",
                    "reason": "",
                    "data": (
                        base64.b64encode(b"info").decode("ascii")
                        if data_available
                        else ""
                    ),
                },
                "statistics_memory": {
                    "available": data_available,
                    "is_text": False,
                    "vulkan_result": 0,
                    "size": 2 if data_available else 0,
                    "encoding": "base64",
                    "reason": "",
                    "data": (
                        base64.b64encode(b"\xde\xad").decode("ascii")
                        if data_available
                        else ""
                    ),
                },
            }
        ],
    }


class TestVgfNeuralStatisticsInspector(unittest.TestCase):
    def test_parse_vgf_neural_statistics_metadata(self) -> None:
        metadata = json.dumps(_metadata_payload()).encode("utf-8")

        parsed = parse_vgf_neural_statistics_metadata(metadata)

        self.assertEqual(parsed["schema"], SCHEMA)
        self.assertEqual(parsed["schema_version"], SCHEMA_VERSION)
        self.assertTrue(parsed["api_available"])
        self.assertTrue(parsed["data_available"])

        segment = parsed["segments"][0]
        self.assertEqual(segment["debug_database"]["raw_data"], b"\x01\x02\x03")
        self.assertEqual(segment["statistics_info"]["raw_data"], b"info")
        self.assertEqual(segment["statistics_memory"]["raw_data"], b"\xde\xad")

    def test_parse_vgf_neural_statistics_unavailable_metadata(self) -> None:
        metadata = json.dumps(_metadata_payload(data_available=False)).encode("utf-8")

        parsed = parse_vgf_neural_statistics_metadata(metadata)

        self.assertEqual(parsed["schema_version"], SCHEMA_VERSION)
        self.assertTrue(parsed["api_available"])
        self.assertFalse(parsed["data_available"])
        self.assertFalse(parsed["available"])
        self.assertEqual(parsed["reason"], "Statistics data unavailable")

        segment = parsed["segments"][0]
        self.assertEqual(segment["debug_database"]["raw_data"], b"")
        self.assertEqual(segment["statistics_info"]["raw_data"], b"")
        self.assertEqual(segment["statistics_memory"]["raw_data"], b"")

    def test_parse_rejects_unsupported_schema_version(self) -> None:
        payload = _metadata_payload()
        payload["schema_version"] = SCHEMA_VERSION + 1
        metadata = json.dumps(payload).encode("utf-8")

        with self.assertRaisesRegex(
            ValueError,
            "Unsupported VGF neural statistics metadata schema version",
        ):
            parse_vgf_neural_statistics_metadata(metadata)

    def test_parse_rejects_malformed_base64(self) -> None:
        payload = _metadata_payload()
        payload["segments"][0]["debug_database"]["data"] = "not-valid-base64!"
        metadata = json.dumps(payload).encode("utf-8")

        with self.assertRaisesRegex(
            ValueError,
            "Malformed base64 data in VGF neural statistics blob",
        ):
            parse_vgf_neural_statistics_metadata(metadata)

    def test_delegate_metadata_parser_ignores_other_metadata(self) -> None:
        metadata = json.dumps(_metadata_payload()).encode("utf-8")

        parsed = parse_vgf_neural_statistics_delegate_metadata(
            [
                b"",
                b"not json",
                b'{"schema":"some.other.backend","schema_version":1}',
                metadata,
            ]
        )

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["schema"], SCHEMA)

    def test_delegate_parser_surfaces_malformed_vgf_metadata(self) -> None:
        malformed_metadata = (
            b'{"schema":"executorch.vgf.neural_statistics",' b'"schema_version":1'
        )

        with self.assertRaisesRegex(
            ValueError,
            "Malformed VGF neural statistics delegate metadata",
        ):
            parse_vgf_neural_statistics_delegate_metadata([malformed_metadata])

    def test_inspector_api_extracts_vgf_neural_statistics(self) -> None:
        metadata = json.dumps(_metadata_payload()).encode("utf-8")

        inspector = Inspector.__new__(Inspector)
        inspector.event_blocks = [
            EventBlock(
                name="forward",
                events=[
                    Event(
                        name="delegate_execute",
                        _delegate_debug_metadatas=[metadata],
                        is_delegated_op=True,
                    )
                ],
            )
        ]

        records = inspector.get_vgf_neural_statistics()

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["schema_version"], SCHEMA_VERSION)
        self.assertTrue(records[0]["data_available"])
        self.assertEqual(
            records[0]["segments"][0]["statistics_memory"]["raw_data"],
            b"\xde\xad",
        )


if __name__ == "__main__":
    unittest.main()
