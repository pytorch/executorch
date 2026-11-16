# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Decode experimental Ethos-U PMU metadata from ETDump profile events.

The wire format and public Python types may change without compatibility
support.

"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from struct import Struct
from typing import Any, Sequence

_HEADER = Struct("<4sIIIIQQI")
_COUNTER = Struct("<IQI")
_MAGIC_PREFIX = b"EUC"
_MAX_COUNTERS = 8


class EthosUProduct(IntEnum):
    """Product IDs from core_driver/include/ethosu_types.h."""

    U55 = 0
    U65 = 1
    U85 = 2


class EthosUPlatform(IntEnum):
    """Platform IDs from EthosUExecutionStats.h."""

    CoreDriver = 0


class EthosUPmuEvent(IntEnum):
    """Symbolic IDs from core_driver/include/ethosu_pmu_types.h.

    Values mirror the order of the core driver's ethosu_pmu_event_type enum.

    """

    ETHOSU_PMU_AO_ACTIVE = 0
    ETHOSU_PMU_AO_ACTIVE_16BIT = 1
    ETHOSU_PMU_AO_ACTIVE_8BIT = 2
    ETHOSU_PMU_AO_STALLED_BY_AB = 3
    ETHOSU_PMU_AO_STALLED_BY_AB_OR_CB = 4
    ETHOSU_PMU_AO_STALLED_BY_ACC = 5
    ETHOSU_PMU_AO_STALLED_BY_ACC_IB = 6
    ETHOSU_PMU_AO_STALLED_BY_BS = 7
    ETHOSU_PMU_AO_STALLED_BY_BS_OR_OB = 8
    ETHOSU_PMU_AO_STALLED_BY_CB = 9
    ETHOSU_PMU_AO_STALLED_BY_IB = 10
    ETHOSU_PMU_AO_STALLED_BY_OB = 11
    ETHOSU_PMU_AO_STALLED_BY_OFMP = 12
    ETHOSU_PMU_AO_STALLED_BY_OFMP_OB = 13
    ETHOSU_PMU_AXI0_ENABLED_CYCLES = 14
    ETHOSU_PMU_AXI0_RD_DATA_BEAT_RECEIVED = 15
    ETHOSU_PMU_AXI0_RD_STALL_LIMIT = 16
    ETHOSU_PMU_AXI0_RD_TRANS_ACCEPTED = 17
    ETHOSU_PMU_AXI0_RD_TRANS_COMPLETED = 18
    ETHOSU_PMU_AXI0_RD_TRAN_REQ_STALLED = 19
    ETHOSU_PMU_AXI0_WR_DATA_BEAT_STALLED = 20
    ETHOSU_PMU_AXI0_WR_DATA_BEAT_WRITTEN = 21
    ETHOSU_PMU_AXI0_WR_STALL_LIMIT = 22
    ETHOSU_PMU_AXI0_WR_TRANS_ACCEPTED = 23
    ETHOSU_PMU_AXI0_WR_TRANS_COMPLETED_M = 24
    ETHOSU_PMU_AXI0_WR_TRANS_COMPLETED_S = 25
    ETHOSU_PMU_AXI0_WR_TRAN_REQ_STALLED = 26
    ETHOSU_PMU_AXI1_ENABLED_CYCLES = 27
    ETHOSU_PMU_AXI1_RD_DATA_BEAT_RECEIVED = 28
    ETHOSU_PMU_AXI1_RD_STALL_LIMIT = 29
    ETHOSU_PMU_AXI1_RD_TRANS_ACCEPTED = 30
    ETHOSU_PMU_AXI1_RD_TRANS_COMPLETED = 31
    ETHOSU_PMU_AXI1_RD_TRAN_REQ_STALLED = 32
    ETHOSU_PMU_AXI1_WR_DATA_BEAT_STALLED = 33
    ETHOSU_PMU_AXI1_WR_DATA_BEAT_WRITTEN = 34
    ETHOSU_PMU_AXI1_WR_STALL_LIMIT = 35
    ETHOSU_PMU_AXI1_WR_TRANS_ACCEPTED = 36
    ETHOSU_PMU_AXI1_WR_TRANS_COMPLETED_M = 37
    ETHOSU_PMU_AXI1_WR_TRANS_COMPLETED_S = 38
    ETHOSU_PMU_AXI1_WR_TRAN_REQ_STALLED = 39
    ETHOSU_PMU_AXI_LATENCY_1024 = 40
    ETHOSU_PMU_AXI_LATENCY_128 = 41
    ETHOSU_PMU_AXI_LATENCY_256 = 42
    ETHOSU_PMU_AXI_LATENCY_32 = 43
    ETHOSU_PMU_AXI_LATENCY_512 = 44
    ETHOSU_PMU_AXI_LATENCY_64 = 45
    ETHOSU_PMU_AXI_LATENCY_ANY = 46
    ETHOSU_PMU_CC_STALLED_ON_BLOCKDEP = 47
    ETHOSU_PMU_CC_STALLED_ON_SHRAM_RECONFIG = 48
    ETHOSU_PMU_CYCLE = 49
    ETHOSU_PMU_ECC_AO_CB = 50
    ETHOSU_PMU_ECC_AO_LUT = 51
    ETHOSU_PMU_ECC_AO_OB = 52
    ETHOSU_PMU_ECC_DMA = 53
    ETHOSU_PMU_ECC_MAC_AB = 54
    ETHOSU_PMU_ECC_MAC_IB = 55
    ETHOSU_PMU_ECC_SB0 = 56
    ETHOSU_PMU_ECC_SB1 = 57
    ETHOSU_PMU_EXT0_ENABLED_CYCLES = 58
    ETHOSU_PMU_EXT0_RD_DATA_BEAT_RECEIVED = 59
    ETHOSU_PMU_EXT0_RD_STALL_LIMIT = 60
    ETHOSU_PMU_EXT0_RD_TRANS_ACCEPTED = 61
    ETHOSU_PMU_EXT0_RD_TRANS_COMPLETED = 62
    ETHOSU_PMU_EXT0_RD_TRAN_REQ_STALLED = 63
    ETHOSU_PMU_EXT0_WR_DATA_BEAT_STALLED = 64
    ETHOSU_PMU_EXT0_WR_DATA_BEAT_WRITTEN = 65
    ETHOSU_PMU_EXT0_WR_STALL_LIMIT = 66
    ETHOSU_PMU_EXT0_WR_TRANS_ACCEPTED = 67
    ETHOSU_PMU_EXT0_WR_TRANS_COMPLETED_M = 68
    ETHOSU_PMU_EXT0_WR_TRANS_COMPLETED_S = 69
    ETHOSU_PMU_EXT0_WR_TRAN_REQ_STALLED = 70
    ETHOSU_PMU_EXT1_ENABLED_CYCLES = 71
    ETHOSU_PMU_EXT1_RD_DATA_BEAT_RECEIVED = 72
    ETHOSU_PMU_EXT1_RD_STALL_LIMIT = 73
    ETHOSU_PMU_EXT1_RD_TRANS_ACCEPTED = 74
    ETHOSU_PMU_EXT1_RD_TRANS_COMPLETED = 75
    ETHOSU_PMU_EXT1_RD_TRAN_REQ_STALLED = 76
    ETHOSU_PMU_EXT1_WR_DATA_BEAT_STALLED = 77
    ETHOSU_PMU_EXT1_WR_DATA_BEAT_WRITTEN = 78
    ETHOSU_PMU_EXT1_WR_STALL_LIMIT = 79
    ETHOSU_PMU_EXT1_WR_TRANS_ACCEPTED = 80
    ETHOSU_PMU_EXT1_WR_TRANS_COMPLETED_M = 81
    ETHOSU_PMU_EXT1_WR_TRANS_COMPLETED_S = 82
    ETHOSU_PMU_EXT1_WR_TRAN_REQ_STALLED = 83
    ETHOSU_PMU_EXT_ENABLED_CYCLES = 84
    ETHOSU_PMU_EXT_RD_DATA_BEAT_RECEIVED = 85
    ETHOSU_PMU_EXT_RD_STALL_LIMIT = 86
    ETHOSU_PMU_EXT_RD_TRANS_ACCEPTED = 87
    ETHOSU_PMU_EXT_RD_TRANS_COMPLETED = 88
    ETHOSU_PMU_EXT_RD_TRAN_REQ_STALLED = 89
    ETHOSU_PMU_EXT_WR_DATA_BEAT_STALLED = 90
    ETHOSU_PMU_EXT_WR_DATA_BEAT_WRITTEN = 91
    ETHOSU_PMU_EXT_WR_STALL_LIMIT = 92
    ETHOSU_PMU_EXT_WR_TRANS_ACCEPTED = 93
    ETHOSU_PMU_EXT_WR_TRANS_COMPLETED_M = 94
    ETHOSU_PMU_EXT_WR_TRANS_COMPLETED_S = 95
    ETHOSU_PMU_EXT_WR_TRAN_REQ_STALLED = 96
    ETHOSU_PMU_MAC_ACTIVE = 97
    ETHOSU_PMU_MAC_ACTIVE_16BIT = 98
    ETHOSU_PMU_MAC_ACTIVE_32BIT = 99
    ETHOSU_PMU_MAC_ACTIVE_8BIT = 100
    ETHOSU_PMU_MAC_DPU_ACTIVE = 101
    ETHOSU_PMU_MAC_STALLED_BY_ACC = 102
    ETHOSU_PMU_MAC_STALLED_BY_IB = 103
    ETHOSU_PMU_MAC_STALLED_BY_INT_ACC = 104
    ETHOSU_PMU_MAC_STALLED_BY_INT_W = 105
    ETHOSU_PMU_MAC_STALLED_BY_W = 106
    ETHOSU_PMU_MAC_STALLED_BY_WD = 107
    ETHOSU_PMU_MAC_STALLED_BY_WD_ACC = 108
    ETHOSU_PMU_MAC_STALLED_BY_W_OR_ACC = 109
    ETHOSU_PMU_NO_EVENT = 110
    ETHOSU_PMU_NPU_ACTIVE = 111
    ETHOSU_PMU_NPU_IDLE = 112
    ETHOSU_PMU_SRAM0_ENABLED_CYCLES = 113
    ETHOSU_PMU_SRAM0_RD_DATA_BEAT_RECEIVED = 114
    ETHOSU_PMU_SRAM0_RD_STALL_LIMIT = 115
    ETHOSU_PMU_SRAM0_RD_TRANS_ACCEPTED = 116
    ETHOSU_PMU_SRAM0_RD_TRANS_COMPLETED = 117
    ETHOSU_PMU_SRAM0_RD_TRAN_REQ_STALLED = 118
    ETHOSU_PMU_SRAM0_WR_DATA_BEAT_STALLED = 119
    ETHOSU_PMU_SRAM0_WR_DATA_BEAT_WRITTEN = 120
    ETHOSU_PMU_SRAM0_WR_STALL_LIMIT = 121
    ETHOSU_PMU_SRAM0_WR_TRANS_ACCEPTED = 122
    ETHOSU_PMU_SRAM0_WR_TRANS_COMPLETED_M = 123
    ETHOSU_PMU_SRAM0_WR_TRANS_COMPLETED_S = 124
    ETHOSU_PMU_SRAM0_WR_TRAN_REQ_STALLED = 125
    ETHOSU_PMU_SRAM1_ENABLED_CYCLES = 126
    ETHOSU_PMU_SRAM1_RD_DATA_BEAT_RECEIVED = 127
    ETHOSU_PMU_SRAM1_RD_STALL_LIMIT = 128
    ETHOSU_PMU_SRAM1_RD_TRANS_ACCEPTED = 129
    ETHOSU_PMU_SRAM1_RD_TRANS_COMPLETED = 130
    ETHOSU_PMU_SRAM1_RD_TRAN_REQ_STALLED = 131
    ETHOSU_PMU_SRAM1_WR_DATA_BEAT_STALLED = 132
    ETHOSU_PMU_SRAM1_WR_DATA_BEAT_WRITTEN = 133
    ETHOSU_PMU_SRAM1_WR_STALL_LIMIT = 134
    ETHOSU_PMU_SRAM1_WR_TRANS_ACCEPTED = 135
    ETHOSU_PMU_SRAM1_WR_TRANS_COMPLETED_M = 136
    ETHOSU_PMU_SRAM1_WR_TRANS_COMPLETED_S = 137
    ETHOSU_PMU_SRAM1_WR_TRAN_REQ_STALLED = 138
    ETHOSU_PMU_SRAM2_ENABLED_CYCLES = 139
    ETHOSU_PMU_SRAM2_RD_DATA_BEAT_RECEIVED = 140
    ETHOSU_PMU_SRAM2_RD_STALL_LIMIT = 141
    ETHOSU_PMU_SRAM2_RD_TRANS_ACCEPTED = 142
    ETHOSU_PMU_SRAM2_RD_TRANS_COMPLETED = 143
    ETHOSU_PMU_SRAM2_RD_TRAN_REQ_STALLED = 144
    ETHOSU_PMU_SRAM2_WR_DATA_BEAT_STALLED = 145
    ETHOSU_PMU_SRAM2_WR_DATA_BEAT_WRITTEN = 146
    ETHOSU_PMU_SRAM2_WR_STALL_LIMIT = 147
    ETHOSU_PMU_SRAM2_WR_TRANS_ACCEPTED = 148
    ETHOSU_PMU_SRAM2_WR_TRANS_COMPLETED_M = 149
    ETHOSU_PMU_SRAM2_WR_TRANS_COMPLETED_S = 150
    ETHOSU_PMU_SRAM2_WR_TRAN_REQ_STALLED = 151
    ETHOSU_PMU_SRAM3_ENABLED_CYCLES = 152
    ETHOSU_PMU_SRAM3_RD_DATA_BEAT_RECEIVED = 153
    ETHOSU_PMU_SRAM3_RD_STALL_LIMIT = 154
    ETHOSU_PMU_SRAM3_RD_TRANS_ACCEPTED = 155
    ETHOSU_PMU_SRAM3_RD_TRANS_COMPLETED = 156
    ETHOSU_PMU_SRAM3_RD_TRAN_REQ_STALLED = 157
    ETHOSU_PMU_SRAM3_WR_DATA_BEAT_STALLED = 158
    ETHOSU_PMU_SRAM3_WR_DATA_BEAT_WRITTEN = 159
    ETHOSU_PMU_SRAM3_WR_STALL_LIMIT = 160
    ETHOSU_PMU_SRAM3_WR_TRANS_ACCEPTED = 161
    ETHOSU_PMU_SRAM3_WR_TRANS_COMPLETED_M = 162
    ETHOSU_PMU_SRAM3_WR_TRANS_COMPLETED_S = 163
    ETHOSU_PMU_SRAM3_WR_TRAN_REQ_STALLED = 164
    ETHOSU_PMU_SRAM_ENABLED_CYCLES = 165
    ETHOSU_PMU_SRAM_RD_DATA_BEAT_RECEIVED = 166
    ETHOSU_PMU_SRAM_RD_STALL_LIMIT = 167
    ETHOSU_PMU_SRAM_RD_TRANS_ACCEPTED = 168
    ETHOSU_PMU_SRAM_RD_TRANS_COMPLETED = 169
    ETHOSU_PMU_SRAM_RD_TRAN_REQ_STALLED = 170
    ETHOSU_PMU_SRAM_WR_DATA_BEAT_STALLED = 171
    ETHOSU_PMU_SRAM_WR_DATA_BEAT_WRITTEN = 172
    ETHOSU_PMU_SRAM_WR_STALL_LIMIT = 173
    ETHOSU_PMU_SRAM_WR_TRANS_ACCEPTED = 174
    ETHOSU_PMU_SRAM_WR_TRANS_COMPLETED_M = 175
    ETHOSU_PMU_SRAM_WR_TRANS_COMPLETED_S = 176
    ETHOSU_PMU_SRAM_WR_TRAN_REQ_STALLED = 177
    ETHOSU_PMU_WD_ACTIVE = 178
    ETHOSU_PMU_WD_PARSE_ACTIVE = 179
    ETHOSU_PMU_WD_PARSE_ACTIVE_SC0 = 180
    ETHOSU_PMU_WD_PARSE_ACTIVE_SC1 = 181
    ETHOSU_PMU_WD_PARSE_ACTIVE_SC2 = 182
    ETHOSU_PMU_WD_PARSE_ACTIVE_SC3 = 183
    ETHOSU_PMU_WD_PARSE_STALLED = 184
    ETHOSU_PMU_WD_PARSE_STALLED_IN = 185
    ETHOSU_PMU_WD_PARSE_STALLED_OUT = 186
    ETHOSU_PMU_WD_PARSE_STALL_IN_SC0 = 187
    ETHOSU_PMU_WD_PARSE_STALL_IN_SC1 = 188
    ETHOSU_PMU_WD_PARSE_STALL_IN_SC2 = 189
    ETHOSU_PMU_WD_PARSE_STALL_IN_SC3 = 190
    ETHOSU_PMU_WD_PARSE_STALL_OUT_SC0 = 191
    ETHOSU_PMU_WD_PARSE_STALL_OUT_SC1 = 192
    ETHOSU_PMU_WD_PARSE_STALL_OUT_SC2 = 193
    ETHOSU_PMU_WD_PARSE_STALL_OUT_SC3 = 194
    ETHOSU_PMU_WD_PARSE_STALL_SC0 = 195
    ETHOSU_PMU_WD_PARSE_STALL_SC1 = 196
    ETHOSU_PMU_WD_PARSE_STALL_SC2 = 197
    ETHOSU_PMU_WD_PARSE_STALL_SC3 = 198
    ETHOSU_PMU_WD_STALLED = 199
    ETHOSU_PMU_WD_STALLED_BY_WD_BUF = 200
    ETHOSU_PMU_WD_STALLED_BY_WS = 201
    ETHOSU_PMU_WD_STALLED_BY_WS_FC = 202
    ETHOSU_PMU_WD_STALLED_BY_WS_SC0 = 203
    ETHOSU_PMU_WD_STALLED_BY_WS_SC1 = 204
    ETHOSU_PMU_WD_STALLED_BY_WS_SC2 = 205
    ETHOSU_PMU_WD_STALLED_BY_WS_SC3 = 206
    ETHOSU_PMU_WD_STALLED_BY_WS_TC = 207
    ETHOSU_PMU_WD_TRANS_DW0 = 208
    ETHOSU_PMU_WD_TRANS_DW1 = 209
    ETHOSU_PMU_WD_TRANS_WB = 210
    ETHOSU_PMU_WD_TRANS_WB0 = 211
    ETHOSU_PMU_WD_TRANS_WB1 = 212
    ETHOSU_PMU_WD_TRANS_WB2 = 213
    ETHOSU_PMU_WD_TRANS_WB3 = 214
    ETHOSU_PMU_WD_TRANS_WBLK = 215
    ETHOSU_PMU_WD_TRANS_WS = 216
    ETHOSU_PMU_WD_TRANS_WS_FC = 217
    ETHOSU_PMU_WD_TRANS_WS_SC0 = 218
    ETHOSU_PMU_WD_TRANS_WS_SC1 = 219
    ETHOSU_PMU_WD_TRANS_WS_SC2 = 220
    ETHOSU_PMU_WD_TRANS_WS_SC3 = 221
    ETHOSU_PMU_WD_TRANS_WS_TC = 222


@dataclass(frozen=True)
class EthosUPmuCounter:
    """One populated Ethos-U PMU event counter (experimental API)."""

    event_type: EthosUPmuEvent
    count: float
    overflow: bool


@dataclass(frozen=True)
class EthosUPmuProfileEvent:
    """PMU measurements for one Ethos-U invocation (experimental API)."""

    product: EthosUProduct
    platform: EthosUPlatform
    macs_per_cycle: int
    success: bool
    register_base: int
    cycle_count: float
    cycle_overflow: bool
    counters: tuple[EthosUPmuCounter, ...]


def parse_pmu_profile_event(metadata: bytes) -> EthosUPmuProfileEvent:
    """Decode one Ethos-U PMU delegate metadata blob (experimental API).

    Args:
        metadata (bytes): Raw ETDump delegate debug metadata.

    Returns:
        EthosUPmuProfileEvent: Decoded NPU cycles and PMU counters.

    Raises:
        ValueError: If the blob has an invalid format, version, or event ID.

    """
    if len(metadata) < 4:
        raise ValueError("Ethos-U PMU metadata is shorter than its header")
    if metadata[:3] != _MAGIC_PREFIX:
        raise ValueError(f"Unsupported Ethos-U PMU metadata magic: {metadata[:4]!r}")
    try:
        version = int(metadata[3:4])
    except ValueError as exc:
        raise ValueError("Invalid Ethos-U PMU metadata version") from exc
    parsers = {1: _parse_pmu_profile_event_v1}
    parser = parsers.get(version)
    if parser is None:
        raise ValueError(
            f"Unsupported Ethos-U PMU metadata version: {version}; "
            f"supported versions: {set(parsers.keys())}"
        )
    else:
        return parser(metadata)


def _parse_pmu_profile_event_v1(metadata: bytes) -> EthosUPmuProfileEvent:
    if len(metadata) < _HEADER.size:
        raise ValueError("Ethos-U PMU metadata is shorter than its header")

    (
        _,
        product,
        macs_per_cycle,
        platform_id,
        flags,
        register_base,
        cycle_count,
        counter_count,
    ) = _HEADER.unpack_from(metadata)
    try:
        platform = EthosUPlatform(platform_id)
    except ValueError as exc:
        raise ValueError(f"Unknown Ethos-U platform ID: {platform_id}") from exc
    try:
        product_type = EthosUProduct(product)
    except ValueError as exc:
        raise ValueError(f"Unknown Ethos-U product ID: {product}") from exc
    if flags & ~0b11:
        raise ValueError(f"Unsupported Ethos-U PMU flags: {flags:#x}")
    if counter_count > _MAX_COUNTERS:
        raise ValueError(f"Too many Ethos-U PMU counters: {counter_count}")

    expected_size = _HEADER.size + counter_count * _COUNTER.size
    if len(metadata) != expected_size:
        raise ValueError(
            f"Ethos-U PMU metadata has {len(metadata)} bytes; expected {expected_size}"
        )

    counters = _parse_counters(metadata, _HEADER.size, counter_count)

    return EthosUPmuProfileEvent(
        product=product_type,
        platform=platform,
        macs_per_cycle=macs_per_cycle,
        success=bool(flags & 2),
        register_base=register_base,
        cycle_count=float("nan") if flags & 1 else float(cycle_count),
        cycle_overflow=bool(flags & 1),
        counters=counters,
    )


def _parse_counters(
    metadata: bytes, offset: int, counter_count: int
) -> tuple[EthosUPmuCounter, ...]:
    counters = []
    for slot in range(counter_count):
        event_id, count, overflow = _COUNTER.unpack_from(
            metadata, offset + slot * _COUNTER.size
        )
        try:
            event_type = EthosUPmuEvent(event_id)
        except ValueError as exc:
            raise ValueError(f"Unknown Ethos-U PMU event ID: {event_id}") from exc
        if overflow not in (0, 1):
            raise ValueError(f"Invalid Ethos-U PMU overflow value: {overflow}")
        counters.append(
            EthosUPmuCounter(
                event_type, float("nan") if overflow else float(count), bool(overflow)
            )
        )
    return tuple(counters)


def parse_ethosu_delegate_pmu_metadata(
    metadata_list: Sequence[bytes | bytearray | str | None],
) -> dict[str, Any]:
    """Decode Ethos-U PMU samples for Inspector (experimental API).

    Args:
        metadata_list (Sequence[bytes | bytearray | str | None]): Raw samples.

    Returns:
        dict[str, Any]: Decoded samples in the `ethosu_pmu` key, or an empty
        dictionary for unrelated delegate metadata. Missing samples retain
        their positions as None.

    """
    records: list[dict[str, Any] | None] = []
    found = False
    for metadata in metadata_list:
        if not isinstance(metadata, (bytes, bytearray)) or not metadata.startswith(
            _MAGIC_PREFIX
        ):
            records.append(None)
            continue

        record = parse_pmu_profile_event(bytes(metadata))
        found = True
        records.append(
            {
                "product": record.product,
                "platform": record.platform,
                "macs_per_cycle": record.macs_per_cycle,
                "success": record.success,
                "register_base": record.register_base,
                "cycle_count": record.cycle_count,
                "cycle_overflow": record.cycle_overflow,
                "counters": [
                    {
                        "event_type": counter.event_type,
                        "event_id": counter.event_type.value,
                        "count": counter.count,
                        "overflow": counter.overflow,
                    }
                    for counter in record.counters
                ],
            }
        )

    return {"ethosu_pmu": records} if found else {}
