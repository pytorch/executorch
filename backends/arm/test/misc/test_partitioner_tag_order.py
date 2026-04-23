# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

from executorch.backends.arm.tosa import partitioner as tosa_partitioner
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.partitioner import TOSAPartitioner


class _FakeCapabilityBasedPartitioner:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def propose_partitions(self):
        return [
            SimpleNamespace(nodes=[SimpleNamespace(meta={}, target=f"op{idx}")])
            for idx in range(3)
        ]


def _make_reporter() -> SimpleNamespace:
    return SimpleNamespace(
        report_reject=lambda *args, **kwargs: None,
        get_table_report=lambda: "",
    )


def test_tag_module_preserves_partition_discovery_order(monkeypatch):
    partitioner = TOSAPartitioner(TosaCompileSpec("TOSA-1.0+FP"))

    monkeypatch.setattr(
        tosa_partitioner, "get_cond_while_submodules_nested", lambda module: []
    )
    monkeypatch.setattr(
        tosa_partitioner, "tosa_support_factory", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        tosa_partitioner,
        "CapabilityBasedPartitioner",
        _FakeCapabilityBasedPartitioner,
    )
    monkeypatch.setattr(
        partitioner,
        "_partition_has_invalid_uint8",
        lambda partition, tag: False,
    )
    monkeypatch.setattr(
        partitioner,
        "_preserve_io_quantization_enabled",
        lambda: False,
    )

    tags = partitioner._tag_module(
        SimpleNamespace(graph=SimpleNamespace(nodes=[])),
        SimpleNamespace(),
        _make_reporter(),
    )

    assert tags == ["tag0", "tag1", "tag2"]


def test_partition_preserves_tag_discovery_order(monkeypatch):
    partitioner = TOSAPartitioner(TosaCompileSpec("TOSA-1.0+FP"))

    monkeypatch.setattr(
        partitioner,
        "_tag_module",
        lambda *args, **kwargs: ["tag2", "tag10"],
    )
    monkeypatch.setattr(tosa_partitioner, "tag_constant_data", lambda program: None)
    monkeypatch.setattr(
        tosa_partitioner, "WhyNoPartitionReporter", _make_reporter
    )

    result = partitioner.partition(SimpleNamespace(graph_module=SimpleNamespace()))

    assert list(result.partition_tags) == ["tag2", "tag10"]
