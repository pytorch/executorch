# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch

from executorch.backends.arm.common.pipeline_config import (
    LeakyReLULoweringConfig,
    SoftmaxDecompositionConfig,
)
from executorch.backends.arm.ethosu import (
    EthosUCompileSpec,
    VelaExternalBlockPlacements,
)
from executorch.backends.arm.tosa.backend import TOSABackend
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
from executorch.backends.arm.vgf import (
    backend as vgf_backend,
    partitioner as vgf_partitioner,
    VgfBackend,
    VgfCompileSpec,
    VgfPartitioner,
)
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.backend_details import PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import PartitionResult
from pytest import mark, raises, warns


def test_compile_spec_u55_INT():
    compile_spec = (
        EthosUCompileSpec(
            "ethos-u55",
            extra_flags=["--my-flag"],
            external_block_placements=VelaExternalBlockPlacements(
                cmd_data="mem1",
                weight_data="mem2",
            ),
        )
        .dump_intermediate_artifacts_to("my_path")
        .dump_debug_info(EthosUCompileSpec.DebugMode.TOSA)
    )
    spec_list = compile_spec._to_list()

    roundtripped = EthosUCompileSpec._from_list(spec_list)
    assert roundtripped == compile_spec
    assert roundtripped.external_block_placements == VelaExternalBlockPlacements(
        cmd_data="mem1",
        weight_data="mem2",
    )
    assert "--my-flag" in compile_spec.compiler_flags
    assert "--output-format=raw" in compile_spec.compiler_flags
    with raises(ValueError, match="Incorrect output format"):
        VgfCompileSpec._from_list(spec_list)

    spec_list.pop(0)
    with raises(ValueError, match="No tosa_spec in compile spec."):
        EthosUCompileSpec._from_list(spec_list)


def test_ethos_u55_defaults_to_stable_softmax_u55_INT():
    """Test that EthosUCompileSpec for U55 defaults to STABLE softmax config."""
    compile_spec = EthosUCompileSpec("ethos-u55-128")
    pipeline_config = compile_spec._get_pass_pipeline_config()
    assert pipeline_config.softmax == SoftmaxDecompositionConfig.STABLE


def test_ethos_u65_defaults_to_high_end_dedicated_sram_u65_INT():
    compile_spec = EthosUCompileSpec("ethos-u65-256")

    assert "--accelerator-config=ethos-u65-256" in compile_spec.compiler_flags
    assert "--system-config=Ethos_U65_High_End" in compile_spec.compiler_flags
    assert "--memory-mode=Dedicated_Sram_384KB" in compile_spec.compiler_flags
    assert compile_spec.tosa_spec.is_U55_subset


def test_ethos_u85_defaults_to_masked_softmax_u85_INT():
    """Test that EthosUCompileSpec for U85 defaults to MASKED softmax config."""
    compile_spec = EthosUCompileSpec("ethos-u85-256")
    pipeline_config = compile_spec._get_pass_pipeline_config()
    roundtripped = EthosUCompileSpec._from_list(compile_spec._to_list())
    assert pipeline_config.softmax == SoftmaxDecompositionConfig.MASKED
    assert roundtripped.external_block_placements == VelaExternalBlockPlacements()


def test_compile_spec_vgf_no_quant():
    compile_spec = (
        VgfCompileSpec(compiler_flags=["--my-flag"])
        .dump_intermediate_artifacts_to("my_path")
        .dump_debug_info(None)
    )
    compile_spec2 = VgfCompileSpec(
        compiler_flags=["--my-flag2"]
    ).dump_intermediate_artifacts_to("my_path")

    spec_list = compile_spec._to_list()

    assert VgfCompileSpec._from_list(spec_list) == compile_spec
    assert VgfCompileSpec._from_list(spec_list) != compile_spec2
    with raises(ValueError, match="Incorrect output format"):
        EthosUCompileSpec._from_list(spec_list)


def test_compile_spec_vgf_defaults_leaky_relu_to_decompose():
    compile_spec = VgfCompileSpec()
    pipeline_config = compile_spec._get_pass_pipeline_config()

    assert pipeline_config.leaky_relu is LeakyReLULoweringConfig.DECOMPOSE


def test_alias_buffer_mutations_roundtrip_vgf_FP_INT():
    compile_spec = VgfCompileSpec(alias_buffer_mutations=True)
    serialized = compile_spec._to_list()
    serialized.append(CompileSpec("mutable_buffer_pairs", b"0:0:0"))
    roundtripped = VgfCompileSpec._from_list(serialized)

    assert roundtripped == compile_spec
    assert roundtripped.alias_buffer_mutations is True


def test_alias_buffer_mutations_with_debug_info_roundtrip_vgf_FP_INT():
    compile_spec = VgfCompileSpec(emit_debug_info=True, alias_buffer_mutations=True)
    roundtripped = VgfCompileSpec._from_list(compile_spec._to_list())

    assert roundtripped == compile_spec
    assert roundtripped.emit_debug_info is True
    assert roundtripped.alias_buffer_mutations is True


def test_alias_buffer_mutations_defaults_off_vgf_FP_INT():
    assert VgfCompileSpec().alias_buffer_mutations is False
    assert VgfCompileSpec() != VgfCompileSpec(alias_buffer_mutations=True)


def test_alias_buffer_mutation_compile_specs_are_partition_local(monkeypatch):
    partitioner = VgfPartitioner(VgfCompileSpec(alias_buffer_mutations=True))
    shared_spec = partitioner.delegation_spec
    tagged_program = object()
    partition_result = PartitionResult(
        tagged_program,
        {
            "tag0": shared_spec,
            "tag1": shared_spec,
        },
    )

    def fake_partition(_self, _exported_program):
        return partition_result

    monkeypatch.setattr(TOSAPartitioner, "partition", fake_partition)
    monkeypatch.setattr(partitioner, "_validate_mutable_buffers", lambda _: None)
    monkeypatch.setattr(vgf_partitioner, "tag_mutated_buffer", lambda _: None)

    result = partitioner.partition(tagged_program)
    tag0_specs = result.partition_tags["tag0"].compile_specs
    tag1_specs = result.partition_tags["tag1"].compile_specs

    assert tag0_specs is not tag1_specs
    assert tag0_specs is not shared_spec.compile_specs
    assert tag1_specs is not shared_spec.compile_specs
    assert CompileSpec(vgf_backend.MUTABLE_BUFFER_OWNERSHIP_KEY, b"1") in tag0_specs
    assert CompileSpec(vgf_backend.MUTABLE_BUFFER_OWNERSHIP_KEY, b"1") in tag1_specs
    assert all(
        spec.key != vgf_backend.MUTABLE_BUFFER_OWNERSHIP_KEY
        for spec in shared_spec.compile_specs
    )

    tag0_specs.append(CompileSpec("mutable_buffer_pairs", b"0:0:0"))
    assert all(spec.key != "mutable_buffer_pairs" for spec in tag1_specs)
    assert all(spec.key != "mutable_buffer_pairs" for spec in shared_spec.compile_specs)


def test_vgf_direct_aliasing_requires_partitioner(monkeypatch):

    class MutatingCache(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("cache", torch.zeros(4, dtype=torch.int8))

        def forward(self, x):
            self.cache.add_(x)
            return self.cache

    monkeypatch.setattr(
        TOSABackend,
        "_preprocess",
        staticmethod(lambda *args: PreprocessResult(processed_bytes=b"tosa")),
    )
    monkeypatch.setattr(
        VgfBackend,
        "_compile_tosa_flatbuffer",
        staticmethod(lambda *args: b"vgf"),
    )
    monkeypatch.setattr(
        vgf_backend, "arm_get_first_delegation_tag", lambda graph_module: "test"
    )

    exported_program = torch.export.export(
        MutatingCache(), (torch.ones(4, dtype=torch.int8),)
    ).run_decompositions()
    original_specs = VgfCompileSpec(alias_buffer_mutations=True)._to_list()
    with raises(ValueError, match="requires VgfPartitioner"):
        to_backend("VgfBackend", exported_program, original_specs)

    partitioned_specs = original_specs + [
        CompileSpec(vgf_backend.MUTABLE_BUFFER_OWNERSHIP_KEY, b"1")
    ]
    first = to_backend("VgfBackend", exported_program, partitioned_specs)

    def pairs(specs):
        return [spec for spec in specs if spec.key == "mutable_buffer_pairs"]

    assert not pairs(original_specs)
    assert len(pairs(first.compile_specs)) == 1
    assert all(
        spec.key != vgf_backend.MUTABLE_BUFFER_OWNERSHIP_KEY
        for spec in first.compile_specs
    )
    with raises(ValueError, match="requires VgfPartitioner"):
        to_backend("VgfBackend", first.original_module, first.compile_specs)

    disabled_specs = VgfCompileSpec()._to_list() + pairs(first.compile_specs)
    without_aliasing = to_backend("VgfBackend", exported_program, disabled_specs)
    assert not pairs(without_aliasing.compile_specs)
    assert len(pairs(disabled_specs)) == 1


def test_compile_spec_tosa_defaults_leaky_relu_to_decompose():
    compile_spec = TosaCompileSpec("TOSA-1.0+INT")
    pipeline_config = compile_spec._get_pass_pipeline_config()

    assert pipeline_config.leaky_relu is LeakyReLULoweringConfig.DECOMPOSE


def test_compile_spec_tosa_INT():
    compile_spec = TosaCompileSpec("TOSA-1.0+INT")
    spec_list = compile_spec._to_list()

    assert TosaCompileSpec._from_list(spec_list) == compile_spec
    with raises(ValueError, match="Incorrect output format"):
        VgfCompileSpec._from_list(spec_list)


def test_preserve_io_quantization_roundtrip_vgf_FP_INT():
    compile_spec = VgfCompileSpec()._set_preserve_io_quantization(True)
    roundtripped = VgfCompileSpec._from_list(compile_spec._to_list())
    assert roundtripped.preserve_io_quantization is True


def test_preserve_tosa_dev_mode_roundtrip_vgf_FP_INT():
    compile_spec = VgfCompileSpec()
    roundtripped = VgfCompileSpec._from_list(compile_spec._to_list())
    assert roundtripped.tosa_dev_mode is True


def test_emit_debug_info_roundtrip_vgf_FP_INT():
    disabled = VgfCompileSpec()
    disabled_roundtripped = VgfCompileSpec._from_list(disabled._to_list())
    assert disabled_roundtripped.emit_debug_info is False

    enabled = VgfCompileSpec(emit_debug_info=True)
    enabled_roundtripped = VgfCompileSpec._from_list(enabled._to_list())
    assert enabled_roundtripped.emit_debug_info is True


def test_preserve_io_quantization_warns_for_u55_INT():
    with warns(
        UserWarning,
        match="preserve_io_quantization=True is redundant for INT-only TOSA",
    ):
        EthosUCompileSpec("ethos-u55-128")._set_preserve_io_quantization(True)


def test_preserve_io_quantization_no_warn_for_vgf_FP_INT():
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        VgfCompileSpec()._set_preserve_io_quantization(True)
    assert len(recorded_warnings) == 0


@mark.parametrize("max_scratch_size", [None, 2097152, 4194304])
def test_ethosu_scratch_capacity_roundtrip(max_scratch_size):
    compile_spec = EthosUCompileSpec("ethos-u55-128", max_scratch_size=max_scratch_size)
    roundtripped = EthosUCompileSpec._from_list(compile_spec._to_list())
    assert roundtripped.max_scratch_size == max_scratch_size
    assert all("max_scratch_size" not in flag for flag in roundtripped.compiler_flags)


@mark.parametrize("max_scratch_size", [0, -1, 1.5, True, "2097152"])
def test_ethosu_scratch_capacity_rejects_invalid_values(max_scratch_size):
    with raises(ValueError, match="max_scratch_size must be a positive integer"):
        EthosUCompileSpec("ethos-u55-128", max_scratch_size=max_scratch_size)
