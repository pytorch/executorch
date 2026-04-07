# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.scripts.ethosu_pte_info import (
    get_ethosu_delegate_config_from_pte,
)
from executorch.exir._serialize import _PTEFile, _serialize_pte_binary
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.schema import (
    BackendDelegate,
    BackendDelegateDataReference,
    BackendDelegateInlineData,
    Buffer,
    Chain,
    ContainerMetadata,
    DataLocation,
    ExecutionPlan,
    Program,
    SubsegmentOffsets,
)
from pytest import raises


def _make_pte_bytes(*, delegates: list[BackendDelegate]) -> bytes:
    program = Program(
        version=0,
        execution_plan=[
            ExecutionPlan(
                name="forward",
                container_meta_type=ContainerMetadata(
                    encoded_inp_str="[]", encoded_out_str="[]"
                ),
                values=[],
                inputs=[],
                outputs=[],
                chains=[Chain(inputs=[], outputs=[], instructions=[], stacktrace=None)],
                operators=[],
                delegates=delegates,
                non_const_buffer_sizes=[0],
            )
        ],
        constant_buffer=[Buffer(storage=b"")],
        backend_delegate_data=[BackendDelegateInlineData(data=b"delegate-data")],
        segments=[],
        constant_segment=SubsegmentOffsets(segment_index=0, offsets=[]),
        mutable_data_segments=[],
        named_data=[],
    )
    return bytes(_serialize_pte_binary(_PTEFile(program)))


def _write_pte(tmp_path, *delegates: BackendDelegate):
    pte_path = tmp_path / "model.pte"
    pte_path.write_bytes(_make_pte_bytes(delegates=list(delegates)))
    return pte_path


def _ethosu_delegate(
    target: str,
    system_config: str,
    memory_mode: str,
    index: int = 0,
) -> BackendDelegate:
    return BackendDelegate(
        id="EthosUBackend",
        processed=BackendDelegateDataReference(
            location=DataLocation.INLINE, index=index
        ),
        compile_specs=EthosUCompileSpec(
            target=target,
            system_config=system_config,
            memory_mode=memory_mode,
        )._to_list(),
    )


def test_ethosu_pte_info_no_target(tmp_path) -> None:
    pte_path = _write_pte(
        tmp_path,
        _ethosu_delegate(
            target="ethos-u85-256",
            system_config="Ethos_U85_SYS_DRAM_Mid",
            memory_mode="Dedicated_Sram_384KB",
        ),
    )

    config = get_ethosu_delegate_config_from_pte(pte_path)

    assert config is not None
    assert config.target == "ethos-u85-256"
    assert config.system_config == "Ethos_U85_SYS_DRAM_Mid"
    assert config.memory_mode == "Dedicated_Sram_384KB"


def test_ethosu_pte_info_returns_none_without_ethosu_delegate_no_target(
    tmp_path,
) -> None:
    pte_path = _write_pte(
        tmp_path,
        BackendDelegate(
            id="OtherBackend",
            processed=BackendDelegateDataReference(
                location=DataLocation.INLINE, index=0
            ),
            compile_specs=[CompileSpec(key="k", value=b"v")],
        ),
    )

    assert get_ethosu_delegate_config_from_pte(pte_path) is None


def test_ethosu_pte_info_rejects_mixed_configs_no_target(tmp_path) -> None:
    pte_path = _write_pte(
        tmp_path,
        _ethosu_delegate(
            target="ethos-u55-128",
            system_config="Ethos_U55_High_End_Embedded",
            memory_mode="Shared_Sram",
        ),
        _ethosu_delegate(
            target="ethos-u85-256",
            system_config="Ethos_U85_SYS_DRAM_Mid",
            memory_mode="Sram_Only",
        ),
    )

    with raises(ValueError, match="multiple Ethos-U delegate compile spec"):
        get_ethosu_delegate_config_from_pte(pte_path)
