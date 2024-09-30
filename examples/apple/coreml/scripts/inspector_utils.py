# Copyright © 2024 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.

import copy
import errno
import json
import os

import subprocess
from dataclasses import dataclass
from pathlib import Path

from typing import Any, Dict, Final, List, Optional, Tuple, Union

import executorch.exir as exir

import pandas as pd
import torch
from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition import CoreMLPartitioner

from executorch.devtools import BundledProgram, generate_etrecord, Inspector
from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)
from executorch.devtools.inspector import Event

from executorch.exir import (
    EdgeProgramManager,
    ExecutorchBackendConfig,
    ExecutorchProgramManager,
    ExirExportedProgram,
    to_edge,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.tracer import Value

from torch.export import export, ExportedProgram

COREML_METADATA_KEYS: Final[List[Tuple[str, str]]] = [
    ("operatorName", "coreml_operator"),
    ("estimatedCost", "coreml_estimated_cost"),
    ("preferredComputeUnit", "coreml_preferred_device"),
    ("supportedComputeUnits", "coreml_supported_devices"),
]


def build_devtools_runner_including_coreml(
    root_dir_path: Path,
    conda_env_name: str,
    force: bool = False,
):
    if not force:
        devtools_executable_path = (
            root_dir_path / "cmake-out" / "examples" / "devtools" / "example_runner"
        )
        print(devtools_executable_path)
        if devtools_executable_path.is_file():
            return

    cd_root_command: str = f"cd {root_dir_path.resolve()}"
    conda_activate_env_command: str = f"source conda activate {conda_env_name}"
    build_devtools_runner_command: str = (
        "./examples/devtools/build_example_runner.sh --coreml"
    )
    build_command: str = (
        f"{cd_root_command} && {conda_activate_env_command} && {build_devtools_runner_command}"
    )
    subprocess.run(
        f'bash -c "{build_command}"', shell=True, check=True
    ).check_returncode()


_EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig(
    _check_ir_validity=False,
    _skip_dim_order=True,
)

_EDGE_BACKEND_CONFIG = exir.ExecutorchBackendConfig(
    extract_delegate_segments=True,
)


def to_core_aten(
    module: torch.nn.Module,
    example_inputs: Tuple[Value, ...],
) -> ExportedProgram:
    core_aten_program = export(
        mod=module,
        args=example_inputs,
    )
    return core_aten_program


def core_aten_to_edge(
    core_aten_program: ExportedProgram,
    edge_compile_config: exir.EdgeCompileConfig,
) -> EdgeProgramManager:
    edge_manager = to_edge(
        programs=core_aten_program,
        compile_config=edge_compile_config,
    )
    return edge_manager


def module_to_edge(
    module: torch.nn.Module,
    example_inputs: Tuple[Value, ...],
    edge_compile_config: exir.EdgeCompileConfig = _EDGE_COMPILE_CONFIG,
) -> EdgeProgramManager:
    module.eval()
    core_aten_program = to_core_aten(
        module=module,
        example_inputs=example_inputs,
    )
    return core_aten_to_edge(
        core_aten_program=core_aten_program,
        edge_compile_config=edge_compile_config,
    )


def lower_and_export_edge_to_coreml(
    edge_program: EdgeProgramManager,
    compile_specs: List[CompileSpec],
    config: ExecutorchBackendConfig,
    skip_ops_for_coreml_delegation: Optional[List[str]] = None,
) -> ExirExportedProgram:
    partitioner = CoreMLPartitioner(
        skip_ops_for_coreml_delegation=skip_ops_for_coreml_delegation,
        compile_specs=compile_specs,
    )
    delegated_program_manager = edge_program.to_backend(
        partitioner,
    )
    executorch_program = delegated_program_manager.to_executorch(
        config=config,
    )
    return executorch_program


def write_to_file(buffer: bytes, file_path: Path):
    with open(file_path.resolve(), "wb") as file:
        file.write(buffer)


def generate_bundled_program(
    executorch_program: ExecutorchProgramManager,
    example_inputs: Tuple[Value, ...],
    method_name: str,
    bundled_program_path: Path,
):
    method_test_suites = [
        MethodTestSuite(
            method_name=method_name,
            test_cases=[MethodTestCase(inputs=example_inputs)],
        )
    ]

    bundled_program = BundledProgram(executorch_program, method_test_suites)
    bundled_program_buffer = serialize_from_bundled_program_to_flatbuffer(
        bundled_program
    )

    write_to_file(buffer=bundled_program_buffer, file_path=bundled_program_path)


def generate_etdump_with_intermediate_values(
    root_dir_path: Path,
    bundled_program_path: Path,
    et_dump_path: Path,
    debug_buffer_path: Path,
    debug_buffer_size: int,
):
    devtools_executable_path = (
        root_dir_path / "cmake-out" / "examples" / "devtools" / "example_runner"
    )
    if not devtools_executable_path.is_file():
        raise FileNotFoundError(
            errno.ENOENT,
            os.strerror(errno.ENOENT),
            str(devtools_executable_path.resolve()),
        )

    devtools_runner_command: str = f"""
    {devtools_executable_path.resolve()} -dump_intermediate_outputs\
    -bundled_program_path {bundled_program_path.resolve()}\
    -etdump_path {et_dump_path.resolve()}\
    -debug_output_path {debug_buffer_path.resolve()}\
    -debug_buffer_size {debug_buffer_size}"""
    subprocess.run(
        f'bash -c "{devtools_runner_command}"', shell=True, check=True
    ).check_returncode()


def create_inspector(
    edge_program: EdgeProgramManager,
    executorch_program: ExecutorchProgramManager,
    example_inputs: Tuple[Value, ...],
    model_name: str,
    root_dir_path: Path,
    working_dir_path: Path,
    method_name: str = "forward",
    debug_buffer_size: int = 1 * 1024 * 1024 * 1024,
    delegate_metadata_parser=None,
    delegate_time_scale_converter=None,
) -> Inspector:
    et_record_path = working_dir_path / f"{model_name}_etrecord.bin"
    generate_etrecord(
        et_record=et_record_path.resolve(),
        edge_dialect_program=edge_program,
        executorch_program=executorch_program,
    )

    bundled_program_path = working_dir_path / f"{model_name}.bpte"
    generate_bundled_program(
        executorch_program=executorch_program,
        example_inputs=example_inputs,
        method_name=method_name,
        bundled_program_path=bundled_program_path,
    )

    et_dump_path: Path = working_dir_path / f"{model_name}_etdump.etdp"
    debug_buffer_path: Path = working_dir_path / f"{model_name}_debug_output.bin"
    generate_etdump_with_intermediate_values(
        root_dir_path=root_dir_path,
        bundled_program_path=bundled_program_path,
        et_dump_path=et_dump_path,
        debug_buffer_path=debug_buffer_path,
        debug_buffer_size=debug_buffer_size,
    )

    return Inspector(
        etdump_path=str(et_dump_path.resolve()),
        etrecord=str(et_record_path.resolve()),
        debug_buffer_path=str(debug_buffer_path.resolve()),
        enable_module_hierarchy=True,
        delegate_metadata_parser=delegate_metadata_parser,
        delegate_time_scale_converter=delegate_time_scale_converter,
    )


def parse_coreml_delegate_metadata(delegate_metadatas: List[str]) -> Dict[str, Any]:
    if len(delegate_metadatas) == 0:
        return
    try:
        coreml_metadata: Dict[str, Any] = json.loads(delegate_metadatas[0])
        result: Dict[str, str] = {}
        for col_key, col_name in COREML_METADATA_KEYS:
            value = coreml_metadata.get(col_key, None)
            if value is not None:
                result[col_name] = value
        return result

    except ValueError:
        return {}


def convert_coreml_delegate_time(
    event_name: Union[str, int], input_time: Union[int, float]
) -> Union[int, float]:
    return input_time / (1000 * 1000)


def create_inspector_coreml(
    edge_program: EdgeProgramManager,
    compile_specs: List[CompileSpec],
    example_inputs: Tuple[Value, ...],
    model_name: str,
    root_dir_path: Path,
    working_dir_path: Path,
    method_name: str = "forward",
    debug_buffer_size: int = 1 * 1024 * 1024 * 1024,
) -> Inspector:
    edge_program_copy = copy.deepcopy(edge_program)
    executorch_program = lower_and_export_edge_to_coreml(
        edge_program=edge_program_copy,
        compile_specs=compile_specs,
        config=_EDGE_BACKEND_CONFIG,
    )
    return create_inspector(
        edge_program=edge_program,
        executorch_program=executorch_program,
        example_inputs=example_inputs,
        root_dir_path=root_dir_path,
        model_name=f"{model_name}_coreml",
        working_dir_path=working_dir_path,
        method_name=method_name,
        debug_buffer_size=debug_buffer_size,
        delegate_metadata_parser=parse_coreml_delegate_metadata,
        delegate_time_scale_converter=convert_coreml_delegate_time,
    )


def create_inspector_reference(
    edge_program: EdgeProgramManager,
    example_inputs: Tuple[Value, ...],
    model_name: str,
    root_dir_path: Path,
    working_dir_path: Path,
    method_name: str = "forward",
    debug_buffer_size: int = 1 * 1024 * 1024 * 1024,
) -> Inspector:
    edge_program_copy = copy.deepcopy(edge_program)
    return create_inspector(
        edge_program=edge_program,
        executorch_program=edge_program_copy.to_executorch(),
        example_inputs=example_inputs,
        root_dir_path=root_dir_path,
        model_name=f"{model_name}_default",
        working_dir_path=working_dir_path,
        method_name=method_name,
        debug_buffer_size=debug_buffer_size,
    )


def get_debug_handle_to_event_map(
    inspector: Inspector,
    event_block_name: str = "Execute",
) -> Dict[int, Event]:
    result = {}

    def is_not_blank(s):
        return bool(s and not s.isspace())

    event_names_to_ignore = {"DELEGATE_CALL", "OPERATOR_CALL"}
    for event_block in inspector.event_blocks:
        if event_block.name == event_block_name:
            for event in event_block.events:
                if is_not_blank(event.name) and event.name not in event_names_to_ignore:
                    debug_handles = []
                    if isinstance(event.debug_handles, int):
                        debug_handles.append(event.debug_handles)
                    elif isinstance(event.debug_handles, list):
                        debug_handles.extend(event.debug_handles)
                    debug_handles.sort()
                    for debug_handle in debug_handles:
                        if len(event.debug_data) > 0:
                            result[debug_handle] = event
    return result


@dataclass
class EventData:
    tag: str
    event: Event


@dataclass
class ComparisonResult:
    datas: List[tuple[EventData, EventData]]

    def to_dataframe(
        self,
        atol: float = 1e-3,
        rtol: float = 1e-3,
    ) -> pd.DataFrame:
        def get_compute_device(event: Event) -> str:
            if event.delegate_backend_name == CoreMLBackend.__name__:
                return event.delegate_debug_metadatas.get(
                    "coreml_preferred_device", "CPU"
                )

            return "CPU"

        if len(self.datas) == 0:
            return

        (data1, data2) = self.datas[0]
        dict = {
            data1.tag: [],
            f"{data1.tag}_compute_unit": [],
            data2.tag: [],
            f"{data2.tag}_compute_unit": [],
            "max_diff": [],
        }

        for data1, data2 in self.datas:
            event1 = data1.event
            event2 = data2.event
            debug_data1 = event1.debug_data[0]
            debug_data2 = event2.debug_data[0]

            if debug_data1.size() != debug_data2.size():
                continue

            max_diff = 0.0
            indices = torch.isclose(
                debug_data1, debug_data2, atol=atol, rtol=rtol
            ).logical_not()

            # Find the maximum difference
            if torch.count_nonzero(indices) > 0:
                values1 = torch.masked_select(debug_data1, indices)
                values2 = torch.masked_select(debug_data2, indices)
                diff = torch.abs(values1 - values2)
                max_diff = torch.max(diff).item()

            dict[f"{data1.tag}_compute_unit"].append(get_compute_device(event1))
            dict[f"{data2.tag}_compute_unit"].append(get_compute_device(event2))
            dict["max_diff"].append(max_diff)
            dict[data1.tag].append(event1.name)
            dict[data2.tag].append(event2.name)

        return pd.DataFrame(dict)


def get_comparison_result(
    inspector1: Inspector,
    tag1: str,
    inspector2: Inspector,
    tag2: str,
) -> ComparisonResult:
    debug_handle_event_map_1 = get_debug_handle_to_event_map(inspector1)
    debug_handle_event_map_2 = get_debug_handle_to_event_map(inspector2)

    event_datas = []
    for handle, event1 in debug_handle_event_map_1.items():
        event2 = debug_handle_event_map_2.get(handle, None)
        if event2 is None:
            continue

        event_data1 = EventData(tag=tag1, event=event1)
        event_data2 = EventData(tag=tag2, event=event2)
        event_datas.append((event_data1, event_data2))

    return ComparisonResult(datas=event_datas)
