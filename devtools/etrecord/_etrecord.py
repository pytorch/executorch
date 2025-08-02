# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import json
import os
import pickle
from typing import BinaryIO, Dict, IO, List, Optional, Union
from zipfile import BadZipFile, ZipFile

import torch

from executorch import exir

from executorch.devtools.bundled_program.config import ConfigValue
from executorch.devtools.bundled_program.core import BundledProgram
from executorch.exir import (
    EdgeProgramManager,
    ExecutorchProgram,
    ExecutorchProgramManager,
    ExirExportedProgram,
    ExportedProgram,
)
from executorch.exir.emit._emitter import _DelegateDebugIdentifierMap

from executorch.exir.serde.export_serialize import SerializedArtifact
from executorch.exir.serde.serialize import deserialize, serialize

ProgramInput = ConfigValue
ProgramOutput = torch.Tensor

try:
    # breaking change introduced in python 3.11
    # pyre-ignore
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        pass


class ETRecordReservedFileNames(StrEnum):
    ETRECORD_IDENTIFIER = "ETRECORD_V0"
    EXPORTED_PROGRAM = "exported_program"
    EXPORT_GRAPH_ID = "export_graph_id"
    EDGE_DIALECT_EXPORTED_PROGRAM = "edge_dialect_exported_program"
    ET_DIALECT_GRAPH_MODULE = "et_dialect_graph_module"
    DEBUG_HANDLE_MAP_NAME = "debug_handle_map"
    DELEGATE_MAP_NAME = "delegate_map"
    REFERENCE_OUTPUTS = "reference_outputs"
    REPRESENTATIVE_INPUTS = "representative_inputs"


class ETRecord:
    def __init__(
        self,
        exported_program: Optional[ExportedProgram] = None,
        export_graph_id: Optional[int] = None,
        edge_dialect_program: Optional[ExportedProgram] = None,
        graph_map: Optional[Dict[str, ExportedProgram]] = None,
        _debug_handle_map: Optional[Dict[int, Union[int, List[int]]]] = None,
        _delegate_map: Optional[
            Dict[str, Dict[int, Dict[str, Union[str, _DelegateDebugIdentifierMap]]]]
        ] = None,
        _reference_outputs: Optional[Dict[str, List[ProgramOutput]]] = None,
        _representative_inputs: Optional[List[ProgramOutput]] = None,
    ):
        self.exported_program = exported_program
        self.export_graph_id = export_graph_id
        self.edge_dialect_program = edge_dialect_program
        self.graph_map = graph_map
        self._debug_handle_map = _debug_handle_map
        self._delegate_map = _delegate_map
        self._reference_outputs = _reference_outputs
        self._representative_inputs = _representative_inputs

    def save(self, path: Union[str, os.PathLike, BinaryIO, IO[bytes]]) -> None:
        """
        Serialize and save the ETRecord to the specified path.

        Args:
            path: Path where the ETRecord file will be saved to.
        """
        if isinstance(path, (str, os.PathLike)):
            # pyre-ignore[6]: In call `os.fspath`, for 1st positional argument, expected `str` but got `Union[PathLike[typing.Any], str]`
            path = os.fspath(path)

        etrecord_zip = ZipFile(path, "w")

        try:
            self._write_identifier(etrecord_zip)
            self._save_programs(etrecord_zip)
            self._save_graph_map(etrecord_zip)
            self._save_metadata(etrecord_zip)
        finally:
            etrecord_zip.close()

    def _write_identifier(self, etrecord_zip: ZipFile) -> None:
        """Write the magic file identifier."""
        etrecord_zip.writestr(ETRecordReservedFileNames.ETRECORD_IDENTIFIER, "")

    def _save_programs(self, etrecord_zip: ZipFile) -> None:
        """Save exported program and edge dialect program."""
        if self.exported_program is not None:
            self._save_exported_program(
                etrecord_zip,
                ETRecordReservedFileNames.EXPORTED_PROGRAM,
                "",
                self.exported_program,
            )

        if self.edge_dialect_program is not None:
            self._save_edge_dialect_program(etrecord_zip, self.edge_dialect_program)

    def _save_graph_map(self, etrecord_zip: ZipFile) -> None:
        """Save graph map if present."""
        if self.graph_map is not None:
            # pyre-ignore[16]: Undefined attribute [16]: `Optional` has no attribute `items`.
            for module_name, export_module in self.graph_map.items():
                if "/" in module_name:
                    base_name, method_name = module_name.rsplit("/", 1)
                    self._save_exported_program(
                        etrecord_zip, base_name, method_name, export_module
                    )
                else:
                    self._save_exported_program(
                        etrecord_zip, module_name, "forward", export_module
                    )

    def _save_metadata(self, etrecord_zip: ZipFile) -> None:
        """Save debug maps, reference outputs, and other metadata."""
        if self._debug_handle_map is not None:
            etrecord_zip.writestr(
                ETRecordReservedFileNames.DEBUG_HANDLE_MAP_NAME,
                json.dumps(self._debug_handle_map),
            )

        if self._delegate_map is not None:
            etrecord_zip.writestr(
                ETRecordReservedFileNames.DELEGATE_MAP_NAME,
                json.dumps(self._delegate_map),
            )

        if self._reference_outputs is not None:
            etrecord_zip.writestr(
                ETRecordReservedFileNames.REFERENCE_OUTPUTS,
                pickle.dumps(self._reference_outputs),
            )

        if self._representative_inputs is not None:
            etrecord_zip.writestr(
                ETRecordReservedFileNames.REPRESENTATIVE_INPUTS,
                pickle.dumps(self._representative_inputs),
            )

        if self.export_graph_id is not None:
            etrecord_zip.writestr(
                ETRecordReservedFileNames.EXPORT_GRAPH_ID,
                json.dumps(self.export_graph_id),
            )

    def _save_exported_program(
        self,
        etrecord_zip: ZipFile,
        module_name: str,
        method_name: str,
        ep: ExportedProgram,
    ) -> None:
        """Save an exported program to the ETRecord zip file."""
        serialized_artifact = serialize(ep)
        assert isinstance(serialized_artifact.exported_program, bytes)

        method_name = f"/{method_name}" if method_name != "" else ""
        base_name = f"{module_name}{method_name}"

        etrecord_zip.writestr(base_name, serialized_artifact.exported_program)
        etrecord_zip.writestr(f"{base_name}_state_dict", serialized_artifact.state_dict)
        etrecord_zip.writestr(f"{base_name}_constants", serialized_artifact.constants)
        etrecord_zip.writestr(
            f"{base_name}_example_inputs", serialized_artifact.example_inputs
        )

    def _save_edge_dialect_program(
        self, etrecord_zip: ZipFile, edge_dialect_program: ExportedProgram
    ) -> None:
        """Save the edge dialect program to the ETRecord zip file."""
        serialized_artifact = serialize(edge_dialect_program)
        assert isinstance(serialized_artifact.exported_program, bytes)

        base_name = ETRecordReservedFileNames.EDGE_DIALECT_EXPORTED_PROGRAM
        etrecord_zip.writestr(base_name, serialized_artifact.exported_program)
        etrecord_zip.writestr(f"{base_name}_state_dict", serialized_artifact.state_dict)
        etrecord_zip.writestr(f"{base_name}_constants", serialized_artifact.constants)
        etrecord_zip.writestr(
            f"{base_name}_example_inputs", serialized_artifact.example_inputs
        )

    def add_extra_export_modules(
        self,
        extra_recorded_export_modules: Dict[
            str,
            Union[
                ExportedProgram,
                ExirExportedProgram,
                EdgeProgramManager,
            ],
        ],
    ) -> None:
        """
        Add extra export modules to the ETRecord after it has been created.

        This method allows users to add more export modules they want to record
        to an existing ETRecord instance. The modules will be added to the graph_map
        and will be included when the ETRecord is saved.

        Args:
            extra_recorded_export_modules: A dictionary of graph modules with the key being
                the user provided name and the value being the corresponding exported module.
                The exported graph modules can be either the output of `torch.export()` or `exir.to_edge()`.
        """
        if self.graph_map is None:
            self.graph_map = {}

        # Now self.graph_map is guaranteed to be non-None
        graph_map = self.graph_map
        for module_name, export_module in extra_recorded_export_modules.items():
            _validate_module_name(module_name)
            _add_module_to_graph_map(graph_map, module_name, export_module)


def _get_reference_outputs(
    bundled_program: BundledProgram,
) -> Dict[str, List[ProgramOutput]]:
    """
    Extracts out the expected outputs from the bundled program, keyed by the method names.
    """
    reference_outputs = {}
    for method_test_suite in bundled_program.method_test_suites:
        reference_outputs[method_test_suite.method_name] = []
        for test_case in method_test_suite.test_cases:
            if not test_case.expected_outputs:
                raise ValueError(
                    f"Missing at least one set of expected outputs for method {method_test_suite.method_name}."
                )
            reference_outputs[method_test_suite.method_name].append(
                test_case.expected_outputs
            )
    return reference_outputs


def _get_representative_inputs(
    bundled_program: BundledProgram,
) -> Optional[List[ProgramInput]]:
    """
    Extracts out the inputs from the bundled program, keyed by the method names.
    """
    for method_test_suite in bundled_program.method_test_suites:
        if method_test_suite.method_name == "forward":
            if not method_test_suite.test_cases:
                raise ValueError(
                    "The 'forward' method is defined, but no corresponding input test cases are provided."
                )
            # Get first example input from the forward method
            test_case = method_test_suite.test_cases[0]
            return test_case.inputs

    # If the forward method is not defined, return None to indicate that there are no representative inputs for the model.
    return None


def generate_etrecord(
    et_record: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    edge_dialect_program: Union[EdgeProgramManager, ExirExportedProgram],
    executorch_program: Union[
        ExecutorchProgram,
        ExecutorchProgramManager,
        BundledProgram,
    ],
    exported_program: Optional[
        Union[ExportedProgram, Dict[str, ExportedProgram]]
    ] = None,
    extra_recorded_export_modules: Optional[
        Dict[
            str,
            Union[
                ExportedProgram,
                ExirExportedProgram,
                EdgeProgramManager,
            ],
        ]
    ] = None,
) -> None:
    """
    Generates an `ETRecord` from the given objects, serializes it and saves it to the given path.
    The objects that will be serialized to an `ETRecord` are all the graph modules present
    in the `extra_recorded_export_modules` dict, the graph module present in the edge dialect program object,
    and also the graph module present in the ExecuTorch program object, which
    is the closest graph module representation of what is eventually run on the device.
    In addition to all the graph modules, we also serialize the program buffer, which the users
    can provide to the ExecuTorch runtime to run the model, and the debug handle map
    for Developer Tools usage.

    Args:
        et_record: Path to where the `ETRecord` file will be saved to.
        edge_dialect_program: `EdgeProgramManager` for this model returned by the call to to_edge()
        executorch_program: The ExecuTorch program for this model returned by the call to `to_executorch()` or the `BundledProgram` of this model
        exported_program: Optional graph module for this model returned by the call to `torch.export` from nn.Module.
        extra_recorded_export_modules [Optional]: **Should be ignored by OSS users**. A dictionary of graph modules with the key being the user provided name and the
            value being the corresponding exported module. The exported graph modules can be either the
            output of `torch.export()` or `exir.to_edge()`.

    Returns:
        None
    """
    # Process all inputs and prepare data for ETRecord construction
    processed_exported_program, export_graph_id = _process_exported_program(
        exported_program
    )
    graph_map = _process_extra_recorded_modules(extra_recorded_export_modules)
    processed_edge_dialect_program = _process_edge_dialect_program(edge_dialect_program)
    debug_handle_map, delegate_map, reference_outputs, representative_inputs = (
        _process_executorch_program(executorch_program)
    )

    # Create ETRecord instance and save
    etrecord = ETRecord(
        exported_program=processed_exported_program,
        export_graph_id=export_graph_id,
        edge_dialect_program=processed_edge_dialect_program,
        graph_map=graph_map if graph_map else None,
        _debug_handle_map=debug_handle_map,
        _delegate_map=delegate_map,
        _reference_outputs=reference_outputs,
        _representative_inputs=representative_inputs,
    )

    etrecord.save(et_record)


def _process_exported_program(
    exported_program: Optional[Union[ExportedProgram, Dict[str, ExportedProgram]]]
) -> tuple[Optional[ExportedProgram], int]:
    """Process exported program and return the processed program and export graph id."""
    processed_exported_program = None
    export_graph_id = 0

    if exported_program is not None:
        if isinstance(exported_program, dict) and "forward" in exported_program:
            processed_exported_program = exported_program["forward"]
        elif isinstance(exported_program, ExportedProgram):
            processed_exported_program = exported_program

        if processed_exported_program is not None:
            export_graph_id = id(processed_exported_program.graph)

    return processed_exported_program, export_graph_id


def _process_extra_recorded_modules(
    extra_recorded_export_modules: Optional[
        Dict[
            str,
            Union[
                ExportedProgram,
                ExirExportedProgram,
                EdgeProgramManager,
            ],
        ]
    ]
) -> Dict[str, ExportedProgram]:
    """Process extra recorded export modules and return graph map."""
    graph_map = {}

    if extra_recorded_export_modules is not None:
        for module_name, export_module in extra_recorded_export_modules.items():
            _validate_module_name(module_name)
            _add_module_to_graph_map(graph_map, module_name, export_module)

    return graph_map


def _validate_module_name(module_name: str) -> None:
    """Validate that module name is not a reserved name."""
    contains_reserved_name = any(
        reserved_name in module_name for reserved_name in ETRecordReservedFileNames
    )
    if contains_reserved_name:
        raise RuntimeError(
            f"The name {module_name} provided in the extra_recorded_export_modules dict is a reserved name in the ETRecord namespace."
        )


def _add_module_to_graph_map(
    graph_map: Dict[str, ExportedProgram],
    module_name: str,
    export_module: Union[ExportedProgram, ExirExportedProgram, EdgeProgramManager],
) -> None:
    """Add export module to graph map based on its type."""
    if isinstance(export_module, ExirExportedProgram):
        graph_map[f"{module_name}/forward"] = export_module.exported_program
    elif isinstance(export_module, ExportedProgram):
        graph_map[f"{module_name}/forward"] = export_module
    elif isinstance(
        export_module,
        (EdgeProgramManager, exir.program._program.EdgeProgramManager),
    ):
        for method in export_module.methods:
            graph_map[f"{module_name}/{method}"] = export_module.exported_program(
                method
            )
    else:
        raise RuntimeError(f"Unsupported graph module type. {type(export_module)}")


def _process_edge_dialect_program(
    edge_dialect_program: Union[EdgeProgramManager, ExirExportedProgram]
) -> ExportedProgram:
    """Process edge dialect program and return the exported program."""
    if isinstance(
        edge_dialect_program,
        (EdgeProgramManager, exir.program._program.EdgeProgramManager),
    ):
        return edge_dialect_program.exported_program()
    elif isinstance(edge_dialect_program, ExirExportedProgram):
        return edge_dialect_program.exported_program
    else:
        raise RuntimeError(
            f"Unsupported type of edge_dialect_program passed in {type(edge_dialect_program)}."
        )


def _process_executorch_program(
    executorch_program: Union[
        ExecutorchProgram, ExecutorchProgramManager, BundledProgram
    ]
) -> tuple[Optional[Dict], Optional[Dict], Optional[Dict], Optional[List]]:
    """Process executorch program and return debug maps and bundled program data."""
    if isinstance(executorch_program, BundledProgram):
        reference_outputs = _get_reference_outputs(executorch_program)
        representative_inputs = _get_representative_inputs(executorch_program)
        # pyre-ignore[16]: Item `None` of `typing.Union[None, exir.program._program.ExecutorchProgram, exir.program._program.ExecutorchProgramManager]` has no attribute `debug_handle_map`
        debug_handle_map = executorch_program.executorch_program.debug_handle_map
        # pyre-ignore[16]: Item `None` of `typing.Union[None, exir.program._program.ExecutorchProgram, exir.program._program.ExecutorchProgramManager]` has no attribute `debug_handle_map`
        delegate_map = executorch_program.executorch_program.delegate_map
        return debug_handle_map, delegate_map, reference_outputs, representative_inputs
    else:
        debug_handle_map = executorch_program.debug_handle_map
        delegate_map = executorch_program.delegate_map
        return debug_handle_map, delegate_map, None, None


def parse_etrecord(etrecord_path: str) -> ETRecord:  # noqa: C901
    """
    Parses an `ETRecord` file and returns an `ETRecord` object that contains the deserialized graph
    modules, program buffer, and a debug handle map.
    In the graph map in the returned `ETRecord` object if a model with multiple entry points was provided
    originally by the user during `ETRecord` generation then each entry point will be stored as a separate
    graph module in the `ETRecord` object with the name being `the original module name + "/" + the
    name of the entry point`.

    Args:
        etrecord_path: Path to the `ETRecord` file.

    Returns:
        `ETRecord` object.
    """

    try:
        etrecord_zip = ZipFile(etrecord_path, "r")
    except BadZipFile:
        raise RuntimeError("Invalid etrecord file passed in.")

    file_list = etrecord_zip.namelist()

    if ETRecordReservedFileNames.ETRECORD_IDENTIFIER not in file_list:
        raise RuntimeError(
            "ETRecord identifier missing from etrecord file passed in. Either an invalid file was passed in or the file is corrupt."
        )

    graph_map: Dict[str, ExportedProgram] = {}
    debug_handle_map = None
    delegate_map = None
    exported_program = None
    edge_dialect_program = None
    reference_outputs = None
    representative_inputs = None
    export_graph_id = 0

    serialized_exported_program_files = set()
    serialized_state_dict_files = set()
    serialized_constants_files = set()
    serialized_example_inputs_files = set()
    for entry in file_list:
        if entry == ETRecordReservedFileNames.DEBUG_HANDLE_MAP_NAME:
            debug_handle_map = json.loads(
                etrecord_zip.read(ETRecordReservedFileNames.DEBUG_HANDLE_MAP_NAME)
            )
        elif entry == ETRecordReservedFileNames.DELEGATE_MAP_NAME:
            delegate_map = json.loads(
                etrecord_zip.read(ETRecordReservedFileNames.DELEGATE_MAP_NAME)
            )
        elif entry == ETRecordReservedFileNames.ETRECORD_IDENTIFIER:
            continue
        elif entry == ETRecordReservedFileNames.EDGE_DIALECT_EXPORTED_PROGRAM:
            serialized_artifact = SerializedArtifact(
                etrecord_zip.read(
                    ETRecordReservedFileNames.EDGE_DIALECT_EXPORTED_PROGRAM
                ),
                etrecord_zip.read(f"{entry}_state_dict"),
                etrecord_zip.read(f"{entry}_constants"),
                etrecord_zip.read(f"{entry}_example_inputs"),
            )
            edge_dialect_program = deserialize(serialized_artifact)
        elif entry == ETRecordReservedFileNames.EXPORTED_PROGRAM:
            serialized_artifact = SerializedArtifact(
                etrecord_zip.read(ETRecordReservedFileNames.EXPORTED_PROGRAM),
                etrecord_zip.read(f"{entry}_state_dict"),
                etrecord_zip.read(f"{entry}_constants"),
                etrecord_zip.read(f"{entry}_example_inputs"),
            )
            exported_program = deserialize(serialized_artifact)
        elif entry == ETRecordReservedFileNames.REFERENCE_OUTPUTS:
            # @lint-ignore PYTHONPICKLEISBAD
            reference_outputs = pickle.loads(
                etrecord_zip.read(ETRecordReservedFileNames.REFERENCE_OUTPUTS)
            )
        elif entry == ETRecordReservedFileNames.REPRESENTATIVE_INPUTS:
            # @lint-ignore PYTHONPICKLEISBAD
            representative_inputs = pickle.loads(
                etrecord_zip.read(ETRecordReservedFileNames.REPRESENTATIVE_INPUTS)
            )
        elif entry == ETRecordReservedFileNames.EXPORT_GRAPH_ID:
            export_graph_id = json.loads(
                etrecord_zip.read(ETRecordReservedFileNames.EXPORT_GRAPH_ID)
            )
        else:
            if entry.endswith("state_dict"):
                serialized_state_dict_files.add(entry)
            elif entry.endswith("constants"):
                serialized_constants_files.add(entry)
            elif entry.endswith("example_inputs"):
                serialized_example_inputs_files.add(entry)
            else:
                serialized_exported_program_files.add(entry)

    for serialized_file in serialized_exported_program_files:
        serialized_state_dict_file = f"{serialized_file}_state_dict"
        serialized_constants_file = f"{serialized_file}_constants"
        serialized_example_inputs_file = f"{serialized_file}_example_inputs"
        assert (
            serialized_state_dict_file in serialized_state_dict_files
        ), f"Could not find corresponding state dict file for {serialized_file}."
        serialized_artifact = SerializedArtifact(
            etrecord_zip.read(serialized_file),
            etrecord_zip.read(serialized_state_dict_file),
            etrecord_zip.read(serialized_constants_file),
            etrecord_zip.read(serialized_example_inputs_file),
        )
        graph_map[serialized_file] = deserialize(serialized_artifact)

    return ETRecord(
        exported_program=exported_program,
        edge_dialect_program=edge_dialect_program,
        graph_map=graph_map,
        _debug_handle_map=debug_handle_map,
        _delegate_map=delegate_map,
        _reference_outputs=reference_outputs,
        _representative_inputs=representative_inputs,
        export_graph_id=export_graph_id,
    )
