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
    INSTRUCTION_ID_TO_NUM_OUTS_MAP_NAME = "instruction_id_to_num_outs_map"
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
        _instruction_id_to_num_outs_map: Optional[
            Dict[str, Dict[int, Union[int, List[int]]]]
        ] = None,
        _reference_outputs: Optional[Dict[str, List[ProgramOutput]]] = None,
        _representative_inputs: Optional[List[ProgramInput]] = None,
    ):
        """
        Please do not construct an ETRecord object directly.

        If you want to create an ETRecord for logging AOT information to further analysis, please mark `generate_etrecord`
        as True in your export api, and get the ETRecord object from the `ExecutorchProgramManager`.
        For exmaple:
        ```python
            exported_program = torch.export.export(model, inputs)
            edge_program = to_edge_transform_and_lower(exported_program, generate_etrecord=True)
            executorch_program = edge_program.to_executorch()
            etrecord = executorch_program.get_etrecord()
        ```

        If user need to create an ETRecord manually, please use the `create_etrecord` function.
        """

        self.exported_program = exported_program
        self.export_graph_id = export_graph_id
        self.edge_dialect_program = edge_dialect_program
        self.graph_map = graph_map
        self._debug_handle_map = _debug_handle_map
        self._delegate_map = _delegate_map
        self._instruction_id_to_num_outs_map = _instruction_id_to_num_outs_map
        self._reference_outputs = _reference_outputs
        self._representative_inputs = _representative_inputs

    def save(self, path: Union[str, os.PathLike, BinaryIO, IO[bytes]]) -> None:
        """
        Serialize and save the ETRecord to the specified path for use in Inspector. The ETRecord
        should contains at least edge dialect program and executorch program information for further
        analysis, otherwise it will raise an exception.

        Args:
            path: Path where the ETRecord file will be saved to.

        Raises:
            RuntimeError: If the ETRecord does not contain essential information for Inpector.
        """
        if isinstance(path, (str, os.PathLike)):
            # pyre-ignore[6]: In call `os.fspath`, for 1st positional argument, expected `str` but got `Union[PathLike[typing.Any], str]`
            path = os.fspath(path)

        if not (self.edge_dialect_program and self._debug_handle_map):
            raise RuntimeError(
                "ETRecord must contain edge dialect program and executorch program to be saved"
            )

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

        if self._instruction_id_to_num_outs_map is not None:
            etrecord_zip.writestr(
                ETRecordReservedFileNames.INSTRUCTION_ID_TO_NUM_OUTS_MAP_NAME,
                json.dumps(self._instruction_id_to_num_outs_map),
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
            _add_module_to_graph_map(graph_map, module_name, export_module)

    def add_executorch_program(
        self,
        executorch_program: Union[
            ExecutorchProgram,
            ExecutorchProgramManager,
            BundledProgram,
        ],
    ) -> None:
        """
        Add executorch program data to the ETRecord after it has been created.

        This method allows users to add executorch program data they want to record
        to an existing ETRecord instance. The executorch program data includes debug handle map,
        delegate map, reference outputs, and representative inputs that will be included
        when the ETRecord is saved.

        Args:
            executorch_program: The ExecuTorch program for this model returned by the call to
                `to_executorch()` or the `BundledProgram` of this model.

        Raises:
            RuntimeError: If executorch program data already exists in the ETRecord.
        """
        # Check if executorch program data already exists
        if (
            self._debug_handle_map is not None
            or self._delegate_map is not None
            or self._instruction_id_to_num_outs_map is not None
            or self._reference_outputs is not None
            or self._representative_inputs is not None
        ):
            raise RuntimeError(
                "Executorch program data already exists in the ETRecord. "
                "Cannot add executorch program data when it already exists."
            )

        # Process executorch program and extract data
        (
            debug_handle_map,
            delegate_map,
            instruction_id_to_num_outs_map,
            reference_outputs,
            representative_inputs,
        ) = _process_executorch_program(executorch_program)

        # Set the extracted data
        self._debug_handle_map = debug_handle_map
        self._delegate_map = delegate_map
        self._instruction_id_to_num_outs_map = instruction_id_to_num_outs_map
        self._reference_outputs = reference_outputs
        self._representative_inputs = representative_inputs

    def add_exported_program(
        self,
        exported_program: Optional[Union[ExportedProgram, Dict[str, ExportedProgram]]],
    ) -> None:
        """
        Add exported program to the ETRecord after it has been created.

        This method allows users to add an exported program they want to record
        to an existing ETRecord instance. The exported program will be included
        when the ETRecord is saved.

        Args:
            exported_program: The exported program for this model returned by the call to
                `torch.export()` or a dictionary with method names as keys and exported programs as values.
                Can be None, in which case no exported program data will be added.

        Raises:
            RuntimeError: If exported program already exists in the ETRecord.
        """
        # Check if exported program already exists
        if self.exported_program is not None or self.export_graph_id is not None:
            raise RuntimeError(
                "Exported program already exists in the ETRecord. "
                "Cannot add exported program when it already exists."
            )

        # Process exported program and extract data
        processed_exported_program, export_graph_id = _process_exported_program(
            exported_program
        )

        # Set the extracted data
        self.exported_program = processed_exported_program
        self.export_graph_id = export_graph_id

    def add_edge_dialect_program(
        self,
        edge_dialect_program: Union[EdgeProgramManager, ExirExportedProgram],
    ) -> None:
        """
        Add edge dialect program to the ETRecord after it has been created.

        This method allows users to add an edge dialect program they want to record
        to an existing ETRecord instance. The edge dialect program will be included
        when the ETRecord is saved.

        Args:
            edge_dialect_program: The edge dialect program for this model returned by the call to
                `to_edge()` or `EdgeProgramManager` for this model.

        Raises:
            RuntimeError: If edge dialect program already exists in the ETRecord.
        """
        # Check if edge dialect program already exists
        if self.edge_dialect_program is not None:
            raise RuntimeError(
                "Edge dialect program already exists in the ETRecord. "
                "Cannot add edge dialect program when it already exists."
            )

        # Process edge dialect program and extract data
        processed_edge_dialect_program = _process_edge_dialect_program(
            edge_dialect_program
        )

        # Set the extracted data
        self.edge_dialect_program = processed_edge_dialect_program

    def update_representative_inputs(
        self,
        representative_inputs: Union[List[ProgramInput], BundledProgram],
    ) -> None:
        """
        Update the representative inputs in the ETRecord.

        This method allows users to customize the representative inputs that will be
        included when the ETRecord is saved. The representative inputs can be provided
        directly as a list or extracted from a BundledProgram.

        Args:
            representative_inputs: Either a list of ProgramInput objects or a BundledProgram
                from which representative inputs will be extracted.
        """
        if isinstance(representative_inputs, BundledProgram):
            self._representative_inputs = _get_representative_inputs(
                representative_inputs
            )
        else:
            self._representative_inputs = representative_inputs

    def update_reference_outputs(
        self,
        reference_outputs: Union[
            Dict[str, List[ProgramOutput]], List[ProgramOutput], BundledProgram
        ],
    ) -> None:
        """
        Update the reference outputs in the ETRecord.

        This method allows users to customize the reference outputs that will be
        included when the ETRecord is saved. The reference outputs can be provided
        directly as a dictionary mapping method names to lists of outputs, as a
        single list of outputs (which will be treated as {"forward": List[ProgramOutput]}),
        or extracted from a BundledProgram.

        Args:
            reference_outputs: Either a dictionary mapping method names to lists of
                ProgramOutput objects, a single list of ProgramOutput objects (treated
                as outputs for the "forward" method), or a BundledProgram from which
                reference outputs will be extracted.
        """
        if isinstance(reference_outputs, BundledProgram):
            self._reference_outputs = _get_reference_outputs(reference_outputs)
        elif isinstance(reference_outputs, list):
            self._reference_outputs = {"forward": reference_outputs}
        else:
            self._reference_outputs = reference_outputs


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
    etrecord = ETRecord()
    etrecord.add_exported_program(exported_program)
    etrecord.add_edge_dialect_program(edge_dialect_program)
    etrecord.add_executorch_program(executorch_program)

    # Add extra export modules if user provided
    if extra_recorded_export_modules is not None:
        etrecord.add_extra_export_modules(extra_recorded_export_modules)

    etrecord.save(et_record)


def _process_exported_program(
    exported_program: Optional[Union[ExportedProgram, Dict[str, ExportedProgram]]]
) -> tuple[Optional[ExportedProgram], Optional[int]]:
    """Process exported program and return the processed program and export graph id."""
    processed_exported_program = None
    export_graph_id = None

    if exported_program is not None:
        if isinstance(exported_program, dict) and "forward" in exported_program:
            processed_exported_program = exported_program["forward"]
        elif isinstance(exported_program, ExportedProgram):
            processed_exported_program = exported_program

        if processed_exported_program is not None:
            export_graph_id = id(processed_exported_program.graph)

    return processed_exported_program, export_graph_id


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
    _validate_module_name(module_name)

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
) -> tuple[
    Optional[Dict], Optional[Dict], Optional[Dict], Optional[Dict], Optional[List]
]:
    """Process executorch program and return debug maps and bundled program data."""
    if isinstance(executorch_program, BundledProgram):
        reference_outputs = _get_reference_outputs(executorch_program)
        representative_inputs = _get_representative_inputs(executorch_program)
        # pyre-ignore[16]: Item `None` of `typing.Union[None, exir.program._program.ExecutorchProgram, exir.program._program.ExecutorchProgramManager]` has no attribute `debug_handle_map`
        debug_handle_map = executorch_program.executorch_program.debug_handle_map
        # pyre-ignore[16]: Item `None` of `typing.Union[None, exir.program._program.ExecutorchProgram, exir.program._program.ExecutorchProgramManager]` has no attribute `debug_handle_map`
        delegate_map = executorch_program.executorch_program.delegate_map
        # pyre-ignore[16]: Item `None` of `typing.Union[None, exir.program._program.ExecutorchProgram, exir.program._program.ExecutorchProgramManager]` has no attribute `instruction_id_to_num_outs_map`
        instruction_id_to_num_outs_map = (
            executorch_program.executorch_program.instruction_id_to_num_outs_map
        )
        return (
            debug_handle_map,
            delegate_map,
            instruction_id_to_num_outs_map,
            reference_outputs,
            representative_inputs,
        )
    else:
        debug_handle_map = executorch_program.debug_handle_map
        delegate_map = executorch_program.delegate_map
        instruction_id_to_num_outs_map = (
            executorch_program.instruction_id_to_num_outs_map
        )
        return (
            debug_handle_map,
            delegate_map,
            instruction_id_to_num_outs_map,
            None,
            None,
        )


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
    instruction_id_to_num_outs_map = None
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
        elif entry == ETRecordReservedFileNames.INSTRUCTION_ID_TO_NUM_OUTS_MAP_NAME:
            instruction_id_to_num_outs_map = json.loads(
                etrecord_zip.read(
                    ETRecordReservedFileNames.INSTRUCTION_ID_TO_NUM_OUTS_MAP_NAME
                )
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
        _instruction_id_to_num_outs_map=instruction_id_to_num_outs_map,
        _reference_outputs=reference_outputs,
        _representative_inputs=representative_inputs,
        export_graph_id=export_graph_id,
    )
