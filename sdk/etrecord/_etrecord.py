# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union
from zipfile import BadZipFile, ZipFile

from executorch import exir
from executorch.exir import (
    EdgeProgramManager,
    ExecutorchProgram,
    ExecutorchProgramManager,
    ExirExportedProgram,
    ExportedProgram,
    MultiMethodExecutorchProgram,
    MultiMethodExirExportedProgram,
)
from executorch.exir.emit._emitter import _DelegateDebugIdentifierMap
from executorch.exir.serde.serialize import deserialize, serialize


class ETRecordReservedFileNames(str, Enum):
    ETRECORD_IDENTIFIER = "ETRECORD_V0"
    PROGRAM_BUFFER = "program_buffer"
    EDGE_DIALECT_EXPORTED_PROGRAM = "edge_dialect_exported_program"
    ET_DIALECT_GRAPH_MODULE = "et_dialect_graph_module"
    DEBUG_HANDLE_MAP_NAME = "debug_handle_map"
    DELEGATE_MAP_NAME = "delegate_map"


@dataclass
class ETRecord:
    edge_dialect_program: Optional[ExportedProgram] = None
    graph_map: Optional[Dict[str, ExportedProgram]] = None
    program_buffer: Optional[bytes] = None
    _debug_handle_map: Optional[Dict[int, Union[int, List[int]]]] = None
    _delegate_map: Optional[
        Dict[str, Dict[int, Dict[str, Union[str, _DelegateDebugIdentifierMap]]]]
    ] = None


def _handle_exported_program(
    etrecord_zip: ZipFile, module_name: str, method_name: str, ep: ExportedProgram
) -> None:
    assert isinstance(ep, ExportedProgram)
    serialized_ep, serialized_state_dict = serialize(ep)
    etrecord_zip.writestr(f"{module_name}/{method_name}", serialized_ep)
    etrecord_zip.writestr(
        f"{module_name}/{method_name}_state_dict", serialized_state_dict
    )


def _handle_multi_method_exported_program(
    etrecord_zip: ZipFile,
    module_name: str,
    multi_method: MultiMethodExirExportedProgram,
) -> None:
    for method_name, ep in multi_method.methods().items():
        _handle_exported_program(
            etrecord_zip, module_name, method_name, ep.exported_program
        )


def _handle_export_module(
    etrecord_zip: ZipFile,
    export_module: Union[
        MultiMethodExirExportedProgram, ExirExportedProgram, EdgeProgramManager
    ],
    module_name: str,
) -> None:
    if isinstance(export_module, MultiMethodExirExportedProgram):
        _handle_multi_method_exported_program(etrecord_zip, module_name, export_module)
    elif isinstance(export_module, ExirExportedProgram):
        _handle_exported_program(
            etrecord_zip, module_name, "forward", export_module.exported_program
        )
    elif isinstance(
        export_module,
        (EdgeProgramManager, exir.program._program.EdgeProgramManager),
    ):
        for method in export_module.methods:
            _handle_exported_program(
                etrecord_zip,
                module_name,
                method,
                export_module.exported_program(method),
            )
    else:
        raise RuntimeError(f"Unsupported graph module type. {type(export_module)}")


def _handle_edge_dialect_exported_program(
    etrecord_zip: ZipFile, edge_dialect_exported_program: ExportedProgram
) -> None:
    serialized_ep, serialized_state_dict = serialize(edge_dialect_exported_program)

    etrecord_zip.writestr(
        ETRecordReservedFileNames.EDGE_DIALECT_EXPORTED_PROGRAM,
        serialized_ep,
    )
    etrecord_zip.writestr(
        f"{ETRecordReservedFileNames.EDGE_DIALECT_EXPORTED_PROGRAM}_state_dict",
        serialized_state_dict,
    )


def generate_etrecord(
    etrecord_path: str,
    edge_dialect_program: Union[EdgeProgramManager, ExirExportedProgram],
    executorch_program: Union[
        ExecutorchProgram, MultiMethodExecutorchProgram, ExecutorchProgramManager
    ],
    export_modules: Optional[
        Dict[
            str,
            Union[
                MultiMethodExirExportedProgram, ExirExportedProgram, EdgeProgramManager
            ],
        ]
    ] = None,
) -> None:
    """
    Generates an `ETRecord` from the given objects, serializes it and saves it to the given path.
    The objects that will be serialized to an `ETRecord` are all the graph modules present
    in the `export_modules` dict, the graph module present in the edge dialect program object,
    and also the graph module present in the ExecuTorch program object, which
    is the closest graph module representation of what is eventually run on the device.
    In addition to all the graph modules, we also serialize the program buffer, which the users
    can provide to the ExecuTorch runtime to run the model, and the debug handle map
    for SDK tooling usage.

    Args:
        etrecord_path: Path to where the `ETRecord` file will be saved to.
        edge_dialect_program: `EdgeProgramManager` for this model returned by the call to to_edge()
        executorch_program: `ExecutorchProgramManager` for this model returned by the call to `to_executorch()`
        export_modules[Optional]: **Should be ignored by OSS users**. A dictionary of graph modules with the key being the user provided name and the
            value being the corresponding exported module. The exported graph modules can be either the
            output of `capture()` or `to_edge()`.

    Returns:
        None
    """

    etrecord_zip = ZipFile(etrecord_path, "w")
    # Write the magic file identifier that will be used to verify that this file
    # is an etrecord when it's used later in the SDK tooling.
    etrecord_zip.writestr(ETRecordReservedFileNames.ETRECORD_IDENTIFIER, "")

    if export_modules is not None:
        for module_name, export_module in export_modules.items():
            contains_reserved_name = any(
                reserved_name in module_name
                for reserved_name in ETRecordReservedFileNames
            )
            if contains_reserved_name:
                raise RuntimeError(
                    f"The name {module_name} provided in the export_modules dict is a reserved name in the ETRecord namespace."
                )
            _handle_export_module(etrecord_zip, export_module, module_name)

    if isinstance(
        edge_dialect_program,
        (EdgeProgramManager, exir.program._program.EdgeProgramManager),
    ):
        _handle_edge_dialect_exported_program(
            etrecord_zip,
            edge_dialect_program.exported_program(),
        )
    elif isinstance(edge_dialect_program, ExirExportedProgram):
        _handle_edge_dialect_exported_program(
            etrecord_zip,
            edge_dialect_program.exported_program,
        )
    else:
        raise RuntimeError(
            f"Unsupported type of edge_dialect_program passed in {type(edge_dialect_program)}."
        )

    etrecord_zip.writestr(
        ETRecordReservedFileNames.DEBUG_HANDLE_MAP_NAME,
        json.dumps(executorch_program.debug_handle_map),
    )

    etrecord_zip.writestr(
        ETRecordReservedFileNames.DELEGATE_MAP_NAME,
        json.dumps(executorch_program.delegate_map),
    )


def parse_etrecord(etrecord_path: str) -> ETRecord:
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
    program_buffer = None
    edge_dialect_program = None

    serialized_exported_program_files = set()
    serialized_state_dict_files = set()
    for entry in file_list:
        if entry == ETRecordReservedFileNames.DEBUG_HANDLE_MAP_NAME:
            debug_handle_map = json.loads(
                etrecord_zip.read(ETRecordReservedFileNames.DEBUG_HANDLE_MAP_NAME)
            )
        elif entry == ETRecordReservedFileNames.DELEGATE_MAP_NAME:
            delegate_map = json.loads(
                etrecord_zip.read(ETRecordReservedFileNames.DELEGATE_MAP_NAME)
            )
        elif entry == ETRecordReservedFileNames.PROGRAM_BUFFER:
            program_buffer = etrecord_zip.read(ETRecordReservedFileNames.PROGRAM_BUFFER)
        elif entry == ETRecordReservedFileNames.ETRECORD_IDENTIFIER:
            continue
        elif entry == ETRecordReservedFileNames.EDGE_DIALECT_EXPORTED_PROGRAM:
            edge_dialect_program = deserialize(
                etrecord_zip.read(
                    ETRecordReservedFileNames.EDGE_DIALECT_EXPORTED_PROGRAM
                ),
                etrecord_zip.read(f"{entry}_state_dict"),
            )
        else:
            if entry.endswith("state_dict"):
                serialized_state_dict_files.add(entry)
            else:
                serialized_exported_program_files.add(entry)

    for serialized_file in serialized_exported_program_files:
        serialized_state_dict_file = f"{serialized_file}_state_dict"
        assert (
            serialized_state_dict_file in serialized_state_dict_files
        ), "Could not find corresponding state dict file for {serialized_file}."
        graph_map[serialized_file] = deserialize(
            etrecord_zip.read(serialized_file),
            etrecord_zip.read(serialized_state_dict_file),
        )

    return ETRecord(
        edge_dialect_program=edge_dialect_program,
        graph_map=graph_map,
        program_buffer=program_buffer,
        _debug_handle_map=debug_handle_map,
        _delegate_map=delegate_map,
    )
