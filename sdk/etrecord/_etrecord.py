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

from executorch.exir import (
    ExecutorchProgram,
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
    ET_DIALECT_GRAPH_MODULE = "et_dialect_graph_module"
    DEBUG_HANDLE_MAP_NAME = "debug_handle_map"
    DELEGATE_MAP_NAME = "delegate_map"


@dataclass
class ETRecord:
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
    export_module: Union[MultiMethodExirExportedProgram, ExirExportedProgram],
    module_name: str,
) -> None:
    if isinstance(export_module, MultiMethodExirExportedProgram):
        _handle_multi_method_exported_program(etrecord_zip, module_name, export_module)
    elif isinstance(export_module, ExirExportedProgram):
        _handle_exported_program(
            etrecord_zip, module_name, "forward", export_module.exported_program
        )
    else:
        raise RuntimeError(f"Unsupported graph module type. {type(export_module)}")


def _handle_program(
    etrecord_zip: ZipFile,
    program: Union[ExecutorchProgram, MultiMethodExecutorchProgram],
) -> None:
    if isinstance(program, MultiMethodExecutorchProgram):
        # Do a dummy read of the program here to make sure that the emitter runs
        # under the hood which will result in the debug handle map being generated.
        program.program

        _handle_multi_method_exported_program(
            etrecord_zip,
            ETRecordReservedFileNames.ET_DIALECT_GRAPH_MODULE,
            program._executorch_dialect_ir_program,
        )

    elif isinstance(program, ExecutorchProgram):
        # Do a dummy read of the program here to make sure that the emitter runs
        # under the hood which will result in the debug handle map being generated.
        program.program

        _handle_exported_program(
            etrecord_zip,
            ETRecordReservedFileNames.ET_DIALECT_GRAPH_MODULE,
            "forward",
            program.dump_exported_program(),
        )

        etrecord_zip.writestr(ETRecordReservedFileNames.PROGRAM_BUFFER, program.buffer)

    else:
        raise RuntimeError(
            f"program passed in should be either ExecutorchProgram or MultiMethodExecutorchProgram. {type(program)}"
        )


def generate_etrecord(
    etrecord_path: str,
    program: Optional[Union[ExecutorchProgram, MultiMethodExecutorchProgram]] = None,
    export_modules: Optional[
        Dict[
            str,
            Union[MultiMethodExirExportedProgram, ExirExportedProgram],
        ]
    ] = None,
) -> None:
    """
    Generates an ETRecord from the given objects and saves it to the given path.
    The objects that will be serialized to an ETRecord are all the graph modules present
    in the export_modules dict and also the graph module present in the program object, which
    is the closest graph module representation of what is eventually run on the device.
    In addition to all the graph modules we also serialize the program buffer which the users
    can provide to the ExecuTorch runtime to run the model and we also serialize the debug handle map
    for SDK tooling usage.

    Args:
        etrecord_path: Path to where the ETRecord file will be saved to.
        program: ExecutorchProgram or MultiMethodExecutorchProgram for this model returned by the
            call to to_executorch()
        export_modules: Dictionary of graph modules with the key being the user provided name and the
            value is the corresponding exported module. The exported graph modules can be either the
            output of capture() or to_edge().
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

    if program is not None:
        _handle_program(etrecord_zip, program)

        etrecord_zip.writestr(
            ETRecordReservedFileNames.DEBUG_HANDLE_MAP_NAME,
            json.dumps(program.debug_handle_map),
        )

        etrecord_zip.writestr(
            ETRecordReservedFileNames.DELEGATE_MAP_NAME,
            json.dumps(program.delegate_map),
        )


def parse_etrecord(etrecord_path: str) -> ETRecord:
    """
    Parses an ETRecord file and returns a ETRecord object that contains the deserialized graph
    modules, program buffer and debug handle map.
    In the graph map in the returned ETRecord object if a model with multiple entry points was provided
    originally by the user during ETRecord generation then each entry point will be stored as a separate
    graph module in the ETRecord object with the name being the original module name + "/" + the
    name of the entry point.
    Args:
        etrecord_path: Path to the ETRecord file.
    Returns:
        ETRecord object
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
        graph_map=graph_map,
        program_buffer=program_buffer,
        _debug_handle_map=debug_handle_map,
        _delegate_map=delegate_map,
    )
