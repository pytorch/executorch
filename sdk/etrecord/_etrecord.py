import json
import pickle
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union
from zipfile import BadZipFile, ZipFile

import torch

from executorch.exir import (
    EdgeDialectProgram,
    ExecutorchProgram,
    ExirExportedProgram,
    MultiMethodExecutorchProgram,
    MultiMethodExirExportedProgram,
)


class ETRecordReservedFileNames(str, Enum):
    ETRECORD_IDENTIFIER = "ETRECORD_V0"
    PROGRAM_BUFFER = "program_buffer"
    ET_DIALECT_GRAPH_MODULE = "et_dialect_graph_module"
    DEBUG_HANDLE_MAP_NAME = "debug_handle_map"


@dataclass
class ETRecord:
    graph_map: Optional[Dict[str, torch.fx.GraphModule]] = None
    program_buffer: Optional[bytes] = None
    _debug_handle_map: Optional[Dict[int, Union[int, List[int]]]] = None


def get_export_module_handler(
    etrecord_zip: ZipFile,
    export_module: Union[
        EdgeDialectProgram, MultiMethodExirExportedProgram, ExirExportedProgram
    ],
):
    export_module_handlers = {
        EdgeDialectProgram: lambda module_name, export_module: etrecord_zip.writestr(
            module_name, pickle.dumps(export_module.graph_module)
        ),
        MultiMethodExirExportedProgram: lambda module_name, export_module: [
            etrecord_zip.writestr(
                module_name + "/" + method_name, pickle.dumps(graph_module)
            )
            for method_name, graph_module in export_module.methods().items()
        ],
        ExirExportedProgram: lambda module_name, export_module: etrecord_zip.writestr(
            module_name, pickle.dumps(export_module.graph_module)
        ),
    }

    handler = export_module_handlers.get(type(export_module))
    return handler


def get_program_handler(
    etrecord_zip: ZipFile,
    program: Union[ExecutorchProgram, MultiMethodExecutorchProgram],
):
    program_handlers = {
        ExecutorchProgram: lambda program: [
            etrecord_zip.writestr(
                ETRecordReservedFileNames.ET_DIALECT_GRAPH_MODULE + "/" + "forward",
                pickle.dumps(program.dump_graph_module()),
            ),
            etrecord_zip.writestr(
                ETRecordReservedFileNames.PROGRAM_BUFFER, program.buffer
            ),
        ],
        MultiMethodExecutorchProgram: lambda program: [
            etrecord_zip.writestr(
                ETRecordReservedFileNames.ET_DIALECT_GRAPH_MODULE + "/" + method_name,
                pickle.dumps(graph_module),
            )
            for method_name, graph_module in program._executorch_dialect_ir_program.methods().items()
        ],
    }

    handler = program_handlers.get(type(program))
    return handler


def generate_etrecord(
    etrecord_path: str,
    program: Optional[Union[ExecutorchProgram, MultiMethodExecutorchProgram]] = None,
    export_modules: Optional[
        Dict[
            str,
            Union[
                EdgeDialectProgram, MultiMethodExirExportedProgram, ExirExportedProgram
            ],
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
            handler = get_export_module_handler(etrecord_zip, export_module)
            if handler:
                handler(module_name, export_module)
            else:
                raise RuntimeError(
                    f"Unsupported graph module type. {type(export_module)}"
                )

    if program is not None:
        handler = get_program_handler(etrecord_zip, program)
        if handler:
            # Do a dummy read of the program here to make sure that the emitter runs
            # under the hood which will result in the debug handle map being generated.
            program.program
            handler(program)
        else:
            raise RuntimeError(
                f"program passed in should be either ExecutorchProgram or MultiMethodExecutorchProgram. {type(program)}"
            )

        etrecord_zip.writestr(
            ETRecordReservedFileNames.DEBUG_HANDLE_MAP_NAME,
            json.dumps(program.debug_handle_map),
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

    graph_map: Dict[str, torch.fx.GraphModule] = {}
    debug_handle_map = None
    program_buffer = None

    for entry in file_list:
        if entry == ETRecordReservedFileNames.DEBUG_HANDLE_MAP_NAME:
            debug_handle_map = json.loads(
                etrecord_zip.read(ETRecordReservedFileNames.DEBUG_HANDLE_MAP_NAME)
            )
        elif entry == ETRecordReservedFileNames.PROGRAM_BUFFER:
            program_buffer = etrecord_zip.read(ETRecordReservedFileNames.PROGRAM_BUFFER)
        elif entry == ETRecordReservedFileNames.ETRECORD_IDENTIFIER:
            continue
        else:
            graph_map[entry] = pickle.loads(etrecord_zip.read(entry))

    return ETRecord(
        graph_map=graph_map,
        program_buffer=program_buffer,
        _debug_handle_map=debug_handle_map,
    )
