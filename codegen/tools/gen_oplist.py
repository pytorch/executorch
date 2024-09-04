# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import sys
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set

import yaml
from torchgen.executorch.parse import strip_et_fields

from torchgen.gen import LineLoader, parse_native_yaml_struct
from torchgen.selective_build.operator import SelectiveBuildOperator
from torchgen.selective_build.selector import merge_et_kernel_metadata

# Output YAML file format:
# ------------------------
#
# <BEGIN FILE CONTENTS>
# include_all_non_op_selectives: False
# include_all_operators: False
# debug_info:
#   - model1@v100
#   - model2@v50
# operators:
#   aten::add:
#     is_root_operator: Yes
#     is_used_for_training: Yes
#     include_all_overloads: No
#     debug_info:
#       - model1@v100
#       - model2@v50
#   aten::add.int:
#     is_root_operator: No
#     is_used_for_training: No
#     include_all_overloads: Yes
# et_kernel_metadata:
#   aten::add.out:
#     # A list of different kernel keys (tensors with dtype-enum/dim-order) combinations used in model
#       - v1/6;0,1|6;0,1|6;0,1|6;0,1  # Float, 0, 1
#       - v1/3;0,1|3;0,1|3;0,1|3;0,1  # Int, 0, 1
#   aten::mul.out:
#       - v1/6;0,1|6;0,1|6;0,1|6;0,1  # Float, 0, 1
# <END FILE CONTENTS>


class ScalarType(IntEnum):
    Byte = 0
    Char = 1
    Short = 2
    Int = 3
    Long = 4
    Float = 6
    Double = 7
    Bool = 11
    # TODO(jakeszwe): Verify these are unused and then remove support
    QInt8 = 12
    QUInt8 = 13
    QInt32 = 14
    QUInt4X2 = 16
    QUInt2X4 = 17
    # Types currently not implemented.
    # Half = 5
    # ComplexHalf = 8
    # ComplexFloat = 9
    # ComplexDouble = 10
    # BFloat16 = 15


class KernelType(IntEnum):
    TENSOR = 5
    TENSOR_LIST = 10
    OPTIONAL_TENSOR_LIST = 11


def _get_operators(model_file: str) -> List[str]:
    from executorch.codegen.tools.selective_build import (
        _get_program_from_buffer,
        _get_program_operators,
    )

    print("Processing model file: ", model_file)
    with open(model_file, "rb") as f:
        buf = f.read()

    program = _get_program_from_buffer(buf)
    operators = _get_program_operators(program)
    print(f"Model file loaded, operators are: {operators}")
    return operators


def _get_kernel_metadata_for_model(model_file: str) -> Dict[str, List[str]]:

    from executorch.codegen.tools.selective_build import (
        _get_io_metadata_for_program_operators,
        _get_program_from_buffer,
        _IOMetaData,
    )

    with open(model_file, "rb") as f:
        buf = f.read()

    program = _get_program_from_buffer(buf)
    operators_with_io_metadata = _get_io_metadata_for_program_operators(program)

    op_kernel_key_list: Dict[str, List[str]] = {}

    specialized_kernels: Set[List[_IOMetaData]]
    for op_name, specialized_kernels in operators_with_io_metadata.items():
        print(op_name)
        if op_name not in op_kernel_key_list:
            op_kernel_key_list[op_name] = []

        for specialized_kernel in specialized_kernels:
            version = "v1"
            kernel_key = version + "/"
            for io_metadata in specialized_kernel:
                if io_metadata.kernel_type in [
                    KernelType.TENSOR,
                    KernelType.TENSOR_LIST,
                    KernelType.OPTIONAL_TENSOR_LIST,
                ]:
                    dim_order = ",".join(map(str, io_metadata.dim_order))
                    kernel_key += f"{io_metadata.dtype};{dim_order}|"
            op_kernel_key_list[op_name].append(kernel_key[:-1])

    return op_kernel_key_list


def _get_et_kernel_metadata_from_ops_yaml(ops_yaml_path: str) -> Dict[str, List[str]]:
    ops = []
    with open(ops_yaml_path, "r") as f:
        es = yaml.load(f, Loader=LineLoader)
        func_entries = []
        for e in es:
            if "op" in e:
                ops.append(("aten::" if "::" not in e.get("op") else "") + e.get("op"))
            else:
                func_entries.append(e)
        strip_et_fields(es)
        parsed_yaml = parse_native_yaml_struct(
            func_entries, set(), None, path=ops_yaml_path, skip_native_fns_gen=True
        )
    ops.extend([f"{f.namespace}::{f.func.name}" for f in parsed_yaml.native_functions])
    # TODO (larryliu): accept the new op yaml syntax
    return {op: ["default"] for op in ops}


def _dump_yaml(
    op_list: List[str],
    output_path: str,
    model_name: Optional[str] = None,
    et_kernel_metadata: Optional[Dict[str, List[str]]] = None,
    include_all_operators: bool = False,
):
    # no debug info yet
    output = {}
    operators: Dict[str, Dict[str, object]] = {}
    for op_name in op_list:
        op = SelectiveBuildOperator.from_yaml_dict(
            op_name,
            {
                "is_root_operator": True,
                "is_used_for_training": True,
                "include_all_overloads": False,
                "debug_info": [model_name],
            },
        )
        operators[op_name] = op.to_dict()

    output["operators"] = operators
    output["custom_classes"] = []
    output["build_features"] = []
    output["include_all_non_op_selectives"] = False
    output["include_all_operators"] = include_all_operators
    output["kernel_metadata"] = {}
    output["et_kernel_metadata"] = et_kernel_metadata
    with open(output_path, "wb") as out_file:
        out_file.write(
            yaml.safe_dump(
                output,
                default_flow_style=False,
            ).encode("utf-8")
        )


def gen_oplist(
    output_path: str,
    model_file_path: Optional[str] = None,
    ops_schema_yaml_path: Optional[str] = None,
    root_ops: Optional[str] = None,
    ops_dict: Optional[str] = None,
    include_all_operators: bool = False,
):
    assert (
        model_file_path
        or ops_schema_yaml_path
        or root_ops
        or ops_dict
        or include_all_operators
    ), "Need to provide either model_file_path or ops_schema_yaml_path or root_ops or ops_dict or include_all_operators."

    assert output_path, "Need to provide output_path for dumped yaml file."
    op_set = set()
    source_name = None
    et_kernel_metadata = {}
    if root_ops:
        # decide delimiter
        delimiter = "," if "," in root_ops else " "
        print(root_ops)
        op_set.update(
            set(filter(lambda x: len(x) > 0, map(str.strip, root_ops.split(delimiter))))
        )
        et_kernel_metadata = merge_et_kernel_metadata(
            et_kernel_metadata, {op: ["default"] for op in op_set}
        )
    if ops_dict:
        ops_and_metadata = json.loads(ops_dict)
        for op, metadata in ops_and_metadata.items():
            op_set.update({op})
            op_metadata = metadata if len(metadata) > 0 else ["default"]
            et_kernel_metadata = merge_et_kernel_metadata(
                et_kernel_metadata, {op: op_metadata}
            )
    if model_file_path:
        assert os.path.isfile(
            model_file_path
        ), f"The value for --model_file_path needs to be a valid file, got {model_file_path}"
        op_set.update(_get_operators(model_file_path))
        source_name = model_file_path
        et_kernel_metadata = merge_et_kernel_metadata(
            et_kernel_metadata, _get_kernel_metadata_for_model(model_file_path)
        )
    if ops_schema_yaml_path:
        assert os.path.isfile(
            ops_schema_yaml_path
        ), f"The value for --ops_schema_yaml_path needs to be a valid file, got {ops_schema_yaml_path}"
        et_kernel_metadata = merge_et_kernel_metadata(
            et_kernel_metadata,
            _get_et_kernel_metadata_from_ops_yaml(ops_schema_yaml_path),
        )
        op_set.update(et_kernel_metadata.keys())
        source_name = ops_schema_yaml_path
    _dump_yaml(
        sorted(op_set),
        output_path,
        os.path.basename(source_name) if source_name else None,
        et_kernel_metadata,
        include_all_operators,
    )


def main(args: List[Any]) -> None:
    """This binary generates selected_operators.yaml which will be consumed by caffe2/torchgen/gen.py.
    It reads the model file, deserialize it and dumps all the operators into selected_operators.yaml so
    it can be used in gen.py.
    """
    parser = argparse.ArgumentParser(
        description="Generate operator list from a model file"
    )
    parser.add_argument(
        "--output_path",
        help=("The path to the output yaml file (selected_operators.yaml)"),
        required=True,
    )
    parser.add_argument(
        "--model_file_path",
        help=("Path to an executorch program"),
        required=False,
    )
    parser.add_argument(
        "--ops_schema_yaml_path",
        help=("Dump operator names from operator schema yaml path"),
        required=False,
    )
    parser.add_argument(
        "--root_ops",
        help=("A comma separated list of root operators used by the model"),
        required=False,
    )
    parser.add_argument(
        "--ops_dict",
        help=(
            "A json object containing operators and their associated dtype and dim order"
        ),
        required=False,
    )
    parser.add_argument(
        "--include-all-operators",
        "--include_all_operators",
        action="store_true",
        default=False,
        help="Set this flag to request inclusion of all operators (i.e. build is not selective).",
        required=False,
    )
    options = parser.parse_args(args)

    try:
        gen_oplist(
            output_path=options.output_path,
            model_file_path=options.model_file_path,
            ops_schema_yaml_path=options.ops_schema_yaml_path,
            root_ops=options.root_ops,
            ops_dict=options.ops_dict,
            include_all_operators=options.include_all_operators,
        )
    except Exception as e:
        command = ["python codegen/tools/gen_oplist.py"]
        if options.model_file_path:
            command.append(f"--model_file_path {options.model_file_path}")
        if options.ops_schema_yaml_path:
            command.append(f"--ops_schema_yaml_path {options.ops_schema_yaml_path}")
        if options.root_ops:
            command.append(f"--root_ops {options.root_ops}")
        if options.ops_dict:
            command.append(f"--ops_dict {options.ops_dict}")
        if options.include_all_operators:
            command.append("--include-all-operators")
        repro_command = " ".join(command)
        raise RuntimeError(
            f"""Failed to generate selected_operators.yaml. Repro command:
            {repro_command}
            """
        ) from e


if __name__ == "__main__":
    main(sys.argv[1:])
