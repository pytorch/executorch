# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Any, Dict, IO, List, Optional, Set, Tuple

import ruamel.yaml

import torch
from executorch.exir.dialects.edge.dtype.runner import DtypeRunner
from executorch.exir.dialects.edge.dtype.supported import regular_tensor_dtypes_to_str
from executorch.exir.dialects.edge.op.api import get_callable
from executorch.exir.dialects.edge.op.sample_input import SAMPLE_INPUT
from executorch.exir.dialects.edge.spec.utils import (
    get_names_for_args_with_dtype,
    type_aggregrate,
)

from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.opinfo.core import (
    BinaryUfuncInfo,
    generate_elementwise_binary_with_scalar_samples,
)

# Ops taking too long to run
BLOCKLISTED_OPS = [
    "_native_batch_norm_legit_no_training.default",
]

name_to_opinfo = {
    op.aten_name if op.aten_name is not None else op.name: op for op in op_db
}

# pyre-ignore
yaml = ruamel.yaml.YAML()

dtr = DtypeRunner()


class test_case_generator:
    def __init__(
        self,
        preset_types: Dict[torch.dtype, List[torch.dtype]],
        test_case_size: List[List[int]],
        *args,
        **kwargs,
    ):
        self.preset_types = preset_types
        self.test_case_size = test_case_size

        for preset_type in self.preset_types.values():
            if len(preset_type) != len(self.test_case_size):
                raise Exception(
                    "Preset type size does not match test case size, get {} and {}".format(
                        len(preset_type), len(self.test_case_size)
                    )
                )
        self.args = args
        self.kwargs = kwargs

    def get_sample_input(self, dtype: torch.dtype):
        if dtype not in self.preset_types:
            raise Exception(f"Unsupported type {dtype}")

        yield [
            torch.randn(tensor_size).to(preset_type)
            for preset_type, tensor_size in zip(
                self.preset_types[dtype], self.test_case_size
            )
        ] + list(self.args), self.kwargs


preset_test_case_generators: Dict[str, test_case_generator] = {
    "aten::lift_fresh_copy": test_case_generator(
        {
            k: [
                k,
            ]
            for k in regular_tensor_dtypes_to_str
        },
        [
            [3, 4, 10],
        ],
    ),
}


def get_func_schema(op_name_may_with_overload: str) -> torch._C.FunctionSchema:
    """Get the function schema given a op name may or may not have overload name."""
    if "." in op_name_may_with_overload:
        op_name, overload_name = op_name_may_with_overload.rsplit(".", 1)
    else:
        op_name, overload_name = (
            op_name_may_with_overload,
            "",
        )

    func_schemas = torch._C._jit_get_schemas_for_operator(op_name)
    found_overload_names = []
    for func_schema in func_schemas:
        found_overload_names.append(func_schema.overload_name)
        if overload_name == func_schema.overload_name:
            return func_schema

    raise ValueError(
        "Cannot find {} with specific overload {}. All overloads we can find are {}".format(
            op_name, overload_name, found_overload_names
        )
    )


def get_func_name_yaml(func_schema: torch._C.FunctionSchema) -> str:
    """Return the operation name in yaml file given its function schema.
    It should consists operator package name plus operator overload name."""
    return func_schema.name + (
        ".{}".format(func_schema.overload_name) if func_schema.overload_name else ""
    )


def get_test_gen_key(op_name: str) -> str:
    """Map the operator name to key of test case generator.

    The test case generator here can be either the preset test case generator at the top of this file, or an entry of opdb.

    Will raise exception if cannot find the corresponding operator in opdb.
    """
    if op_name in preset_test_case_generators:
        return op_name

    opdb_key = op_name.split("::")[-1].strip("_")
    if opdb_key.endswith("_copy"):
        opdb_key = opdb_key[:-5]
    elif opdb_key == "sym_size":
        opdb_key = "resize_"
    elif opdb_key == "sym_numel":
        opdb_key = "abs"
    elif opdb_key == "convolution":
        opdb_key = "conv_transpose2d"
    elif opdb_key == "embedding":
        opdb_key = "nn.functional.embedding"

    if opdb_key not in name_to_opinfo:
        # current function is unsupported: can not find it in opdb
        raise Exception(
            "Can not find operator {} in the opdb using key {}".format(
                op_name, opdb_key
            )
        )
    return opdb_key


def get_sample_input(key: str, overload_name: str, edge_type: torch.dtype):
    """Given a key and a specific edge_type,
    return a set of testcase for this operator in the certain type"""

    if key in preset_test_case_generators:
        yield next(preset_test_case_generators[key].get_sample_input(edge_type))
    else:
        opdb_key = key
        op_info = name_to_opinfo[opdb_key]
        if overload_name == "Scalar" and isinstance(op_info, BinaryUfuncInfo):
            sample_input = next(
                generate_elementwise_binary_with_scalar_samples(
                    op_info,
                    device=torch.device("cpu"),
                    dtype=edge_type,
                    requires_grad=False,
                )
            )
        else:
            sample_input = next(
                op_info.sample_inputs(
                    torch.device("cpu"), edge_type, required_grad=False
                )
            )
        sample_args = [sample_input.input] + list(sample_input.args)
        sample_kwargs = sample_input.kwargs
        if opdb_key in ["log_softmax", "softmax"]:
            sample_args.append(False)
            sample_kwargs = {}
        elif opdb_key == "resize_":
            sample_args[-1] = 0
        elif opdb_key == "to":
            for dtype in regular_tensor_dtypes_to_str:
                sample_args = sample_args[:1]
                sample_kwargs = {"dtype": dtype}
                yield sample_args, sample_kwargs
        elif opdb_key == "clamp":
            sample_args = sample_args[:1] + [1]
        elif opdb_key == "conv_transpose2d":
            sample_kwargs = {
                "stride": (2, 2),
                "padding": (2, 2),
                "output_padding": (1, 1),
                "groups": 1,
                "dilation": (1, 1),
                "transposed": True,
            }
        elif opdb_key == "split":
            sample_args[1] = 1
        elif opdb_key == "scalar_tensor":
            del sample_kwargs["requires_grad"]
        yield sample_args, sample_kwargs


def in_legal_edge_type(vals: List[Any]) -> bool:
    """Given a list of object, check the tensors in it are in edge type or not.
    Return false if any of it not in edge type; true if otherwise"""
    is_in_legal_type = True
    for val in vals:
        is_in_legal_type = is_in_legal_type and (
            (not isinstance(val, torch.Tensor))
            or val.dtype in regular_tensor_dtypes_to_str
        )
    return is_in_legal_type


def seq(*args):
    """Convert a list into a yaml sequence to make the yaml file more structure."""
    s = ruamel.yaml.comments.CommentedSeq(args)
    s.fa.set_flow_style()
    return s


def print_error_msg(unsupported_funcs: List[str]):
    """Print unsupported funciton name in current model"""
    if unsupported_funcs:
        print("*********************************")
        print(
            "Unsupport following functions, please read the error messages above for details:"
        )
        for f in unsupported_funcs:
            print(f)


def is_not_dype_exception(exc: BaseException, dtype_str: str) -> bool:
    """Check if an exception about unsupported dtype."""

    # alias dtype means the alias name of dtype str, like "Boolean" is the alias name of "Bool".
    # Set default alias_dtype as twice of str(exc) to make sure default alias dtype is not part of str(exc)
    alias_dtype = 2 * str(exc)
    if dtype_str == "Bool":
        alias_dtype = "Boolean"

    return not (
        ("not supported" in str(exc) or "not implemented" in str(exc))
        and (
            dtype_str in str(exc)
            or alias_dtype in str(exc)
            or dtype_str.lower() in str(exc)
        )
    )


class EdgeOpYamlInfo:
    def __init__(
        self,
        func_name: str,
        tensor_variable_names: List[str],
        allowed_types: Set[Tuple[str, ...]],
        inherits: str = "",
        custom: str = "",
    ) -> None:
        """
        Record all information for single function in edge.yaml file
        func_name: name of current Edge function (e.g add.Tensor)
        tensor_variable_names: all names for function's variable in tensor type, including inputs and outputs
            (e.g. self, other, __ret, first two are tensor inputs and the last one is tensor output)
        inherits/custom: the place the function is implemented; if we want to reuse the existing function,
            set inherits as the target function (e.g. aten::add.Tensor); otherwise, set custom as the target
            (e.g. edge::add.Tensor). Noticed that must one and only one of the inherits and custom attribute can be set.
        allowed_types: all combinations of types tensor variables allowed. The length of each list in allow_types should
            be same as number of variables, and each element should be one of the allowed types in string, a.k.a one of
            the values in regular_tensor_dtypes_to_str.
        """

        self.func_name = func_name
        self.tensor_variable_names = tensor_variable_names

        assert bool(inherits) ^ bool(
            custom
        ), "Must set one and only one of the inherits and custom attribute."
        self.inherits = inherits
        self.custom = custom

        assert all(
            len(self.tensor_variable_names) == len(type_combination)
            for type_combination in allowed_types
        ), "{}'s tensor_variable_names length must be the same as number of allowed types, but got {} vs {}: {}.".format(
            self.inherits,
            self.tensor_variable_names,
            allowed_types,
            [
                len(self.tensor_variable_names) == type_combination
                for type_combination in allowed_types
            ],
        )

        self.type_alias, self.type_constraint = type_aggregrate(allowed_types)

    def to_yaml(self) -> Dict[str, Any]:
        """Convert self to a dicitionary for yaml lib to dump"""
        try:
            impl_source_key = "inherits" if self.inherits else "custom"
            impl_source_value = self.inherits if self.inherits else self.custom

            type_alias_yaml = {
                "T{}".format(i): seq(*sorted(set(ts)))
                for i, ts in enumerate(self.type_alias)
            }
            type_constraint_yaml = [
                {
                    self.tensor_variable_names[tensor_idx]: "T{}".format(type_idx)
                    for tensor_idx, type_idx in enumerate(self.type_constraint[j])
                }
                for j in range(len(self.type_constraint))
            ]

            yaml_dict: Dict[str, Any] = {
                "func": self.func_name,
                "namespace": "edge",
                impl_source_key: impl_source_value,
                "type_alias": type_alias_yaml,
                "type_constraint": type_constraint_yaml,
            }
            return yaml_dict
        except BaseException:
            print(
                "Operator {} inherited from {} failed convert to yaml".format(
                    self.func_name, self.inherits
                )
            )
            print(self)
            return {}

    def __str__(self) -> str:
        my_str: str = "\nop_yaml_info: \n"
        my_str += "name: {}\n".format(self.func_name)
        my_str += "tensor_variable_names: {}\n".format(self.tensor_variable_names)
        my_str += "inherits: {}\n".format(self.inherits)
        my_str += "custom: {}\n".format(self.custom)
        my_str += "type_alias: {}\n".format(self.type_alias)
        my_str += "type_constraint: {}\n".format(self.type_constraint)
        return my_str


class EdgeYamlInfo:
    def __init__(self):
        """
        All info for a single edge dialect yaml file.
        """
        self.all_op_yaml_info: List[EdgeOpYamlInfo] = []

    def append(self, op_yaml_info: EdgeOpYamlInfo) -> None:
        self.all_op_yaml_info.append(op_yaml_info)

    def to_yaml(self, yaml_stream: IO) -> List[str]:
        tag = "generated"
        heading = f"# @{tag} by //executorch/exir/dialects/edge/spec/gen.py\n\n"

        yaml_stream.write(heading)
        yaml_stream.write(
            "# This yaml file is auto-generated by //executorch/exir/dialects/edge/spec/gen.py\n"
        )
        yaml_stream.write("# Please do not update it manually.\n")
        yaml_stream.write(
            "# If anything is not up-to-date, please rerun the binary target. Optional argument: --regenerate.\n"
        )

        yaml_list: List[Dict[str, Any]] = []
        failed_operator: List[str] = []
        for op_yaml_info in self.all_op_yaml_info:
            op_yaml = op_yaml_info.to_yaml()
            if op_yaml:
                yaml_list.append(op_yaml)
            else:
                failed_operator.append(op_yaml_info.inherits)

        yaml_list = sorted(yaml_list, key=lambda d: d["func"])

        for idx, op_yaml in enumerate(yaml_list):
            yaml.dump(
                [
                    op_yaml,
                ],
                yaml_stream,
            )
            if idx != len(yaml_list) - 1:
                yaml_stream.write("\n")

        return failed_operator

    def _str__(self) -> str:
        return "\n\n".join(list(map(str, self.all_op_yaml_info)))


def try_all_dtypes_input_samples(
    op_name: str,
) -> Set[Tuple[str]]:
    """Input samples given test generate key in all possible dtypes on given operation"""
    valid_type_combinations: Set[Tuple[str, ...]] = set()
    assert (
        op_name in SAMPLE_INPUT
    ), f"{op_name} does not have a sample input in SAMPLE_INPUT."
    inputs = SAMPLE_INPUT[op_name]
    sample_args: List[Any] = []
    sample_kwargs: Dict[Any, Any] = {}

    result = dtr.run(op_name, inputs)
    for success, _, valid_dtypes, _, _ in result:
        if success and not any(dtype is None for dtype in valid_dtypes):
            valid_type_combinations.add(
                tuple(regular_tensor_dtypes_to_str[t] for t in valid_dtypes)
            )
    if not valid_type_combinations:
        # current function is unsupported: error test case from opdb
        print(
            f"{op_name} is unsupported: no legal test case has been found from runner.py"
        )
        if (not sample_args) and (not sample_kwargs):
            print("Can not get sample input case.")
        else:
            print("One of the sample inputs is", sample_args, sample_kwargs)
    return valid_type_combinations


def gen_op_yaml(op_name: str) -> Optional[EdgeOpYamlInfo]:
    """Generate yaml info for given operator.
    Arguments:
        op_name: The name of operator. Needs to conform the convention of "<name>.<overload_name>".
                If no overload name for the operator, needs to use "default" as overload name.
    Return the yaml info for given operator if generation succeed. Otherwise return None.
    """

    try:
        func_schema: torch._C.FunctionSchema = get_callable(op_name)._schema
    except BaseException as e:
        # Can not find operator schema, or can not find operator based on op_name.
        # Return None to append it into unsupport_funcs and skip.
        raise RuntimeError(f"Can not find operator schema for {op_name}") from e

    valid_type_combinations = try_all_dtypes_input_samples(op_name)

    if not valid_type_combinations:
        return

    func_name_yaml = get_func_name_yaml(func_schema)
    tensor_variable_names = get_names_for_args_with_dtype(op_name, func_schema)
    inherits = func_schema.name + (
        ".{}".format(func_schema.overload_name) if func_schema.overload_name else ""
    )

    try:
        op_yaml_info = EdgeOpYamlInfo(
            func_name=func_name_yaml,
            tensor_variable_names=tensor_variable_names,
            inherits=inherits,
            allowed_types=valid_type_combinations,
        )
    except BaseException as e:
        # Failed to create yaml file for current function.
        # Append it to unsupported_funcs.
        print("Failed to create yaml file for current function:", op_name)
        print("Error msg:", str(e))
        return

    return op_yaml_info


def gen_edge_yaml(op_names: List[str], yaml_out_stream: IO) -> List[str]:
    """Generate yaml file of edge dialect operators for target model.

    Given a list of operator names, generate a yaml file edge.yaml that describes all allowed tensor dtypes for those operators.

    Args:
        op_names: The list of operator names.
        yaml_out_stream: The place the yaml file will be stored. e.g. a file.

    Returns:
        A list of incompatible operators that can not be auto-generated.

    """

    print("************************************************************")
    print("These are ops used by current model: ")
    print(op_names)
    print("************************************************************")

    edge_yaml_info = EdgeYamlInfo()

    # Record all functions in the model whose yaml file can not be auto-generated.
    unsupported_funcs: List[str] = []

    for i, op_name in enumerate(op_names):
        ret = gen_op_yaml(op_name)
        if ret is None:
            # Skip this op. Return None means it cannot be auto-generated
            print(f"Skipping op ({i+1}/{len(op_names)}): {op_name}")
            unsupported_funcs.append(op_name)
        else:
            print(
                f"Generating dtype constraints for op ({i+1}/{len(op_names)}): {op_name}"
            )
            # Append the generated yaml info for op to edge_yaml_info
            edge_yaml_info.append(ret)

    unsupported_funcs += edge_yaml_info.to_yaml(yaml_out_stream)
    return unsupported_funcs


def main():
    parser = argparse.ArgumentParser(
        description="Generate allowed tensor dtypes for core ATen ops"
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Whether to regenerate edge.yaml, based on all edge ops used in ASR models. By default we reuses operators in existing edge.yaml file.",
    )
    options = parser.parse_args()

    yaml_path = "executorch/exir/dialects/edge/edge.yaml"
    if options.regenerate:
        # TODO(larryliu0820): Use all core ATen ops here.
        op_names = [op for op in SAMPLE_INPUT.keys() if op not in BLOCKLISTED_OPS]
    else:
        with open(yaml_path, "r") as f:
            obj = yaml.load(f)
            if not obj:
                raise Exception("YAML file is empty!")
            op_names = [e["inherits"] for e in obj]

    with open(yaml_path, "w") as stream:
        unsupported_funcs = gen_edge_yaml(op_names, stream)
    print_error_msg(unsupported_funcs)


if __name__ == "__main__":
    main()
