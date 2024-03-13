# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Set, Union

import pkg_resources

import torch

from executorch.exir.dialects.edge.dtype.supported import regular_tensor_str_to_dtypes
from executorch.exir.dialects.edge.op.api import to_variant
from executorch.exir.dialects.edge.spec.utils import get_tensor_variable_names

# pyre-ignore
from ruamel.yaml import YAML
from torchgen.model import SchemaKind


class AllowedDtypeSet:
    """All legal dtypes for current type alias.

    This class is a wrapper of Set[torch.dtype]. Normally it is a set of all legal types listed in
    edge/edge.yaml file for each type alias. If one of the argument under the type alias receiving
    its actual type, AllowedDtypeSet will be degenerated to the set of only the actual type.

    TODO(gasoonjia): Prevent users from misusing.

    Public Attributes:
        types: a set of all allowed dtypes listed in edge/edge.yaml.

    Private Attributes:
        _reduced_type: the actual type this type alias currently represents. 0 means unrestricted,
                       each type in self.types is legal.

    """

    def __init__(self, types: Set[torch.dtype]):
        self.types: Set[torch.dtype] = types
        self._reduced_type: Union[torch.dtype, int] = 0

    def reduce_to(self, t: torch.dtype) -> bool:
        """Reduce the legal dtype to given t.
        t must be a legal type for this type alias.

        return True if reduction succeed; otherwise False.
        """
        if self.__contains__(t):
            self._reduced_type = t
            return True
        else:
            return False

    def clear(self):
        """Derestrict AllowedDtypeSet to all allowed dtypes in yaml."""
        self._reduced_type = 0

    def __contains__(self, key: torch.dtype):
        """Check if key is a legal type of this type alias"""
        if self._reduced_type:
            return key == self._reduced_type
        return key in self.types


class FunctionDtypeConstraint:
    """Dtype constraint for each EdgeDialect ops.

    Arguments:
        essential_tensor_io_names: All names of essential tensor inputs and outputs.
        optional_tensor_io_names: All names of optional tensor inputs.
        type_alias: Dict of type alias name to corresponding list of dtypes.
        type_constraint: List of dict containing dtype constraint represented in type alias for each arg name.
    """

    def __init__(
        self,
        essential_tensor_io_names: List[str],
        optional_tensor_io_names: List[str],
        type_alias: Dict[str, List[torch.dtype]],
        type_constraint: List[Dict[str, str]],
    ):
        self.essential_tensor_io_names: List[str] = essential_tensor_io_names
        self.optional_tensor_io_names: List[str] = optional_tensor_io_names
        self.type_alias: Dict[str, AllowedDtypeSet] = {
            alias: AllowedDtypeSet(set(types)) for alias, types in type_alias.items()
        }
        self.type_constraint: List[Dict[str, str]] = type_constraint
        # type_constraint's non return entries should include all tensor-like arguments.
        for t_constraint in self.type_constraint:
            type_constraint_names = set(t_constraint)
            all_tensor_arg_names = set(
                self.essential_tensor_io_names + self.optional_tensor_io_names
            )
            if not all_tensor_arg_names.issubset(type_constraint_names):
                raise RuntimeError(
                    "Input entries of type_constraint must contain all tensor-like arguments, "
                    + f"but get {type_constraint_names} and {all_tensor_arg_names}"
                )

    def validate(self, types: Dict[str, Optional[torch.dtype]]) -> bool:
        """Check if the given input type combination a legal one of current function.

        Args:
            types: A dict of arg name to its current dtype.

        Returns:
            True iff a. types are legal for current operator b. all arg name can be found
            in current operator and c. input contains all essential tensor inputs; False otherwise.

            The essential tensor inputs here mean non-optional inputs in tensor and tensor list.
        """

        # Every arg name in `types` should be one of the tensor ios in current function.
        for arg_name in types:
            if not self.__contains__(arg_name):
                return False

        # Any essential tensor input should exist in current `type` input.
        for io_name in self.essential_tensor_io_names:
            if io_name not in types:
                return False

        valid_type = False
        for constraint in self.type_constraint:
            if valid_type:
                break

            valid_type = True
            # Narrow down the type_alias based on contraint and actual input
            for arg_name, arg_type in types.items():
                if arg_type is None:
                    # None means the user didn't set dtype for this argment
                    # (i.e. empty tensorlist), skipping the validation.
                    continue
                elif arg_type in self.type_alias[constraint[arg_name]]:
                    self.type_alias[constraint[arg_name]].reduce_to(arg_type)
                else:
                    valid_type = False
                    break

            for alias in self.type_alias.values():
                alias.clear()

        return valid_type

    def __contains__(self, key: str):
        return key in self.type_constraint[0]

    def __getitem__(self, arg_name: str) -> Set[torch.dtype]:
        """Return all legal types for given arg name.
        Return all its legal type in a set, or an empty set if can not find
        the arg_name in current function."""

        if arg_name not in self.type_constraint[0]:
            return set()

        valid_dtype: Set[torch.dtype] = set()
        for constraint in self.type_constraint:
            valid_dtype = self.type_alias[constraint[arg_name]].types | valid_dtype

        return valid_dtype


def _load_edge_dialect_info() -> Dict[str, Dict[str, Any]]:
    # pyre-ignore
    yaml = YAML(typ="safe")
    edge_dialect_yaml_info = yaml.load(
        pkg_resources.resource_string(__name__, "edge.yaml").decode("utf8")
    )
    if edge_dialect_yaml_info:
        return {
            edge_op_yaml_info["inherits"]: edge_op_yaml_info
            for edge_op_yaml_info in edge_dialect_yaml_info
        }
    else:
        return {}


_edge_dialect_info: Dict[str, Dict[str, Any]] = _load_edge_dialect_info()


class EdgeDialectArgument:
    """Argument class for EdgeDialect ops.
    Wraps around torch._C.Argument with dtype constraints.
    Redirects all `getattr` calls to torch._C.Argument.
    """

    def __init__(self, argument: torch._C.Argument, allowed_types: Set[torch.dtype]):
        self.argument = argument
        self.allowed_types = allowed_types

    def __getattr__(self, name):
        if name == "allowed_types":  # arg.allowed_types
            return self.allowed_types
        return getattr(self.argument, name)


class EdgeDialectFunctionSchema:
    """FunctionSchema class for EdgeDialect ops.
    Wraps around torch._C.FunctionSchema with Tensor dtype constraints.
    In constructor, walk through all Tensor arguments and returns in the original schema
    for ATen operator, replace the argument with EdgeDialectArgument.
    """

    def __init__(
        self,
        schema: torch._C.FunctionSchema,
    ):
        self.schema = schema
        edge_op_full_name = schema.name + (
            ".{}".format(schema.overload_name) if schema.overload_name else ""
        )

        (
            essential_tensor_io_names,
            optional_tensor_io_names,
            all_tensor_io_names,
        ) = get_tensor_variable_names(self.schema)

        if edge_op_full_name in _edge_dialect_info:
            # Directly use the information from edge.yaml if available.
            _edge_op_info = _edge_dialect_info[edge_op_full_name]
            type_alias = {
                alias: [regular_tensor_str_to_dtypes[t] for t in types]
                for alias, types in _edge_op_info["type_alias"].items()
            }
            type_constraint = _edge_op_info["type_constraint"]
        else:
            # Not get the info from edge.yaml
            # Create a dtype constraint for this operator that allows any dtype
            # combinations as long as any dtype is legal in ExecuTorch.
            type_alias = {
                f"T{idx}": list(regular_tensor_str_to_dtypes.values())
                for idx in range(len(all_tensor_io_names))
            }
            type_constraint = [
                {io_name: f"T{idx}" for idx, io_name in enumerate(all_tensor_io_names)}
            ]

        self.dtype_constraint = FunctionDtypeConstraint(
            essential_tensor_io_names=essential_tensor_io_names,
            optional_tensor_io_names=optional_tensor_io_names,
            type_alias=type_alias,
            type_constraint=type_constraint,
        )

        arg_list: List[Union[torch._C.Argument, EdgeDialectArgument]] = []
        for argument in self.schema.arguments:
            if argument.name in self.dtype_constraint:
                arg_list.append(
                    EdgeDialectArgument(
                        argument,
                        self.dtype_constraint[argument.name],
                    )
                )
            else:
                arg_list.append(argument)
        self.arguments = arg_list
        return_names = sorted(
            n
            for n in self.dtype_constraint.type_constraint[0].keys()
            if n.startswith("__ret")
        )
        ret_list: List[Union[torch._C.Argument, EdgeDialectArgument]] = []
        ret_iter = iter(return_names)
        for ret in self.schema.returns:
            if isinstance(ret.type, torch.TensorType):
                name = next(ret_iter, None)
                if name:
                    ret_list.append(
                        EdgeDialectArgument(ret, self.dtype_constraint[name])
                    )
                    continue
            ret_list.append(ret)
        self.returns = ret_list

    def __getattr__(self, name):
        if name == "arguments":
            return self.arguments
        if name == "returns":
            return self.returns
        if name == "dtype_constraint":
            return self.dtype_constraint
        return getattr(self.schema, name)

    def __str__(self):
        return str(self.schema)


class EdgeOpOverload:
    """OpOverload for edge ops.
    Contains API to find the out variant of this operator overload.
    """

    def __init__(
        self,
        op: torch._ops.OpOverload,
        schema: EdgeDialectFunctionSchema,
    ):
        self._schema = schema
        self._op = op
        self.__name__ = f"{self.namespace}.{self._op.__name__}"

    def to_out_variant(self) -> torch._ops.OpOverload:
        """Find out the out-variant of this operator and return it.
        TODO (larryliu): Implement execution dialect class and let this function return that.
        This implementation assumes the out variant is available in torch.ops.*.

        Raises:
            RuntimeError: if we could't find the out variant, raise an exception.
            TODO (larryliu): Catch this in BackendDialect and generate an operator definition
            for missing out variant.
        Returns:
            torch._ops.OpOverload: The out-variant operator of self.
        """

        # return if already found
        if "_out_variant" in self.__dict__ and self._out_variant:
            return self._out_variant
        out_variant = to_variant(self._op, SchemaKind.out)
        self._out_variant = out_variant
        return out_variant

    def __getattr__(self, name):
        if name == "_schema":
            return self._schema
        else:
            return getattr(self._op, name)

    def __call__(self, *args, **kwargs):
        return self._op(*args, **kwargs)

    def __repr__(self):
        return "<EdgeOpOverload: {}>: schema = {}".format(
            self.__name__, self._schema.schema
        )

    __str__ = __repr__


class EdgeOpOverloadPacket:
    """OpOverloadPacket for edge ops.
    Wraps torch._ops.OpOverloadPacket and overrides __getattr__ to return OpOverload
    for Edge ops. The main difference between an Edge op and its corresponding ATen op
    is that Edge op contains a different schema (see EdgeDialectFunctionSchema).
    """

    def __init__(
        self,
        qualified_op_name: str,  # e.g., edge::aten::add
        op_name: str,
        parent_overload_packet: torch._ops.OpOverloadPacket,
    ):
        self._parent_overload_packet = parent_overload_packet
        self._parent_qualified_op_name = parent_overload_packet._qualified_op_name
        self._qualified_op_name = qualified_op_name
        self.__name__ = self._qualified_op_name.replace("::", ".")
        self._op = parent_overload_packet._op
        self._overload_names = parent_overload_packet._overload_names
        self._dir = []

    def __repr__(self):
        return "<EdgeOpOverloadPacket(op='{}', parent_op='{}')>".format(
            self._qualified_op_name.replace("::", "."),
            self._parent_qualified_op_name.replace("::", "."),
        )

    def __hash__(self):
        return hash(self._op)

    def __str__(self):
        return "{}".format(self._qualified_op_name.replace("::", "."))

    @property
    def op(self):
        return self._op

    def __getattr__(self, key):
        # It is not a valid op_name when __file__ is passed in
        if key == "__file__":
            return "exir.ops.edge"
        try:
            parent_overload = getattr(self._parent_overload_packet, key)
        except AttributeError:
            raise AttributeError(
                "The underlying op of '{}' has no overload name '{}'".format(
                    str(self), key
                )
            ) from None

        edge_schema = EdgeDialectFunctionSchema(
            parent_overload._schema,
        )  # create a new schema based on parent op schema
        overload = EdgeOpOverload(
            parent_overload,
            edge_schema,
        )
        # cache the overload object
        setattr(self, key, overload)
        self._dir.append(key)
        return overload

    def __call__(self, *args, **kwargs):
        return self._parent_overload_packet(*args, **kwargs or {})
