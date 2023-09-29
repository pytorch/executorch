# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import logging
import re
import sys
from contextlib import contextmanager
from typing import Dict, Generator, List, Optional, Tuple, TypeVar, Union

import torch


@contextmanager
def no_dispatch() -> Generator[None, None, None]:
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard


def get_schema_for_operators(ops: List[str]) -> Dict[str, str]:
    r"""
    Accept a list of operator names fetched from the Graph Module (these are of
    the form torch.ops.aten.cat.default, and return a dict of operator name (in
    the form namespace::op_name.overload_name) to operator schema string.

    Note: This method should only be used for debugging errors in export, and
    not in a production context.
    """
    d = {}
    pat = re.compile(r"^torch.ops.([^\.]+)\.(.*)")

    aten_ops = []
    for op in ops:
        aten_ops.append(re.sub(pat, r"\1::\2", op))

    all_schemas = torch._C._jit_get_all_schemas()

    schema_dict = {}
    for s in all_schemas:
        n = s.name
        if s.overload_name != "":
            n = n + "." + s.overload_name
        else:
            n = n + ".default"
        schema_dict[n] = str(s)

    for op in aten_ops:
        d[op] = "<No Schema Found>"
        if op in schema_dict:
            d[op] = schema_dict[op]

    return d


T = TypeVar("T")  # Declare type variable


def extract_out_arguments(
    schema: torch._C.FunctionSchema, keyword_args: Dict[str, T]
) -> Union[Tuple[str, T], List[Tuple[str, T]]]:
    # Given a possible out schema, find all out arguments and return them as tuple of
    # the arg name and the actual value.
    out_args = []
    for arg in schema.arguments:
        name = arg.name
        if arg.is_out and name in keyword_args:
            out_args.append((name, keyword_args[name]))

    # TODO (tmanlaibaatar) There are 3 ops with TensorList as the storage for aliased tensor
    # which was added after is_out logic. Until we fix that implementation,
    # hack to manually add out args
    if len(out_args) == 0:
        if "out" in keyword_args:
            out_args.append(("out", keyword_args["out"]))

    if len(out_args) == 1:
        return out_args[0]

    return out_args


def format_schema_name(schema: torch._C.FunctionSchema) -> str:
    if schema.overload_name != "":
        return schema.name + "." + schema.overload_name
    return schema.name


@contextmanager
def override_logger(
    newLevel: int = logging.DEBUG,
    fmtstr: str = "%(message)s",
    filename: Optional[str] = None,
) -> Generator[None, None, None]:
    """
    If an nonempty filename string is provided, the log wil also be written to
    that file besides stderr.
    """
    try:
        oldLevel = logging.root.level
        logging.root.setLevel(newLevel)
        if fmtstr:
            newformatter = logging.Formatter(fmtstr, None, "%")
            oldFormatters = []
            for handler in logging.root.handlers:
                oldFormatters.append(handler.formatter)
                handler.formatter = newformatter
        filehandler = None
        if filename:
            filehandler = logging.FileHandler(filename, mode="w")
            logging.root.addHandler(filehandler)
        yield
    finally:
        logging.root.setLevel(oldLevel)
        if fmtstr:
            # pyre-fixme[61]: `oldFormatters` is undefined, or not always defined.
            for handler, formatter in zip(logging.root.handlers, oldFormatters):
                handler.formatter = formatter
        if filehandler:
            logging.root.removeHandler(filehandler)


@contextmanager
def setting_python_recursive_limit(limit: int = 10000) -> Generator[None, None, None]:
    """
    Temporarily increase the python interpreter stack recursion limit.
    This is mostly used for pickling large scale modules.
    """
    default = sys.getrecursionlimit()
    if limit > default:
        sys.setrecursionlimit(limit)
    try:
        yield
    finally:
        sys.setrecursionlimit(default)
