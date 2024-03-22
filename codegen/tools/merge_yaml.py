# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

import yaml

try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader  # type: ignore[misc]


class BlankLineDumper(yaml.SafeDumper):
    def write_line_break(self, data=None):
        super().write_line_break(data)
        # insert a new line between entries.
        if len(self.indents) == 1:
            super().write_line_break()


def merge(functions_yaml_path: str, fallback_yaml_path: Optional[str], output_dir: str):
    output_file = os.path.join(output_dir, "merged.yaml")

    def get_canonical_opname(func: object) -> str:
        """get the canonical name of an operator
        "op" and "func" are two keywords we are supporting for yaml files.
        To give an example:
        - op: add.Tensor # mostly used for binding ATen ops to kernels
        - func: add.Tensor(Tensor self, Tensor other, Scalar alpha) # mostly used for
            defining custom ops.

        These two will be supported
        Args:
            func (object): yaml object

        Returns:
            str: canonical name of the operator
        """
        # pyre-ignore
        opname = func["op"] if "op" in func else func["func"].split("(")[0]
        if "::" not in opname:
            opname = "aten::" + opname
        return opname

    with open(functions_yaml_path) as f:
        functions_obj = yaml.load(f, Loader=Loader)
        functions_dict: Dict[str, object] = defaultdict(object)
        for func in functions_obj:
            functions_dict[get_canonical_opname(func)] = func
    if fallback_yaml_path is not None and os.path.exists(fallback_yaml_path):
        with open(fallback_yaml_path) as f:
            fallback_obj = yaml.load(f, Loader=Loader)
            for func in fallback_obj:
                opname = get_canonical_opname(func)
                if opname not in functions_dict:
                    functions_dict[opname] = func

    with open(output_file, "w") as f:
        yaml.dump(
            list(functions_dict.values()),
            f,
            Dumper=BlankLineDumper,
            default_flow_style=False,
            sort_keys=False,
            width=1000,
        )


def main(argv: List[Any]) -> None:
    """Merge functions.yaml and fallback yaml. The output yaml will be a union of all entries in functions.yaml and fallback yaml, with operator entries in functions.yaml overriding entries with the same op name in fallback yaml.
    E.g.,
    functions.yaml:
    - op: add.Tensor
      - kernel: add_impl

    fallback yaml:
    - op: add.Tensor
      - kernel: add_fallback
    - op: relu
      - kernel: relu_fallback

    Merged:
    - op: add.Tensor
      - kernel: add_impl
    - op: relu
      - kernel: relu_fallback

    """
    parser = argparse.ArgumentParser(
        description="Merge functions.yaml, custom_ops.yaml with fallback yaml, for codegen to consume."
    )
    parser.add_argument(
        "--functions-yaml-path",
        "--functions_yaml_path",
        help="path to the functions.yaml file to use.",
        required=True,
    )
    parser.add_argument(
        "--fallback-yaml-path",
        "--fallback_yaml_path",
        help="path to fallback yaml file.",
        required=False,
    )
    parser.add_argument(
        "--output_dir",
        help=("The directory to store the output yaml file"),
        required=True,
    )

    options = parser.parse_args(argv)
    assert options.functions_yaml_path is not None and os.path.exists(
        options.functions_yaml_path
    )
    merge(options.functions_yaml_path, options.fallback_yaml_path, options.output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
