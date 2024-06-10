# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from typing import Any, List

import pkg_resources
import yaml


def _to_c_bool(s: Any):
    if s in [True, False]:
        return str(s).lower()
    return s


def generate_header(d: dict):
    """Generates a supported features header file"""
    ini_path = os.path.join(
        os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))),
        "supported_features_header.ini",
    )
    if os.path.isfile(ini_path):
        header_file = open(ini_path, encoding="utf-8").read()
    else:
        header_file = pkg_resources.resource_string(
            __package__, "supported_features_header.ini"
        ).decode("utf-8")

    return header_file.replace("$header_entries", "".join(generate_header_entry(d)))


def generate_header_entry(d: dict):
    for entry in d:
        namespace = entry["namespace"]
        for feature, properties in entry.items():
            if feature == "namespace":
                # we handled namespace previously
                continue
            yield generate_header_entry_text(namespace, feature, properties)


def generate_header_entry_text(namespace: str, feature: str, properties: dict):
    if namespace == "global":
        full_name = feature
    else:
        full_name = "_".join([namespace, feature])
    if "default" in properties:
        default = _to_c_bool(properties["default"])
        default = f" = {default}"
    else:
        default = ""
    if "docstring" in properties:
        docstring = properties["docstring"]
    else:
        docstring = "TODO: add docstring for this entry"
    t = properties["type"]
    entry = f"{t} {full_name}{default};\n"
    return f"""
  // {docstring}
  {entry}
"""


def generate_definition(d: dict):
    ini_path = os.path.join(
        os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))),
        "supported_features_definition.ini",
    )
    if os.path.isfile(ini_path):
        definition_file = open(ini_path, encoding="utf-8").read()
    else:
        definition_file = pkg_resources.resource_string(
            __package__, "supported_features_definition.ini"
        ).decode("utf-8")

    return definition_file.replace(
        "$definition_entries", "".join(generate_definition_entry(d))
    )


def generate_definition_entry(d: dict):
    if not d:
        return []  # noqa: B901
    for entry in d:
        namespace = entry["namespace"]
        for feature, value in entry.items():
            if feature == "namespace":
                # we handled namespace previously
                continue
            yield generate_definition_entry_text(namespace, feature, value)


def generate_definition_entry_text(namespace: str, feature: str, value: Any):
    if namespace == "global":
        full_name = feature
    else:
        full_name = "_".join([namespace, feature])
    value = _to_c_bool(value)
    return f"""
  .{full_name} = {value},
"""


def main(args: List[Any]) -> None:
    """
    This binary generates the supported_features.h from supported_features.yaml
    from this (//executorch/kernels/test) directory. Then for a specific kernel,
    we need to supply the overridden supported_features_def.yaml for the kernel.
    """
    with open(args[0]) as f:
        y = yaml.full_load(f)
        if "supported_features_def" in args[0]:
            print(generate_definition(y))
        else:
            print(generate_header(y))


def invoke_main() -> None:
    if len(sys.argv) != 2:
        print(
            "Usage: gen_supported_features.py <path-to>/supported_features{_def}.yaml"
        )
        exit(1)
    main(sys.argv[1:])


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
