# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import yaml

kernels = {
    "aten": "fbcode/executorch/kernels/test/supported_features_def_aten.yaml",
    "optimized": "fbcode/executorch/kernels/optimized/test/supported_features_def.yaml",
    "portable": "fbcode/executorch/kernels/portable/test/supported_features_def.yaml",
    "quantized": "fbcode/executorch/kernels/quantized/test/supported_features_def.yaml",
    "custom_kernel_example": "fbcode/executorch/kernels/test/custom_kernel_example/supported_features_def.yaml",
}

definitions_yaml = "fbcode/executorch/kernels/test/supported_features.yaml"


def gen_overriden_values():
    overriden_values = {}
    for name, path in kernels.items():
        with open(path) as f:
            overrides = yaml.full_load(f)
            if not overrides:
                continue
            for entry in overrides:
                namespace = entry["namespace"]
                for feature, value in entry.items():
                    if feature == "namespace":
                        # we handled namespace previously
                        continue
                    overriden_values[namespace, feature, name] = value
    return overriden_values


def make_md_table():
    print("# Supported features table")

    overriden_values = gen_overriden_values()
    with open(definitions_yaml) as f:
        definitions = yaml.full_load(f)

    print("|", "|".join(["feature"] + list(kernels.keys())), "|")
    print("|", "|".join(["---"] * (len(kernels) + 1)), "|")
    for entry in definitions:
        namespace = entry["namespace"]
        print(
            "|", "|".join([f"**namespace {namespace}**"] + ["---"] * len(kernels)), "|"
        )
        for feature, value in entry.items():
            if feature == "namespace":
                # we handled namespace previously
                continue
            values = [
                str(overriden_values.get((namespace, feature, k), value["default"]))
                for k in kernels
            ]
            print("|", "|".join([feature] + values), "|")

    kernels_str = "\n".join(map(str, ((k, v) for k, v in kernels.items())))
    print(
        f"""
# Source
All of supported features are defined in fbcode/executorch/kernels/test/supported_features.yaml.

Each kernel can have its own overrides, which are defined in
{kernels_str}
"""
    )


if __name__ == "__main__":
    make_md_table()
