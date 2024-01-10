# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import inspect
import os
import sys
from typing import Any, Dict, List, Type

import torch
from executorch.exir import CaptureConfig
from executorch.exir.passes import MemoryPlanningPass
from executorch.test.end2end.exported_module import ExportedModule
from torch import nn
from torch._export import dynamic_dim

"""Traces and exports nn.Modules to ExecuTorch .pte program files.

This tool mainly exists to export programs for C++ tests, but can also
be used to export models manually.
"""

#
# Module definitions.
#
# If we ever have more than a handful, consider splitting into multiple files.
#


class ModuleBasic(nn.Module):
    def __init__(self):
        super(ModuleBasic, self).__init__()

    def forward(self, x):
        return torch.sin(x).max()

    def get_random_inputs(self):
        return (torch.randn(100),)

    @staticmethod
    def get_export_kwargs() -> Dict[str, Any]:
        """Returns custom trace params for ExportedModule."""
        return {
            # aten::max.default does not have an out variant.
            "ignore_to_out_var_failure": True,
        }


class ModuleIndex(nn.Module):
    def __init__(self):
        super(ModuleIndex, self).__init__()

    def forward(self, x):
        # Weird index that happens to generate a None in torch.index.Tensor_out
        # which is desirable for deserialization testing. A modified form of
        # an example index from https://pytorch.org/cppdocs/notes/tensor_indexing.html.
        return x[1::2, torch.tensor([1, 2])]

    def get_random_inputs(self):
        return (torch.randn(10, 10, 10),)


class ModuleNoOp(nn.Module):
    def __init__(self):
        super(ModuleNoOp, self).__init__()

    def forward(self, x, y):
        return (x, y)

    def get_random_inputs(self):
        return (torch.randn(2, 2), torch.randn(2, 2))


class ModuleAdd(nn.Module):
    def __init__(self):
        super(ModuleAdd, self).__init__()

    def forward(self, x, y, alpha):
        return torch.add(x, y, alpha=alpha)

    def get_random_inputs(self):
        return (torch.randn(2, 2), torch.randn(2, 2), 1.0)


class ModuleDynamicCatUnallocatedIO(nn.Module):
    def __init__(self):
        super(ModuleDynamicCatUnallocatedIO, self).__init__()
        # TODO(T163238401)
        self._inputs = (torch.randn(3, 4),)

    def forward(self, k):
        k = torch.cat((k, torch.ones(1, 4)))
        return k

    def get_random_inputs(self):
        return self._inputs

    def get_constraints(self):
        return [
            dynamic_dim(self._inputs[0], 0) <= 3,
        ]

    def get_memory_planning_pass(self):
        return MemoryPlanningPass(
            memory_planning_algo="greedy",
            alloc_graph_input=False,
            alloc_graph_output=False,
        )

    @staticmethod
    def get_export_kwargs():
        return {"capture_config": CaptureConfig(pt2_mode=True, enable_aot=True)}


class ModuleLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = 3 * torch.ones(2, 2, dtype=torch.float)
        self.b = 2 * torch.ones(2, 2, dtype=torch.float)

    def forward(self, x: torch.Tensor):
        out_1 = torch.mul(self.a, x)
        out_2 = torch.add(out_1, self.b)
        return out_2

    def get_random_inputs(self):
        return (torch.ones(2, 2, dtype=torch.float),)


class ModuleMultipleEntry(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = 3 * torch.ones(2, 2, dtype=torch.float)
        self.b = 2 * torch.ones(2, 2, dtype=torch.float)

    def forward(self, x: torch.Tensor):
        return x + self.a

    def forward2(self, x: torch.Tensor):
        return x + self.a + self.b

    def get_random_inputs(self):
        return (torch.ones(2, 2, dtype=torch.float),)

    @staticmethod
    def get_method_names_to_export() -> List[str]:
        return ["forward", "forward2"]


#
# Main logic.
#


def export_module_to_program(
    module_class: Type[nn.Module], extract_constant_segment: bool
):
    """Exports the module and returns the serialized program data."""
    # Look for an optional @staticmethod that defines custom trace params.
    export_kwargs: Dict[str, Any] = {}
    if hasattr(module_class, "get_export_kwargs"):
        # pyre-ignore[16]: pyre doesn't know about get_export_kwargs.
        export_kwargs = module_class.get_export_kwargs()
    if hasattr(module_class, "get_method_names_to_export"):
        # pyre-ignore[16]: pyre doesn't know about get_export_kwargs.
        methods = module_class.get_method_names_to_export()
    else:
        methods = ["forward"]
    module = ExportedModule.export(
        module_class,
        methods,
        extract_constant_segment=extract_constant_segment,
        **export_kwargs,
    )
    return module.executorch_program.buffer


def main() -> None:
    # These args are optimized for genrule usage. There's a lot of startup
    # overhead for this tool, so it's faster to export multiple models at once
    # when possible.
    parser = argparse.ArgumentParser(
        prog="export_program",
        description="Exports nn.Module models to ExecuTorch .pte files",
    )
    parser.add_argument(
        "--modules",
        help="Comma-separated list of model class names to export; "
        + "e.g., '--modules=ModuleBasic,ModuleAdd'",
        type=lambda s: [item.strip() for item in s.split(",")],
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Path to the directory to write <classname>.pte files to",
    )
    args = parser.parse_args()

    # Find the classes to export. Only looks in this module for now, but could
    # be extended to look in other modules if helpful.
    module_names_to_classes: Dict[str, Type[nn.Module]] = {}
    for module in args.modules:
        module_class = getattr(sys.modules[__name__], module, None)
        if not (inspect.isclass(module_class) and issubclass(module_class, nn.Module)):
            raise NameError(f"Could not find nn.Module class named '{module}'")
        module_names_to_classes[module] = module_class

    # Export and write to the output files.
    os.makedirs(args.outdir, exist_ok=True)
    for module_name, module_class in module_names_to_classes.items():
        for extract_constant_segment in (True, False):
            suffix = "" if extract_constant_segment else "-no-constant-segment"
            outfile = os.path.join(args.outdir, f"{module_name}{suffix}.pte")
            with open(outfile, "wb") as fp:
                fp.write(
                    export_module_to_program(
                        module_class, extract_constant_segment=extract_constant_segment
                    )
                )
            print(f"Exported {module_name} and wrote program data to {outfile}")


if __name__ == "__main__":
    main()
