# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import argparse
import inspect
import os
import sys
from typing import Any, Dict, List, Type

import torch
from executorch.exir import CaptureConfig
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.program._program import ExecutorchProgramManager
from torch import nn
from torch.export import Dim

from ..end2end.exported_module import ExportedModule

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


class ModuleAddHalf(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, alpha):
        return torch.add(x, y, alpha=alpha)

    def get_random_inputs(self):
        return (
            torch.randn(2, 2).half(),
            torch.randn(2, 2).half(),
            1.0,
        )


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

    def get_dynamic_shapes(self):
        return ({0: Dim("dim0_k", max=3)},)

    def get_memory_planning_pass(self):
        return MemoryPlanningPass(
            alloc_graph_input=False,
            alloc_graph_output=False,
        )

    @staticmethod
    def get_export_kwargs():
        return {"capture_config": CaptureConfig(pt2_mode=True, enable_aot=True)}


class ModuleAddMul(torch.nn.Module):
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


# Used for program-data-separation.
class ModuleLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x: torch.Tensor):
        return self.linear(x)

    def get_random_inputs(self):
        return (torch.randn(3),)


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


class ModuleSimpleTrain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        return self.loss(self.linear(x).softmax(dim=0), y)

    def get_random_inputs(self):
        return (torch.randn(3), torch.tensor([1.0, 0.0, 0.0]))

    @staticmethod
    def export_joint():
        return True


class ModuleStateful(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("state", torch.zeros(1, dtype=torch.int32))

    def forward(self, x):
        self.state.add_(1)
        return x + self.state

    def get_random_inputs(self):
        return (torch.ones(1),)

    @staticmethod
    def export_state_names():
        return True


# Mimicking LLM with forward taking tokens and input_pos
class ModuleKVCacheInputPos(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x, input_pos):
        return (self.linear(x.to(torch.float)).to(torch.long) + input_pos).to(
            torch.float
        )

    def get_random_inputs(self):
        return (
            torch.randint(100, [1, 3], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
        )


# Mimicking LLM with forward taking tokens and cache_positions
class ModuleKVCacheCachePos(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x, cache_positions):
        return (self.linear(x.to(torch.float)).to(torch.long) + cache_positions).to(
            torch.float
        )

    def get_random_inputs(self):
        return (
            torch.randint(100, [1, 3], dtype=torch.long),
            torch.arange(3, dtype=torch.long),
        )


# Mimicking LLM with forward taking only tokens
class ModuleNoKVCache(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x):
        return self.linear(x.to(torch.float))

    def get_random_inputs(self):
        return (torch.randint(100, [1, 3], dtype=torch.long),)


#
# Main logic.
#


def export_module_to_program(
    module_class: Type[nn.Module],
    external_constants: bool = False,
) -> ExecutorchProgramManager:
    """Exports the module and returns the serialized program data."""
    torch.manual_seed(0)
    # Look for an optional @staticmethod that defines custom trace params.
    export_kwargs: Dict[str, Any] = {}
    if hasattr(module_class, "get_export_kwargs"):
        # pyre-ignore[16]: pyre doesn't know about get_export_kwargs.
        export_kwargs = module_class.get_export_kwargs()
    export_joint = False
    export_state_names = False
    if hasattr(module_class, "export_joint"):
        export_joint = module_class.export_joint()  # pyre-ignore
    if hasattr(module_class, "export_state_names"):
        export_state_names = module_class.export_state_names()
    if hasattr(module_class, "get_method_names_to_export"):
        # pyre-ignore[16]: pyre doesn't know about get_export_kwargs.
        methods = module_class.get_method_names_to_export()
    else:
        methods = ["forward"]
    module = ExportedModule.export(
        module_class,
        methods,
        export_joint_graph=export_joint,
        external_constants=external_constants,
        export_state_names=export_state_names,
        **export_kwargs,
    )
    return module.executorch_program


def main() -> None:
    # These args are optimized for genrule usage. There's a lot of startup
    # overhead for this tool, so it's faster to export multiple models at once
    # when possible.
    torch.manual_seed(0)
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
        help="Path to the directory to write <classname>.pte files and .ptd files to",
    )
    parser.add_argument(
        "--external-constants",
        action="store_true",
        help="Export the model with external constants",
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
        if args.external_constants:
            module_name = f"{module_name}Program"
        outfile = os.path.join(args.outdir, f"{module_name}.pte")
        prog = export_module_to_program(
            module_class,
            external_constants=args.external_constants,
        )
        with open(outfile, "wb") as fp:
            prog.write_to_file(fp)
            print(f"Exported {module_name} and wrote program data to {outfile}")

        if args.external_constants:
            # current infra doesnt easily allow renaming this file, so just hackily do it here.
            prog._tensor_data[f"{module_name}"] = prog._tensor_data.pop(
                "_default_external_constant"
            )
        prog.write_tensor_data_to_file(args.outdir)


if __name__ == "__main__":
    main()
