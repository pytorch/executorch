# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import inspect
import os
import sys
from typing import Dict, final, Optional, Sequence, Type

import executorch.exir as exir

import torch
from executorch.exir import to_edge
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.test.backend_with_compiler_demo import (
    BackendWithCompilerDemo,
)
from torch import nn
from torch.export import export

"""Traces and exports delegated nn.Modules to ExecuTorch .pte program files.

Creates two versions of each file:
- <module-name>.pte: Delegate data stored in segments outside of the flatbuffer data.
- <module-name>-nosegments.pte: Delegate data is stored directly in the flatbuffer data.

This tool mainly exists to export programs for C++ tests, but can also
be used to export models manually.
"""

#
# Modules
#


class ModuleAddMul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, a: torch.Tensor, x: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        y: torch.Tensor = torch.mm(a, x)
        z: torch.Tensor = torch.add(y, b)
        return z

    def get_random_inputs(self) -> Sequence[torch.Tensor]:
        return (torch.ones(2, 2), 2 * torch.ones(2, 2), 3 * torch.ones(2, 2))


#
# Backends
#


@final
class StubBackend(BackendDetails):
    """No-op backend to test serialization/init."""

    @staticmethod
    def preprocess(*args, **kwargs) -> PreprocessResult:
        return PreprocessResult(processed_bytes=b"StubBackend:data")


#
# Program logic
#


def export_module_to_program(
    module_class: Type[nn.Module],
    *,
    backend_id: str,
    extract_delegate_segments: bool,
    constant_tensor_alignemnt: Optional[int] = None,
    delegate_alignment: Optional[int] = None,
    method: str = "forward",
) -> bytes:
    eager_module = module_class().eval()
    inputs = ()
    if hasattr(eager_module, "get_random_inputs"):
        inputs = eager_module.get_random_inputs()

    class WrapperModule(torch.nn.Module):
        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def forward(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

    edge: exir.EdgeProgramManager = to_edge(
        export(WrapperModule(getattr(eager_module, method)), args=inputs)
    )

    lowered_module = to_backend(backend_id, edge.exported_program(), compile_specs=[])

    class CompositeModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.lowered_module = lowered_module

        def forward(self, *args, **kwargs):
            return self.lowered_module(*args, **kwargs)

    composite_module = CompositeModule()
    composite_module(*inputs)

    executorch_program = to_edge(export(composite_module, args=inputs)).to_executorch(
        config=exir.ExecutorchBackendConfig(
            extract_delegate_segments=extract_delegate_segments,
            constant_tensor_alignment=constant_tensor_alignemnt,
            delegate_alignment=delegate_alignment,
        )
    )

    return executorch_program.buffer


def main() -> None:
    known_backend_ids = [
        BackendWithCompilerDemo.__name__,
        StubBackend.__name__,
    ]

    # These args are optimized for genrule usage. There's a lot of startup
    # overhead for this tool, so it's faster to export multiple models at once
    # when possible.
    parser = argparse.ArgumentParser(
        prog="export_delegated_program",
        description="Exports delegated nn.Module models to ExecuTorch .pte files",
    )
    parser.add_argument(
        "--modules",
        help="Comma-separated list of model class names to export; "
        + "e.g., '--modules=ModuleOne,ModuleTwo'",
        type=lambda s: [item.strip() for item in s.split(",")],
    )
    parser.add_argument(
        "--backend_id",
        type=str,
        default=StubBackend.__name__,
        help="ID of the backend to use for delegation; "
        + f"one of {known_backend_ids}",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Path to the directory to write <classname>[-<suffix>[...]].pte "
        + "files to.",
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
        for extract_delegate_segments in (True, False):
            suffix = "" if extract_delegate_segments else "-nosegments"
            # Create files with the default alignment, and a large alignment.
            # This alignment should be so large that it's extremely unlikely for
            # the data to accidentally be aligned to it in the default case.
            for delegate_alignment in (None, 1024):
                suffix += f"-da{delegate_alignment}" if delegate_alignment else ""
                outfile = os.path.join(args.outdir, f"{module_name}{suffix}.pte")
                with open(outfile, "wb") as fp:
                    fp.write(
                        export_module_to_program(
                            module_class,
                            backend_id=args.backend_id,
                            extract_delegate_segments=extract_delegate_segments,
                            delegate_alignment=delegate_alignment,
                        )
                    )
                print(f"Exported {module_name} and wrote program data to {outfile}")


if __name__ == "__main__":
    main()
