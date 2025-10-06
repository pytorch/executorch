# Copyright © 2023 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.

import argparse
import collections
import copy
from typing import Any, Dict, List, Optional, Tuple, Union
import pathlib
import sys

import coremltools as ct

import executorch.exir as exir

import torch
from torch._inductor.decomposition import conv1d_to_conv2d

from executorch.exir.backend.partitioner import Partitioner

# pyre-fixme[21]: Could not find module `executorch.backends.apple.coreml.compiler`.
from executorch.backends.cuda.cuda_backend import CudaBackend

# pyre-fixme[21]: Could not find module `executorch.backends.apple.coreml.partition`.
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.devtools.etrecord import generate_etrecord
from executorch.exir import to_edge

from executorch.exir.backend.backend_api import to_backend
from executorch.extension.export_util.utils import save_pte_program

from executorch.examples.models import MODEL_NAME_TO_MODEL
from executorch.examples.models.model_factory import EagerModelFactory

# Script to export a model with coreml delegation.

_EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig(
    _check_ir_validity=False,
    _skip_dim_order=True,  # TODO(T182928844): enable dim_order in backend
)
aten = torch.ops.aten


def is_fbcode():
    return not hasattr(torch.version, "git_version")


_CAN_RUN_WITH_PYBINDINGS = (sys.platform == "darwin") and not is_fbcode()
if _CAN_RUN_WITH_PYBINDINGS:
    from executorch.runtime import Runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help=f"Provide model name. Valid ones: {list(MODEL_NAME_TO_MODEL.keys())}",
    )
    parser.add_argument(
      "--output_dir",
      type=pathlib.Path,
      default=pathlib.Path("./"),
      help="Output directory for the exported model",
    )
    parser.add_argument("--generate_etrecord", action=argparse.BooleanOptionalAction)
    parser.add_argument("--save_processed_bytes", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    return args


def save_processed_bytes(processed_bytes, base_name: str):
    filename = f"{base_name}.bin"
    print(f"Saving processed bytes to {filename}")
    with open(filename, "wb") as file:
        file.write(processed_bytes)
    return


def main():
    args = parse_args()

    if args.model_name not in MODEL_NAME_TO_MODEL:
        raise RuntimeError(
            f"Model {args.model_name} is not a valid name. "
            f"Available models are {list(MODEL_NAME_TO_MODEL.keys())}."
        )

    valid_compute_units = [compute_unit.name.lower() for compute_unit in ct.ComputeUnit]
    if args.compute_unit not in valid_compute_units:
        raise RuntimeError(
            f"{args.compute_unit} is invalid. "
            f"Valid compute units are {valid_compute_units}."
        )

    model, example_args, example_kwargs, dynamic_shapes = (
        EagerModelFactory.create_model(*MODEL_NAME_TO_MODEL[args.model_name])
    )
    if not args.dynamic_shapes:
        dynamic_shapes = None

    model = model.eval()
    exported_programs = torch.export.export(
        model,
        args=example_args,
        kwargs=example_kwargs,
        dynamic_shapes=dynamic_shapes,
    )
    print(exported_programs)

    partitioners: Dict[str, List[Partitioner]] = {
        key: [CudaPartitioner([CudaBackend.generate_method_name_compile_spec(key)])]
        for key in exported_programs.keys()
    }
    # Add decompositions for triton to generate kernels.
    for key, ep in exported_programs.items():
        exported_programs[key] = ep.run_decompositions(
            {
                aten.conv1d.default: conv1d_to_conv2d,
            }
        )
    with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]):
        et_prog = to_edge_transform_and_lower(
            exported_programs,
            partitioner=partitioners,
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                _skip_dim_order=True,
            ),
            constant_methods=metadata,
            transform_passes=[RemovePaddingIdxEmbeddingPass()],
            generate_etrecord=args.generate_etrecord,
        )
    exec_program = delegated_program.to_executorch()
    save_pte_program(exec_program, args.model_name, args.output_dir)
    if args.generate_etrecord:
        exec_program.get_etrecord().save(f"{args.model_name}_cuda_etrecord.bin")



if __name__ == "__main__":
    main()
