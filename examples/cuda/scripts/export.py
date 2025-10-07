# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer with CUDA delegate.

import argparse
import pathlib

import torch

from executorch.backends.cuda.cuda_backend import CudaBackend

from executorch.backends.cuda.cuda_partitioner import CudaPartitioner

from executorch.examples.models import MODEL_NAME_TO_MODEL
from executorch.examples.models.model_factory import EagerModelFactory

from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

from executorch.extension.export_util.utils import save_pte_program
from torch._inductor.decomposition import conv1d_to_conv2d
from torch.nn.attention import SDPBackend

# Script to export a model with coreml delegation.

_EDGE_COMPILE_CONFIG = EdgeCompileConfig(
    _check_ir_validity=False,
    _skip_dim_order=True,  # TODO(T182928844): enable dim_order in backend
)


def is_fbcode():
    return not hasattr(torch.version, "git_version")


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

    (
        model,
        example_args,
        example_kwargs,
        dynamic_shapes,
    ) = EagerModelFactory.create_model(*MODEL_NAME_TO_MODEL[args.model_name])
    model = model.eval()
    exported_programs = torch.export.export(
        model,
        args=example_args,
        kwargs=example_kwargs,
        dynamic_shapes=dynamic_shapes,
    )
    print(exported_programs)

    partitioner = CudaPartitioner(
        [CudaBackend.generate_method_name_compile_spec(args.model_name)]
    )
    # Add decompositions for triton to generate kernels.
    exported_programs = exported_programs.run_decompositions(
        {
            torch.ops.aten.conv1d.default: conv1d_to_conv2d,
        }
    )
    with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]):
        et_prog = to_edge_transform_and_lower(
            exported_programs,
            partitioner=[partitioner],
            compile_config=_EDGE_COMPILE_CONFIG,
            generate_etrecord=args.generate_etrecord,
        )
    exec_program = et_prog.to_executorch()
    save_pte_program(exec_program, args.model_name, args.output_dir)
    if args.generate_etrecord:
        exec_program.get_etrecord().save(f"{args.model_name}_etrecord.bin")


if __name__ == "__main__":
    main()
