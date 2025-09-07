# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def get_default_model_resource_dir(model_file_path: str) -> Path:
    """
    Get the default path to resouce files (which contain files such as the
    checkpoint and param files), either:
    1. Uses the path from importlib.resources, only works with buck2
    2. Uses default path located in examples/models/llama/params

    Expected to be called from with a `model.py` file located in a
    `executorch/examples/models/<model_name>` directory.

    Args:
        model_file_path: The file path to the eager model definition.
            For example, `executorch/examples/models/llama/model.py`,
            where `executorch/examples/models/llama` contains all
            the llama2-related files.

    Returns:
        The path to the resource directory containing checkpoint, params, etc.
    """

    try:
        import importlib.resources as _resources

        # 1st way: If we can import this path, we are running with buck2 and all resources can be accessed with importlib.resources.
        # pyre-ignore
        from executorch.examples.models.llama import params  # noqa

        # Get the model name from the cwd, assuming that this module is called from a path such as
        # examples/models/<model_name>/model.py.
        model_name = Path(model_file_path).parent.name
        model_dir = _resources.files(f"executorch.examples.models.{model_name}")
        with _resources.as_file(model_dir) as model_path:
            resource_dir = model_path / "params"
        assert resource_dir.exists()

    except Exception:
        # 2nd way:
        resource_dir = Path(model_file_path).absolute().parent / "params"

    return resource_dir


def get_checkpoint_dtype(checkpoint: Dict[str, Any]) -> Optional[torch.dtype]:
    """
    Get the dtype of the checkpoint, returning "None" if the checkpoint is empty.
    """
    dtype = None
    if len(checkpoint) > 0:
        first_key = next(iter(checkpoint))
        first = checkpoint[first_key]
        dtype = first.dtype
        mismatched_dtypes = [
            (key, value.dtype)
            for key, value in checkpoint.items()
            if hasattr(value, "dtype") and value.dtype != dtype
        ]
        if len(mismatched_dtypes) > 0:
            print(
                f"Mixed dtype model. Dtype of {first_key}: {first.dtype}. Mismatches in the checkpoint: {mismatched_dtypes}"
            )
    return dtype


def load_checkpoint_from_pytorch_model(input_dir: str) -> Dict:
    index_path = os.path.join(input_dir, "pytorch_model.bin.index.json")
    if os.path.exists(index_path):
        # Sharded checkpoint.
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        checkpoint_shards = sorted(set(weight_map.values()))

        # Load all the shards into memory
        shard_to_weights = {}
        for shard in checkpoint_shards:
            shard_to_weights[shard] = torch.load(
                os.path.join(input_dir, shard),
                weights_only=True,
                map_location=torch.device("cpu"),
            )

        # Merge tensors into consolidated state dict.
        merged_state_dict = {}
        for weight_name, shard in weight_map.items():
            tensor = shard_to_weights[shard][weight_name]
            merged_state_dict[weight_name] = tensor
        return merged_state_dict

    # Single checkpoint
    model_path = os.path.join(input_dir, "pytorch_model.bin")
    if os.path.exists(model_path):
        state_dict = torch.load(
            model_path, weights_only=True, map_location=torch.device("cpu")
        )
        return state_dict

    raise FileNotFoundError(f"Could not find pytorch_model checkpoint in {input_dir}")
