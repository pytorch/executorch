# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from pathlib import Path
from typing import Any, Dict, Optional


def get_default_model_resource_dir(model_file_path: str) -> Path:
    """
    Get the default path to resouce files (which contain files such as the
    checkpoint and param files), either:
    1. Uses the path from pkg_resources, only works with buck2
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
        import pkg_resources

        # 1st way: If we can import this path, we are running with buck2 and all resources can be accessed with pkg_resources.
        # pyre-ignore
        from executorch.examples.models.llama import params  # noqa

        # Get the model name from the cwd, assuming that this module is called from a path such as
        # examples/models/<model_name>/model.py.
        model_name = Path(model_file_path).parent.name
        resource_dir = Path(
            pkg_resources.resource_filename(
                f"executorch.examples.models.{model_name}", "params"
            )
        )
    except:
        # 2nd way.
        resource_dir = Path(model_file_path).absolute().parent / "params"

    return resource_dir


def get_checkpoint_dtype(checkpoint: Dict[str, Any]) -> Optional[str]:
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
            if value.dtype != dtype
        ]
        if len(mismatched_dtypes) > 0:
            print(
                f"Mixed dtype model. Dtype of {first_key}: {first.dtype}. Mismatches in the checkpoint: {mismatched_dtypes}"
            )
    return dtype
