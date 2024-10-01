# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from pathlib import Path
from typing import Any, Dict, Optional

def get_default_model_resource_dir() -> str:
    """
    Get the default path to resouce files (which contain files such as the
    checkpoint and param files), either:
    1. Uses the path from pkg_resources, only works with buck2
    2. Uses default path located in examples/models/llama2/params
    """

    try:
        # 2nd way: If we can import this path, we are running with buck2 and all resources can be accessed with pkg_resources.
        # pyre-ignore
        import pgk_resources
        from executorch.examples.models.llama2 import params

        ckpt_dir = Path(
            pkg_resources.resource_filename(
                "executorch.examples.models.llama2", "params"
            )
        )
    except:
        # 3rd way.
        ckpt_dir = Path(__file__).absolute().parent / "params"

    return ckpt_dir   

def get_checkpoint_dtype(checkpoint: Dict[str, Any]) -> Optional[str]:
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
            raise ValueError(
                f"Mixed dtype model. Dtype of {first_key}: {first.dtype}. Mismatches in the checkpoint: {mismatched_dtypes}"
            )
    return dtype
