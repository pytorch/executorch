# Copyright © 2024 Apple Inc. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import shutil
from pathlib import Path

from typing import List, Optional

import executorchcoreml

from executorch.backends.apple.coreml.compiler import CoreMLBackend

from executorch.exir._serialize._program import deserialize_pte_binary

from executorch.exir.schema import (
    BackendDelegate,
    BackendDelegateDataReference,
    DataLocation,
)


def extract_coreml_models(pte_data: bytes):
    program = deserialize_pte_binary(pte_data)
    delegates: List[BackendDelegate] = sum(
        [execution_plan.delegates for execution_plan in program.execution_plan], []
    )
    coreml_delegates: List[BackendDelegate] = [
        delegate for delegate in delegates if delegate.id == CoreMLBackend.__name__
    ]
    model_index: int = 1
    for coreml_delegate in coreml_delegates:
        coreml_delegate_data: BackendDelegateDataReference = coreml_delegate.processed
        coreml_processed_bytes: Optional[bytes] = None
        match coreml_delegate_data.location:
            case DataLocation.INLINE:
                coreml_processed_bytes = program.backend_delegate_data[
                    coreml_delegate_data.index
                ].data

            case _:
                AssertionError("The loaded Program must have inline data.")

        model_name: str = f"model_{model_index}"
        model_path: Path = Path() / "extracted_coreml_models" / model_name
        if model_path.exists():
            shutil.rmtree(model_path.absolute())
        os.makedirs(model_path.absolute())

        if executorchcoreml.unflatten_directory_contents(
            coreml_processed_bytes, str(model_path.absolute())
        ):
            print(f"Core ML models are extracted and saved to path = {model_path}")
        model_index += 1

    if len(coreml_delegates) == 0:
        print("The model isn't delegated to Core ML.")


if __name__ == "__main__":
    """
    Extracts the Core ML models embedded in the ``.pte`` file and saves them to the
    file system.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_path",
        required=True,
        help="Input must be a .pte file.",
    )

    args = parser.parse_args()
    model_path = str(args.model_path)
    with open(model_path, mode="rb") as pte_file:
        pte_data = pte_file.read()
        extract_coreml_models(pte_data)
