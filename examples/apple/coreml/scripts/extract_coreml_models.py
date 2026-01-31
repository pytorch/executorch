# Copyright © 2024 Apple Inc. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import shutil
from pathlib import Path

from typing import Dict, List, Optional

from executorch.backends.apple.coreml import executorchcoreml
from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.schema import (
    BackendDelegate,
    BackendDelegateDataReference,
    DataLocation,
)


def extract_coreml_models(pte_data: bytes):
    pte_file = deserialize_pte_binary(pte_data)
    program = pte_file.program

    # Build a map from named_data keys to their data for multifunction model support.
    # Multifunction models store a JSON reference in processed_bytes that points to
    # the actual model data in named_data.
    # After deserialization, pte_file.named_data is a NamedDataStoreOutput containing
    # buffers and pte_data (key -> DataEntry mapping).
    named_data_map: Dict[str, bytes] = {}
    if pte_file.named_data is not None:
        for key, data_entry in pte_file.named_data.pte_data.items():
            named_data_map[key] = pte_file.named_data.buffers[data_entry.buffer_index]

    delegates: List[BackendDelegate] = sum(
        [execution_plan.delegates for execution_plan in program.execution_plan], []
    )
    coreml_delegates: List[BackendDelegate] = [
        delegate for delegate in delegates if delegate.id == CoreMLBackend.__name__
    ]

    # Track extracted models to avoid duplicates (multifunction models share partitions)
    extracted_keys: set = set()
    model_index: int = 1

    for coreml_delegate in coreml_delegates:
        coreml_delegate_data: BackendDelegateDataReference = coreml_delegate.processed
        coreml_processed_bytes: Optional[bytes] = None
        model_name: Optional[str] = None

        match coreml_delegate_data.location:
            case DataLocation.INLINE:
                raw_bytes = program.backend_delegate_data[
                    coreml_delegate_data.index
                ].data

                # Check if this is a JSON reference to named_data (multifunction models)
                # JSON references are prefixed with "CMJR" magic number
                MAGIC_NUMBER = b"CMJR"
                if raw_bytes.startswith(MAGIC_NUMBER):
                    # Strip magic number and parse JSON
                    json_bytes = raw_bytes[len(MAGIC_NUMBER) :]
                    try:
                        reference = json.loads(json_bytes.decode("utf-8"))
                        key = reference.get("key")
                        if key in extracted_keys:
                            # Already extracted this partition, skip
                            continue
                        if key in named_data_map:
                            coreml_processed_bytes = named_data_map[key]
                            model_name = key  # Use the key as model name
                            extracted_keys.add(key)
                        else:
                            print(
                                f"Warning: Named data key '{key}' not found in program"
                            )
                            continue
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        print(f"Warning: Failed to parse JSON reference: {e}")
                        continue
                else:
                    # Not a JSON reference, treat as raw model data (legacy format)
                    coreml_processed_bytes = raw_bytes

            case _:
                AssertionError("The loaded Program must have inline data.")

        if coreml_processed_bytes is None:
            continue

        if model_name is None:
            model_name = f"model_{model_index}"

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
