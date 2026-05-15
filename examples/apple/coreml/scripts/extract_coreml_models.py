# Copyright © 2024 Apple Inc. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import shutil
from pathlib import Path

from typing import Dict, List, Optional, Union

from executorch.backends.apple.coreml import executorchcoreml
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.schema import (
    BackendDelegate,
    BackendDelegateDataReference,
    DataLocation,
)

COREML_BACKEND_ID = "CoreMLBackend"
# JSON references to named_data (multifunction models) are prefixed with this.
_NAMED_DATA_MAGIC = b"CMJR"


def extract_coreml_models(
    pte_data: bytes,
    out_dir: Optional[Union[str, Path]] = None,
) -> List[Path]:
    """Extract every Core ML partition embedded in a .pte to ``out_dir``.

    Multifunction models share partitions across delegates via a JSON reference
    into ``named_data``; duplicates are deduplicated by that key. ``out_dir``
    defaults to ``./extracted_coreml_models`` (CLI behaviour). Returns the list
    of extracted model directories, suitable for passing to
    ``MLComputePlan.load_from_path`` or to ``ct.models.MLModel``.
    """
    out_root = Path(out_dir) if out_dir is not None else Path("extracted_coreml_models")
    out_root.mkdir(parents=True, exist_ok=True)

    pte_file = deserialize_pte_binary(pte_data)
    program = pte_file.program

    named_data_map: Dict[str, bytes] = {}
    if pte_file.named_data is not None:
        for key, data_entry in pte_file.named_data.pte_data.items():
            named_data_map[key] = pte_file.named_data.buffers[data_entry.buffer_index]

    delegates: List[BackendDelegate] = sum(
        [execution_plan.delegates for execution_plan in program.execution_plan], []
    )
    coreml_delegates: List[BackendDelegate] = [
        delegate for delegate in delegates if delegate.id == COREML_BACKEND_ID
    ]

    extracted_paths: List[Path] = []
    extracted_keys: set = set()
    model_index: int = 1

    for coreml_delegate in coreml_delegates:
        coreml_delegate_data: BackendDelegateDataReference = coreml_delegate.processed
        if coreml_delegate_data.location != DataLocation.INLINE:
            continue

        raw_bytes = program.backend_delegate_data[coreml_delegate_data.index].data
        coreml_processed_bytes: Optional[bytes] = None
        model_name: Optional[str] = None

        if raw_bytes.startswith(_NAMED_DATA_MAGIC):
            try:
                reference = json.loads(
                    raw_bytes[len(_NAMED_DATA_MAGIC) :].decode("utf-8")
                )
                key = reference.get("key")
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"Warning: Failed to parse JSON reference: {e}")
                continue
            if key in extracted_keys:
                continue
            if key not in named_data_map:
                print(f"Warning: Named data key '{key}' not found in program")
                continue
            extracted_keys.add(key)
            coreml_processed_bytes = named_data_map[key]
            model_name = key
        else:
            coreml_processed_bytes = raw_bytes

        if coreml_processed_bytes is None:
            continue

        if model_name is None:
            model_name = f"model_{model_index}"

        model_path = out_root / model_name
        if model_path.exists():
            shutil.rmtree(model_path)
        model_path.mkdir(parents=True)

        if executorchcoreml.unflatten_directory_contents(
            coreml_processed_bytes, str(model_path.absolute())
        ):
            extracted_paths.append(model_path)

        model_index += 1

    return extracted_paths


def main() -> None:
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
        extracted_paths = extract_coreml_models(pte_data)

    if extracted_paths:
        for path in extracted_paths:
            print(f"Core ML models are extracted and saved to path = {path}")
    else:
        print("The model isn't delegated to Core ML.")


if __name__ == "__main__":
    main()
