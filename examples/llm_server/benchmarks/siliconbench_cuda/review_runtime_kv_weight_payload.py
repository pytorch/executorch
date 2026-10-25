"""Compare every model-level FQN weight storage in two PTD packages on CPU.

Generated shared objects and library-local TensorConstants are reported
separately. This verifies identical original quantized and unquantized model
storage bytes; it does not infer GPU residency from a file's size.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def stream_digest(file, offset=0, size=None):
    file.seek(offset)
    h = hashlib.sha256()
    while size is None or size:
        chunk = file.read(
            8 * 1024 * 1024 if size is None else min(size, 8 * 1024 * 1024)
        )
        if not chunk:
            assert size is None, "Truncated PTD storage"
            break
        h.update(chunk)
        if size is not None:
            size -= len(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(args.source.parent))
    from executorch.extension.flat_tensor.serialize.serialize import (
        FlatTensorHeader,
        _deserialize_to_flat_tensor,
    )

    def read(path):
        with path.open("rb") as file:
            header = FlatTensorHeader.from_bytes(file.read(48)[8:])
            assert header.is_valid()
            assert (
                header.segment_base_offset + header.segment_data_size
                <= path.stat().st_size
            )
            file.seek(0)
            metadata = _deserialize_to_flat_tensor(
                file.read(header.flatbuffer_offset + header.flatbuffer_size)
            )
            assert metadata.version == 0
            weights, other = {}, []
            cached = {}
            for entry in metadata.named_data:
                assert 0 <= entry.segment_index < len(metadata.segments)
                segment = metadata.segments[entry.segment_index]
                assert (
                    0 <= segment.offset
                    and segment.offset + segment.size <= header.segment_data_size
                )
                pieces = entry.key.split(":", 2)
                # cuda_weight_collector._storage_key gives original model FQNs
                # stable keys. A library-local _tensor_constant is scoped by an
                # extra library key and is not an original model parameter.
                model_weight = (
                    len(pieces) == 3
                    and pieces[0] == "cuda_fqn_weight"
                    and pieces[1] in ["cpu", "cuda"]
                    and ":" not in pieces[2]
                    and not pieces[2].startswith("_tensor_constant")
                )
                if not model_weight:
                    other.append(dict(key=entry.key, bytes=segment.size))
                    continue
                assert entry.key not in weights
                if entry.segment_index not in cached:
                    cached[entry.segment_index] = stream_digest(
                        file, header.segment_base_offset + segment.offset, segment.size
                    )
                weights[entry.key] = dict(
                    bytes=segment.size, sha256=cached[entry.segment_index]
                )
            assert weights and any(
                "parametrizations.weight.original0" in key for key in weights
            )
            return dict(
                PTD_sha256=stream_digest(file),
                model_storages=weights,
                unique_model_storage_count=len(cached),
                other_entries=other,
            )

    fixed, runtime = read(args.fixed), read(args.runtime)
    assert fixed["model_storages"].keys() == runtime["model_storages"].keys()
    mismatches = [
        key
        for key in fixed["model_storages"]
        if fixed["model_storages"][key] != runtime["model_storages"][key]
    ]
    result = dict(
        fixed_PTD_sha256=fixed["PTD_sha256"],
        runtime_PTD_sha256=runtime["PTD_sha256"],
        all_quantized_and_nonquantized_storage_sha256_equal=not mismatches,
        all_model_level_FQN_storage_keys_equal=True,
        model_level_FQN_storage_count=len(fixed["model_storages"]),
        mismatching_model_storages=mismatches,
        fixed=fixed,
        runtime=runtime,
        CPU_only=True,
        model_weights_or_quantization_modified=False,
        scope="Every original model-level FQN storage, including quantized bytes/scales/zeros and BF16 exceptions. Generated library-local constants and shared objects are disclosed separately; PTD bytes are not GPU memory.",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    assert not mismatches, mismatches
    print(
        json.dumps(
            {
                "model_weight_storages": result["model_level_FQN_storage_count"],
                "identical": not mismatches,
            }
        )
    )


if __name__ == "__main__":
    main()
