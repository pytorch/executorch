# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compile-only metadata placeholders for explicitly runtime-owned KV buffers."""

import contextlib
import re

import torch
from torch.multiprocessing.reductions import StorageWeakRef


class RuntimeOwnedKVCompileBuffers:
    """Identify buffers by the lowered program's FQNs and storage identities.

    A graph-owned alias excludes a storage from this optimization. Cloned
    placeholders retain the original shape, stride and offset but have no
    payload; only the runtime can supply that payload. Keep this object scoped
    to one program's compilation, including structured-weight collection.
    """

    def __init__(self, program, fqn_prefix: str):
        named = (*program.state_dict.items(), *program.constants.items())
        owned, other = set(), set()
        for name, tensor in named:
            if not isinstance(tensor, torch.Tensor):
                continue
            key = StorageWeakRef(tensor.untyped_storage())
            target = owned if name.startswith(fqn_prefix) else other
            target.add(key)
        self._storages = owned - other
        self._clones = []

    def owns_empty(self, tensor) -> bool:
        return (
            isinstance(tensor, torch.Tensor)
            and tensor.numel() > 0
            and tensor.untyped_storage().nbytes() == 0
            and StorageWeakRef(tensor.untyped_storage()) in self._storages
        )

    def cpu_placeholder(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.owns_empty(tensor):
            raise ValueError("Expected an empty runtime-owned KV buffer")
        required = tensor.storage_offset() + 1
        for size, stride in zip(tensor.shape, tensor.stride()):
            required += (size - 1) * stride
        # Reserve an uninitialized view, then immediately release its storage.
        # Unlike zeros(), this never initializes a model-sized CPU KV cache.
        backing = torch.empty(int(required), dtype=tensor.dtype, device="cpu")
        value = torch.as_strided(
            backing, tensor.shape, tensor.stride(), tensor.storage_offset()
        )
        value.untyped_storage().resize_(0)
        self._storages.add(StorageWeakRef(value.untyped_storage()))
        self._clones.append(value)
        return value

    @contextlib.contextmanager
    def constant_info_sizes(self, wrapper, tensors, alignment: int):
        """Keep AOTI's declared extent independent from empty payload storage.

        Inductor emits constant data_size from storage.nbytes(), which is zero
        for these views. The CUDA runtime checks the declared extent before
        binding its own storage. Change only that field for identified empty
        runtime buffers; leave shape, dtype, offset and blob packing untouched.
        """
        if alignment not in (1, 64):
            raise ValueError("Unsupported AOTI constant alignment")
        sizes = {}
        for index, tensor in enumerate(tensors):
            if self.owns_empty(tensor):
                extent = tensor.storage_offset() + 1
                for size, stride in zip(tensor.shape, tensor.stride()):
                    extent += (size - 1) * stride
                size_bytes = extent * tensor.element_size()
                sizes[index] = (size_bytes + alignment - 1) // alignment * alignment
        if not sizes:
            yield
            return
        lines = wrapper.prefix.get_lines_ref()
        first = len(lines)
        yield
        replacements = {}
        seen = set()
        for line_index in range(first, len(lines)):
            line = lines[line_index]
            match = (
                re.fullmatch(r"(\s*)constants_info_\[(\d+)\]\.data_size = (\d+);", line)
                if isinstance(line, str)
                else None
            )
            if match is not None and int(match[2]) in sizes:
                index = int(match[2])
                if int(match[3]) != 0 or index in seen:
                    raise RuntimeError(
                        "Unexpected runtime-owned AOTI constant size emission"
                    )
                replacements[line_index] = (
                    f"{match[1]}constants_info_[{index}].data_size = {sizes[index]};"
                )
                seen.add(index)
        if seen != sizes.keys():
            raise RuntimeError("Missing runtime-owned AOTI constant size emission")
        for line_index, line in replacements.items():
            lines[line_index] = line
