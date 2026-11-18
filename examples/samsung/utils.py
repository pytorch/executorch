import logging
import os

import torch


def _save_tensor(tensor: torch.Tensor, path: str) -> None:
    """Serialize a single tensor to a .bin file."""
    tensor.cpu().detach().numpy().tofile(path)


def save_tensors(tensors, prefix: str, artifact_dir: str) -> None:
    """Recursively save all tensors from arbitrary nested structures.

    Accepts tensor, list/tuple of tensors, dict of tensors, or any combination
    of the above at arbitrary depth. Each leaf tensor is written as a .bin file
    under *artifact_dir* with a name derived from *prefix* and the navigation path.
    """

    def _collect(obj, path_parts):
        if isinstance(obj, torch.Tensor):
            name = "_".join([prefix, *path_parts]) + ".bin"
            _save_tensor(obj, os.path.join(artifact_dir, name))
        elif isinstance(obj, dict):
            for i, value in enumerate(obj.values()):
                _collect(value, path_parts + [str(i)])
        elif isinstance(obj, (list, tuple)):
            for i, value in enumerate(obj):
                _collect(value, path_parts + [str(i)])
        else:
            logging.warning("Skipping unsupported type: %s", type(obj))

    _collect(tensors, [])


def save_position_embeddings(tensors, path: str) -> None:
    """Concatenate position-embedding tables into a single raw .bin file.

    ``tensors`` is a flat sequence ordered as the runtime expects the tables
    (e.g. global cos, global sin, local cos, local sin). Each tensor is
    written as C-contiguous raw bytes, back to back, without any header.
    """
    with open(path, "wb") as f:
        for tensor in tensors:
            f.write(tensor.cpu().detach().contiguous().numpy().tobytes())
