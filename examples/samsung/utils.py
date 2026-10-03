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
            _save_tensor(
                obj,
                os.path.join(
                    artifact_dir, prefix + "_" + "_".join(map(str, path_parts)) + ".bin"
                ),
            )
        elif isinstance(obj, dict):
            for key, value in obj.items():
                _collect(value, path_parts + [key])
        elif isinstance(obj, (list, tuple)):
            for i, value in enumerate(obj):
                _collect(value, path_parts + [str(i)])
        else:
            logging.warning("Skipping unsupported type: %s", type(obj))

    _collect(tensors, [])
