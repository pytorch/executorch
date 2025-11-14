import collections
import logging
import os

import torch


def save_tensors(tensors, prefix, artifact_dir):
    if isinstance(tensors, tuple):
        for index, output in enumerate(tensors):
            save_path = prefix + "_" + str(index) + ".bin"
            output.detach().numpy().tofile(os.path.join(artifact_dir, save_path))
    elif isinstance(tensors, torch.Tensor):
        tensors.detach().numpy().tofile(os.path.join(artifact_dir, prefix + ".bin"))
    elif isinstance(tensors, collections.OrderedDict):
        for index, output in enumerate(tensors.values()):
            save_path = prefix + "_" + str(index) + ".bin"
            output.detach().numpy().tofile(os.path.join(artifact_dir, save_path))
    else:
        logging.warning("Unsupported type (", type(tensors), ") skip saving tensor. ")
