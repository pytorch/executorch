import copy
import logging

from typing import Dict, Optional

from executorch.backends.test.harness.stages.stage import Stage, StageType
from executorch.exir import ExecutorchProgramManager

from torch.utils._pytree import tree_flatten

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
try:
    from executorch.extension.pybindings.portable_lib import (  # @manual
        _load_for_executorch_from_buffer,
        Verification,
    )
except ImportError as e:
    logger.warning(f"{e=}")
    pass


class Serialize(Stage):
    def __init__(self):
        self.buffer = None
        self.data_files: Dict[str, bytes] = {}

    def stage_type(self) -> StageType:
        return StageType.SERIALIZE

    def run(self, artifact: ExecutorchProgramManager, inputs=None) -> None:
        self.buffer = artifact.buffer
        # Capture external data files (e.g., .ptd files for CUDA backend)
        self.data_files = artifact.data_files

    @property
    def artifact(self) -> bytes:
        return self.buffer

    @property
    def graph_module(self) -> None:
        return None

    def run_artifact(self, inputs):
        inputs_flattened, _ = tree_flatten(inputs)

        # Combine all external data files into a single buffer for data_map_buffer
        # Most backends have at most one external data file, but we concatenate
        # in case there are multiple (though this may not be fully supported)
        data_map_buffer = None
        if self.data_files:
            # If there's exactly one data file, use it directly
            # Otherwise, log a warning - multiple external files may need special handling
            if len(self.data_files) == 1:
                data_map_buffer = list(self.data_files.values())[0]
            else:
                # For multiple files, we use the first one and warn
                # This is a limitation - proper handling would need runtime support
                logger.warning(
                    f"Multiple external data files found ({list(self.data_files.keys())}). "
                    f"Using the first one. This may not work correctly for all backends."
                )
                data_map_buffer = list(self.data_files.values())[0]

        executorch_module = _load_for_executorch_from_buffer(
            self.buffer,
            data_map_buffer=data_map_buffer,
            program_verification=Verification.Minimal,
        )
        executorch_output = copy.deepcopy(
            executorch_module.run_method("forward", tuple(inputs_flattened))
        )
        return executorch_output

    def dump_artifact(self, path_to_dump: Optional[str]):
        """
        dump_artifact is overridden to dump the serialized bytes into pte file
        """
        if not path_to_dump:
            raise RuntimeError("path_to_dump file not provided")
        else:
            with open(path_to_dump, "wb") as f:
                f.write(self.artifact)
