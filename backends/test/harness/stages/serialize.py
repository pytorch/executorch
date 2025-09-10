import copy
import logging

from typing import Optional

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

    def stage_type(self) -> StageType:
        return StageType.SERIALIZE

    def run(self, artifact: ExecutorchProgramManager, inputs=None) -> None:
        self.buffer = artifact.buffer

    @property
    def artifact(self) -> bytes:
        return self.buffer

    @property
    def graph_module(self) -> None:
        return None

    def run_artifact(self, inputs):
        inputs_flattened, _ = tree_flatten(inputs)
        executorch_module = _load_for_executorch_from_buffer(
            self.buffer, program_verification=Verification.Minimal
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
