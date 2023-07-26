import copy
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
from executorch import exir
from executorch.backends.backend_api import to_backend, validation_disabled
from executorch.backends.partitioner import Partitioner
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackFloatingPointPartitioner,
)
from executorch.backends.xnnpack.utils.configs import (
    get_xnnpack_capture_config,
    get_xnnpack_edge_compile_config,
)
from executorch.exir import (
    CaptureConfig,
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    ExecutorchProgram,
    ExirExportedProgram,
)
from executorch.exir.passes.spec_prop_pass import SpecPropPass
from executorch.exir.serialize import serialize_to_flatbuffer

# pyre-ignore[21]: Could not find module `executorch.pybindings.portable`.
from executorch.extension.pybindings.portable import (  # @manual
    _load_for_executorch_from_buffer,
)
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.backend_config.executorch import (
    get_executorch_backend_config,
)
from torch.ao.quantization.pt2e.quantizer import XNNPACKQuantizer
from torch.ao.quantization.pt2e.quantizer.quantizer import QuantizationConfig, Quantizer
from torch.ao.quantization.pt2e.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
)
from torch.ao.quantization.qconfig_mapping import (
    _get_symmetric_qnnpack_qconfig_mapping,
    QConfigMapping,
)

from torch.ao.quantization.quantize_fx import (
    _convert_to_reference_decomposed_fx,
    prepare_fx,
)
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.testing import FileCheck
from torch.utils._pytree import tree_flatten


class Stage(ABC):
    """
    Interface for a Stage in the PT2.0 lowering pipeline
    """

    @property
    @abstractmethod
    def artifact(self):
        pass

    @abstractmethod
    def run(self, artifact, inputs):
        pass

    # TODO: use __repr__
    @abstractmethod
    def check_str(self):
        pass


_stages_: Dict[str, Stage] = {}


def register_stage(stage: Stage):
    assert isinstance(stage, type)
    name = stage.__qualname__
    if name in _stages_:
        raise RuntimeError(f"Duplicate stage in Tester, {name}")
    _stages_[name] = stage
    return stage


@register_stage
class Quantize(Stage):
    def __init__(
        self,
        qconfig_mapping: Optional[QConfigMapping] = None,
        backend_config: Optional[BackendConfig] = None,
    ):
        self.qconfig_mapping = (
            qconfig_mapping or _get_symmetric_qnnpack_qconfig_mapping()
        )
        self.backend_config = backend_config or get_executorch_backend_config()
        self.converted = None

    @property
    def artifact(self) -> torch.fx.GraphModule:
        return self.converted

    def run(self, artifact: torch.nn.Module, inputs: Tuple[torch.Tensor]) -> None:
        prepared = prepare_fx(
            artifact, self.qconfig_mapping, inputs, backend_config=self.backend_config
        )
        self.converted = _convert_to_reference_decomposed_fx(
            prepared, backend_config=self.backend_config
        )

    def check_str(self) -> str:
        return self.converted.code


@register_stage
class Quantize2(Stage):
    def __init__(
        self,
        quantizer: Optional[Quantizer] = None,
        quantization_config: Optional[QuantizationConfig] = None,
    ):
        self.quantizer = quantizer or XNNPACKQuantizer()
        self.quantization_config = (
            quantization_config or get_symmetric_quantization_config()
        )

        self.quantizer.set_global(self.quantization_config)

        self.converted_program = None

    @property
    def artifact(self) -> ExirExportedProgram:
        return self.converted_program

    def run(
        self, artifact: ExirExportedProgram, inputs: Optional[Tuple[torch.Tensor]]
    ) -> None:
        prepared = prepare_pt2e(artifact.graph_module, self.quantizer)
        converted = convert_pt2e(prepared)
        artifact.graph_module = converted
        self.converted_program = artifact

    def check_str(self) -> str:
        return self.converted_program.graph_module.code


@register_stage
class Export(Stage):
    def __init__(self, capture_config: Optional[CaptureConfig] = None):
        self.capture_conf = capture_config or get_xnnpack_capture_config()
        self.exir_exported_program = None

    @property
    def artifact(self) -> ExirExportedProgram:
        return self.exir_exported_program

    def run(self, artifact: torch.nn.Module, inputs) -> None:
        self.exir_exported_program = exir.capture(artifact, inputs, self.capture_conf)

    def check_str(self) -> str:
        return self.exir_exported_program.graph_module.code


@register_stage
class ToEdge(Stage):
    def __init__(self, edge_compile_config: Optional[EdgeCompileConfig] = None):
        self.edge_compile_conf = (
            edge_compile_config or get_xnnpack_edge_compile_config()
        )
        self.edge_dialect_program = None

    @property
    def artifact(self) -> ExirExportedProgram:
        return self.edge_dialect_program

    def run(self, artifact: ExirExportedProgram, inputs=None) -> None:
        self.edge_dialect_program = artifact.to_edge(self.edge_compile_conf)

    def check_str(self) -> str:
        return self.edge_dialect_program.graph_module.code


@register_stage
class Partition(Stage):
    def __init__(self, partitioner: Optional[Partitioner] = None):
        self.partitioner = partitioner or XnnpackFloatingPointPartitioner
        self.delegate_module = None

    @property
    def artifact(self) -> torch.fx.GraphModule:
        return self.delegate_module

    def run(self, artifact: ExirExportedProgram, inputs=None):
        with validation_disabled():
            self.delegate_module = to_backend(artifact, self.partitioner)

    def check_str(self) -> str:
        return self.delegate_module.code


@register_stage
class ToExecutorch(Stage):
    def __init__(
        self,
        config: Optional[ExecutorchBackendConfig] = None,
    ):
        self.config = config or ExecutorchBackendConfig(
            passes=[SpecPropPass()],
        )
        self.exported_program = None

    @property
    def artifact(self) -> ExecutorchProgram:
        return self.exported_program

    def run(self, artifact: ExirExportedProgram, inputs=None):
        self.exported_program = artifact.to_executorch(self.config)

    def check_str(self) -> str:
        return self.exported_program.graph_module.code


@register_stage
class Serialize(Stage):
    def __init__(self):
        self.buffer = None

    @property
    def artifact(self) -> bytes:
        return self.buffer

    def run(self, artifact, inputs=None) -> None:
        self.buffer = serialize_to_flatbuffer(artifact)

    def check_str(self) -> str:
        return ""


class Tester:
    def __init__(
        self,
        module: torch.nn.Module,
        inputs: Tuple[torch.Tensor],
    ):
        self.module = module
        self.inputs = inputs
        self.stages: Dict[str, Stage] = OrderedDict.fromkeys(list(_stages_.keys()))

        # Current stage name
        self.cur: str = ""

        # Reference output from Eager mode
        self.reference_output = None

        # Output by running a serialized/lowered module on ET
        self.executorch_output = None

    @staticmethod
    def _stage_name(stage) -> str:
        t = stage if isinstance(stage, type) else type(stage)
        return t.__qualname__

    def _pre(self, stage):
        name = self._stage_name(stage)
        assert name in self.stages and not self.stages[name]

        # TODO: do a basic state machine check

        last_artifact = self.get_artifact() if self.cur else self.module
        self.cur = name
        return last_artifact

    def _post(self, stage):
        name = self._stage_name(stage)
        assert name in self.stages
        self.stages[name] = stage

    @staticmethod
    def _assert_outputs_equal(model_output, ref_output, atol=1e-03, rtol=1e-03):
        """
        Helper testing function that asserts that the model output and the reference output
        are equal with some tolerance. Due to numerical differences between eager mode and
        the XNNPACK's backend, we relax the detal such that absolute tolerance is 1e-3. and
        relative tolerance is 1e-3.
        """

        # Compare the result from executor and eager mode direclty
        if isinstance(ref_output, tuple) or isinstance(ref_output, list):
            # Multiple outputs executor always returns tuple, even if there is one output
            assert len(ref_output) == len(model_output)
            for i in range(len(ref_output)):
                assert torch.allclose(
                    model_output[i],
                    ref_output[i],
                    atol=atol,
                    rtol=rtol,
                )
        else:
            # If one output, eager returns tensor while executor returns a tuple(tensor) of size 1
            assert torch.allclose(model_output[0], ref_output, atol=atol, rtol=rtol)

    # Stages
    def quantize(self, quantize_stage: Optional[Quantize] = None):
        quantize_stage = quantize_stage or Quantize()
        last = self._pre(quantize_stage)
        quantize_stage.run(last, self.inputs)
        self._post(quantize_stage)
        return self

    def export(self, export_stage: Optional[Export] = None):
        export_stage = export_stage or Export()
        last = self._pre(export_stage)
        export_stage.run(last, self.inputs)
        self._post(export_stage)
        return self

    def quantize2(self, quantize_stage: Optional[Quantize2] = None):
        quantize_stage = quantize_stage or Quantize2()
        last = self._pre(quantize_stage)
        quantize_stage.run(last, self.inputs)
        self._post(quantize_stage)
        return self

    def to_edge(self, to_edge_stage: Optional[ToEdge] = None):
        to_edge_stage = to_edge_stage or ToEdge()
        last = self._pre(to_edge_stage)
        to_edge_stage.run(last)
        self._post(to_edge_stage)
        return self

    def partition(self, partition_stage: Optional[Partition] = None):
        partition_stage = partition_stage or Partition()
        last = self._pre(partition_stage)
        partition_stage.run(last)
        self._post(partition_stage)
        return self

    def to_executorch(self, to_executorch_stage: Optional[ToExecutorch] = None):
        to_executorch_stage = to_executorch_stage or ToExecutorch()
        last = self._pre(to_executorch_stage)
        to_executorch_stage.run(last)
        self._post(to_executorch_stage)
        return self

    def serialize(self, serialize_stage: Optional[Serialize] = None):
        serialize_stage = serialize_stage or Serialize()
        last = self._pre(serialize_stage)
        serialize_stage.run(last.program)
        self._post(serialize_stage)
        return self

    # Util functions
    def get_artifact(self, stage: Optional[str] = None):
        stage = stage or self.cur
        return self.stages[stage].artifact

    def check(self, input: List[str]):
        for key in input:
            FileCheck().check(key).run(self.stages[self.cur].check_str())
        return self

    def check_not(self, input: List[str]):
        for key in input:
            FileCheck().check_not(key).run(self.stages[self.cur].check_str())
        return self

    def check_count(self, input: Dict[Any, int]):
        # TODO target checks similar to checkGraphModuleNodes()
        for key, count in input.items():
            FileCheck().check_count(key, count, exactly=True).run(
                self.stages[self.cur].check_str()
            )
        return self

    def run_method(self, method="forward"):
        # Reference
        delegated_module = self.get_artifact(self._stage_name(Partition))
        self.reference_output = delegated_module(*self.inputs)

        # Executorch
        inputs_flattened, _ = tree_flatten(self.inputs)
        serialized_buffer = self.get_artifact(self._stage_name(Serialize))
        executorch_module = _load_for_executorch_from_buffer(serialized_buffer)
        self.executorch_output = copy.deepcopy(
            executorch_module.run_method(method, tuple(inputs_flattened))
        )
        return self

    def compare_outputs(self, atol=1e-03, rtol=1e-03):
        assert self.reference_output is not None
        assert self.executorch_output is not None
        self._assert_outputs_equal(
            self.executorch_output, self.reference_output, atol=atol, rtol=rtol
        )
        return self
