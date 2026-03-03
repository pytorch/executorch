from typing import Any, Dict, Iterable, Sequence

import torch
import torch.fx.passes.operator_support as ops
import torch.fx.passes.splitter_base as splitter_base
from torch.fx.passes.tools_common import get_acc_ops_name, Tensors

from .. import (
    CONVERTERS,
    InputTensorSpec,
    NO_EXPLICIT_BATCH_DIM_SUPPORT,
    NO_IMPLICIT_BATCH_DIM_SUPPORT,
    TRTInterpreter,
    TRTModule,
)
from ..tools.trt_minimizer import TensorRTMinimizer


def create_trt_operator_support(
    use_implicit_batch_dim=True,
    exclude_support_node_name: set = (),
) -> ops.OperatorSupportBase:
    """Creates an `OperatorSupportBase` instance used for TRT splitting purpose."""
    # Create an `OperatorSupport` that declares a node supported if it
    # finds a registered TRT converter.
    support_dict: Dict[str, None] = {}
    for k in CONVERTERS.keys():
        if use_implicit_batch_dim:
            if k not in NO_IMPLICIT_BATCH_DIM_SUPPORT.keys():
                support_dict[get_acc_ops_name(k)] = None
        elif k not in NO_EXPLICIT_BATCH_DIM_SUPPORT.keys():
            support_dict[get_acc_ops_name(k)] = None
    supported_if_converter_registered = ops.OperatorSupport(support_dict=support_dict)

    return ops.chain(
        ops.OpSupports.decline_if_node_in_names(exclude_support_node_name),
        # 1. Node is not supported if it has args with int64 or float64 dtype:
        ops.OpSupports.decline_if_input_dtype(torch.int64),
        ops.OpSupports.decline_if_input_dtype(torch.float64),
        # 2. Node is supported if it has TRT converter:
        supported_if_converter_registered,
    )


class TRTSplitterSetting(splitter_base._SplitterSettingBase):
    def __init__(self):
        super().__init__()

        # Determines what batch mode we'll use for lowering.
        # During split, we'll split out the operators that
        # don't support the batch dim.
        self.use_implicit_batch_dim: bool = True
        self.exclude_support_node_name: set = set()
        self.use_experimental_rt: bool = False

        if self.use_experimental_rt and self.use_implicit_batch_dim:
            raise ValueError(
                "The experimental unifed runtime only supports explicit batch. Please make sure to set use_implicit_batch_dim=False when use_experimental_rt=True"
            )


class TRTSplitter(splitter_base._SplitterBase):
    def __init__(
        self,
        module: torch.fx.GraphModule,
        sample_input: Sequence[Any],
        operator_support: ops.OperatorSupportBase = None,
        settings: TRTSplitterSetting = None,
    ):
        if not settings:
            settings = TRTSplitterSetting()
        if not operator_support:
            operator_support = create_trt_operator_support(
                settings.use_implicit_batch_dim, settings.exclude_support_node_name
            )
        super().__init__(
            module,
            sample_input,
            operator_support,
            settings,
            non_acc_submodule_name="_run_on_gpu_",
        )

    def _lower_model_to_backend(
        self, mod: torch.fx.GraphModule, inputs: Iterable[torch.Tensor]
    ):
        """
        Lower a GraphModule `mod` to TensorRT with `inputs`.
        """
        # Current code for lowering is place-holder, subject to future change
        # based on feeds model's actual status
        interp = TRTInterpreter(mod, InputTensorSpec.from_tensors(inputs))
        interpreter_result = interp.run(*inputs)
        if self.settings.use_experimental_rt:
            import io

            from torch_tensorrt._Device import Device
            from torch_tensorrt.dynamo._TorchTensorRTModule import TorchTensorRTModule

            with io.BytesIO() as engine_bytes:
                engine_bytes.write(interpreter_result.engine.serialize())
                engine_str = engine_bytes.getvalue()

            return TorchTensorRTModule(
                engine_str,
                name=str(type(mod)),
                input_binding_names=interpreter_result.input_names,
                output_binding_names=interpreter_result.output_names,
                target_device=Device(f"cuda:{torch.cuda.current_device()}"),
                # cuda_graph_batch_size=lower_setting.cuda_graph_batch_size, # NOTE: Not sure what this is supposed to do
            )
        else:
            return TRTModule(
                interpreter_result.engine,
                interpreter_result.input_names,
                interpreter_result.output_names,
            )

    def _find_culprit(self, mod: torch.fx.GraphModule, inputs: Tensors):
        """
        This function serves the preview functionality in Splitter. When previewing
        splitting result, if something wrong happens during lowering model to TensorRT
        or running a TensorRT model, this function will be called to find any culprit
        that is responsible for the error.
        """
        # Since we don't care about accuracy here, we pass in a dummy compare function.
        minimizer = TensorRTMinimizer(mod, inputs, lambda a, b, c: (1, True))
        minimizer.settings.traverse_method = "sequential"
        minimizer.settings.find_all = True
        culprits = minimizer.minimize()

        if len(culprits) == 0:
            reports = "Unable to find a culprit!\n"
        else:
            reports = "Found some problematic nodes:\n"
            for node in culprits:
                reports += f"{node.format_node()}\n"

        return reports
