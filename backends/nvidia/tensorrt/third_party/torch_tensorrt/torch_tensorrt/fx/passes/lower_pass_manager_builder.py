import datetime
import logging
from functools import partial, wraps
from typing import Any, Callable, Optional, Sequence

import torch
from torch import nn
from torch.fx.passes.pass_manager import PassManager, inplace_wrapper
from torch.fx.passes.shape_prop import ShapeProp
from torch.fx.passes.splitter_base import SplitResult, generate_inputs_for_submodules
from torch_tensorrt.fx.passes.pass_utils import apply_bfloat_float_conversion
from torch_tensorrt.fx.utils import LowerPrecision

from ..input_tensor_spec import generate_input_specs
from ..lower_setting import LowerSetting
from ..observer import Observer
from ..passes.remove_duplicate_output_args import remove_duplicate_output_args
from .graph_opts import common_subexpression_elimination
from .lower_basic_pass import (  # noqa
    fix_clamp_numerical_limits_to_fp16,
    fix_reshape_batch_dim,
    replace_mutable_op,
    replace_op_with_indices,
    run_const_fold,
)

_LOGGER: logging.Logger = logging.getLogger(__name__)


Input = Sequence[Any]


# ----------------------------------------------------------------------
# OBSERVERS
# ----------------------------------------------------------------------
# List of observers. We can subscribe to them by calling its `add(callback)`
# function from anywhere in code:
#
# >>> from torch_tensorrt.fx.lower import FUSE_PASSES_POST_OBSERVER
# >>> with FUSE_PASSES_POST_OBSERVER.add(print_module_and_input):
# >>>     # print_module_and_input will be called right after the fuse passes
# >>>     lower(module, sample_input)

# Observer for the model after the fuse passes.
FUSE_PASSES_POST_OBSERVER: Observer[Callable[[nn.Module, Input], None]] = Observer(
    "FUSE_PASSES_POST_OBSERVER"
)

# Observer for the TRT split submodules before lowering
LOWER_SPLIT_PRE_OBSERVER: Observer[Callable[[str, nn.Module, Input], None]] = Observer(
    "LOWER_SPLIT_PRE_OBSERVER"
)

# Observer for the TRT split submodules after lowering
LOWER_SPLIT_POST_OBSERVER: Observer[Callable[[str, nn.Module, Input], None]] = Observer(
    "LOWER_SPLIT_POST_OBSERVER"
)
# ----------------------------------------------------------------------


def wrapper(fn: Callable, input) -> Callable:
    @wraps(fn)
    def wrapped_fn(gm):
        if isinstance(gm, torch.fx.GraphModule):
            ShapeProp(gm).propagate(*input)
        return fn(gm, input)

    return wrapped_fn


class LowerPassManagerBuilder:
    """
    Build PassManager for lowering.

     Attributes:
        lower_setting: Setting that will be used during process of lowering, see lower_setting.py for the details.
        _trace_func: fx trace function for TRT conversion.
        _split_func: the fx2trt split function.
        _lower_func: function to create and run `TRTInterpreter` to convert `fx.GraphModule`
            into a TensorRT engine.

    """

    def __init__(
        self,
        lower_setting: LowerSetting,
        trace_func: Callable,
        split_func: Callable,
        lower_func: Callable,
    ):
        self.lower_setting = lower_setting
        self._trace_func = trace_func
        self._split_func = split_func
        self._lower_func = lower_func

    def _const_fold_pass(self) -> PassManager:
        passes = [
            wrapper(self._trace_func, self._input),
            run_const_fold,
        ]
        return PassManager.build_from_passlist(passes)

    def graph_optimization_pass(self) -> PassManager:
        passes = [
            wrapper(self._trace_func, self._input),
        ]
        for p in self.lower_setting.customized_fuse_pass.passes:
            passes.append(wrapper(p, self._input))
        for p in self.lower_setting.lower_basic_fuse_pass.passes:
            passes.append(wrapper(p, self._input))
        if (
            hasattr(self.lower_setting, "lower_precision")
            and self.lower_setting.lower_precision is LowerPrecision.FP16
        ) or (
            hasattr(self.lower_setting, "precision")
            and self.lower_setting.precision is LowerPrecision.FP16
        ):
            passes.append(wrapper(fix_clamp_numerical_limits_to_fp16, self._input))

        passes.append(inplace_wrapper(common_subexpression_elimination))
        passes.append(
            inplace_wrapper(lambda m: FUSE_PASSES_POST_OBSERVER.observe(m, self._input))
        )
        passes.append(fix_reshape_batch_dim)

        return PassManager.build_from_passlist(passes)

    def graph_optimization_pass_aten(self) -> PassManager:
        passes = []

        for p in self.lower_setting.customized_fuse_pass.passes:
            passes.append(wrapper(p, self._input))
        for p in self.lower_setting.lower_basic_fuse_pass.passes:
            passes.append(wrapper(p, self._input))
        # TODO fix this pass for aten graph
        # if (
        #     hasattr(self.lower_setting, "lower_precision")
        #     and self.lower_setting.lower_precision is LowerPrecision.FP16
        # ) or (
        #     hasattr(self.lower_setting, "precision")
        #     and self.lower_setting.precision is LowerPrecision.FP16
        # ):
        #     passes.append(wrapper(fix_clamp_numerical_limits_to_fp16, self._input))

        passes.append(
            inplace_wrapper(lambda m: FUSE_PASSES_POST_OBSERVER.observe(m, self._input))
        )
        # TODO we most likely do not need it for aten
        # passes.append(fix_reshape_batch_dim)

        return PassManager.build_from_passlist(passes)

    def _split_pass(self) -> PassManager:
        passes = [
            partial(
                self._split_func, inputs=self._input, lower_setting=self.lower_setting
            )
        ]
        passes.append(
            inplace_wrapper(
                lambda split_result: remove_duplicate_output_args(
                    split_result.split_module, split_result.submodule_inputs.keys()
                )
            )
        )
        return PassManager.build_from_passlist(passes)

    def _trt_lower_pass(self) -> PassManager:
        def lower_func(split_result: SplitResult) -> nn.Module:
            if (
                hasattr(self.lower_setting, "explicit_batch_dimension")
                and self.lower_setting.explicit_batch_dimension
                and self._additional_input
            ):
                additional_submodule_inputs = generate_inputs_for_submodules(
                    split_result.split_module,
                    self._additional_input,
                    list(split_result.submodule_inputs.keys()),
                )
            else:
                additional_submodule_inputs = None

            for submod_name, submod_inputs in split_result.submodule_inputs.items():
                submod = getattr(split_result.split_module, submod_name)

                LOWER_SPLIT_PRE_OBSERVER.observe(submod_name, submod, submod_inputs)

                # Only acc submodules will be lowered.
                if not submod_name.startswith(split_result.non_acc_submodule_prefix):
                    _LOGGER.info(f"Now lowering submodule {submod_name}")
                    lowering_start_time = datetime.datetime.now()

                    self.lower_setting.input_specs = generate_input_specs(
                        submod_inputs,
                        self.lower_setting,
                        (
                            additional_submodule_inputs[submod_name]
                            if additional_submodule_inputs
                            else None
                        ),
                    )
                    lowered_module = self._lower_func(
                        submod, submod_inputs, self.lower_setting, submod_name
                    )
                    setattr(split_result.split_module, submod_name, lowered_module)
                    LOWER_SPLIT_POST_OBSERVER.observe(
                        submod_name, lowered_module, submod_inputs
                    )
                    _LOGGER.info(
                        f"Lowering submodule {submod_name} elapsed time {datetime.datetime.now() - lowering_start_time}"
                    )

            return split_result.split_module

        return PassManager.build_from_passlist([lower_func])

    def _default_lower_pass(self) -> PassManager:
        def lower_func(split_result: SplitResult) -> nn.Module:
            if self._additional_input:
                additional_submodule_inputs = generate_inputs_for_submodules(
                    split_result.split_module,
                    self._additional_input,
                    list(split_result.submodule_inputs.keys()),
                )
            else:
                additional_submodule_inputs = None

            for submod_name, submod_inputs in split_result.submodule_inputs.items():
                submod = getattr(split_result.split_module, submod_name)

                LOWER_SPLIT_PRE_OBSERVER.observe(submod_name, submod, submod_inputs)
                # Only acc submodules will be lowered.
                if not submod_name.startswith(split_result.non_acc_submodule_prefix):
                    _LOGGER.info(f"ACC submodule graph: {submod.graph}")
                    lowering_start_time = datetime.datetime.now()

                    self.lower_setting.additional_inputs = (
                        (
                            additional_submodule_inputs[submod_name]
                            if additional_submodule_inputs
                            else None
                        ),
                    )

                    lowered_module = self._lower_func(
                        submod, submod_inputs, self.lower_setting, submod_name
                    )
                    setattr(split_result.split_module, submod_name, lowered_module)
                    LOWER_SPLIT_POST_OBSERVER.observe(
                        submod_name, lowered_module, submod_inputs
                    )
                    _LOGGER.info(
                        f"Lowering submodule {submod_name} elapsed time {datetime.datetime.now() - lowering_start_time}"
                    )
                else:
                    _LOGGER.info(f"GPU submodule graph: {submod.graph}")
                    apply_bfloat_float_conversion(submod, submod_inputs, submod_name)

            return split_result.split_module

        return PassManager.build_from_passlist([lower_func])

    def _default_replace_mutable_op_pass(self) -> PassManager:
        return PassManager.build_from_passlist([replace_mutable_op])

    def build_trt_lower_pipeline(
        self, input: Input, additional_input: Optional[Input] = None
    ) -> PassManager:
        self._input = input
        self._additional_input = additional_input
        passes = []

        passes.append(self._default_replace_mutable_op_pass())
        passes.append(self._const_fold_pass())
        passes.append(self.graph_optimization_pass())
        passes.append(self._split_pass())
        passes.append(self._trt_lower_pass())

        pm = PassManager.build_from_passlist(passes)
        return pm

    def build_aten2trt_lower_pipeline(
        self, input: Input, additional_input: Optional[Input] = None
    ) -> PassManager:
        self._input = input
        self._additional_input = additional_input
        passes = []
        passes.append(
            wrapper(self._trace_func, self._input),
        )
        passes.append(self.graph_optimization_pass_aten())
        passes.append(self._split_pass())
        passes.append(self._trt_lower_pass())

        pm = PassManager.build_from_passlist(passes)
        return pm

    def build_default_lower_pipeline(
        self, input: Input, additional_input: Optional[Input] = None
    ) -> PassManager:
        self._input = input
        self._additional_input = additional_input
        passes = []

        passes.append(self._default_replace_mutable_op_pass())
        passes.append(self._const_fold_pass())
        passes.append(self.graph_optimization_pass())
        passes.append(self._split_pass())
        passes.append(self._default_lower_pass())

        pm = PassManager.build_from_passlist(passes)
        return pm
