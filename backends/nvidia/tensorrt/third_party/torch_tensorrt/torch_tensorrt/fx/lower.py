import dataclasses as dc
import logging
from typing import Any, Callable, Optional, Sequence

# @manual=//deeplearning/trt/python:py_tensorrt
import tensorrt as trt
import torch
import torch.fx as fx
import torch.nn as nn
import torch_tensorrt.fx.tracer.dispatch_tracer.aten_tracer as aten_tracer
from torch.fx.passes.splitter_base import SplitResult

from .fx2trt import TRTInterpreter, TRTInterpreterResult
from .lower_setting import LowerSetting
from .passes.lower_pass_manager_builder import LowerPassManagerBuilder
from .passes.pass_utils import PassFunc, validate_inference
from .tools.timing_cache_utils import TimingCacheManager
from .tools.trt_splitter import TRTSplitter, TRTSplitterSetting
from .tracer.acc_tracer import acc_tracer
from .trt_module import TRTModule
from .utils import LowerPrecision

logger = logging.getLogger(__name__)

Input = Sequence[Any]


def compile(
    module: nn.Module,
    input,
    min_acc_module_size: int = 10,
    max_batch_size: int = 2048,
    max_workspace_size=1 << 25,
    explicit_batch_dimension=False,
    lower_precision=LowerPrecision.FP16,
    verbose_log=False,
    timing_cache_prefix="",
    save_timing_cache=False,
    cuda_graph_batch_size=-1,
    dynamic_batch=True,
    is_aten=False,
    use_experimental_fx_rt=False,
    correctness_atol=1e-1,
    correctness_rtol=1e-1,
) -> nn.Module:
    """
    Takes in original module, input and lowering setting, run lowering workflow to turn module
    into lowered module, or so called TRTModule.

    Args:
        module: Original module for lowering.
        input: Input for module.
        max_batch_size: Maximum batch size (must be >= 1 to be set, 0 means not set)
        min_acc_module_size: Minimal number of nodes for an accelerated submodule
        max_workspace_size: Maximum size of workspace given to TensorRT.
        explicit_batch_dimension: Use explicit batch dimension in TensorRT if set True, otherwise use implicit batch dimension.
        lower_precision: lower_precision config given to TRTModule.
        verbose_log: Enable verbose log for TensorRT if set True.
        timing_cache_prefix: Timing cache file name for timing cache used by fx2trt.
        save_timing_cache: Update timing cache with current timing cache data if set to True.
        cuda_graph_batch_size: Cuda graph batch size, default to be -1.
        dynamic_batch: batch dimension (dim=0) is dynamic.
        use_experimental_fx_rt: Uses the next generation TRTModule which supports both Python and TorchScript based execution (including in C++).
    Returns:
        A torch.nn.Module lowered by TensorRT.
    """
    if use_experimental_fx_rt and not explicit_batch_dimension:
        raise ValueError(
            "The experimental unifed runtime only supports explicit batch. Please make sure to set explicit_batch_dimension=True when use_experimental_fx_rt=True"
        )

    lower_setting = LowerSetting(
        max_batch_size=max_batch_size,
        min_acc_module_size=min_acc_module_size,
        max_workspace_size=max_workspace_size,
        explicit_batch_dimension=explicit_batch_dimension,
        lower_precision=lower_precision,
        verbose_log=verbose_log,
        timing_cache_prefix=timing_cache_prefix,
        save_timing_cache=save_timing_cache,
        cuda_graph_batch_size=cuda_graph_batch_size,
        dynamic_batch=dynamic_batch,
        is_aten=is_aten,
        use_experimental_rt=use_experimental_fx_rt,
        correctness_atol=correctness_atol,
        correctness_rtol=correctness_rtol,
    )
    lowerer = Lowerer.create(lower_setting=lower_setting)
    return lowerer(module, input)


@dc.dataclass
class LowerTrtInterpreter:
    lower_setting: LowerSetting
    timing_cache_manager: TimingCacheManager

    @classmethod
    def create(cls, lower_setting):
        timing_cache_manager = TimingCacheManager(
            lower_setting.timing_cache_prefix, lower_setting.save_timing_cache
        )
        return LowerTrtInterpreter(lower_setting, timing_cache_manager)

    def __call__(self, mod, input, split_name) -> TRTInterpreterResult:
        assert self.lower_setting.input_specs, "Can't find input specs for lowering!"
        logger.info(
            f"split_name={split_name}, input_specs={self.lower_setting.input_specs}"
        )

        # Prepare algorithm selector and timing_cache for TRTInterpreter
        algo_selector = None
        if self.lower_setting.algo_selector:
            algo_selector = self.lower_setting.algo_selector(f"{split_name}.json")
        cache_data = None
        if self.timing_cache_manager:
            try:
                cache_data = self.timing_cache_manager.get_timing_cache_trt(split_name)
                logger.info("Timing cache is used!")
            except Exception as e:
                logger.warning(f"Cannot load timing cache for {split_name}: {str(e)}")
                cache_data = None

        interpreter = TRTInterpreter(
            mod,
            input_specs=self.lower_setting.input_specs,
            explicit_batch_dimension=self.lower_setting.explicit_batch_dimension,
            explicit_precision=self.lower_setting.explicit_precision,
            logger_level=(
                trt.Logger.VERBOSE
                if self.lower_setting.verbose_log
                else trt.Logger.WARNING
            ),
        )

        interp_result: TRTInterpreterResult = interpreter.run(
            max_batch_size=self.lower_setting.max_batch_size,
            max_workspace_size=self.lower_setting.max_workspace_size,
            lower_precision=self.lower_setting.lower_precision,
            strict_type_constraints=self.lower_setting.strict_type_constraints,
            algorithm_selector=algo_selector,
            timing_cache=cache_data,
            profiling_verbosity=(
                trt.ProfilingVerbosity.DETAILED
                if self.lower_setting.verbose_profile
                else trt.ProfilingVerbosity.LAYER_NAMES_ONLY
            ),
            tactic_sources=self.lower_setting.tactic_sources,
        )

        # Update timing cache file if needed
        timing_cache = interp_result.serialized_cache
        if timing_cache and self.timing_cache_manager:
            self.timing_cache_manager.update_timing_cache(split_name, timing_cache)

        return interp_result


def default_split_function(
    model: fx.GraphModule, inputs: Input, lower_setting: LowerSetting
) -> SplitResult:
    splitter_setting = TRTSplitterSetting()
    splitter_setting.use_implicit_batch_dim = not lower_setting.explicit_batch_dimension
    splitter_setting.min_acc_module_size = lower_setting.min_acc_module_size
    splitter_setting.use_experimental_rt = lower_setting.use_experimental_rt
    splitter = TRTSplitter(model, inputs, settings=splitter_setting)
    splitter.node_support_preview()
    return splitter.generate_split_results()


def create_lower_trt_interpreter(lower_setting: LowerSetting) -> LowerTrtInterpreter:
    return LowerTrtInterpreter.create(lower_setting)


def default_lower_pass(
    create_trt_interpreter: Callable[[LowerSetting], LowerTrtInterpreter],
) -> PassFunc:
    def lower_pass(
        mod: nn.Module, input: Input, lower_setting: LowerSetting, module_name: str
    ) -> nn.Module:
        """
        Create a module transformation pass which lowers an `fx.GraphModule` into a
        `TRTModule`
        """
        interpreter = create_trt_interpreter(lower_setting)
        interp_res: TRTInterpreterResult = interpreter(mod, input, module_name)
        if lower_setting.use_experimental_rt:
            import io

            from torch_tensorrt._Device import Device
            from torch_tensorrt.dynamo._TorchTensorRTModule import TorchTensorRTModule

            with io.BytesIO() as engine_bytes:
                engine_bytes.write(interp_res.engine.serialize())
                engine_str = engine_bytes.getvalue()

            trt_module = TorchTensorRTModule(
                engine_str,
                name=module_name,
                input_binding_names=interp_res.input_names,
                output_binding_names=interp_res.output_names,
                target_device=Device(f"cuda:{torch.cuda.current_device()}"),
                # cuda_graph_batch_size=lower_setting.cuda_graph_batch_size, # NOTE: Not sure what this is supposed to do
            )
            return trt_module

        else:
            trt_module = TRTModule(
                engine=interp_res.engine,
                input_names=interp_res.input_names,
                output_names=interp_res.output_names,
                cuda_graph_batch_size=lower_setting.cuda_graph_batch_size,
            )
            return trt_module

    return lower_pass


@dc.dataclass(frozen=True)
class Lowerer:
    """Lowers a module using fx2trt.

    This is a composable class to facilitate fx2trt. A normal fx2trt process
    composes of the following passes to transform an `fx.GraphModule`:

        1. trace - use torch.fx to trace the module so we can get the graph
            representation of the model.
        2. split - the graph module is split into several submodules,
            running either via TensorRT, or via regular CUDA.

    For each split that need to run via TRT, the following passes are
    invoked:

        3. `TRTInterpreter` - build the TRT engine for the submodule that
            can be supported through `TRTInterpreter`.
        4. Wraps the executable TRT engine into `TRTModule`, which is an `nn.Module`.
        5. The converted submodule is then set back onto the top-level module

    """

    lower_pass_manager_builder: LowerPassManagerBuilder

    @classmethod
    def create(
        cls,
        lower_setting: LowerSetting,
        interpreter_builder: Callable = create_lower_trt_interpreter,
        split_func: Callable = default_split_function,
    ) -> "Lowerer":
        """Instantiate a `Lowerer` instance."""
        if not lower_setting.is_aten:
            return cls(
                lower_pass_manager_builder=LowerPassManagerBuilder(
                    lower_setting=lower_setting,
                    trace_func=lambda module, inputs: acc_tracer.trace(
                        module,
                        inputs,  # type: ignore[arg-type]
                        ast_rewriter_allow_list=lower_setting.ast_rewriter_allow_list,
                        leaf_module_list=lower_setting.leaf_module_list,
                    ),
                    split_func=split_func,
                    lower_func=default_lower_pass(interpreter_builder),
                )
            )
        # proxytensor_trace
        else:
            return cls(
                lower_pass_manager_builder=LowerPassManagerBuilder(
                    lower_setting=lower_setting,
                    trace_func=lambda module, inputs: aten_tracer.opt_trace(
                        module, inputs
                    ),
                    split_func=split_func,
                    lower_func=default_lower_pass(interpreter_builder),
                )
            )

    def __call__(
        self,
        module: nn.Module,
        inputs: Input,
        additional_inputs: Optional[Input] = None,
        fp16_conversion_fn: Optional[Callable[[Input], Input]] = None,
    ) -> nn.Module:
        lower_setting = self.lower_pass_manager_builder.lower_setting
        atol = lower_setting.correctness_atol
        rtol = lower_setting.correctness_rtol

        @validate_inference(
            atol=atol,
            rtol=rtol,
        )
        def do_lower(module: nn.Module, inputs: Input) -> nn.Module:
            module.eval()
            if (
                self.lower_pass_manager_builder.lower_setting.lower_precision
                == LowerPrecision.FP16
            ):
                module.half()
                # A custom conversion function can be passed to the lowerer to
                # handle inputs with custom types. By default, just handle
                # tensors and NoneType.
                if fp16_conversion_fn is None:
                    conversion_fn = lambda x: (
                        x.half() if x is not None and x.dtype == torch.float32 else x
                    )
                else:
                    conversion_fn = fp16_conversion_fn

                inputs = tuple(conversion_fn(x) for x in inputs)
            if lower_setting.is_aten:
                pm = self.lower_pass_manager_builder.build_aten2trt_lower_pipeline(
                    inputs, additional_inputs
                )
            else:
                pm = self.lower_pass_manager_builder.build_trt_lower_pipeline(
                    inputs, additional_inputs
                )
            lower_result = pm(module)
            return lower_result

        return do_lower(module, inputs)
