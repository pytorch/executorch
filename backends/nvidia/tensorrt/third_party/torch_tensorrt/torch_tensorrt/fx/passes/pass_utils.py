import contextlib
import io
import json
import logging
import tempfile
from datetime import datetime
from functools import wraps
from traceback import TracebackException
from typing import Any, Callable, List, Optional

import torch
import torch_tensorrt.fx.diagnostics as diagnostics
from torch import fx
from torch.fx.node import Node
from torch.fx.passes.shape_prop import ShapeProp

# Create an alias for module input type to avoid littering pyre-ignore for Any
# throughout the file.
Input = Any
_LOGGER: logging.Logger = logging.getLogger(__name__)

PassFunc = Callable[[fx.GraphModule, Input], fx.GraphModule]

RELAX_ACCURACY_FAILURE: bool = False
FINAL_CHECK_ATOL_MULTIPLIER: float = 10
FINAL_CHECK_RTOL_MULTIPLIER: float = 10

# A global override of the alternative batch size used in validate_variable_batch_sizes
ALTERNATIVE_BATCH_SIZE_OVERRIDE: Optional[int] = None
# If exception during validate_variable_batch_sizes should be thrown
ALTERNATIVE_BATCH_SIZE_EXCEPTION_SHOULD_THROW: bool = False


class RelaxAccuracyCheckMode:
    """
    Basically a context manager that controls a global variable that controls
    the accuracy check mode. Use it like
    with RelaxAccuracyCheckMode(True):
        fx2trt()
    """

    def __init__(
        self,
        mode: bool,
        final_atol_multiplier: Optional[float] = None,
        final_rtol_multiplier: Optional[float] = None,
    ):
        """
        Arguments:
        mode: whether we relax the immediate accuracy check failure or not. If yes, we will do an extra
        accruacy check by raising the tolerance by the multipler times and only raise error if that fails.
        This is to avoid catastrophic errors.
        final_atol_multiplier [optional]: set FINAL_CHECK_ATOL_MULTIPLIER if specifier.
        final_rtol_multiplier [optional]: set FINAL_CHECK_RTOL_MULTIPLIER if specifier.
        """
        global RELAX_ACCURACY_FAILURE
        global FINAL_CHECK_ATOL_MULTIPLIER
        global FINAL_CHECK_RTOL_MULTIPLIER
        self._old_mode = (
            RELAX_ACCURACY_FAILURE,
            FINAL_CHECK_ATOL_MULTIPLIER,
            FINAL_CHECK_RTOL_MULTIPLIER,
        )
        RELAX_ACCURACY_FAILURE = mode
        FINAL_CHECK_ATOL_MULTIPLIER = (
            final_atol_multiplier
            if final_atol_multiplier
            else FINAL_CHECK_ATOL_MULTIPLIER
        )
        FINAL_CHECK_RTOL_MULTIPLIER = (
            final_rtol_multiplier
            if final_rtol_multiplier
            else FINAL_CHECK_RTOL_MULTIPLIER
        )
        _LOGGER.info(
            f"Set new relaxed accuracy check mode: {RELAX_ACCURACY_FAILURE=}, {FINAL_CHECK_ATOL_MULTIPLIER=}, {FINAL_CHECK_RTOL_MULTIPLIER=}"
        )

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        global RELAX_ACCURACY_FAILURE
        global FINAL_CHECK_ATOL_MULTIPLIER
        global FINAL_CHECK_RTOL_MULTIPLIER
        (
            RELAX_ACCURACY_FAILURE,
            FINAL_CHECK_ATOL_MULTIPLIER,
            FINAL_CHECK_RTOL_MULTIPLIER,
        ) = self._old_mode
        _LOGGER.info(
            f"Restored old relaxed accuracy check mode: {RELAX_ACCURACY_FAILURE=}, {FINAL_CHECK_ATOL_MULTIPLIER=}, {FINAL_CHECK_RTOL_MULTIPLIER=}"
        )


@contextlib.contextmanager
def override_alternative_batch_size(alternative_batch_size: int = -1):
    """
    A context manager to override alternative_batch_size

    Example:

    >>> # disables run_alternative_batch_size verification
    >>> with override_alternative_batch_size(-1):
    >>>     fx2ait()
    """

    global ALTERNATIVE_BATCH_SIZE_OVERRIDE
    old_value = ALTERNATIVE_BATCH_SIZE_OVERRIDE
    ALTERNATIVE_BATCH_SIZE_OVERRIDE = alternative_batch_size
    _LOGGER.info(f"Override {ALTERNATIVE_BATCH_SIZE_OVERRIDE=} ({old_value=})")
    try:
        yield
    finally:
        ALTERNATIVE_BATCH_SIZE_OVERRIDE = old_value
        _LOGGER.info(f"Restored old value: {ALTERNATIVE_BATCH_SIZE_OVERRIDE=})")


@contextlib.contextmanager
def override_alternative_batch_size_exception_should_throw(
    exception_should_throw: bool,
):
    """
    A context manager to set if exception during alternative batch size verification
    should be thrown.
    """
    global ALTERNATIVE_BATCH_SIZE_EXCEPTION_SHOULD_THROW
    old_value = ALTERNATIVE_BATCH_SIZE_EXCEPTION_SHOULD_THROW
    ALTERNATIVE_BATCH_SIZE_EXCEPTION_SHOULD_THROW = exception_should_throw
    try:
        yield
    finally:
        ALTERNATIVE_BATCH_SIZE_EXCEPTION_SHOULD_THROW = old_value


def chain_passes(*passes: PassFunc) -> PassFunc:
    """
    Chains a sequence of pass functions to form a single pass function
    """

    def parent_pass(module: fx.GraphModule, input: Input) -> fx.GraphModule:
        for pass_ in passes:
            if isinstance(module, torch.fx.GraphModule):
                ShapeProp(module).propagate(*input)
            module = pass_(module, input)
        return module

    return parent_pass


# (TODO(shirongwu): Add exception notification for fblearner flow when available, notify oncall
# on pass that failed accuracy check.
def validate_inference(
    rtol=None, atol=None, run_alternative_batch_size: int = -1
) -> "Decorator":
    """
    Returns a decorator on a PassFunc to sanity check the model outputs
    difference before/after the transformation is within tolerance.

    Args:
        rtol: reletive tolerance
        atol: absoluate tolerance
        run_alternative_batch_size (int):
            In addition to running inference at original batch size in the
            input, also run at an alternative batch size. If set to -1, do not
            run at alternative batch size. It must be smaller than the original
            batch size. This is useful to check the model can run at different
            batch sizes. Usually we can set this to 1.
    """

    def _validate_inference(pass_: PassFunc) -> PassFunc:
        """
        A decorator to wrap a pass function to validate that its inference
        results before and after the pass run should be `close`.
        """

        @wraps(pass_)
        def pass_with_validation(
            module: fx.GraphModule,
            input: Input,
            *args,
            **kwargs,
        ) -> fx.GraphModule:
            res0 = module(*input)
            processed_module = pass_(module, input, *args, **kwargs)
            res1 = processed_module(*input)

            tensor_res_0 = _collect_tensors(res0)
            tensor_res_1 = _collect_tensors(res1)
            relax_accuracy_check_failure = RELAX_ACCURACY_FAILURE

            for kk, (x, y) in enumerate(zip(tensor_res_0, tensor_res_1)):
                kwargs2 = {"equal_nan": True}
                if rtol:
                    kwargs2["rtol"] = rtol
                if atol:
                    kwargs2["atol"] = atol
                kwargs2["msg"] = (
                    lambda msg: f"Pass {pass_} failed correctness check due at output {kk}:\n{msg}"
                )
                # If tensors are on different devices, make sure to compare
                # their copies that are on the same device.
                if x.get_device() != y.get_device():
                    x = x.cpu()
                    y = y.cpu()
                try:
                    torch.testing.assert_close(x, y, **kwargs2)
                except Exception as e:
                    if relax_accuracy_check_failure:
                        _LOGGER.error(f"{e}")
                        kwargs2["rtol"] *= FINAL_CHECK_RTOL_MULTIPLIER
                        kwargs2["atol"] *= FINAL_CHECK_ATOL_MULTIPLIER
                        new_atol = kwargs2["atol"]
                        new_rtol = kwargs2["rtol"]
                        _LOGGER.info(
                            f"Do a sanity check to see whether things are completely wrong with {new_atol=}, {new_rtol=}"
                        )
                        torch.testing.assert_close(x, y, **kwargs2)
                        return processed_module
                    else:
                        raise e

            return processed_module

        return pass_with_validation

    return _validate_inference


def validate_variable_batch_sizes(run_alternative_batch_size: int = -1) -> "Decorator":
    """
    Returns a decorator on a PassFunc to verify the model can run with
    different batch sizes before/after the transformation is within tolerance.

    Args:
        run_alternative_batch_size (int):
            In addition to running inference at original batch size in the
            input, also run at an alternative batch size. If set to -1, do not
            run at alternative batch size. It must be smaller than the original
            batch size. This is useful to check the model can run at different
            batch sizes. Usually we can set this to 1.

            If the global variable `ALTERNATIVE_BATCH_SIZE_OVERRIDE` is set, it
            overrides `run_alternative_batch_size`.
            `ALTERNATIVE_BATCH_SIZE_OVERRIDE` can be set via:

                with override_alternative_batch_size(...): ...
    """

    def _run_alternative_batch_size(pass_: PassFunc) -> PassFunc:
        """
        A decorator for PassFunc to check that the model (both before and after
        the transformation by pass func) can run at alternative batch size.
        """

        @wraps(pass_)
        def pass_with_validation(
            module: fx.GraphModule,
            input: Input,
            *args,
            **kwargs,
        ) -> fx.GraphModule:
            _run_alternative_batch_size = (
                ALTERNATIVE_BATCH_SIZE_OVERRIDE
                if ALTERNATIVE_BATCH_SIZE_OVERRIDE is not None
                else run_alternative_batch_size
            )

            if _run_alternative_batch_size < 0:
                return pass_(module, input, *args, **kwargs)

            if not isinstance(input, (list, tuple)):
                _LOGGER.info(
                    f"Skip run_alternative_batch_size: input must be list, tuple. Actual: {type(input)}"
                )
                return pass_(module, input, *args, **kwargs)

            if not all(isinstance(x, torch.Tensor) for x in input):
                _LOGGER.info(
                    "Skip run_alternative_batch_size: input elements must all be tensors"
                )
                return pass_(module, input, *args, **kwargs)

            if not all(len(x.shape) > 0 for x in input):
                _LOGGER.info(
                    "Skip run_alternative_batch_size: some input tensor(s) are scalar"
                )
                return pass_(module, input, *args, **kwargs)

            batch_size_candidates = {x.shape[0] for x in input}
            if len(batch_size_candidates) > 1:
                _LOGGER.info(
                    f"Skip run_alternative_batch_size: input tensors' first dim must be the same, actual: {batch_size_candidates}"
                )
                return pass_(module, input, *args, **kwargs)

            batch_size = next(iter(batch_size_candidates))
            assert (
                _run_alternative_batch_size <= batch_size
            ), f"{_run_alternative_batch_size=} must be smaller or equal to {batch_size=}"

            input_alt_bs = [x[:_run_alternative_batch_size, ...] for x in input]

            def run_module(mod, stage: str):
                """Run module with full bs and alternative bs"""
                _LOGGER.info(
                    f"Running {stage} model at alternative batch size: {_run_alternative_batch_size}"
                )
                try:
                    mod(*input)
                    mod(*input_alt_bs)
                except Exception as e:
                    _LOGGER.warning(
                        f"Failed running {stage} module at full or alternative batch size: {e}"
                    )
                    diagnostics.write(
                        "lowering_diagnostics",
                        json.dumps(
                            {
                                "validate_variable_batch_sizes_exception": repr(e),
                                "validate_variable_batch_sizes_exception_type": type(
                                    e
                                ).__name__,
                                "validate_variable_batch_sizes_exception_traceback": "".join(
                                    TracebackException.from_exception(e).format()
                                ),
                            }
                        ),
                    )
                    if ALTERNATIVE_BATCH_SIZE_EXCEPTION_SHOULD_THROW:
                        raise

            run_module(module, "original")
            module_after = pass_(module, input, *args, **kwargs)
            run_module(module_after, "transformed")

            return module_after

        return pass_with_validation

    return _run_alternative_batch_size


Decorator = Callable[[Callable], Callable]


def decorate_method(dec_for_function: Decorator) -> Decorator:
    def dec_for_method(unbounded_method) -> Callable:
        def decorated_unbounded_method(self, *args, **kwargs):
            @dec_for_function
            def bounded_method(*args, **kwargs):
                return unbounded_method(self, *args, **kwargs)

            return bounded_method(*args, **kwargs)

        return decorated_unbounded_method

    return dec_for_method


def log_perf_before_after(pass_: PassFunc) -> PassFunc:
    """
    Wraps a pass function to log perf of the module before and after the pass
    """

    @wraps(pass_)
    def check_perf_with_before_after_log(
        module: fx.GraphModule, input: Input
    ) -> fx.GraphModule:
        def benchmark_torch_function(iters: int, f, *args) -> float:
            """Estimates the average time duration for a single inference call in second

            If the input is batched, then the estimation is for the batches inference call.

            Args:
                iters: number of inference iterations to run
                f: a function to perform a single inference call

            Returns:
                estimated average time duration in second for a single inference call
            """
            with torch.inference_mode():
                f(*args)
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            # print("== Start benchmark iterations")
            with torch.inference_mode():
                start_event.record()
                for _ in range(iters):
                    f(*args)
                end_event.record()
            torch.cuda.synchronize()
            # print("== End benchmark iterations")
            return (start_event.elapsed_time(end_event) * 1.0e-3) / iters

        time_before = benchmark_torch_function(100, lambda: module(*input))
        _LOGGER.info(f"[{pass_}] Perf Before(eager mode): {time_before}")

        module = pass_(module, input)
        time_after = benchmark_torch_function(100, lambda: module(*input))
        _LOGGER.info(f"[{pass_}] Perf After(eager mode): {time_after}")
        return module

    return check_perf_with_before_after_log


def log_before_after(pass_: PassFunc) -> PassFunc:
    """
    Wraps a pass function to log the module graph before and after the pass
    """

    @wraps(pass_)
    def pass_with_before_after_log(
        module: fx.GraphModule, input: Input
    ) -> fx.GraphModule:
        before_io = io.StringIO()
        after_io = io.StringIO()
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as f:
            print(f"[{pass_}] Before:\n{module.graph}", file=f)
            print(module.graph, file=before_io)
            start_time = datetime.now()
            module = pass_(module, input)
            t_elapsed = datetime.now() - start_time
            print(f"[{pass_}] After:\n{module.graph}", file=f)
            print(module.graph, file=after_io)
            t = before_io.getvalue() == after_io.getvalue()
            _LOGGER.info(
                f"== Log pass {pass_} before/after graph to {f.name}, before/after are the same = {t}, time elapsed = {t_elapsed}"
            )
            return module

    return pass_with_before_after_log


def _collect_tensors(arg: fx.node.Argument) -> List[torch.Tensor]:
    """Collects all the tensors found in a nested container object"""
    res: List[torch.Tensor] = []

    def collect(x: fx.node.Argument) -> fx.node.Argument:
        if isinstance(x, torch.Tensor):
            res.append(x)
        return x

    fx.node.map_aggregate(arg, collect)
    return res


class InputOutputDtypeInferInterpreter(torch.fx.Interpreter):
    """
    Interprete a graph to propagate the output tensor dtype from its inputs, extracing
    input and output graph node that need dtype cast to float32/bfloat16.
    """

    def __init__(self, module: torch.fx.GraphModule):
        super().__init__(module)
        self.need_cast_to_float32 = []
        self.need_cast_to_bfloat = []

    def _need_cast(self, node: Node, run_result) -> None:
        if node.op == "placeholder" and (
            run_result.dtype not in (torch.int32, torch.int64)
        ):
            _LOGGER.info(
                f"Encountered node: {node.format_node()} need dtype cast to float32."
            )
            self.need_cast_to_float32.append(node)
        # Process node that will be used as final output
        elif "output" in set(i.name for i in node.users.keys()):
            if run_result.dtype not in (torch.int32, torch.int64):
                _LOGGER.info(
                    f"Encountered node: {node.format_node()} need dtype cast to bfloat16."
                )
                self.need_cast_to_bfloat.append(node)

    def run_node(self, n: Node) -> Any:
        run_result = super().run_node(n)

        if torch.is_tensor(run_result):
            n.meta["tensor_dtype"] = run_result.dtype
            self._need_cast(n, run_result)
        return run_result


def apply_bfloat_float_conversion(
    gm: torch.fx.GraphModule, inputs: Any, name: str
) -> None:
    _LOGGER.info("Apply bfloat-float32 conversion on {name}")
    interpreter = InputOutputDtypeInferInterpreter(gm)
    interpreter.run(*inputs)

    def to_bfloat(x):
        return x.to(torch.bfloat16)

    def to_float(x):
        return x.to(torch.float32)

    for node in interpreter.need_cast_to_float32:
        with gm.graph.inserting_after(node):
            cast = gm.graph.call_function(
                to_float,
                (node,),
                {},
            )
            node.replace_all_uses_with(cast)

    for node in interpreter.need_cast_to_bfloat:
        with gm.graph.inserting_after(node):
            cast = gm.graph.call_function(to_bfloat, (node,), {})
            node.replace_all_uses_with(cast)
