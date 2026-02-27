import dataclasses as dc
from typing import List, Optional, Set, Type

from torch import nn
from torch.fx.passes.pass_manager import PassManager

from .input_tensor_spec import InputTensorSpec
from .passes.lower_basic_pass import fuse_permute_linear, fuse_permute_matmul
from .utils import LowerPrecision


@dc.dataclass
class LowerSettingBasic:
    """
    Basic class for lowering.
    max_batch_size: The maximum batch size for lowering job.
                    If run with TensorRT lowering, this is the maximum
                    batch size which can be used at execution time,
                    and also the batch size for which the ICudaEngine
                    will be optimized.
                    If run with AITemplate lowering, this the max batch_size
                    for the model.
    lower_precision: lower precision dtype during lowering.
    min_acc_module_size(int): minimal number of nodes for an accelerate submodule.
    ast_rewriter_allow_list (Optional[Set[nn.Module]]): Optional allow list of
    modules that need AST rewriting. This is aiming to eliminate input variable involve in
    exception checking control flow.
    leaf_module_list (Optional[Set[nn.Module]]): Optional leaf module list where
    modules will not be traced into.
    verbose_profile (bool): verbosity of profiler, default to False.
    """

    max_batch_size: int = 2048
    lower_precision: LowerPrecision = LowerPrecision.FP32
    min_acc_module_size: int = 10
    ast_rewriter_allow_list: Optional[Set[Type[nn.Module]]] = None
    leaf_module_list: Optional[Set[Type[nn.Module]]] = None
    verbose_profile: bool = False
    is_aten: bool = False


@dc.dataclass
class LowerSetting(LowerSettingBasic):
    """
    Basic configuration for lowering stack.
    Args:
    input_specs: Specs for inputs to engine, can either be a single size or a
    range defined by Min, Optimal, Max sizes.
    explicit_batch_dimension: Use explicit batch dimension during lowering.
    explicit_precision: Use explicit precision during lowering.
    max_workspace_size: The maximum workspace size. The maximum GPU temporary
    memory which the TensorRT engine can use at execution time.
    strict_type_constraints: Require TensorRT engine to strictly follow data type
    setting at execution time.
    customized_fuse_pass: List of custmozied pass to apply during lowering process.
    lower_basic_fuse_pass: Enable basic pass fuse duirng lowering, i.e. fuse multiple operations
    as (a->b->c->d)=>(e). Current basic fuse patterns are:
    permute->linear
    permute->matmul
    verbose_log: Enable TensorRT engine verbose log mode.
    algo_selector: Enable TensorRT algorithm selector at execution time.
    timing_cache_prefix: TensorRT timing cache file path. TensorRT engine will use timing
    cache file at execution time if valid timing cache file is provided.
    save_timing_cache: Save updated timing cache data into timing cache file if the timing
    cache file is provided.
    cuda_graph_batch_size (int): Cuda graph batch size, default to be -1.
    preset_lowerer (str): when specified, use a preset logic to build the
    instance of Lowerer.
    only used by explicit batch dim with dynamic shape mode. In general, we use 2 GPU setting with
    2 stream on each. Set total number to 8 as a safe default value.
    dynamic_batch: enable the dynamic shape in TRT with dim=-1 for the 1st dimension.
    tactic_sources: tactic sources for TensorRT kernel selection. Default to None,
    meaning all possible tactic sources.
    correctness_atol: absolute tolerance for correctness check
    correctness_rtol: relative tolerance for correctness check
    use_experimental_rt: Uses the next generation TRTModule which supports both Python and TorchScript based execution (including in C++).
    """

    input_specs: List[InputTensorSpec] = dc.field(default_factory=list)
    explicit_batch_dimension: bool = True
    explicit_precision: bool = False
    max_workspace_size: int = 1 << 30
    strict_type_constraints: bool = False
    customized_fuse_pass: PassManager = dc.field(
        default_factory=lambda: PassManager.build_from_passlist([])
    )
    lower_basic_fuse_pass: PassManager = dc.field(
        default_factory=lambda: PassManager.build_from_passlist(
            [fuse_permute_matmul, fuse_permute_linear]
        )
    )
    verbose_log: bool = False
    algo_selector = None
    timing_cache_prefix: str = ""
    save_timing_cache: bool = False
    cuda_graph_batch_size: int = -1
    preset_lowerer: str = ""
    opt_profile_replica: int = 8
    dynamic_batch: bool = True
    tactic_sources: Optional[int] = None
    correctness_atol: float = 0.1
    correctness_rtol: float = 0.1
    use_experimental_rt: bool = False
