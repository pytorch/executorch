# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Configurations for exporting Llama.

Uses dataclasses, which integrate with OmegaConf and Hydra.
"""

import argparse
import ast
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, List, Optional


################################################################################
################################## BaseConfig ##################################
################################################################################


class ModelType(str, Enum):
    STORIES110M = "stories110m"
    LLAMA2 = "llama2"
    LLAMA3 = "llama3"
    LLAMA3_1 = "llama3_1"
    LLAMA3_2 = "llama3_2"
    LLAMA3_2_VISION = "llama3_2_vision"
    STATIC_LLAMA = "static_llama"
    QWEN2_5 = "qwen2_5"
    QWEN3_0_6B = "qwen3-0_6b"
    QWEN3_1_7B = "qwen3-1_7b"
    QWEN3_4B = "qwen3-4b"
    PHI_4_MINI = "phi_4_mini"
    SMOLLM2 = "smollm2"


class PreqMode(str, Enum):
    """
    If you are dealing with pre-quantized checkpoints, this used to
    be the way to specify them. Now you don't need to specify these
    options if you use a TorchAo-prequantized checkpoint, but they
    are still around to preserve backward compatibility.
    """

    PREQ_8DA4W = "8da4w"
    PREQ_8DA4W_OUT_8DA8W = "8da4w_output_8da8w"


@dataclass
class BaseConfig:
    """
    Configurations specific to the model, e.g. whether it’s Qwen3 or Phi-4-mini,
    and are the minimal set of parameters needed to load the pretrained
    eager model and its weights.

    Attributes:
        model_class: Which model to to export.
        params: Model parameters, such as n_layers, hidden_size, etc.
            If left empty will use defaults specified in model_args.py.
        checkpoint: Path to the checkpoint file.
            If left empty, the model will be initialized with random weights.
        checkpoint_dir: Path to directory containing sharded checkpoint files.
        tokenizer_path: Path to the tokenizer file.
        metadata: Json string containing metadata information.
            e.g. '{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}'
        use_lora: Rank of the LoRA, if set to 0 then this means no LoRA. For use with QAT.
        fairseq2: For legacy internal use cases, this is safe to ignore.
        preq_mode: Legacy option to specify how prequantized weights are loaded.
            Going forward, ExecuTorch supports loading weights prequantized through
            TorchAo as-is, without any special handling.
        preq_group_size: Legacy option to specify the group size of prequantized weights.
        preq_embedding_quantize: Legacy option to specify how prequantized embeddings
            are loaded.
    """

    model_class: ModelType = ModelType.LLAMA3
    params: Optional[str] = None
    checkpoint: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    tokenizer_path: Optional[str] = None
    metadata: Optional[str] = None
    use_lora: int = int
    fairseq2: bool = False
    preq_mode: Optional[PreqMode] = None
    preq_group_size: int = 32
    preq_embedding_quantize: str = "8,0"


################################################################################
################################# ModelConfig ##################################
################################################################################


class DtypeOverride(str, Enum):
    """
    DType of the model. Highly recommended to use "fp32", unless you want to
    export without a backend, in which case you can also use "bf16". "fp16"
    is not recommended.
    """

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


@dataclass
class ModelConfig:
    """
    Configurations not necessarily specific to the model, but are needed to
    finish off the rest of the model configuration in eager. You can think
    of these like optimizations / actual configurations. The same ModelConfig
    can be applied to multiple models.

    Attributes:
        dtype_override: dtype to cast the model to.
        enable_dynamic_shape: whether to enable dynamic shapes on the sequence
            length so that the model can handle arbitrary prefill lengths and
            token generation.
        use_shared_embeddings: whether the embedding/output weights should be
            shared. Only available with torchao kernels, e.g. when
            qmode set to use a "torchao:8da(\\d+)w" pattern.
        use_sdpa_with_kv_cache: Whether to use flash attention by substituting
            for our custom SDPA op. Note that the naming is poor and this
            doesn't actually have anything to do with the kv_cache at the moment.
        expand_rope_table: Temporary workaround to expand sin/cos table in head
            dim to take vectorized path in optimized kernels.
        use_attention_sink: Whether to use attention sink to support multi-round
            conversation. Structured as:
            '<sink_size>,<window_size>,<batch_eviction_size>',
            e.g., '4,2044,1024'.
        output_prune_map: Path to the output pruning token mapping file (token_map.json).
        input_prune_map: Path to the output pruning token mapping file (token_map.json).
        use_kv_cache: Whether to use KV cache.
        quantize_kv_cache: Whether to perform int8 per token quantization on the KV cache.
        local_global_attention: List of integers specifying local and global attention pattern.
            e.g., [0, 16, 0, 16] to specify that every other layer is sliding window of 16.
            [0, 16, 32] pattern specifies 2nd and 3rd layers have sliding windows of 16 and 32.
            [16] pattern specifies all layers have a sliding window of 16.
    """

    dtype_override: DtypeOverride = DtypeOverride.FP32
    enable_dynamic_shape: bool = True
    use_shared_embedding: bool = False
    use_sdpa_with_kv_cache: bool = False
    expand_rope_table: bool = False
    use_attention_sink: Optional[str] = None
    output_prune_map: Optional[str] = None
    input_prune_map: Optional[str] = None
    use_kv_cache: bool = False
    quantize_kv_cache: bool = False
    local_global_attention: Optional[List[int]] = None

    def __post_init__(self):
        self._validate_attention_sink()
        self._validate_local_global_attention()

        if self.quantize_kv_cache and not self.use_kv_cache:
            raise ValueError(
                "Cannot quantize the KV cache (quantize_kv_cache) without enabling the KV cache (use_kv_cache)"
            )

        if self.local_global_attention and not self.use_kv_cache:
            raise ValueError(
                "Cannot use local_global_attention without enabling the KV cache (use_kv_cache)"
            )

    def _validate_attention_sink(self):
        if self.use_attention_sink:
            attention_sink_params = self.use_attention_sink.split(",")
            if len(attention_sink_params) != 3:
                raise ValueError(
                    "The value of use_attention_sink must be structured like '<sink_size>,<window_size>,<batch_eviction_size>'"
                )

    def _validate_local_global_attention(self):
        if self.local_global_attention:
            local_global_err = "The value of local_global_attention must be a list of integers, e.g., [0, 16, 0, 16]"
            try:
                parsed = ast.literal_eval(self.local_global_attention)
                if not (
                    isinstance(parsed, list) and all(isinstance(i, int) for i in parsed)
                ):
                    raise ValueError(local_global_err)
            except Exception:
                raise ValueError(local_global_err)


################################################################################
################################ ExportConfig ##################################
################################################################################


@dataclass
class ExportConfig:
    """
    Configures properties relevant to the export process.

    Attributes:
        max_seq_length: Maximum length of sequence to evaluate.
        max_context_length: Maximum of context for the model to remember.
        output_dir: Output dir to save the exported .pte file to.
        output_name: File name to override the exported .pte file.
        so_library: Shared library to specify custom quantized operators.
        export_only: Whether to stop right after torch.export() and
            just save the exported .pt2 graph file.
    """

    max_seq_length: int = 128
    max_context_length: int = 128
    output_dir: Optional[str] = None
    output_name: Optional[str] = None
    so_library: Optional[str] = None
    export_only: bool = False

    def __post_init__(self):
        if self.max_context_length > self.max_seq_length:
            raise ValueError(
                f"max_context_length of {self.max_context_length} cannot be greater than max_seq_length of {self.max_seq_length}"
            )


################################################################################
################################# DebugConfig ##################################
################################################################################


@dataclass
class DebugConfig:
    """
    Configures options to debug the export process.

    Attributes:
        profile_memory: Whether to generate a chrome trace of activation memory
            for intermediate tensors.
        profile_path: Use cProfile to profile the export. Results are saved to
            profile_path as an html file.
        generate_etrecord: Whether to generate an ETRecord debug artifact.
        generate_full_logits: Whether to keep the full logits, potentially useful
            for debugging purposes. Kept off by default to save memory.
        verbose: Whether to log the export process verbosely (log level >= INFO).
    """

    profile_memory: bool = False
    profile_path: Optional[str] = None
    generate_etrecord: bool = False
    generate_full_logits: bool = False
    verbose: bool = False


################################################################################
############################# QuantizationConfig ###############################
################################################################################


class Pt2eQuantize(str, Enum):
    """
    Type of backend-specific Pt2e quantization strategy to use.

    Pt2e uses a different quantization library that is graph-based
    compared to `qmode`, which is also specified in the QuantizationConfig
    and is source transform-based.
    """

    XNNPACK_DYNAMIC = "xnnpack_dynamic"
    XNNPACK_DYNAMIC_QC4 = "xnnpack_dynamic_qc4"
    QNN_8A8W = "qnn_8a8w"
    QNN_16A16W = "qnn_16a16w"
    QNN_16A4W = "qnn_16a4w"
    COREML_C4W = "coreml_c4w"
    COREML_8A_C8W = "coreml_8a_c8w"
    COREML_8A_C4W = "coreml_8a_c4w"
    COREML_BASELINE_8A_C8W = "coreml_baseline_8a_c8w"
    COREML_BASELINE_8A_C4W = "coreml_baseline_8a_c4w"
    VULKAN_8W = "vulkan_8w"


class SpinQuant(str, Enum):
    CUDA = "cuda"
    NATIVE = "native"


@dataclass
class QuantizationConfig:
    """
    Configures how the model should be quantized (PTQ).

    Attributes:
        qmode: Quantization mode using TorchAo, expressed as a string.
            See the __post_init__ validation for available qmode options.
        embedding_quantize: Type of embedding quantization.
            Must be of the format '<bitwidth>,<groupsize>', e.g., '8,1024'.
        pt2e_quantize: Quantization mode using pt2e, which is an alternative
            to TorchAo that uses backend-aware graph mode quantization rather
            than source transformation quantization.
        group_size: Group size for quantization.
        use_spin_quant: Which spin quant mode to use. If unspecified, don't use
            spin quant.
        use_qat: Whether the checkpoint is quantization-awarely trained.
        calibration_tasks: Tasks for GPTQ calibration from lm_eval.
        calibration_limit: Number of samples used for calibration from lm_eval.
        calibration_seq_length: Sequence length for GPTQ calibration from lm_eval.
        calibration_data: Prompts use for calibration.
    """

    # Constants.
    QMODE_OPTIONS: ClassVar[List[str]] = ["int8", "8da4w", "8da4w-gptq", "vulkan_4w"]
    AO_QUANT_PATTERNS: ClassVar[List[str]] = [
        r"torchao:8da(\d+)w",
        r"torchao:fpa(\d+)w",
    ]

    qmode: Optional[str] = None
    embedding_quantize: Optional[str] = None
    pt2e_quantize: Optional[Pt2eQuantize] = None
    group_size: Optional[int] = None
    use_spin_quant: Optional[SpinQuant] = None
    use_qat: bool = False
    calibration_tasks: Optional[List[str]] = None
    calibration_limit: Optional[int] = None
    calibration_seq_length: Optional[int] = None
    calibration_data: str = "Once upon a time"

    def __post_init__(self):
        if self.qmode:
            self._validate_qmode()

    def _validate_qmode(self) -> None:
        if not self.qmode:
            return

        if self.qmode in self.QMODE_OPTIONS:
            return

        # If qmode is one of these below patterns, this means that we
        # are using ARM-based torchao ops.
        for pattern in self.AO_QUANT_PATTERNS:
            matches = re.findall(pattern, self.qmode)
            if len(matches) == 1:
                return

        raise ValueError(
            f"Got qmode {self.qmode}, but expected one of {self.QMODE_OPTIONS}, or one of the regex patterns {self.AO_QUANT_PATTERNS}."
        )

    def _validate_embedding_quantize(self):
        if len(self.embedding_quantize.split(",")) != 2:
            raise ValueError(
                f'embedding_quantize of {self.embedding_quantize} must follow the following format: "<bitwidth>,<groupsize>"'
            )


################################################################################
############################### BackendConfig ##################################
################################################################################


@dataclass
class XNNPackConfig:
    """
    Configures the XNNPack backend.

    Attributes:
        enabled: :)
        extended_ops: Whether to match more types of ops to delegates to XNNPack.
    """

    enabled: bool = False
    extended_ops: bool = False


class CoreMLQuantize(str, Enum):
    B4W = "b4w"
    C4W = "c4w"


class CoreMLComputeUnit(str, Enum):
    CPU_ONLY = "cpu_only"
    CPU_AND_GPU = "cpu_and_gpu"
    CPU_AND_NE = "cpu_and_ne"
    ALL = "all"


@dataclass
class CoreMLConfig:
    """
    Configures the CoreML backend.
    """

    enabled: bool = False
    enable_state: bool = False
    preserve_sdpa: bool = False
    quantize: Optional[CoreMLQuantize] = None
    ios: int = 15
    compute_units: CoreMLComputeUnit = CoreMLComputeUnit.CPU_ONLY

    def __post_init__(self):
        if self.ios not in (15, 16, 17, 18):
            raise ValueError(f"Invalid coreml ios version: {self.ios}")


@dataclass
class VulkanConfig:
    """
    Configures the Vulkan backend.
    """

    enabled: bool = False


@dataclass
class QNNConfig:
    """
    Configures the QNN backend.
    """

    enabled: bool = False
    use_sha: bool = False
    soc_model: str = "SM8650"
    use_qnn_sha: bool = False
    optimized_rotation_path: Optional[str] = None
    num_sharding: int = 0


@dataclass
class MPSConfig:
    """
    Configures the MPS backend.
    """

    enabled: bool = False


@dataclass
class BackendConfig:
    """
    Configures which backends should be used and how the backends
    should be set up.
    """

    xnnpack: XNNPackConfig = field(default_factory=XNNPackConfig)
    coreml: CoreMLConfig = field(default_factory=CoreMLConfig)
    vulkan: VulkanConfig = field(default_factory=VulkanConfig)
    qnn: QNNConfig = field(default_factory=QNNConfig)
    mps: MPSConfig = field(default_factory=MPSConfig)


################################################################################
################################## LlmConfig ###################################
################################################################################


@dataclass
class LlmConfig:
    """
    The overall configuration for customizing the LLM export process.
    """

    base: BaseConfig = field(default_factory=BaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "LlmConfig":  # noqa: C901
        """
        To support legacy purposes, this function converts CLI args from
        argparse to an LlmConfig, which is used by the LLM export process.
        """
        llm_config = LlmConfig()

        # BaseConfig
        if hasattr(args, "model"):
            llm_config.base.model_class = ModelType(args.model)
        if hasattr(args, "params"):
            llm_config.base.params = args.params
        if hasattr(args, "checkpoint"):
            llm_config.base.checkpoint = args.checkpoint
        if hasattr(args, "checkpoint_dir"):
            llm_config.base.checkpoint_dir = args.checkpoint_dir
        if hasattr(args, "tokenizer_path"):
            llm_config.base.tokenizer_path = args.tokenizer_path
        if hasattr(args, "metadata"):
            llm_config.base.metadata = args.metadata
        if hasattr(args, "use_lora"):
            llm_config.base.use_lora = args.use_lora
        if hasattr(args, "fairseq2"):
            llm_config.base.fairseq2 = args.fairseq2

        # PreqMode settings
        if hasattr(args, "preq_mode") and args.preq_mode:
            llm_config.base.preq_mode = PreqMode(args.preq_mode)
            if hasattr(args, "preq_group_size"):
                llm_config.base.preq_group_size = args.preq_group_size
            if hasattr(args, "preq_embedding_quantize"):
                llm_config.base.preq_embedding_quantize = args.preq_embedding_quantize

        # ModelConfig
        if hasattr(args, "dtype_override"):
            llm_config.model.dtype_override = DtypeOverride(args.dtype_override)
        if hasattr(args, "enable_dynamic_shape"):
            llm_config.model.enable_dynamic_shape = args.enable_dynamic_shape
        if hasattr(args, "use_shared_embedding"):
            llm_config.model.use_shared_embedding = args.use_shared_embedding
        if hasattr(args, "use_sdpa_with_kv_cache"):
            llm_config.model.use_sdpa_with_kv_cache = args.use_sdpa_with_kv_cache
        if hasattr(args, "expand_rope_table"):
            llm_config.model.expand_rope_table = args.expand_rope_table
        if hasattr(args, "use_attention_sink"):
            llm_config.model.use_attention_sink = args.use_attention_sink
        if hasattr(args, "output_prune_map"):
            llm_config.model.output_prune_map = args.output_prune_map
        if hasattr(args, "input_prune_map"):
            llm_config.model.input_prune_map = args.input_prune_map
        if hasattr(args, "use_kv_cache"):
            llm_config.model.use_kv_cache = args.use_kv_cache
        if hasattr(args, "quantize_kv_cache"):
            llm_config.model.quantize_kv_cache = args.quantize_kv_cache
        if hasattr(args, "local_global_attention"):
            llm_config.model.local_global_attention = args.local_global_attention

        # ExportConfig
        if hasattr(args, "max_seq_length"):
            llm_config.export.max_seq_length = args.max_seq_length
        if hasattr(args, "max_context_length"):
            llm_config.export.max_context_length = args.max_context_length
        if hasattr(args, "output_dir"):
            llm_config.export.output_dir = args.output_dir
        if hasattr(args, "output_name"):
            llm_config.export.output_name = args.output_name
        if hasattr(args, "so_library"):
            llm_config.export.so_library = args.so_library
        if hasattr(args, "export_only"):
            llm_config.export.export_only = args.export_only

        # QuantizationConfig
        if hasattr(args, "quantization_mode"):
            llm_config.quantization.qmode = args.quantization_mode
        if hasattr(args, "embedding_quantize"):
            llm_config.quantization.embedding_quantize = args.embedding_quantize
        if hasattr(args, "pt2e_quantize") and args.pt2e_quantize:
            llm_config.quantization.pt2e_quantize = Pt2eQuantize(args.pt2e_quantize)
        if hasattr(args, "group_size"):
            llm_config.quantization.group_size = args.group_size
        if hasattr(args, "use_spin_quant") and args.use_spin_quant:
            llm_config.quantization.use_spin_quant = SpinQuant(args.use_spin_quant)
        if hasattr(args, "use_qat"):
            llm_config.quantization.use_qat = args.use_qat
        if hasattr(args, "calibration_tasks"):
            llm_config.quantization.calibration_tasks = args.calibration_tasks
        if hasattr(args, "calibration_limit"):
            llm_config.quantization.calibration_limit = args.calibration_limit
        if hasattr(args, "calibration_seq_length"):
            llm_config.quantization.calibration_seq_length = args.calibration_seq_length
        if hasattr(args, "calibration_data"):
            llm_config.quantization.calibration_data = args.calibration_data

        # BackendConfig - XNNPack
        if hasattr(args, "xnnpack"):
            llm_config.backend.xnnpack.enabled = args.xnnpack
        if hasattr(args, "xnnpack_extended_ops"):
            llm_config.backend.xnnpack.extended_ops = args.xnnpack_extended_ops

        # CoreML
        if hasattr(args, "coreml"):
            llm_config.backend.coreml.enabled = args.coreml
        llm_config.backend.coreml.enable_state = getattr(
            args, "coreml_enable_state", False
        )
        llm_config.backend.coreml.preserve_sdpa = getattr(
            args, "coreml_preserve_sdpa", False
        )
        if hasattr(args, "coreml_quantize") and args.coreml_quantize:
            llm_config.backend.coreml.quantize = CoreMLQuantize(args.coreml_quantize)
        if hasattr(args, "coreml_ios"):
            llm_config.backend.coreml.ios = args.coreml_ios
        if hasattr(args, "coreml_compute_units"):
            llm_config.backend.coreml.compute_units = CoreMLComputeUnit(
                args.coreml_compute_units
            )

        # Vulkan
        if hasattr(args, "vulkan"):
            llm_config.backend.vulkan.enabled = args.vulkan

        # QNN
        if hasattr(args, "qnn"):
            llm_config.backend.qnn.enabled = args.qnn
        if hasattr(args, "use_qnn_sha"):
            llm_config.backend.qnn.use_sha = args.use_qnn_sha
        if hasattr(args, "soc_model"):
            llm_config.backend.qnn.soc_model = args.soc_model
        if hasattr(args, "optimized_rotation_path"):
            llm_config.backend.qnn.optimized_rotation_path = (
                args.optimized_rotation_path
            )
        if hasattr(args, "num_sharding"):
            llm_config.backend.qnn.num_sharding = args.num_sharding

        # MPS
        if hasattr(args, "mps"):
            llm_config.backend.mps.enabled = args.mps

        # DebugConfig
        if hasattr(args, "profile_memory"):
            llm_config.debug.profile_memory = args.profile_memory
        if hasattr(args, "profile_path"):
            llm_config.debug.profile_path = args.profile_path
        if hasattr(args, "generate_etrecord"):
            llm_config.debug.generate_etrecord = args.generate_etrecord
        if hasattr(args, "generate_full_logits"):
            llm_config.debug.generate_full_logits = args.generate_full_logits
        if hasattr(args, "verbose"):
            llm_config.debug.verbose = args.verbose

        return llm_config

    def __post_init__(self):
        self._validate_low_bit()

    def _validate_low_bit(self):
        if not self.quantization.qmode:
            return

        using_lowbit_ops = False
        for pattern in self.quantization.AO_QUANT_PATTERNS:
            matches = re.findall(pattern, self.quantization.qmode)
            if len(matches) == 1:
                using_lowbit_ops = True

        # If we are using Ao's low bit quantization kernels for ARM,
        # we do not want to also be delegating to a CPU backend (XNNPack).
        if using_lowbit_ops and self.backend.xnnpack.enabled:
            raise ValueError(
                "Cannot use low-bit Ao ops (from qmode=torchao:...) while also delegating to XNNPack."
            )

        # Also we can only use shared embeddings if we are using low bit kernels.
        if self.model.use_shared_embedding and not using_lowbit_ops:
            raise ValueError(
                "Can only use shared embeddings with low-bit ops (with qmode=torchao:...)."
            )
