# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed configuration bridge between ``PipelineContext`` and the LLM flow.

The LLM export/quantize/compile/eval code is driven by the
``argparse.Namespace`` produced by ``llama.py``'s parser: components read
attributes off it directly (``control_args.max_seq_len``,
``args.enable_x86_64``, ...) and ``QnnConfig.load_config`` reflects over it with
``vars()``. ``ControlArgs`` is a typed dataclass holding the same attributes, so
it can be built from a ``PipelineContext`` and handed to that code unchanged.

It **subclasses** ``argparse.Namespace`` deliberately. ``QnnConfig.load_config``
dispatches on ``isinstance(config, argparse.Namespace)`` and raises
``TypeError`` for anything else, so a bare dataclass would break the device
path. Subclassing keeps ``isinstance`` true and ``vars()`` working while adding
field names, defaults and type annotations.

Defaults mirror ``llama.py``'s parser exactly; ``tests/test_control_args.py``
asserts that against the real parser so the two cannot drift.

:meth:`ControlArgs.build_parser` derives an ``argparse`` parser from these same
fields, so a command-line entry point gets one default per setting rather than
maintaining a second table of its own. Callers needing extra arguments pass the
result to their own parser via ``parents=[...]`` and add them there.

.. note::
    This is a second configuration surface alongside ``PipelineContext``, and
    that duplication is temporary: it exists so the pipeline can drive the
    existing LLM components while they are migrated. As components move behind
    pipeline interfaces, their fields leave this dataclass. New fields should be
    added only when an existing component genuinely reads them.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field, fields, MISSING
from typing import (
    Any,
    Dict,
    get_args,
    get_origin,
    get_type_hints,
    List,
    Optional,
    TYPE_CHECKING,
    Union,
)

if TYPE_CHECKING:
    from executorch.backends.qualcomm.genai_pipeline.pipeline_context import (
        PipelineContext,
    )

logger = logging.getLogger(__name__)

# Defaults, mirroring llama.py's argparse configuration. Named rather than
# inlined so callers can reference them instead of repeating literals.
DEFAULT_ARTIFACT = "./llama_qnn"
DEFAULT_BACKEND = "htp"
DEFAULT_BATCH_SIZE = 1
DEFAULT_CALIB_HF_LIMIT = 1
DEFAULT_CALIB_LIMIT = 1
DEFAULT_DTYPE_OVERRIDE = "fp32"
DEFAULT_EVAL_LIMIT = 1
DEFAULT_EVAL_METHOD = "prompt_eval"
DEFAULT_GCAP = 8
DEFAULT_HTP_PERFORMANCE_MODE = 2
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_MODEL_MODE = "hybrid"
DEFAULT_NGRAM = 5
DEFAULT_PORT = -1
DEFAULT_PREFILL_AR_LEN = 32
DEFAULT_PROFILE_LEVEL = 0
DEFAULT_TARGET = "aarch64-android"
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TRAIN_HF_LIMIT = 1000
DEFAULT_TRAIN_LIMIT = 1
DEFAULT_TRAIN_VAL_RATIO = 1.0
DEFAULT_WINDOW = 8

# Fields whose default is deliberately not ``llama.py``'s. Its defaults for
# these two are paths to YAML files that live under ``examples/``, which this
# package must not reference; ``None`` means "let the consumer fall back to its
# own default", which is what those paths are.
_PARSER_DEFAULT_EXEMPT_FIELDS = frozenset(
    {
        "lr_config",
        "train_config",
    }
)

# Fields ``from_pipeline_context`` fills from the context itself. They are read
# back out of the resulting ``ControlArgs`` by the LLM components *and* out of
# the context by the stage configs, so the two must agree: ``extra_options``
# cannot set them, and an explicit override is the only way to differ.
_CONTEXT_OWNED_FIELDS = frozenset(
    {
        "artifact",
        "decoder_model",
        "prompt",
        "soc_model",
    }
)

# Fields ``build_parser`` gives ``nargs="+"``. Their annotation is a list, so a
# command line supplies them as repeated values rather than one string.
_LIST_VALUED_FIELDS = frozenset(
    {
        "audio_path",
        "calib_samples",
        "calib_tasks",
        "eval_methods",
        "eval_tasks",
        "image_path",
        "prompt",
        "train_tasks",
    }
)

# Fields whose value is constrained. ``argparse`` rejects anything else, so a
# typo fails at parse time rather than deep inside the flow.
_FIELD_CHOICES = {
    "backend": ("htp", "gpu"),
    "dtype_override": ("fp32", "fp16"),
    "model_mode": ("kv", "hybrid", "lookahead"),
}


def _parser_type(hint: Any) -> Any:
    """Map a field's annotation to the ``type`` its argparse argument takes.

    The annotation rather than the default is the source of truth: an
    ``Optional[int]`` field defaults to ``None``, which carries no type, and
    typing it ``str`` would hand the consumer ``"1024"`` where it does
    arithmetic on ``max_context_len``.

    Args:
        hint: The field's resolved annotation.

    Returns:
        ``int`` or ``float`` for numeric fields, ``str`` otherwise.
    """
    if get_origin(hint) is Union:
        args = [arg for arg in get_args(hint) if arg is not type(None)]
        if len(args) == 1:
            hint = args[0]

    return hint if hint in (int, float) else str


@dataclass
class ControlArgs(argparse.Namespace):
    """Configuration for the LLM export, quantization and evaluation flow.

    Every field is read by at least one existing component; the grouping below
    follows which part of the flow consumes it. Field defaults match
    ``llama.py``'s parser, so ``ControlArgs()`` is equivalent to running that
    parser with only its required arguments supplied.
    """

    # --- Model identification and input paths ---
    decoder_model: str = ""
    artifact: str = DEFAULT_ARTIFACT
    checkpoint: Optional[str] = None
    params: Optional[str] = None
    tokenizer_model: Optional[str] = None
    tokenizer_bin: Optional[str] = None

    # --- Graph shapes and modes ---
    model_mode: str = DEFAULT_MODEL_MODE
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN
    max_context_len: Optional[int] = None
    prefill_ar_len: int = DEFAULT_PREFILL_AR_LEN
    dtype_override: str = DEFAULT_DTYPE_OVERRIDE
    # Lookahead decoding shape parameters.
    ngram: int = DEFAULT_NGRAM
    window: int = DEFAULT_WINDOW
    gcap: int = DEFAULT_GCAP

    # --- Quantization ---
    use_fp16: bool = False
    embedding_quantize: Optional[str] = None
    quant_recipe_suggestion: bool = False
    batch_size: int = DEFAULT_BATCH_SIZE

    # --- Calibration data selection ---
    calib_tasks: Optional[List[str]] = None
    calib_limit: int = DEFAULT_CALIB_LIMIT
    calib_num_fewshot: Optional[int] = None
    calib_samples: Optional[List[str]] = None
    calib_hf_dataset: Optional[str] = None
    calib_hf_limit: int = DEFAULT_CALIB_HF_LIMIT

    # --- Quantization-aware training ---
    #
    # ``train_config`` and ``lr_config`` name YAML files. ``llama.py`` defaults
    # them to files under its own directory; here they default to ``None``,
    # which the consumer reads as "use your own default", because this package
    # does not reference paths in ``examples/``. See
    # :data:`_PARSER_DEFAULT_EXEMPT_FIELDS`.
    qat: bool = False
    train_config: Optional[str] = None
    lr_config: Optional[str] = None
    train_tasks: Optional[List[str]] = None
    train_limit: int = DEFAULT_TRAIN_LIMIT
    train_hf_dataset: Optional[str] = None
    train_hf_limit: int = DEFAULT_TRAIN_HF_LIMIT
    train_val_ratio: float = DEFAULT_TRAIN_VAL_RATIO
    freeze_all_params: bool = False

    # --- Features ---
    use_attention_sink: Optional[str] = None

    # --- Runtime prompts and multimodal inputs ---
    prompt: List[str] = field(default_factory=list)
    system_prompt: str = ""
    temperature: float = DEFAULT_TEMPERATURE
    audio_path: List[str] = field(default_factory=list)
    image_path: List[str] = field(default_factory=list)

    # --- Evaluation ---
    eval_methods: List[str] = field(default_factory=lambda: [DEFAULT_EVAL_METHOD])
    eval_tasks: Optional[List[str]] = None
    eval_limit: int = DEFAULT_EVAL_LIMIT
    eval_num_fewshot: Optional[int] = None

    # --- Backend and SoC selection ---
    #
    # ``soc_model`` stays a string here, as it is in ``PipelineContext``; the
    # conversion to ``QcomChipset`` happens in the compilation adapter layer.
    soc_model: Optional[str] = None
    backend: str = DEFAULT_BACKEND
    online_prepare: bool = False
    htp_performance_mode: int = DEFAULT_HTP_PERFORMANCE_MODE

    # --- Flow control ---
    compile_only: bool = False
    pre_gen_pte: Optional[str] = None
    verbose: bool = False

    # --- Device and runner settings ---
    #
    # These are consumed by ``QnnConfig``, which reflects over this object with
    # ``vars()`` and asserts that ``soc_model`` and ``build_folder`` are set.
    build_folder: Optional[str] = None
    direct_build_folder: Optional[str] = None
    target: str = DEFAULT_TARGET
    host: Optional[str] = None
    device: Optional[str] = None
    enable_x86_64: bool = False
    shared_buffer: bool = False
    skip_push: bool = False
    config_file: Optional[str] = None

    # --- Partitioning overrides ---
    skip_delegate_node_ids: Optional[str] = None
    skip_delegate_node_ops: Optional[str] = None

    # --- Debug and CI ---
    dump_intermediate_outputs: bool = False
    profile_level: int = DEFAULT_PROFILE_LEVEL
    ci: bool = False
    seed: Optional[int] = None
    # IPC endpoint the CI harness listens on for results.
    ip: str = ""
    port: int = DEFAULT_PORT

    @classmethod
    def field_names(cls) -> frozenset:
        """The set of field names this dataclass defines."""
        return frozenset(f.name for f in fields(cls))

    @classmethod
    def build_parser(cls, **parser_kwargs: Any) -> argparse.ArgumentParser:
        """Derive an ``argparse`` parser from this dataclass's fields.

        Each field becomes ``--field-name`` (also accepting ``--field_name``)
        with this dataclass's default, so a command-line entry point does not
        maintain a second table of defaults that can drift from these. Booleans
        become ``store_true`` flags; list-valued fields take ``nargs="+"``;
        :data:`_FIELD_CHOICES` constrains the rest.

        The parser sets no help text: the authoritative description of each
        setting is this dataclass's field grouping and comments. Entry points
        wanting help strings, or arguments this dataclass does not carry, pass
        this parser as a parent (``ArgumentParser(parents=[...])``).

        Args:
            **parser_kwargs: Forwarded to ``argparse.ArgumentParser``.

        Returns:
            A parser whose ``parse_args`` result :meth:`from_namespace` accepts.
        """
        parser = argparse.ArgumentParser(**parser_kwargs)
        hints = get_type_hints(cls)

        for spec in fields(cls):
            if spec.default is not MISSING:
                default = spec.default
            elif spec.default_factory is not MISSING:
                default = spec.default_factory()
            else:
                default = None

            flags = [f"--{spec.name.replace('_', '-')}"]
            if "_" in spec.name:
                flags.append(f"--{spec.name}")

            if isinstance(default, bool):
                parser.add_argument(
                    *flags, dest=spec.name, action="store_true", default=default
                )
                continue

            kwargs: Dict[str, Any] = {"dest": spec.name, "default": default}
            if spec.name in _LIST_VALUED_FIELDS:
                kwargs["nargs"] = "+"
                kwargs["type"] = str
            elif spec.name in _FIELD_CHOICES:
                kwargs["choices"] = _FIELD_CHOICES[spec.name]
                kwargs["type"] = str
            else:
                kwargs["type"] = _parser_type(hints[spec.name])
            parser.add_argument(*flags, **kwargs)

        return parser

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> "ControlArgs":
        """Build from an ``argparse.Namespace``, ignoring unknown attributes.

        Lets the legacy entry point and the pipeline share one configuration
        type: ``llama.py``'s parser produces a superset of these fields, and
        attributes without a matching field are dropped.

        Args:
            namespace: A parsed namespace, e.g. from ``llama.py``'s parser.

        Returns:
            A ``ControlArgs`` carrying every recognised attribute.
        """
        known = cls.field_names()
        supplied = vars(namespace)

        ignored = sorted(set(supplied) - known)
        if ignored:
            logger.debug("Ignoring unrecognised arguments: %s", ignored)

        return cls(**{k: v for k, v in supplied.items() if k in known})

    @classmethod
    def from_pipeline_context(
        cls,
        context: "PipelineContext",
        **overrides: Any,
    ) -> "ControlArgs":
        """Build from a ``PipelineContext`` plus per-stage overrides.

        The context supplies the four settings it owns; anything else the LLM
        components need is taken from ``context.extra_options`` when the key
        names a field, and ``overrides`` wins over both.

        ``extra_options`` cannot reach the context-owned settings (see
        :data:`_CONTEXT_OWNED_FIELDS`). Those have one authoritative source, and
        letting a loosely-typed dict silently replace one would let the flow
        compile for a SoC other than the one the context reports and the
        quantization stage validated against -- a mismatch that only surfaces on
        device. A caller that genuinely means to change them passes an explicit
        override, which is deliberate and traceable at the call site.

        Args:
            context: The pipeline context holding user inputs.
            **overrides: Field values taking precedence over the context.

        Returns:
            A populated ``ControlArgs``.

        Raises:
            TypeError: If an override does not name a field, which would
                otherwise be silently dropped.
        """
        known = cls.field_names()

        unknown = sorted(set(overrides) - known)
        if unknown:
            raise TypeError(
                f"ControlArgs has no field(s) {unknown}; valid fields: "
                f"{sorted(known)}"
            )

        kwargs: Dict[str, Any] = {
            "decoder_model": context.model_name,
            "soc_model": context.soc_model,
            "artifact": context.artifact_dir,
            "prompt": list(context.prompt),
        }

        extra = context.extra_options or {}
        for key, value in extra.items():
            if key in _CONTEXT_OWNED_FIELDS:
                logger.debug(
                    "extra_options['%s'] ignored: the context owns this field. "
                    "Pass it as an explicit override to change it.",
                    key,
                )
            elif key in known:
                kwargs[key] = value
            else:
                logger.debug("extra_options['%s'] is not a ControlArgs field", key)

        kwargs.update(overrides)

        return cls(**kwargs)
