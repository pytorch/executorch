# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``sqnr_eval``: score runner logits against host-side golden logits."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_TEXT_DECODER,
    DECODE_QDQ_FILENAME,
)
from executorch.backends.qualcomm.genai_pipeline.configs.inference_input_config import (
    InferenceInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.control_args import (
    ControlArgs,
    SQNR_EVAL,
)
from executorch.backends.qualcomm.genai_pipeline.evaluation.evaluator import (
    DeployedModel,
    EvaluationScore,
)
from executorch.backends.qualcomm.genai_pipeline.graph_names import GRAPH_FORWARD
from executorch.backends.qualcomm.genai_pipeline.inference.runner_io import (
    ForwardRequest,
    RunnerMetadata,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HostReference:
    """Host-side goldens that runner logits are scored against.

    Attributes:
        fp32_module: The prepared FP32 text decoder.
        decoder_inputs: Structured example inputs of the decoder's calibration
            graph, ``(tokens, attn_mask, pos_ids, k_caches, v_caches)``.
        max_context_len: The calibration graph's sequence length.
        use_i64_token: Whether the decoder consumes int64 tokens.
        qdq_path: Where the quantization stage saves the QDQ graph. The file
            may be absent, in which case the CPU QDQ comparison is skipped.
    """

    fp32_module: torch.nn.Module
    decoder_inputs: Tuple[Any, ...]
    max_context_len: int
    use_i64_token: bool
    qdq_path: Path


def build_host_reference(
    control_args: ControlArgs,
    input_config: InferenceInputConfig,
) -> HostReference:
    """Build the SQNR goldens from the host model the inference stage carries.

    Args:
        control_args: Control args supplying the context length, embedding
            quantization, and ``pre_gen_pte``.
        input_config: Inference configuration carrying model preparation's
            FP32 modules and their example inputs, keyed by component and
            graph, and the artifact directory. The QDQ graph is read from the
            ``--pre-gen-pte`` directory when given, else the artifact directory.

    Returns:
        The host reference.

    Raises:
        ValueError: If model preparation's FP32 module is unavailable.
    """
    if input_config.model_module is None or input_config.example_inputs is None:
        raise ValueError("sqnr_eval requires the FP32 module from model preparation")

    qdq_dir = control_args.pre_gen_pte or input_config.artifact_dir
    return HostReference(
        fp32_module=input_config.model_module[ARTIFACT_TEXT_DECODER],
        decoder_inputs=input_config.example_inputs[ARTIFACT_TEXT_DECODER][
            GRAPH_FORWARD
        ],
        max_context_len=control_args.max_context_len,
        use_i64_token=control_args.embedding_quantize is not None,
        qdq_path=Path(qdq_dir) / DECODE_QDQ_FILENAME,
    )


def build_prompt_token_ids(
    tokenizer: Any, prompt: str, system_prompt: str
) -> List[int]:
    """Tokenize a prompt the way the legacy calibration and eval path does.

    Args:
        tokenizer: The ``TokenizerWrapper`` from model preparation.
        prompt: The raw user prompt.
        system_prompt: The system prompt, or an empty string.

    Returns:
        Token IDs with BOS prepended and no EOS.

    Raises:
        ValueError: If ``tokenizer`` is not a ``TokenizerWrapper``.
    """
    from executorch.examples.qualcomm.oss_scripts.llama.tokenizer import (
        TokenizerWrapper,
    )

    if not isinstance(tokenizer, TokenizerWrapper):
        raise ValueError(
            "sqnr_eval requires a TokenizerWrapper with chat-template support; "
            f"got {type(tokenizer).__name__}"
        )
    messages = tokenizer.make_chat_template(prompt, system_prompt)
    text = tokenizer.apply_chat_template(messages)
    return tokenizer.tokenizer.encode(text, bos=True, eos=False)


def forward_prompt(
    module: torch.nn.Module,
    decoder_example_inputs: tuple,
    token_ids: List[int],
    max_context_len: int,
    use_i64_token: bool,
) -> torch.Tensor:
    """Run one full-sequence forward and return the prompt-position logits.

    Args:
        module: The FP32 decoder or the loaded QDQ calibration graph.
        decoder_example_inputs: The structured calibration-graph example inputs
            ``(tokens, attn_mask, pos_ids, k_caches, v_caches)``.
        token_ids: The prompt token IDs.
        max_context_len: The calibration graph's sequence length.
        use_i64_token: Whether the decoder consumes int64 tokens.

    Returns:
        Logits of shape ``(1, len(token_ids), vocab)``.
    """
    from executorch.backends.qualcomm.genai_pipeline.datasets.collators.llm_collator import (
        LLMDatasetCollector,
    )
    from executorch.examples.qualcomm.oss_scripts.llama.inference import (
        DecoderInference,
    )

    collator = LLMDatasetCollector().create_collators(
        {ARTIFACT_TEXT_DECODER: decoder_example_inputs},
        max_context_len,
    )[ARTIFACT_TEXT_DECODER]
    batch = collator([(token_ids,)])
    decoder = DecoderInference(
        get_example_inputs=lambda: decoder_example_inputs,
        max_context_len=max_context_len,
        use_i64_token=use_i64_token,
    )
    logits = decoder.predict_step(
        module,
        batch["input_ids"],
        attn_mask=batch["attention_mask"],
    )
    return logits[:, : len(token_ids), :]


def score_sqnr(
    reference: HostReference,
    token_ids: List[int],
    qnn_logits: torch.Tensor,
) -> Tuple[float, Optional[float]]:
    """Score runner logits against the FP32 module and the saved QDQ graph.

    Args:
        reference: The host goldens to score against.
        token_ids: The prompt token IDs fed to the runner.
        qnn_logits: Dequantized runner logits over ``token_ids``.

    Returns:
        ``(sqnr, qdq_sqnr)``; ``qdq_sqnr`` is ``None`` when the QDQ graph is
        not found.
    """
    from torchao.quantization.utils import compute_error

    def forward(module: torch.nn.Module) -> torch.Tensor:
        return forward_prompt(
            module,
            reference.decoder_inputs,
            token_ids,
            reference.max_context_len,
            reference.use_i64_token,
        )

    fp32_logits = forward(reference.fp32_module)
    sqnr = compute_error(fp32_logits, qnn_logits).item()
    logger.info("SQNR Eval Score between FP32 nn.Module and QNN: %s", sqnr)

    if not reference.qdq_path.is_file():
        logger.warning(
            "Skipping SQNR between CPU QDQ and QNN: no QDQ graph at %s. The "
            "quantization stage saves it, so it is missing when quantization was "
            "skipped (e.g. --use_fp16) or when --pre-gen-pte points to a directory "
            "from a run that did not save it.",
            reference.qdq_path,
        )
        return sqnr, None
    qdq_logits = forward(torch.export.load(reference.qdq_path).module())
    qdq_sqnr = compute_error(qdq_logits, qnn_logits).item()
    logger.info("SQNR Eval Score between CPU QDQ and QNN: %s", qdq_sqnr)
    logger.info(
        "SQNR Eval Score between FP32 nn.Module and CPU QDQ: %s",
        compute_error(fp32_logits, qdq_logits).item(),
    )
    return sqnr, qdq_sqnr


class SqnrEvaluator:
    """Score one KV-mode runner forward against the host goldens.

    Feeds one ``ForwardRequest`` over the chat-templated prompt and compares
    the predicted logits with the FP32 module and, when saved, the QDQ graph.
    """

    method = SQNR_EVAL

    def evaluate(
        self,
        model: DeployedModel,
        control_args: ControlArgs,
        input_config: InferenceInputConfig,
        metadata: Optional[RunnerMetadata],
    ) -> EvaluationScore:
        """Forward the chat-templated prompt and score its logits.

        Args:
            model: Runs the forward on the deployed runner.
            control_args: Control args that supply the prompt and system prompt.
            input_config: Inference configuration carrying the host model and
                the tokenizer.
            metadata: Unused; the adapter bounds the input by the decoder's
                context length.

        Returns:
            The SQNR against the FP32 module, and against the QDQ graph when
            it is saved.

        Raises:
            ValueError: If model preparation's FP32 module is unavailable.
        """
        reference = build_host_reference(control_args, input_config)
        token_ids = build_prompt_token_ids(
            input_config.tokenizer, control_args.prompt[0], control_args.system_prompt
        )
        prediction = model(ForwardRequest(token_ids=token_ids))

        sqnr, qdq_sqnr = score_sqnr(reference, token_ids, prediction.logits)
        metrics: Dict[str, Any] = {"sqnr": sqnr}
        if qdq_sqnr is not None:
            metrics["qdq_sqnr"] = qdq_sqnr
        return EvaluationScore(metrics=metrics)
