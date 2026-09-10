# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class DefaultModelLoaderAdapter:
    """Default adapter delegating to HuggingFace transformers for model loading.

    Wraps ``AutoModelForCausalLM`` and ``AutoTokenizer`` for production use.
    Requires ``transformers`` to be installed.

    .. note::
        This adapter is designed for **text-only causal LM** models.
        For multimodal models (vision/audio encoders), use a specialized adapter
        (e.g., ``MultiModalModelLoaderAdapter``) that handles separate encoder/decoder
        loading via ``AutoModel`` or ``AutoModelForSpeechSeq2Seq``.

    .. note::
        Calibration data is **not** produced here -- see
        ``CalibrationDataAdapter``.
    """

    #: Batch size and sequence length of the generated example inputs. HTP has no
    #: dynamic shapes, so these dimensions are baked into the exported graph.
    DEFAULT_BATCH_SIZE = 1
    DEFAULT_AR_LEN = 1

    #: Preferred runtime tokenizer file names, in priority order.
    #: ``pytorch_tokenizers.get_tokenizer`` dispatches on the file extension
    #: (``.json`` -> ``HuggingFaceTokenizer``, otherwise Llama2c/Tiktoken), so the
    #: file we hand back selects the runtime tokenizer implementation.
    RUNTIME_TOKENIZER_NAMES = ("tokenizer.json", "tokenizer.model")

    def load_model(
        self,
        model_name: str,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Load a causal LM model from HuggingFace hub or local path.

        Args:
            model_name: HuggingFace model ID or local path.
            extra_options: Additional options. Supported keys:
                - ``torch_dtype``: dtype for model weights (default: float32).
                - ``device_map``: device placement strategy.
                - ``attn_implementation``: attention implementation to use.

        Returns:
            The loaded nn.Module in eval mode.
        """
        import torch
        from transformers import AutoModelForCausalLM

        extra_options = extra_options or {}
        torch_dtype = extra_options.get("torch_dtype", torch.float32)
        device_map = extra_options.get("device_map", "cpu")
        attn_impl = extra_options.get("attn_implementation")

        logger.info("Loading model '%s' with dtype=%s", model_name, torch_dtype)

        kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
        }
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl

        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        model.eval()

        logger.info("Model loaded successfully")
        return model

    def load_tokenizer(
        self,
        model_name: str,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Load the tokenizer for the given model.

        Args:
            model_name: HuggingFace model ID or local path.
            extra_options: Additional tokenizer options.

        Returns:
            The loaded tokenizer instance.
        """
        from transformers import AutoTokenizer

        extra_options = extra_options or {}

        logger.info("Loading tokenizer for '%s'", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, **extra_options)

        logger.info("Tokenizer loaded successfully")
        return tokenizer

    def get_example_inputs(
        self,
        model: Any,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, ...]:
        """Build example inputs for ``torch.export`` from the model itself.

        Prefers the model's own ``get_example_inputs()`` when it exposes one, so
        models that already describe their export signature (the LLM wrappers
        build a flat ``(tokens, attn_mask, pos_ids, *k_caches, *v_caches)``
        tuple) stay authoritative. Otherwise a minimal ``(input_ids,)`` is
        synthesized, which is the correct signature for a plain HuggingFace
        causal LM without an external KV cache.

        Args:
            model: The module returned by :meth:`load_model`.
            extra_options: Additional options. Supported keys:
                - ``batch_size``: Batch dimension (default:
                  ``DEFAULT_BATCH_SIZE``).
                - ``ar_len``: Sequence length / autoregressive window
                  (default: ``DEFAULT_AR_LEN``).

        Returns:
            A flat tuple positionally matching ``model.forward``.
        """
        import torch

        extra_options = extra_options or {}

        model_provided = getattr(model, "get_example_inputs", None)
        if callable(model_provided):
            logger.info("Using example inputs provided by the model")
            return tuple(model_provided())

        batch_size = extra_options.get("batch_size", self.DEFAULT_BATCH_SIZE)
        ar_len = extra_options.get("ar_len", self.DEFAULT_AR_LEN)

        logger.info(
            "Synthesizing example inputs with batch_size=%d, ar_len=%d",
            batch_size,
            ar_len,
        )
        # int64 token ids: the embedding lookup indexes with them.
        return (torch.zeros((batch_size, ar_len), dtype=torch.int64),)

    def export_tokenizer(
        self,
        tokenizer: Any,
        output_dir: Path,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Export tokenizer to disk and return the runtime tokenizer file.

        ``save_pretrained`` writes several files and returns the tuple of paths it
        wrote. Both ``llm::load_tokenizer`` and ``pytorch_tokenizers.get_tokenizer``
        expect a **single file**, not the containing directory, so one artifact has
        to be singled out.

        The file is chosen **by name** -- ``tokenizer.json`` first, then
        ``tokenizer.model`` -- rather than by position in the returned tuple.
        ``get_tokenizer`` dispatches on the extension, so picking the wrong
        artifact silently constructs the wrong tokenizer class instead of raising,
        and ``save_pretrained``'s ordering is an implementation detail that varies
        with the tokenizer (fast vs slow, whether ``added_tokens.json`` is
        written). ``artifacts[-1]`` remains a last-resort fallback for tokenizers
        that emit neither name, mirroring ``TokenizerWrapper._from_hf``.

        Args:
            tokenizer: The tokenizer instance to export.
            output_dir: Directory to write the exported tokenizer artifacts to.
            extra_options: Additional export options.

        Returns:
            Path to the runtime tokenizer file (e.g. ``tokenizer.json``).

        Raises:
            FileNotFoundError: If no tokenizer artifacts were written.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Exporting tokenizer to %s", output_dir)
        artifacts = tokenizer.save_pretrained(str(output_dir))
        if not artifacts:
            raise FileNotFoundError(
                f"save_pretrained() reported no tokenizer artifacts in {output_dir}."
            )

        runtime_tokenizer_path = self._select_runtime_tokenizer(artifacts)
        logger.info("Tokenizer exported to %s", runtime_tokenizer_path)
        return runtime_tokenizer_path

    @classmethod
    def _select_runtime_tokenizer(cls, artifacts: Any) -> Path:
        """Pick the runtime tokenizer file out of ``save_pretrained``'s artifacts.

        Args:
            artifacts: The paths reported by ``save_pretrained``.

        Returns:
            The first artifact matching :attr:`RUNTIME_TOKENIZER_NAMES`, falling
            back to the last artifact when none matches.
        """
        paths = [Path(artifact) for artifact in artifacts]

        for name in cls.RUNTIME_TOKENIZER_NAMES:
            for path in paths:
                if path.name == name:
                    return path

        logger.warning(
            "None of %s found among tokenizer artifacts; falling back to %s.",
            ", ".join(cls.RUNTIME_TOKENIZER_NAMES),
            paths[-1],
        )
        return paths[-1]
