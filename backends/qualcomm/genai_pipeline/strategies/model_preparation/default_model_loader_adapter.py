# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

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

    def export_tokenizer(
        self,
        tokenizer: Any,
        output_dir: Path,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Export tokenizer to disk and return the runtime tokenizer file.

        ``save_pretrained`` writes several files and returns the tuple of paths
        it wrote, with the tokenizer file last. Both ``llm::load_tokenizer`` and
        ``pytorch_tokenizers.get_tokenizer`` expect that **single file**, not the
        containing directory, so we return it -- mirroring the existing
        ``TokenizerWrapper._from_hf`` flow.

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

        runtime_tokenizer_path = Path(artifacts[-1])
        logger.info("Tokenizer exported to %s", runtime_tokenizer_path)
        return runtime_tokenizer_path
