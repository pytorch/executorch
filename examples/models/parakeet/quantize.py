"""Quantization utilities for Parakeet model export.

Re-exports quantize_model_ from the shared ExecuTorch LLM export library.
"""

from executorch.extension.llm.export.quantize import quantize_model_

__all__ = ["quantize_model_"]
