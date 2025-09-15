# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Python bindings for ExecuTorch MultimodalRunner.

This module provides a Python interface to the ExecuTorch multimodal LLM runner,
enabling processing of mixed inputs (text, images, audio) and text generation.
"""

from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import numpy as np

try:
    from PIL import Image as PILImage

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    # Import shared components from the compiled C++ extension
    from executorch.extension.llm.runner._llm_runner import (  # noqa: F401
        GenerationConfig,
        Image,
        make_image_input,
        make_text_input,
        MultimodalInput,
        MultimodalRunner as _MultimodalRunnerCpp,
        Stats,
    )
except ImportError:
    raise RuntimeError(
        "LLM runner is not installed. Please build ExecuTorch from source with EXECUTORCH_BUILD_PYBIND=ON"
    )


# Define the high-level Python wrapper for MultimodalRunner
class MultimodalRunner:
    """
    High-level Python wrapper for the ExecuTorch MultimodalRunner.

    This class provides a convenient interface for running multimodal language models
    that can process text, images, and other modalities to generate text output.

    Args:
        model_path: Path to the ExecuTorch model file (.pte)
        tokenizer_path: Path to the tokenizer file
        temperature: Default temperature for text generation (default: 0.8)
        device: Device to run on (currently only 'cpu' is supported)

    Example:
        >>> runner = MultimodalRunner("model.pte", "tokenizer.bin")
        >>> inputs = [
        ...     runner.create_text_input("Describe this image:"),
        ...     runner.create_image_input("image.jpg")
        ... ]
        >>> response = runner.generate_text(inputs, max_new_tokens=100)
        >>> print(response)
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        tokenizer_path: Union[str, Path],
        temperature: float = 0.8,
        device: str = "cpu",
    ):
        """Initialize the MultimodalRunner."""
        if device != "cpu":
            raise ValueError(
                f"Currently only 'cpu' device is supported, got '{device}'"
            )

        # Convert paths to strings
        model_path = str(Path(model_path).resolve())
        tokenizer_path = str(Path(tokenizer_path).resolve())

        # Validate paths exist
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not Path(tokenizer_path).exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

        # Initialize the C++ runner
        self._runner = _MultimodalRunnerCpp(model_path, tokenizer_path, temperature)
        self._model_path = model_path
        self._tokenizer_path = tokenizer_path
        self._default_temperature = temperature

    def create_text_input(self, text: str):
        """
        Create a text input for multimodal processing.

        Args:
            text: The input text string

        Returns:
            A MultimodalInput object containing the text
        """
        return make_text_input(text)

    def create_image_input(  # noqa: C901
        self, image: Union[str, Path, np.ndarray, "PILImage.Image"]
    ):
        """
        Create an image input for multimodal processing.

        Args:
            image: Can be:
                - Path to an image file (str or Path)
                - NumPy array with shape (H, W, C) where C is 3 (RGB) or 4 (RGBA)
                - PIL Image object

        Returns:
            A MultimodalInput object containing the image

        Raises:
            ValueError: If the image format is not supported
            FileNotFoundError: If the image file doesn't exist
        """
        if isinstance(image, (str, Path)):
            # Load image from file
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            if HAS_PIL:
                pil_image = PILImage.open(image_path)
                # Convert to RGB if necessary
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")
                image = np.array(pil_image, dtype=np.uint8)
            else:
                # Try to use cv2 if available
                try:
                    import cv2

                    image = cv2.imread(str(image_path))
                    if image is None:
                        raise ValueError(f"Failed to load image: {image_path}")
                    # Convert BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                except ImportError:
                    raise ImportError(
                        "Either PIL or OpenCV is required to load images from files. "
                        "Install with: pip install pillow or pip install opencv-python"
                    )

        elif HAS_PIL and isinstance(image, PILImage.Image):
            # Convert PIL Image to numpy array
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = np.array(image, dtype=np.uint8)

        elif isinstance(image, np.ndarray):
            # Validate numpy array
            if image.ndim != 3:
                raise ValueError(
                    f"Image array must be 3-dimensional (H, W, C), got shape {image.shape}"
                )
            if image.shape[2] not in [3, 4]:
                raise ValueError(
                    f"Image must have 3 (RGB) or 4 (RGBA) channels, got {image.shape[2]}"
                )
            if image.dtype != np.uint8:
                # Convert to uint8 if necessary
                if image.max() <= 1.0:
                    # Assume normalized [0, 1] range
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        return make_image_input(image)

    def generate(
        self,
        inputs: List[Any],
        config: Optional[GenerationConfig] = None,
        token_callback: Optional[Callable[[str], None]] = None,
        stats_callback: Optional[Callable[[Any], None]] = None,
    ):
        """
        Generate text from multimodal inputs with streaming callbacks.

        Args:
            inputs: List of multimodal inputs (text, images, etc.)
            config: Generation configuration (uses defaults if None)
            token_callback: Function called for each generated token
            stats_callback: Function called with generation statistics
        """
        if config is None:
            config = GenerationConfig()
            config.temperature = self._default_temperature

        self._runner.generate(inputs, config, token_callback, stats_callback)

    def generate_text(
        self,
        inputs: List[Any],
        config: Optional[GenerationConfig] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        Generate text from multimodal inputs and return the complete result.

        Args:
            inputs: List of multimodal inputs (text, images, etc.)
            config: Generation configuration (overrides other parameters if provided)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Top-p sampling parameter
            **kwargs: Additional generation parameters

        Returns:
            The generated text as a string
        """
        if config is None:
            config = GenerationConfig()
            config.temperature = temperature or self._default_temperature
            if max_new_tokens is not None:
                config.max_new_tokens = max_new_tokens
            if top_p is not None:
                config.top_p = top_p

            # Set any additional parameters
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        return self._runner.generate_text(inputs, config)

    def stop(self):
        """Stop the current generation process."""
        self._runner.stop()

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size of the model."""
        return self._runner.get_vocab_size()

    @property
    def model_path(self) -> str:
        """Get the path to the loaded model."""
        return self._model_path

    @property
    def tokenizer_path(self) -> str:
        """Get the path to the loaded tokenizer."""
        return self._tokenizer_path

    def __repr__(self) -> str:
        return (
            f"MultimodalRunner(model='{Path(self._model_path).name}', "
            f"tokenizer='{Path(self._tokenizer_path).name}', "
            f"vocab_size={self.vocab_size})"
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.stop()
        return False


# Import utility functions
from .utils import create_generation_config, load_image_from_file, preprocess_image

__all__ = [
    "MultimodalRunner",
    "GenerationConfig",
    "Stats",
    "Image",
    "MultimodalInput",
    "make_text_input",
    "make_image_input",
    "load_image_from_file",
    "preprocess_image",
    "create_generation_config",
]

__version__ = "0.1.0"
