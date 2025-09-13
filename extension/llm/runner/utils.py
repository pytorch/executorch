# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions for the MultimodalRunner Python bindings.

This module provides helper functions for common tasks like image preprocessing,
configuration creation, and data conversion.
"""

from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np

try:
    from PIL import Image as PILImage

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from ._llm_runner import GenerationConfig


def load_image_from_file(
    image_path: Union[str, Path],
    target_size: Optional[Tuple[int, int]] = None,
    mode: str = "RGB",
) -> np.ndarray:
    """
    Load an image from file and optionally resize it.

    Args:
        image_path: Path to the image file
        target_size: Optional (width, height) tuple to resize the image
        mode: Image mode ('RGB', 'RGBA', 'L' for grayscale)

    Returns:
        NumPy array with shape (H, W, C) for color or (H, W) for grayscale

    Raises:
        FileNotFoundError: If the image file doesn't exist
        ImportError: If neither PIL nor OpenCV is available
        ValueError: If the image cannot be loaded
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if HAS_PIL:
        # Use PIL/Pillow
        image = PILImage.open(image_path)

        # Convert to requested mode
        if image.mode != mode:
            image = image.convert(mode)

        # Resize if requested
        if target_size is not None:
            image = image.resize(target_size, PILImage.Resampling.LANCZOS)

        # Convert to numpy array
        return np.array(image, dtype=np.uint8)
    else:
        # Try OpenCV
        try:
            import cv2

            # Read image
            if mode == "L":
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            else:
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            # Convert BGR to RGB if needed
            if mode == "RGB" and len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif mode == "RGBA" and len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

            # Resize if requested
            if target_size is not None:
                image = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)

            return image.astype(np.uint8)

        except ImportError:
            raise ImportError(
                "Either PIL or OpenCV is required to load images from files. "
                "Install with: pip install pillow or pip install opencv-python"
            )


def preprocess_image(
    image: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = False,
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None,
) -> np.ndarray:
    """
    Preprocess an image array for model input.

    Args:
        image: Input image as numpy array (H, W, C)
        target_size: Optional (width, height) tuple to resize the image
        normalize: Whether to normalize pixel values to [0, 1]
        mean: Mean values for normalization (per channel)
        std: Standard deviation values for normalization (per channel)

    Returns:
        Preprocessed image array

    Raises:
        ValueError: If image dimensions are invalid
    """
    if image.ndim != 3:
        raise ValueError(
            f"Image must be 3-dimensional (H, W, C), got shape {image.shape}"
        )

    # Resize if needed
    if target_size is not None:
        if HAS_PIL:
            # Use PIL for resizing
            pil_image = PILImage.fromarray(image)
            pil_image = pil_image.resize(target_size, PILImage.Resampling.LANCZOS)
            image = np.array(pil_image)
        else:
            # Try OpenCV
            try:
                import cv2

                image = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
            except ImportError:
                # Simple nearest neighbor resize as fallback
                from scipy import ndimage

                factors = (
                    target_size[1] / image.shape[0],
                    target_size[0] / image.shape[1],
                    1,
                )
                image = ndimage.zoom(image, factors, order=1)

    # Convert to float for normalization
    if normalize or mean is not None or std is not None:
        image = image.astype(np.float32)

        if normalize:
            image = image / 255.0

        if mean is not None:
            mean_arr = np.array(mean).reshape(1, 1, -1)
            image = image - mean_arr

        if std is not None:
            std_arr = np.array(std).reshape(1, 1, -1)
            image = image / std_arr

    return image


def create_generation_config(
    max_new_tokens: int = 1000,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 40,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    echo: bool = False,
    seed: Optional[int] = None,
    **kwargs,
) -> GenerationConfig:
    """
    Create a GenerationConfig with sensible defaults.

    Args:
        max_new_tokens: Maximum number of tokens to generate (default: 1000)
        temperature: Sampling temperature, higher = more random (default: 0.8)
        top_p: Nucleus sampling parameter (default: 0.95)
        top_k: Top-k sampling parameter (default: 40)
        repetition_penalty: Penalty for repeating tokens (default: 1.0)
        presence_penalty: Penalty for using tokens that appear in the prompt (default: 0.0)
        frequency_penalty: Penalty based on token frequency (default: 0.0)
        echo: Whether to echo the input prompt (default: False)
        seed: Random seed for reproducibility (default: None)
        **kwargs: Additional parameters to set on the config

    Returns:
        A configured GenerationConfig object

    Example:
        >>> config = create_generation_config(
        ...     max_new_tokens=100,
        ...     temperature=0.7,
        ...     top_p=0.9
        ... )
    """
    config = GenerationConfig()

    # Set all parameters
    config.max_new_tokens = max_new_tokens
    config.temperature = temperature
    config.top_p = top_p
    config.top_k = top_k
    config.repetition_penalty = repetition_penalty
    config.presence_penalty = presence_penalty
    config.frequency_penalty = frequency_penalty
    config.echo = echo

    if seed is not None:
        config.seed = seed

    # Set any additional parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"GenerationConfig has no parameter '{key}'")

    return config


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """
    Estimate the number of tokens in a text string.

    This is a rough approximation and actual token count may vary
    depending on the tokenizer used.

    Args:
        text: Input text string
        chars_per_token: Average characters per token (default: 4.0)

    Returns:
        Estimated number of tokens
    """
    return max(1, int(len(text) / chars_per_token))


def format_stats(stats: Any) -> str:
    """
    Format generation statistics for display.

    Args:
        stats: Stats object from the runner

    Returns:
        Formatted string with statistics
    """
    lines = [
        "Generation Statistics:",
        f"  Model load time: {stats.get_model_load_time_ms():.2f} ms",
        f"  Prompt eval time: {stats.get_prompt_eval_time_ms():.2f} ms",
        f"  Generation time: {stats.get_eval_time_ms():.2f} ms",
        f"  Sampling time: {stats.get_sampling_time_ms():.2f} ms",
        f"  Total inference time: {stats.get_inference_time_ms():.2f} ms",
        f"  Prompt tokens: {stats.num_prompt_tokens}",
        f"  Generated tokens: {stats.num_generated_tokens}",
        f"  Tokens per second: {stats.get_tokens_per_second():.2f}",
    ]
    return "\n".join(lines)
