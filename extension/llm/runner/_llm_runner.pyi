"""
Type stubs for _llm_runner module.

This file provides type annotations for the ExecuTorch LLM Runner Python bindings.
"""

from typing import Callable, List, Optional, overload

import torch

class GenerationConfig:
    """Configuration for text generation."""

    echo: bool
    """Whether to echo the input prompt in the output."""

    max_new_tokens: int
    """Maximum number of new tokens to generate (-1 for auto)."""

    warming: bool
    """Whether this is a warmup run (affects perf benchmarking)."""

    seq_len: int
    """Maximum number of total tokens (-1 for auto)."""

    temperature: float
    """Temperature for sampling (higher = more random)."""

    num_bos: int
    """Number of BOS tokens to add to the prompt."""

    num_eos: int
    """Number of EOS tokens to add to the prompt."""

    def __init__(
        self,
        *,
        echo: bool = True,
        max_new_tokens: int = -1,
        warming: bool = False,
        seq_len: int = -1,
        temperature: float = 0.8,
        num_bos: int = 0,
        num_eos: int = 0,
    ) -> None:
        """Initialize GenerationConfig with optional keyword arguments for all fields."""
        ...

    def resolve_max_new_tokens(
        self, max_context_len: int, num_prompt_tokens: int
    ) -> int:
        """
        Resolve the maximum number of new tokens to generate based on constraints.

        Args:
            max_context_len: The maximum context length supported by the model
            num_prompt_tokens: The number of tokens in the input prompt

        Returns:
            The resolved maximum number of new tokens to generate
        """
        ...

    def __repr__(self) -> str: ...

class Stats:
    """Statistics for LLM generation performance."""

    SCALING_FACTOR_UNITS_PER_SECOND: int
    """Scaling factor for timestamps (1000 for milliseconds)."""

    model_load_start_ms: int
    """Start time of model loading in milliseconds."""

    model_load_end_ms: int
    """End time of model loading in milliseconds."""

    inference_start_ms: int
    """Start time of inference in milliseconds."""

    token_encode_end_ms: int
    """End time of tokenizer encoding in milliseconds."""

    model_execution_start_ms: int
    """Start time of model execution in milliseconds."""

    model_execution_end_ms: int
    """End time of model execution in milliseconds."""

    prompt_eval_end_ms: int
    """End time of prompt evaluation in milliseconds."""

    first_token_ms: int
    """Timestamp when the first generated token is emitted."""

    inference_end_ms: int
    """End time of inference/generation in milliseconds."""

    aggregate_sampling_time_ms: int
    """Total time spent in sampling across all tokens."""

    num_prompt_tokens: int
    """Number of tokens in the input prompt."""

    num_generated_tokens: int
    """Number of tokens generated."""

    def on_sampling_begin(self) -> None:
        """Mark the beginning of a sampling operation."""
        ...

    def on_sampling_end(self) -> None:
        """Mark the end of a sampling operation."""
        ...

    def reset(self, all_stats: bool = False) -> None:
        """
        Reset statistics.

        Args:
            all_stats: If True, reset all stats including model load times.
                      If False, preserve model load times.
        """
        ...

    def to_json_string(self) -> str:
        """Convert stats to JSON string representation."""
        ...

    def __repr__(self) -> str: ...

class Image:
    """Container for image data."""

    @overload
    def __init__(self) -> None:
        """Initialize an empty Image."""
        ...

    @overload
    def __init__(self, data: List[int], width: int, height: int, channels: int) -> None:
        """Initialize an Image with uint8 data."""
        ...

    @overload
    def __init__(
        self, data: List[float], width: int, height: int, channels: int
    ) -> None:
        """Initialize an Image with float data."""
        ...

    def is_uint8(self) -> bool:
        """Check if image data is uint8 format."""
        ...

    def is_float(self) -> bool:
        """Check if image data is float format."""
        ...

    @property
    def width(self) -> int:
        """Image width in pixels."""
        ...

    @property
    def height(self) -> int:
        """Image height in pixels."""
        ...

    @property
    def channels(self) -> int:
        """Number of color channels (3 for RGB, 4 for RGBA)."""
        ...

    @property
    def uint8_data(self) -> List[int]:
        """Raw image data as uint8 values."""
        ...

    @property
    def float_data(self) -> List[float]:
        """Raw image data as float values."""
        ...

    def __repr__(self) -> str: ...

class Audio:
    """Container for preprocessed audio data."""

    data: List[int]
    """Raw audio data as a list of uint8 values."""

    batch_size: int
    """Batch size of the audio data."""

    n_bins: int
    """Number of frequency bins (for spectrograms)."""

    n_frames: int
    """Number of time frames."""

    @overload
    def __init__(self) -> None:
        """Initialize an empty Audio."""
        ...

    @overload
    def __init__(
        self, data: List[int], batch_size: int, n_bins: int, n_frames: int
    ) -> None:
        """Initialize Audio with preprocessed data."""
        ...

    def __repr__(self) -> str: ...

class RawAudio:
    """Container for raw audio data."""

    data: List[int]
    """Raw audio data as a list of uint8 values."""

    batch_size: int
    """Batch size of the audio data."""

    n_channels: int
    """Number of audio channels (1 for mono, 2 for stereo)."""

    n_samples: int
    """Number of audio samples."""

    @overload
    def __init__(self) -> None:
        """Initialize an empty RawAudio."""
        ...

    @overload
    def __init__(
        self, data: List[int], batch_size: int, n_channels: int, n_samples: int
    ) -> None:
        """Initialize RawAudio with raw data."""
        ...

    def __repr__(self) -> str: ...

class MultimodalInput:
    """Container for multimodal input data (text, image, audio, etc.)."""

    @overload
    def __init__(self, text: str) -> None:
        """
        Create a MultimodalInput with text.

        Args:
            text: The input text string
        """
        ...

    @overload
    def __init__(self, image: Image) -> None:
        """
        Create a MultimodalInput with an image.

        Args:
            image: The input image
        """
        ...

    @overload
    def __init__(self, audio: Audio) -> None:
        """
        Create a MultimodalInput with preprocessed audio.

        Args:
            audio: The input audio data
        """
        ...

    @overload
    def __init__(self, raw_audio: RawAudio) -> None:
        """
        Create a MultimodalInput with raw audio.

        Args:
            raw_audio: The input raw audio data
        """
        ...

    def is_text(self) -> bool:
        """Check if this input contains text."""
        ...

    def is_image(self) -> bool:
        """Check if this input contains an image."""
        ...

    def is_audio(self) -> bool:
        """Check if this input contains preprocessed audio."""
        ...

    def is_raw_audio(self) -> bool:
        """Check if this input contains raw audio."""
        ...

    def get_text(self) -> Optional[str]:
        """
        Get the text content if this is a text input.

        Returns:
            The text string if this is a text input, None otherwise
        """
        ...

    def get_image(self) -> Optional[Image]:
        """
        Get the image content if this is an image input.

        Returns:
            The Image object if this is an image input, None otherwise
        """
        ...

    def get_audio(self) -> Optional[Audio]:
        """
        Get the audio content if this is an audio input.

        Returns:
            The Audio object if this is an audio input, None otherwise
        """
        ...

    def get_raw_audio(self) -> Optional[RawAudio]:
        """
        Get the raw audio content if this is a raw audio input.

        Returns:
            The RawAudio object if this is a raw audio input, None otherwise
        """
        ...

    def __repr__(self) -> str: ...

class MultimodalRunner:
    """Runner for multimodal language models."""

    def __init__(
        self, model_path: str, tokenizer_path: str, data_path: Optional[str] = None
    ) -> None:
        """
        Initialize a MultimodalRunner.

        Args:
            model_path: Path to the model file (.pte)
            tokenizer_path: Path to the tokenizer file
            data_path: Optional path to additional data file
        Raises:
            RuntimeError: If initialization fails
        """
        ...

    def generate(
        self,
        inputs: List[MultimodalInput],
        config: GenerationConfig,
        token_callback: Optional[Callable[[str], None]] = None,
        stats_callback: Optional[Callable[[Stats], None]] = None,
    ) -> None:
        """
        Generate text from multimodal inputs.

        Args:
            inputs: List of multimodal inputs (text, images, etc.)
            config: Generation configuration
            token_callback: Optional callback called for each generated token
            stats_callback: Optional callback called with generation statistics

        Raises:
            RuntimeError: If generation fails
        """
        ...

    def generate_hf(
        self,
        inputs: dict,
        config: GenerationConfig,
        token_callback: Optional[Callable[[str], None]] = None,
        stats_callback: Optional[Callable[[Stats], None]] = None,
        image_token_id: Optional[int] = None,
    ) -> None:
        """
        Generate text directly from a HuggingFace processor dict.

        Expects at least 'input_ids' (torch.Tensor). If 'pixel_values' is provided,
        an 'image_token_id' (or 'image_token_index') must also be present to locate
        the image position(s) in input_ids.

        Args:
            inputs: HF processor outputs (e.g., from AutoProcessor.apply_chat_template)
            config: Generation configuration
            token_callback: Optional per-token callback
            stats_callback: Optional stats callback
            image_token_id: Optional image token ID (or index)

        Raises:
            RuntimeError: If required keys are missing, shapes are invalid, or generation fails
        """
        ...

    def prefill(self, inputs: List[MultimodalInput]) -> None:
        """
        Prefill multimodal inputs (e.g., to rebuild KV cache from chat history)
        without generating tokens.

        Args:
            inputs: List of multimodal inputs to prefill

        Raises:
            RuntimeError: If prefill fails
        """
        ...

    def generate_text(
        self, inputs: List[MultimodalInput], config: GenerationConfig
    ) -> str:
        """
        Generate text and return the complete result as a string.

        Args:
            inputs: List of multimodal inputs (text, images, etc.)
            config: Generation configuration

        Returns:
            The generated text as a string

        Raises:
            RuntimeError: If generation fails
        """
        ...

    def generate_text_hf(
        self, inputs: dict, config: GenerationConfig, image_token_id
    ) -> str:
        """
        Generate text directly from a HuggingFace processor dict and return as string.

        See generate_hf(inputs: dict, ...) for expected keys and constraints.
        """
        ...

    def stop(self) -> None:
        """Stop the current generation process."""
        ...

    def reset(self) -> None:
        """Reset the runner state and KV cache."""
        ...

    def get_vocab_size(self) -> int:
        """
        Get the vocabulary size of the model.

        Returns:
            The vocabulary size, or -1 if not available
        """
        ...

    def __repr__(self) -> str: ...

def make_text_input(text: str) -> MultimodalInput:
    """
    Create a text input for multimodal processing.

    Args:
        text: The input text string

    Returns:
        A MultimodalInput containing the text
    """
    ...

def make_image_input(image_tensor: torch.Tensor) -> MultimodalInput:
    """
    Create an image input from a torch tensor.

    Args:
        image_tensor: Torch tensor with shape (H, W, C), (1, H, W, C), (C, H, W), or (1, C, H, W)

    Returns:
        A MultimodalInput containing the image

    Raises:
        RuntimeError: If the tensor has invalid dimensions or number of channels
    """
    ...

def make_audio_input(audio_tensor: torch.Tensor) -> MultimodalInput:
    """
    Create a preprocessed audio input from a torch tensor.

    Args:
        audio_tensor: Torch tensor with shape (batch_size, n_bins, n_frames)

    Returns:
        A MultimodalInput containing the preprocessed audio

    Raises:
        RuntimeError: If the tensor has invalid dimensions or dtype
    """
    ...

def make_raw_audio_input(audio_tensor: torch.Tensor) -> MultimodalInput:
    """
    Create a raw audio input from a torch tensor.

    Args:
        audio_tensor: Torch tensor with shape (batch_size, n_channels, n_samples)

    Returns:
        A MultimodalInput containing the raw audio

    Raises:
        RuntimeError: If the tensor has invalid dimensions or dtype
    """
    ...
