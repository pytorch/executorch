"""
Type stubs for _llm_runner module.

This file provides type annotations for the ExecuTorch LLM Runner Python bindings.
"""

from typing import List, Optional, Callable, Union
import numpy as np
from numpy.typing import NDArray

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
    
    def __init__(self) -> None:
        """Initialize GenerationConfig with default values."""
        ...
    
    def resolve_max_new_tokens(self, max_context_len: int, num_prompt_tokens: int) -> int:
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
    
    data: List[int]
    """Raw image data as a list of uint8 values."""
    
    width: int
    """Image width in pixels."""
    
    height: int
    """Image height in pixels."""
    
    channels: int
    """Number of color channels (3 for RGB, 4 for RGBA)."""
    
    def __init__(self) -> None:
        """Initialize an empty Image."""
        ...
    
    def __repr__(self) -> str: ...


class MultimodalInput:
    """Container for multimodal input data (text, image, etc.)."""
    
    def __init__(self, text: str) -> None:
        """
        Create a MultimodalInput with text.
        
        Args:
            text: The input text string
        """
        ...
    
    def __init__(self, image: Image) -> None:
        """
        Create a MultimodalInput with an image.
        
        Args:
            image: The input image
        """
        ...
    
    def is_text(self) -> bool:
        """Check if this input contains text."""
        ...
    
    def is_image(self) -> bool:
        """Check if this input contains an image."""
        ...
    
    def get_text(self) -> Optional[str]:
        """
        Get the text content if this is a text input.
        
        Returns:
            The text string if this is a text input, None otherwise
        """
        ...
    
    def __repr__(self) -> str: ...


class MultimodalRunner:
    """Runner for multimodal language models."""
    
    def __init__(
        self, 
        model_path: str, 
        tokenizer_path: str, 
        data_path: Optional[str] = None
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
        stats_callback: Optional[Callable[[Stats], None]] = None
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
    
    def generate_text(
        self,
        inputs: List[MultimodalInput],
        config: GenerationConfig
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


def make_image_input(image_array: NDArray[np.uint8]) -> MultimodalInput:
    """
    Create an image input from a numpy array.
    
    Args:
        image_array: Numpy array with shape (H, W, C) where C is 3 (RGB) or 4 (RGBA)
        
    Returns:
        A MultimodalInput containing the image
        
    Raises:
        RuntimeError: If the array has invalid dimensions or number of channels
    """
    ...