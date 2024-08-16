import os
import sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
import warnings

from models.llm_models.configuration_base import BaseConfig


# flake8: noqa: E721


def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return f"{category.__name__}: {message}\n"


warnings.formatwarning = warning_on_one_line


def check_all_chunks_same_num_layer(num_blocks_per_chunk):
    for i in range(1, len(num_blocks_per_chunk)):
        if num_blocks_per_chunk[i] != num_blocks_per_chunk[0]:
            print("num_blocks_per_chunk:", num_blocks_per_chunk)
            raise RuntimeError(
                "This version of the sdk doesn't support different number of "
                "decoder layers per chunk, as shape fixer stage will fail. If you require this support,"
                " please contact Mediatek sdk owner."
            )


def check_between_exclusive(num, min_, max_, message=None):
    if not (type(num) == type(min_) == type(max_)):
        raise TypeError(
            f"Got different types for num ({type(num)}), min ({type(min_)}), and max ({type(max_)})"
        )
    if not (min_ < num < max_):
        if message is None:
            raise ValueError(
                f"Expected number between {min_} and {max_} exclusive, but got: {num}"
            )
        else:
            raise ValueError(
                f"{message} must be between {min_} and {max_} exclusive, but got: {num}"
            )


def check_between_inclusive(num, min_, max_, message=None):
    if not (type(num) == type(min_) == type(max_)):
        raise TypeError(
            f"Got different types for num ({type(num)}), min ({type(min_)}), and max ({type(max_)})"
        )
    if not (min_ <= num <= max_):
        if message is None:
            raise ValueError(
                f"Expected number between {min_} and {max_} inclusive, but got: {num}"
            )
        else:
            raise ValueError(
                f"{message} must be between {min_} and {max_} inclusive, but got: {num}"
            )


def check_exist(file_or_folder, message=None):
    if not os.path.exists(file_or_folder):
        if message is None:
            raise FileNotFoundError(f"{file_or_folder} does not exist.")
        else:
            raise FileNotFoundError(f"{message} does not exist: {file_or_folder}")


def check_ext(file, ext, message=None):
    if not file.endswith(ext):
        if message is None:
            raise RuntimeError(f"Expected {ext} file, but got: {file}")
        else:
            raise RuntimeError(f"Expected {ext} file for {message}, but got: {file}")


def check_isdir(folder, message=None):
    if not os.path.isdir(folder):
        if message is None:
            raise FileNotFoundError(f"{folder} is not a directory.")
        else:
            raise RuntimeError(f"Expected directory for {message}, but got: {folder}")


def check_old_arg(path):
    if os.path.isdir(path):
        raise RuntimeError(
            "This package's main usage has changed starting from v0.8.0. Please use"
            " model's config.json as main argument instead of weight directory."
        )


def check_shapes(shapes):
    if not isinstance(shapes, list):
        raise TypeError(f"Expected shapes to be a list, but got {type(shapes)} instead")
    for shape in shapes:
        if shape.count("t") != 1 or shape.count("c") != 1:
            raise RuntimeError(
                f"Shape {shape} is in the wrong format. Every shape needs to be of"
                "the format: xtyc where x and y are integers. (e.g. 32t512c)"
            )
        try:
            _ = int(shape.split("t")[0])
        except ValueError:
            raise RuntimeError(
                f"Shape {shape} is in the wrong format. Every shape needs to be of"
                "the format: xtyc where x and y are integers. (e.g. 32t512c)"
            )

        try:
            _ = int(shape.split("t")[1].split("c")[0])
        except ValueError:
            raise RuntimeError(
                f"Shape {shape} is in the wrong format. Every shape needs to be of"
                "the format: xtyc where x and y are integers. (e.g. 32t512c)"
            )


def check_supported_model(config):
    SUPPORTED_MODELS = [
        "llama",
        "bloom",
        "baichuan",
        "qwen",
        "qwen1.5",
        "qwen2",
        "milm",
    ]
    if not isinstance(config, BaseConfig):
        raise RuntimeError(
            f"Unsupported config class: {type(config)}. "
            "config needs to be subclassed from BaseConfig"
        )

    if config.model_type not in SUPPORTED_MODELS:
        raise RuntimeError(
            f"Unsupported model: {config.model_type}. Supported models: "
            f"{SUPPORTED_MODELS}"
        )


def check_supported_tokenizer(config):
    SUPPORTED_TOKENIZERS = [
        "default",
        "bloom",
        "baichuan",
        "gpt2",
        "gpt2_fast",
        "qwen",
        "qwen2",
        "qwen2_fast",
        "llama",
        "pretrained_fast",
    ]
    if not isinstance(config, BaseConfig):
        raise RuntimeError(
            f"Unsupported config class: {type(config)}. "
            "config needs to be subclassed from BaseConfig"
        )

    if config.tokenizer not in SUPPORTED_TOKENIZERS:
        raise RuntimeError(
            f"Unsupported tokenizer: {config.tokenizer}. Supported tokenizers: "
            f"{SUPPORTED_TOKENIZERS}"
        )


def check_tokenizer_exist(folder):
    model = config = False
    for f in os.listdir(folder):
        if f == "tokenizer.model" or f == "tokenizer.json" or f.endswith(".tiktoken"):
            model = True
        if f == "tokenizer_config.json":
            config = True
    if not model:
        raise FileNotFoundError(
            f"Tokenizer not found in {folder}. Expected tokenizer.model, "
            "tokenizer.json, or tokenizer.tiktoken"
        )
    if not config:
        raise FileNotFoundError(
            f"Tokenizer config not found in {folder}. Expected " "tokenizer_config.json"
        )


def check_weights_exist(weight_dir):
    if (
        len(
            [
                f
                for f in os.listdir(weight_dir)
                if (
                    (f.startswith("pytorch_model") and f.endswith(".bin"))
                    or (f.startswith("model") and f.endswith(".safetensors"))
                )
            ]
        )
        == 0
    ):
        raise FileNotFoundError(
            f"No weight files found in {weight_dir}! Weight files should be either .bin or .safetensors file types."
        )
    safetensors_l = [f for f in os.listdir(weight_dir) if f.endswith(".safetensors")]
    bin_l = [f for f in os.listdir(weight_dir) if f.endswith(".bin")]
    if len(safetensors_l) & len(bin_l):
        raise RuntimeError(
            "Weights should only be in either .bin or .safetensors format, not both."
        )
