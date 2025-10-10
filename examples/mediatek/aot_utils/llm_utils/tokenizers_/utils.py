# Copyright (C) 2024 MediaTek Inc. All rights reserved.
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenizer utilities."""

# flake8: noqa: C901

import functools
import os
import sys
import types
from collections import UserDict
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Union
from uuid import uuid4

import huggingface_hub
import numpy as np
from huggingface_hub import _CACHED_NO_EXIST, hf_hub_download, try_to_load_from_cache

get_torch_version = False
is_flax_available = False
is_mlx_available = False
is_tf_available = False
is_torch_available = True
is_torch_fx_proxy = False
is_offline_mode = True
is_protobuf_available = False


def is_remote_url(_):
    """Check if the given input is a remote URL.

    Args:
        _ (Any): The input to check.

    Returns:
        bool: Always returns False.
    """
    return False


def add_model_info_to_auto_map(auto_map, repo_id):
    """Adds the information of the repo_id to a given auto map."""
    for key, value in auto_map.items():
        if isinstance(value, (tuple, list)):
            auto_map[key] = [
                f"{repo_id}--{v}" if (v is not None and "--" not in v) else v
                for v in value
            ]
        elif value is not None and "--" not in value:
            auto_map[key] = f"{repo_id}--{value}"

    return auto_map


def add_model_info_to_custom_pipelines(custom_pipeline, repo_id):
    """Adds the information of the repo_id to a given custom pipeline."""
    # {custom_pipelines : {task: {"impl": "path.to.task"},...} }
    for task in custom_pipeline:
        if "impl" in custom_pipeline[task]:
            module = custom_pipeline[task]["impl"]
            if "--" not in module:
                custom_pipeline[task]["impl"] = f"{repo_id}--{module}"
    return custom_pipeline


def infer_framework_from_repr(x):
    """Guess the framework from repr.

    Tries to guess the framework of an object `x` from its repr (brittle but will help in `is_tensor` to try the
    frameworks in a smart order, without the need to import the frameworks).
    """
    representation = str(type(x))
    if representation.startswith("<class 'torch."):
        return "pt"
    if representation.startswith("<class 'tensorflow."):
        return "tf"
    if representation.startswith("<class 'jax"):
        return "jax"
    if representation.startswith("<class 'numpy."):
        return "np"
    if representation.startswith("<class 'mlx."):
        return "mlx"
    raise ValueError(f"Unsupported repr {x}")


def _get_frameworks_and_test_func(x):
    """Get frameworks and test function.

    Returns an (ordered since we are in Python 3.7+) dictionary framework to test function, which places the framework
    we can guess from the repr first, then Numpy, then the others.
    """
    framework_to_test = {
        "pt": is_torch_tensor,
        "tf": is_tf_tensor,
        "jax": is_jax_tensor,
        "np": is_numpy_array,
        "mlx": is_mlx_array,
    }
    preferred_framework = infer_framework_from_repr(x)
    # We will test this one first, then numpy, then the others.
    frameworks = [] if preferred_framework is None else [preferred_framework]
    if preferred_framework != "np":
        frameworks.append("np")
    frameworks.extend(
        [f for f in framework_to_test if f not in [preferred_framework, "np"]]
    )
    return {f: framework_to_test[f] for f in frameworks}


def is_tensor(x):
    """Verify whether input is a Tensor.

    Tests if `x` is a `torch.Tensor`, `tf.Tensor`, `jaxlib.xla_extension.DeviceArray`, `np.ndarray` or `mlx.array`
    in the order defined by `infer_framework_from_repr`
    """
    # This gives us a smart order to test the frameworks with the corresponding tests.
    framework_to_test_func = _get_frameworks_and_test_func(x)
    for test_func in framework_to_test_func.values():
        if test_func(x):
            return True

    if is_flax_available:
        from jax.core import Tracer

        if isinstance(x, Tracer):
            return True

    return False


def _is_numpy(x):
    return isinstance(x, np.ndarray)


def is_numpy_array(x):
    """Tests if `x` is a numpy array or not."""
    return _is_numpy(x)


def _is_torch(x):
    import torch

    return isinstance(x, torch.Tensor)


def is_torch_tensor(x):
    """Tests if `x` is a torch tensor or not. Safe to call even if torch is not installed."""
    return False if not is_torch_available else _is_torch(x)


def _is_torch_device(x):
    import torch

    return isinstance(x, torch.device)


def is_torch_device(x):
    """Tests if `x` is a torch device or not. Safe to call even if torch is not installed."""
    return False if not is_torch_available else _is_torch_device(x)


def _is_torch_dtype(x):
    import torch

    if isinstance(x, str):
        if hasattr(torch, x):
            x = getattr(torch, x)
        else:
            return False
    return isinstance(x, torch.dtype)


def is_torch_dtype(x):
    """Tests if `x` is a torch dtype or not. Safe to call even if torch is not installed."""
    return False if not is_torch_available else _is_torch_dtype(x)


def _is_tensorflow(x):
    import tensorflow as tf

    return isinstance(x, tf.Tensor)


def is_tf_tensor(x):
    """Tests if `x` is a tensorflow tensor or not. Safe to call even if tensorflow is not installed."""
    return False if not is_tf_available else _is_tensorflow(x)


def _is_tf_symbolic_tensor(x):
    import tensorflow as tf

    # the `is_symbolic_tensor` predicate is only available starting with TF 2.14
    if hasattr(tf, "is_symbolic_tensor"):
        return tf.is_symbolic_tensor(x)
    return isinstance(x, tf.Tensor)


def is_tf_symbolic_tensor(x):
    """Tests if `x` is a tensorflow symbolic tensor or not (ie. not eager).

    Safe to call even if tensorflow is not installed.
    """
    return False if not is_tf_available else _is_tf_symbolic_tensor(x)


def _is_jax(x):
    import jax.numpy as jnp

    return isinstance(x, jnp.ndarray)


def is_jax_tensor(x):
    """Tests if `x` is a Jax tensor or not. Safe to call even if jax is not installed."""
    return False if not is_flax_available else _is_jax(x)


def _is_mlx(x):
    import mlx.core as mx

    return isinstance(x, mx.array)


def is_mlx_array(x):
    """Tests if `x` is a mlx array or not. Safe to call even when mlx is not installed."""
    return False if not is_mlx_available else _is_mlx(x)


def to_py_obj(obj):
    """Convert a TensorFlow tensor, PyTorch tensor, Numpy array or python list to a python list."""
    framework_to_py_obj = {
        "pt": lambda obj: obj.detach().cpu().tolist(),
        "tf": lambda obj: obj.numpy().tolist(),
        "jax": lambda obj: np.asarray(obj).tolist(),
        "np": lambda obj: obj.tolist(),
    }

    if isinstance(obj, (dict, UserDict)):
        return {k: to_py_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return obj

    # This gives us a smart order to test the frameworks with the corresponding tests.
    framework_to_test_func = _get_frameworks_and_test_func(obj)
    for framework, test_func in framework_to_test_func.items():
        if test_func(obj):
            return framework_to_py_obj[framework](obj)

    # tolist also works on 0d np arrays
    if isinstance(obj, np.number):
        return obj.tolist()
    return obj


def to_numpy(obj):
    """Convert a TensorFlow tensor, PyTorch tensor, Numpy array or python list to a Numpy array."""
    framework_to_numpy = {
        "pt": lambda obj: obj.detach().cpu().numpy(),
        "tf": lambda obj: obj.numpy(),
        "jax": lambda obj: np.asarray(obj),
        "np": lambda obj: obj,
    }

    if isinstance(obj, (dict, UserDict)):
        return {k: to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return np.array(obj)

    # This gives us a smart order to test the frameworks with the corresponding tests.
    framework_to_test_func = _get_frameworks_and_test_func(obj)
    for framework, test_func in framework_to_test_func.items():
        if test_func(obj):
            return framework_to_numpy[framework](obj)

    return obj


class ExplicitEnum(str, Enum):
    """Enum with more explicit error message for missing values."""

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class PaddingStrategy(ExplicitEnum):
    """Possible values for the `padding` argument in [`PreTrainedTokenizerBase.__call__`].

    Useful for tab-completion in an IDE.
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


class TensorType(ExplicitEnum):
    """Possible values for the `return_tensors` argument in [`PreTrainedTokenizerBase.__call__`].

    Useful for tab-completion in an IDE.
    """

    PYTORCH = "pt"
    TENSORFLOW = "tf"
    NUMPY = "np"
    JAX = "jax"
    MLX = "mlx"


def add_end_docstrings(*docstr):
    """Decorator to add end docstrings to a function.

    Args:
        *docstr (str): Variable length docstring arguments to add to the function's docstring.

    Returns:
        function: The decorated function with the appended docstrings.
    """

    def docstring_decorator(fn):
        fn.__doc__ = (fn.__doc__ if fn.__doc__ is not None else "") + "".join(docstr)
        return fn

    return docstring_decorator


def http_user_agent(user_agent: Union[Dict, str, None] = None) -> str:
    """Formats a user-agent string with basic info about a request."""
    ua = f"python/{sys.version.split()[0]}; session_id/{uuid4().hex}"
    if huggingface_hub.constants.HF_HUB_DISABLE_TELEMETRY:
        return ua + "; telemetry/off"
    # CI will set this value to True
    if isinstance(user_agent, dict):
        ua += "; " + "; ".join(f"{k}/{v}" for k, v in user_agent.items())
    elif isinstance(user_agent, str):
        ua += "; " + user_agent
    return ua


def cached_file(
    path_or_repo_id: Union[str, os.PathLike],
    filename: str,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: Optional[bool] = None,
    proxies: Optional[Dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    subfolder: str = "",
    repo_type: Optional[str] = None,
    user_agent: Optional[Union[str, Dict[str, str]]] = None,
    _raise_exceptions_for_gated_repo: bool = True,
    _raise_exceptions_for_missing_entries: bool = True,
    _raise_exceptions_for_connection_errors: bool = True,
    _commit_hash: Optional[str] = None,
    **deprecated_kwargs,
) -> Optional[str]:
    """Tries to locate a file in a local folder and repo, downloads and cache it if necessary.

    Args:
        path_or_repo_id (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a model repo on huggingface.co.
            - a path to a *directory* potentially containing the file.
        filename (`str`):
            The name of the file to locate in `path_or_repo`.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download:
            Deprecated and ignored. All downloads are now resumed by default when possible.
            Will be removed in v5 of Transformers.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request. @lint-ignore
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        subfolder (`str`, *optional*, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
            specify the folder name here.
        repo_type (`str`, *optional*):
            Specify the repo type (useful when downloading from a space for instance).
        user_agent (`str`): Use agent.
        deprecated_kwargs: Deprecated kwargs.

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Optional[str]`: Returns the resolved file (to the cache folder if downloaded from a repo).

    Examples:
    ```python
    # Download a model weight from the Hub and cache it.
    model_weights_file = cached_file("google-bert/bert-base-uncased", "pytorch_model.bin")
    ```
    """
    use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        if token is not None:
            raise ValueError(
                "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
            )
        token = use_auth_token

    if is_offline_mode and not local_files_only:
        # print("Offline mode: forcing local_files_only=True")
        local_files_only = True
    if subfolder is None:
        subfolder = ""

    path_or_repo_id = str(path_or_repo_id)
    full_filename = os.path.join(subfolder, filename)
    if os.path.isdir(path_or_repo_id):
        resolved_file = os.path.join(os.path.join(path_or_repo_id, subfolder), filename)
        if not os.path.isfile(resolved_file):
            if _raise_exceptions_for_missing_entries and filename not in [
                "config.json",
                f"{subfolder}/config.json",
            ]:
                raise OSError(
                    f"{path_or_repo_id} does not appear to have a file named {full_filename}. Checkout "
                    f"'https://huggingface.co/{path_or_repo_id}/tree/{revision}' for available files."
                )
            return None
        return resolved_file

    if cache_dir is None:
        cache_dir = os.getenv(
            "TRANSFORMERS_CACHE",
            os.getenv(
                "PYTORCH_TRANSFORMERS_CACHE", huggingface_hub.constants.HF_HUB_CACHE
            ),
        )
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if _commit_hash is not None and not force_download:
        # If the file is cached under that commit hash, we return it directly.
        resolved_file = try_to_load_from_cache(
            path_or_repo_id,
            full_filename,
            cache_dir=cache_dir,
            revision=_commit_hash,
            repo_type=repo_type,
        )
        if resolved_file is not None:
            if resolved_file is not _CACHED_NO_EXIST:
                return resolved_file
            if not _raise_exceptions_for_missing_entries:
                return None
            raise OSError(f"Could not locate {full_filename} inside {path_or_repo_id}.")

    user_agent = http_user_agent(user_agent)

    return hf_hub_download(
        path_or_repo_id,
        filename,
        subfolder=None if len(subfolder) == 0 else subfolder,
        repo_type=repo_type,
        revision=revision,
        cache_dir=cache_dir,
        user_agent=user_agent,
        force_download=force_download,
        proxies=proxies,
        resume_download=resume_download,
        token=token,
        local_files_only=local_files_only,
    )


def copy_func(f):
    """Returns a copy of a function f."""
    # Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)
    g = types.FunctionType(
        f.__code__,
        f.__globals__,
        name=f.__name__,
        argdefs=f.__defaults__,
        closure=f.__closure__,
    )
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g
