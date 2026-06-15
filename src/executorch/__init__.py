# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ExecuTorch: A unified ML stack for on-device inference."""

try:
    # ``version.py`` is generated at build time from ``version.txt`` (the single
    # source of truth) by ``setup.py``, mirroring ``torch/version.py``.
    from executorch.version import __version__, git_version  # noqa: F401
except ImportError:
    # Source tree / namespace import without a build: fall back to the installed
    # distribution metadata, then to a sentinel, so ``import executorch`` never
    # fails just because the version is unknown.
    try:
        from importlib.metadata import PackageNotFoundError, version as _version

        try:
            __version__ = _version("executorch")
        except PackageNotFoundError:
            __version__ = "0.0.0+unknown"
    except Exception:
        __version__ = "0.0.0+unknown"
    git_version = None

__all__ = ["__version__", "git_version"]
