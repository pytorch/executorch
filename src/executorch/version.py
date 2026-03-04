"""
Version information for the ExecuTorch package.

This module exposes the public package version as well as the
corresponding Git commit hash used to build the release.
"""

from typing import Optional

__all__ = ["__version__", "git_version"]

#: Human-readable package version (PEP 440 compatible).
__version__: str = "1.2.0a0+a20cb0d"

#: Git commit hash associated with this build, if available.
git_version: Optional[str] = "a20cb0dbb6a41b239e24dea0142b657eaf913f5f"
