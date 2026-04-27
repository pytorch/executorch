# Copyright (c) 2026 iote.ai
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Pytest configuration and shared fixtures for the Nordic AXON backend tests."""

from __future__ import annotations

import os

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_sdk: test requires Nordic sdk-edge-ai"
    )
    config.addinivalue_line(
        "markers", "requires_hardware: test requires nRF54LM20DK hardware"
    )


@pytest.fixture
def sdk_edge_ai_path() -> str | None:
    """Path to Nordic sdk-edge-ai, or None if not available."""
    path = os.environ.get("SDK_EDGE_AI_PATH", "")
    if path and os.path.isdir(path):
        return path
    return None


@pytest.fixture
def require_sdk(sdk_edge_ai_path):
    """Skip the test if Nordic SDK is not available."""
    if sdk_edge_ai_path is None:
        pytest.skip(
            "Nordic sdk-edge-ai not found. Set SDK_EDGE_AI_PATH to enable."
        )
    return sdk_edge_ai_path
