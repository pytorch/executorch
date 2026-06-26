# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional


@dataclass
class DnsAidResolverConfig:
    """Configuration for DNS-AID based cloud endpoint resolution.

    ExecuTorch edge agents use this to resolve cloud inference endpoints
    (vLLM, Ray Serve, remote MCP servers) via DNS-AID SVCB lookups.
    """

    agent_name: str
    domain: str
    protocol: str = "mcp"
    min_context_len: Optional[int] = None
    fallback_url: Optional[str] = None
