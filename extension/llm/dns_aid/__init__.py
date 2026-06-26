# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DNS-AID client resolver for ExecuTorch cloud endpoint discovery.

Enables edge agents to resolve cloud inference endpoints (vLLM, Ray Serve,
remote MCP servers) via DNS-AID SVCB lookups rather than hard-coded URLs.

Install the optional dependency: pip install executorch[dns-aid]
"""

from executorch.extension.llm.dns_aid.config import DnsAidResolverConfig
from executorch.extension.llm.dns_aid.resolver import DnsAidResolver

__all__ = ["DnsAidResolver", "DnsAidResolverConfig"]
