# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
from typing import Any, List, Tuple
from urllib.parse import urlparse

from executorch.extension.llm.dns_aid.config import DnsAidResolverConfig


class DnsAidResolver:
    """Resolves cloud inference endpoints via DNS-AID SVCB lookups.

    Client-only: ExecuTorch edge agents use this to discover cloud services
    (vLLM, Ray Serve, remote MCP servers) without hard-coding URLs.
    """

    def __init__(self, config: DnsAidResolverConfig) -> None:
        try:
            import dns_aid

            self._dns_aid = dns_aid
        except ImportError:
            raise ImportError(
                "dns-aid is required for DNS-AID endpoint resolution. "
                "Install it with: pip install executorch[dns-aid]"
            )
        self.config = config

    async def resolve(self) -> Tuple[str, int]:
        """Resolve the configured agent name to a (host, port) tuple.

        Queries DNS-AID SVCB records, filters by optional capability
        requirements, and returns the best match. Falls back to
        fallback_url if no records match.

        Raises:
            RuntimeError: If no matching endpoint is found and no fallback
                is configured.
        """
        result = await self._dns_aid.discover(
            domain=self.config.domain,
            protocol=self.config.protocol,
            name=self.config.agent_name,
        )

        agents = result.agents
        if self.config.min_context_len is not None:
            agents = self._filter_by_min_context_len(agents)

        if agents:
            agent = agents[0]
            return agent.target_host, agent.port

        if self.config.fallback_url:
            parsed = urlparse(self.config.fallback_url)
            host = parsed.hostname or "localhost"
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            return host, port

        raise RuntimeError(
            f"No DNS-AID records found for agent '{self.config.agent_name}' "
            f"at domain '{self.config.domain}' and no fallback_url configured."
        )

    def resolve_sync(self) -> Tuple[str, int]:
        """Synchronous wrapper around resolve().

        Creates a new event loop if none is running. If called from within
        an async context, use ``await resolve()`` instead.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            raise RuntimeError(
                "resolve_sync() cannot be called from within a running event loop. "
                "Use 'await resolver.resolve()' instead."
            )
        return asyncio.run(self.resolve())

    def _filter_by_min_context_len(self, agents: List[Any]) -> List[Any]:
        """Filter agents whose capabilities include a context_len >= threshold.

        Expects capabilities entries formatted as "context_len:<int>".
        Agents without a matching entry are excluded.
        """
        min_ctx = self.config.min_context_len
        if min_ctx is None:
            return agents
        filtered = []
        for agent in agents:
            for cap in agent.capabilities:
                if cap.startswith("context_len:"):
                    try:
                        val = int(cap.split(":", 1)[1])
                    except ValueError:
                        continue
                    if val >= min_ctx:
                        filtered.append(agent)
                        break
        return filtered
