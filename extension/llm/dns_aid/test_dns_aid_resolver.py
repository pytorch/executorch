# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import importlib
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from executorch.extension.llm.dns_aid.config import DnsAidResolverConfig


def _make_agent(
    target_host="vllm.example.com",
    port=443,
    capabilities=None,
):
    agent = MagicMock()
    agent.target_host = target_host
    agent.port = port
    agent.capabilities = capabilities or []
    agent.endpoint_url = f"https://{target_host}:{port}"
    return agent


def _make_discovery_result(agents=None):
    result = MagicMock()
    result.agents = agents or []
    return result


@patch.dict("sys.modules", {"dns_aid": MagicMock()})
class TestDnsAidResolver(unittest.TestCase):
    def _config(self, **overrides):
        defaults = {
            "agent_name": "vllm-llama3-70b",
            "domain": "example.internal",
        }
        defaults.update(overrides)
        return DnsAidResolverConfig(**defaults)

    def _resolver(self, **config_overrides):
        from executorch.extension.llm.dns_aid.resolver import DnsAidResolver

        return DnsAidResolver(self._config(**config_overrides))

    def test_resolve_returns_host_port(self):
        resolver = self._resolver()
        agent = _make_agent("vllm.example.com", 8080)
        resolver._dns_aid.discover = AsyncMock(
            return_value=_make_discovery_result([agent])
        )

        host, port = asyncio.run(resolver.resolve())
        self.assertEqual(host, "vllm.example.com")
        self.assertEqual(port, 8080)

    def test_resolve_sync(self):
        resolver = self._resolver()
        agent = _make_agent("sync.example.com", 443)
        resolver._dns_aid.discover = AsyncMock(
            return_value=_make_discovery_result([agent])
        )

        host, port = resolver.resolve_sync()
        self.assertEqual(host, "sync.example.com")
        self.assertEqual(port, 443)

    def test_resolve_sync_raises_inside_event_loop(self):
        resolver = self._resolver()

        async def _inner():
            resolver.resolve_sync()

        with self.assertRaises(RuntimeError) as ctx:
            asyncio.run(_inner())
        self.assertIn(
            "cannot be called from within a running event loop", str(ctx.exception)
        )

    def test_fallback_url_used_when_no_records(self):
        resolver = self._resolver(fallback_url="https://fallback.example.com:9090/v1")
        resolver._dns_aid.discover = AsyncMock(return_value=_make_discovery_result([]))

        host, port = asyncio.run(resolver.resolve())
        self.assertEqual(host, "fallback.example.com")
        self.assertEqual(port, 9090)

    def test_error_when_no_records_and_no_fallback(self):
        resolver = self._resolver()
        resolver._dns_aid.discover = AsyncMock(return_value=_make_discovery_result([]))

        with self.assertRaises(RuntimeError) as ctx:
            asyncio.run(resolver.resolve())
        self.assertIn("No DNS-AID records found", str(ctx.exception))

    def test_filter_by_min_context_len(self):
        resolver = self._resolver(min_context_len=32768)

        small = _make_agent("small.example.com", 443, capabilities=["context_len:8192"])
        large = _make_agent(
            "large.example.com", 443, capabilities=["context_len:65536"]
        )

        filtered = resolver._filter_by_min_context_len([small, large])
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].target_host, "large.example.com")

    def test_filter_by_min_context_len_noop_when_none(self):
        resolver = self._resolver(min_context_len=None)
        agents = [_make_agent(), _make_agent()]

        filtered = resolver._filter_by_min_context_len(agents)
        self.assertEqual(len(filtered), 2)

    def test_resolve_applies_capability_filter(self):
        resolver = self._resolver(min_context_len=32768)

        small = _make_agent("small.example.com", 443, capabilities=["context_len:8192"])
        large = _make_agent(
            "large.example.com", 8080, capabilities=["context_len:65536"]
        )
        resolver._dns_aid.discover = AsyncMock(
            return_value=_make_discovery_result([small, large])
        )

        host, port = asyncio.run(resolver.resolve())
        self.assertEqual(host, "large.example.com")
        self.assertEqual(port, 8080)


class TestDnsAidResolverImportGuard(unittest.TestCase):
    def test_import_error_without_dns_aid(self):
        original = sys.modules.get("dns_aid")
        sys.modules["dns_aid"] = None
        try:
            import executorch.extension.llm.dns_aid.resolver as resolver_mod

            importlib.reload(resolver_mod)
            with self.assertRaises(ImportError) as ctx:
                resolver_mod.DnsAidResolver(
                    DnsAidResolverConfig(agent_name="test", domain="example.com")
                )
            self.assertIn("dns-aid is required", str(ctx.exception))
        finally:
            if original is not None:
                sys.modules["dns_aid"] = original
            else:
                sys.modules.pop("dns_aid", None)


if __name__ == "__main__":
    unittest.main()
