# DNS-AID Cloud Endpoint Resolution

ExecuTorch edge agents can resolve cloud inference endpoints at runtime via
[DNS-AID](https://dns-aid.org) SVCB lookups instead of hard-coding URLs.

This is a **client-only** integration: ExecuTorch resolves records but does not
publish them. Cloud services (vLLM, Ray Serve, remote MCP servers) publish their
own DNS-AID records.

## Installation

```bash
pip install executorch[dns-aid]
```

## Quick Start

```python
from executorch.extension.llm.dns_aid import DnsAidResolver, DnsAidResolverConfig

config = DnsAidResolverConfig(
    agent_name="vllm-llama3-70b",
    domain="inference.example.internal",
    fallback_url="https://vllm-backup.example.com:8080/v1",
)

resolver = DnsAidResolver(config)
host, port = resolver.resolve_sync()
print(f"Resolved endpoint: {host}:{port}")
```

## Configuration

`DnsAidResolverConfig` fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `agent_name` | `str` | required | Agent name to resolve (e.g., `"vllm-llama3-70b"`) |
| `domain` | `str` | required | DNS-AID domain (e.g., `"inference.example.internal"`) |
| `protocol` | `str` | `"mcp"` | Protocol filter for discovery |
| `min_context_len` | `int \| None` | `None` | Filter agents by minimum context length capability |
| `fallback_url` | `str \| None` | `None` | Static URL to use if DNS resolution fails |

## Capability Filtering

Filter endpoints by capability hints published in DNS-AID records:

```python
config = DnsAidResolverConfig(
    agent_name="vllm-llama3-70b",
    domain="inference.example.internal",
    min_context_len=32768,  # only endpoints with >= 32K context
)
```

Capability filtering matches against `context_len:<value>` entries in the
agent's `capabilities` list.

## Fallback Behavior

If no DNS-AID records are found:

1. If `fallback_url` is set, the resolver parses it and returns its host/port.
2. If no fallback is configured, a `RuntimeError` is raised.

## Async Usage

The resolver is async-native. Use `resolve()` in async contexts:

```python
import asyncio
from executorch.extension.llm.dns_aid import DnsAidResolver, DnsAidResolverConfig

async def main():
    config = DnsAidResolverConfig(
        agent_name="vllm-llama3-70b",
        domain="inference.example.internal",
    )
    resolver = DnsAidResolver(config)
    host, port = await resolver.resolve()
    print(f"Resolved: {host}:{port}")

asyncio.run(main())
```

For synchronous code, use `resolve_sync()`. Note that `resolve_sync()` cannot be
called from within a running event loop — use `await resolve()` instead.

## Without dns-aid Installed

The module is importable without dns-aid installed. `DnsAidResolverConfig` works
standalone. `DnsAidResolver` raises a helpful `ImportError` only when
instantiated without the package:

```
ImportError: dns-aid is required for DNS-AID endpoint resolution.
Install it with: pip install executorch[dns-aid]
```
