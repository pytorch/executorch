#!/usr/bin/env python3
"""Render a Gemma4 chat prompt using the official vLLM jinja template.

Outputs the fully-rendered string to stdout, ready to pipe into the
gemma4_runner via `--prompt "$(...)"`. Supports system prompts, tool
specifications, and reasoning ("thinking") mode.

Examples:

    # Plain user message
    python render_chat.py --user "What is the capital of France?"

    # With system prompt
    python render_chat.py \\
        --system "You are a helpful assistant." \\
        --user "What is the capital of France?"

    # Reasoning mode
    python render_chat.py --user "Solve x^2=4" --enable-thinking

    # Tools (JSON file with OpenAI-style function specs)
    python render_chat.py --user "What's the weather?" --tools tools.json

The rendered prompt already contains BOS and the model-turn opener; pass it
straight to the runner without adding any prefixes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------

DEFAULT_TEMPLATE = Path(__file__).resolve().parent / "chat_template.jinja"


def render(
    user: str,
    system: str | None = None,
    tools: list | None = None,
    enable_thinking: bool = False,
    bos_token: str = "<bos>",
    template_path: Path = DEFAULT_TEMPLATE,
) -> str:
    """Render the Gemma4 chat template into a single prompt string.

    Uses jinja2 directly (no transformers dependency required) so the helper
    can be invoked from any environment that just has Python + jinja2.
    """
    from jinja2 import Environment

    template_src = template_path.read_text()
    env = Environment()
    template = env.from_string(template_src)

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    return template.render(
        messages=messages,
        tools=tools,
        bos_token=bos_token,
        enable_thinking=enable_thinking,
        add_generation_prompt=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True, help="User message content.")
    parser.add_argument("--system", default=None, help="Optional system prompt.")
    parser.add_argument(
        "--tools",
        default=None,
        help="Path to a JSON file with OpenAI-style tool/function specs.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable reasoning mode (adds <|think|> to system turn).",
    )
    parser.add_argument(
        "--template",
        default=str(DEFAULT_TEMPLATE),
        help="Path to the jinja chat template (default: bundled).",
    )
    parser.add_argument(
        "--bos-token",
        default="<bos>",
        help="Token string to prepend (the runner's tokenizer also auto-adds BOS).",
    )
    args = parser.parse_args()

    tools = json.loads(Path(args.tools).read_text()) if args.tools else None
    out = render(
        user=args.user,
        system=args.system,
        tools=tools,
        enable_thinking=args.enable_thinking,
        bos_token=args.bos_token,
        template_path=Path(args.template),
    )
    sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
