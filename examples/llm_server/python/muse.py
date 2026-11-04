# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Launch the installed Muse CLI with the serving model's actual context limit.

python -m executorch.examples.llm_server.python.muse --base-url http://127.0.0.1:8000/v1
Append ``exec --json ...`` for a noninteractive run. No request proxy is used.
"""

import argparse
import json
import os
import shutil
import urllib.request


def command(muse_bin, base_url, model, extra):
    if model.get("api_dialect") != "openrouter":
        raise ValueError("Start the serving endpoint with --api-dialect openrouter")
    context = model.get("context_length")
    if type(context) is not int or context <= 1:
        raise ValueError(
            "Server model metadata must include a context_length greater than one"
        )
    mode = []
    extra = list(extra)
    if extra[:1] == ["--"]:
        extra.pop(0)
    if extra[:1] == ["exec"]:
        mode, extra = extra[:1], extra[1:]
    return [
        muse_bin,
        *mode,
        "--provider",
        "openrouter",
        "--base-url",
        base_url,
        "--model",
        model["id"],
        "-c",
        f"context_compaction.provider_context_limit_tokens={context}",
        *extra,
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model")
    parser.add_argument(
        "--muse-bin", default=os.environ.get("MUSE_BIN") or shutil.which("muse")
    )
    args, extra = parser.parse_known_args()
    if not args.muse_bin:
        parser.error("Install Muse or supply --muse-bin")
    base_url = args.base_url.rstrip("/")
    try:
        with urllib.request.urlopen(base_url + "/models", timeout=10) as response:
            models = json.load(response)["data"]
        if not isinstance(models, list) or any(
            not isinstance(model, dict) or not isinstance(model.get("id"), str)
            for model in models
        ):
            raise ValueError("invalid model list")
        models = [m for m in models if args.model is None or m["id"] == args.model]
        if len(models) != 1:
            raise ValueError(
                f"Model {args.model!r} is not served"
                if args.model and not models
                else "Select a served model with --model"
            )
        argv = command(args.muse_bin, base_url, models[0], extra)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"Cannot configure Muse from {base_url}: {error}")
    os.environ["OPENROUTER_API_KEY"] = "local"
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    main()
