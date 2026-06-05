#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Local Zephyr SDK release proxy for `west sdk install`.

This avoids GitHub releases API rate limits by serving a tiny synthetic releases
API locally. Asset requests are served from a cache directory and populated with
`wget` on cache miss.

Examples:

  # Prepopulate CI cache only.
  .ci/scripts/zephyr_sdk_release_proxy.py \
      --version 1.0.1 \
      --cache-dir .cache/zephyr-sdk/v1.0.1 \
      --download-only

  # Serve cached assets and release metadata.
  .ci/scripts/zephyr_sdk_release_proxy.py \
      --version 1.0.1 \
      --cache-dir .cache/zephyr-sdk/v1.0.1 \
      --port 8765

  west sdk install \
      --version 1.0.1 \
      --api-url http://127.0.0.1:8765/releases \
      --gnu-toolchains arm-zephyr-eabi
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import os
import platform
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path


class AssetVerificationError(Exception):
    pass


def host_tuple() -> tuple[str, str]:
    system = platform.system()
    machine = platform.machine()

    if system != "Linux":
        raise SystemExit(f"Unsupported host OS: {system}")

    if machine in ("x86_64", "AMD64"):
        return "linux", "x86_64"
    if machine in ("aarch64", "arm64"):
        return "linux", "aarch64"

    raise SystemExit(f"Unsupported host architecture: {machine}")


def release_base_url(version: str) -> str:
    return f"https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v{version}"


def asset_names(version: str, toolchain: str) -> list[str]:
    host_os, host_arch = host_tuple()
    return [
        "sha256.sum",
        f"zephyr-sdk-{version}_{host_os}-{host_arch}_minimal.tar.xz",
        f"hosttools_{host_os}-{host_arch}.tar.xz",
        f"toolchain_gnu_{host_os}-{host_arch}_{toolchain}.tar.xz",
    ]


def run_wget(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".tmp")
    cmd = [
        "wget",
        "--continue",
        "--progress=bar:force:noscroll",
        "--output-document",
        str(partial),
        url,
    ]
    subprocess.run(cmd, check=True)
    partial.replace(output)


def ensure_asset(cache_dir: Path, version: str, filename: str) -> Path:
    path = cache_dir / filename
    if path.exists():
        return path

    url = f"{release_base_url(version)}/{filename}"
    print(f"Downloading {url}", flush=True)
    run_wget(url, path)
    return path


def parse_sha256_sum(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in path.read_text().splitlines():
        match = re.match(r"^([0-9a-fA-F]{64})\s+(.+)$", line.strip())
        if match:
            checksums[match.group(2)] = match.group(1).lower()
    return checksums


def verify_asset(cache_dir: Path, checksums: dict[str, str], filename: str) -> None:
    if filename == "sha256.sum":
        return

    expected = checksums.get(filename)
    if expected is None:
        raise AssetVerificationError(f"No checksum entry for {filename}")

    hasher = hashlib.sha256()
    with (cache_dir / filename).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()

    if digest != expected:
        raise AssetVerificationError(
            f"Checksum mismatch for {filename}: expected {expected}, got {digest}"
        )


def populate_cache(cache_dir: Path, version: str, toolchain: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    names = asset_names(version, toolchain)

    sha_file = ensure_asset(cache_dir, version, "sha256.sum")
    checksums = parse_sha256_sum(sha_file)

    for name in names:
        ensure_asset(cache_dir, version, name)
        verify_asset(cache_dir, checksums, name)


def release_json(version: str, toolchain: str, base_url: str) -> bytes:
    assets = [
        {
            "name": name,
            "browser_download_url": f"{base_url}/assets/{name}",
        }
        for name in asset_names(version, toolchain)
    ]
    return json.dumps([{"tag_name": f"v{version}", "assets": assets}]).encode()


class SdkProxyHandler(http.server.SimpleHTTPRequestHandler):
    version: str
    toolchain: str
    cache_dir: Path

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}", file=sys.stderr)

    def send_bytes(self, data: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/releases":
            query = urllib.parse.parse_qs(parsed.query)
            page = query.get("page", ["1"])[0]
            if page == "1":
                host = self.headers.get("Host", "127.0.0.1")
                data = release_json(self.version, self.toolchain, f"http://{host}")
            else:
                data = b"[]"
            self.send_bytes(data, "application/json")
            return

        if parsed.path.startswith("/assets/"):
            filename = Path(urllib.parse.unquote(parsed.path[len("/assets/") :])).name
            allowed = set(asset_names(self.version, self.toolchain))
            if filename not in allowed:
                self.send_error(404, f"Unknown asset: {filename}")
                return

            try:
                path = ensure_asset(self.cache_dir, self.version, filename)
                if filename != "sha256.sum":
                    sha_file = ensure_asset(self.cache_dir, self.version, "sha256.sum")
                    verify_asset(self.cache_dir, parse_sha256_sum(sha_file), filename)
            except (
                AssetVerificationError,
                subprocess.CalledProcessError,
                OSError,
            ) as error:
                self.send_error(500, str(error))
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            with path.open("rb") as f:
                while chunk := f.read(1024 * 1024):
                    self.wfile.write(chunk)
            return

        self.send_error(404, "Not found")


def default_cache_dir(version: str) -> Path:
    root = os.environ.get("XDG_CACHE_HOME")
    if root:
        return Path(root) / "zephyr-sdk" / f"v{version}"
    return Path.home() / ".cache" / "zephyr-sdk" / f"v{version}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=os.environ.get("SDK_VERSION", "1.0.1"))
    parser.add_argument("--toolchain", default="arm-zephyr-eabi")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--download-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir or default_cache_dir(args.version)

    if args.download_only:
        try:
            populate_cache(cache_dir, args.version, args.toolchain)
        except AssetVerificationError as error:
            raise SystemExit(str(error)) from error
        print(f"Cached Zephyr SDK assets in {cache_dir}")
        return

    SdkProxyHandler.version = args.version
    SdkProxyHandler.toolchain = args.toolchain
    SdkProxyHandler.cache_dir = cache_dir

    server = http.server.ThreadingHTTPServer((args.host, args.port), SdkProxyHandler)
    print(
        f"Serving Zephyr SDK release metadata at http://{args.host}:{args.port}/releases",
        flush=True,
    )
    print(f"Cache directory: {cache_dir}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
