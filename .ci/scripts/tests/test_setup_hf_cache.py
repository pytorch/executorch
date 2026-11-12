# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "cache_variable", ["HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"]
)
def test_setup_hf_cache_reuses_models_and_allows_new_downloads(
    tmp_path, cache_variable
):
    shared = tmp_path / "shared"
    hub = shared / "hub"
    cached_model = hub / "models--test--cached"
    cached_model.mkdir(parents=True)
    (cached_model / "config.json").write_text('{"cached": true}')
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"}
    }
    environment.update(
        {
            cache_variable: str(shared if cache_variable == "HF_HOME" else hub),
            "TMPDIR": str(scratch),
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        }
    )
    probe = """
import json
from pathlib import Path
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.utils._fixes import WeakFileLock

cache = Path(HF_HUB_CACHE)
assert (cache / "models--test--cached" / "config.json").read_text() == '{"cached": true}'
lock = cache / ".locks" / "download.lock"
lock.parent.mkdir(parents=True)
with WeakFileLock(str(lock)):
    destination = cache / "models--test--new" / "blobs" / "tokenizer.json"
    destination.parent.mkdir(parents=True)
    destination.write_text("downloaded")
print(json.dumps({"cache": str(cache), "wrote": destination.read_text()}))
"""
    for path in (cached_model, hub, shared):
        path.chmod(0o555)
    try:
        result = subprocess.run(
            [
                "bash",
                "-eu",
                "-c",
                f"source {shlex.quote(str(ROOT / '.ci/scripts/setup-hf-cache.sh'))}\n"
                f"{shlex.quote(sys.executable)} - <<'PY'\n{probe}\nPY",
            ],
            env=environment,
            capture_output=True,
            text=True,
        )
    finally:
        for path in (shared, hub, cached_model):
            path.chmod(0o755)
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads(result.stdout)
    assert Path(observed["cache"]).parent.parent == scratch
    assert observed["wrote"] == "downloaded"
    assert not (hub / "models--test--new").exists()
    assert not (hub / ".locks").exists()
