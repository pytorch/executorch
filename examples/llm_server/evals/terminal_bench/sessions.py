# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-trial KV sessions for Harbor's unchanged mini-SWE-agent loop."""

import json
from pathlib import Path
from urllib.parse import quote

import httpx

from harbor.agents.installed.mini_swe_agent import MiniSweAgent


class SessionMiniSweAgent(MiniSweAgent):
    def __init__(self, *, logs_dir: Path, config_file: str, **kwargs):
        config = json.loads(Path(config_file).read_text())
        headers = config["model"]["model_kwargs"].setdefault("extra_headers", {})
        # Harbor assigns this identity after constructing the agent.
        headers["X-ExecuTorch-Session-ID"] = f"{logs_dir.parent.name}__agent"
        super().__init__(logs_dir=logs_dir, config=config, **kwargs)


class SessionCleanup:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def on_job_start(self, job):
        # END runs after container shutdown and before Harbor admits another trial.
        job.on_trial_ended(self.on_trial_end)

    async def on_trial_end(self, event):
        session_id = quote(f"{event.trial_name}__agent", safe="")
        async with httpx.AsyncClient(trust_env=False, timeout=20) as client:
            response = await client.delete(f"{self.base_url}/sessions/{session_id}")
            response.raise_for_status()

    async def on_job_end(self, job_result):
        pass
