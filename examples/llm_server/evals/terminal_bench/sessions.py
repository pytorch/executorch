# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-trial KV sessions for Harbor's unchanged mini-SWE-agent loop."""

import asyncio
import logging
from urllib.parse import quote

import httpx
import yaml

from harbor.agents.installed.mini_swe_agent import MiniSweAgent

RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


class SessionMiniSweAgent(MiniSweAgent):
    async def run(self, instruction, environment, context):
        if self.context_id is None:
            raise ValueError("Harbor must assign the agent's trial context_id")
        config = yaml.safe_load(self._config_yaml) if self._config_yaml else {}
        headers = (
            config.setdefault("model", {})
            .setdefault("model_kwargs", {})
            .setdefault("extra_headers", {})
        )
        headers["X-ExecuTorch-Session-ID"] = str(self.context_id)
        self._config_yaml = yaml.safe_dump(config, sort_keys=False)
        await super().run(instruction, environment, context)


class SessionCleanup:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def on_job_start(self, job):
        # Harbor awaits END hooks before releasing the trial's concurrency slot.
        job.on_trial_ended(self.on_trial_end)

    async def on_trial_end(self, event):
        session_id = quote(str(event.trial_id), safe="")
        try:
            async with (
                asyncio.timeout(120),
                httpx.AsyncClient(trust_env=False, timeout=20) as client,
            ):
                delay = 1
                while True:
                    try:
                        response = await client.delete(
                            f"{self.base_url}/sessions/{session_id}"
                        )
                    except httpx.TransportError as error:
                        reason = str(error)
                    else:
                        if response.status_code not in RETRYABLE_STATUS_CODES:
                            response.raise_for_status()
                            return
                        reason = f"HTTP {response.status_code}"
                    logging.getLogger(__name__).warning(
                        "Retrying session cleanup for trial %s (%s) in %ss: %s",
                        event.trial_name,
                        session_id,
                        delay,
                        reason,
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 5)
        except (httpx.HTTPError, TimeoutError) as error:
            raise RuntimeError(
                f"Could not release session {session_id} for trial {event.trial_name}; "
                "aborting evaluation to avoid silent session leaks"
            ) from error

    async def on_job_end(self, job_result):
        pass
