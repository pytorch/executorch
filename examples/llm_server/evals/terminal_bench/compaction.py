# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Install the native compaction hook in Harbor's mini-swe-agent environment."""

from pathlib import Path

from harbor.agents.installed.mini_swe_agent import MiniSweAgent

from .sessions import SessionMiniSweAgent


class CliffMiniSweAgent(MiniSweAgent):
    async def install(self, environment):
        await super().install(environment)
        for name in ("cliff_agent.py", "LICENSE.cliffcompaction"):
            await environment.upload_file(
                Path(__file__).with_name(name), f"/installed-agent/{name}"
            )


class SessionCliffMiniSweAgent(CliffMiniSweAgent, SessionMiniSweAgent):
    pass
