"""
Synchronous closed-loop simulation rollout.

The step order mirrors lerobot ``lerobot_eval.rollout()``:

    preprocess_observation -> task -> env_preprocessor -> preprocessor
    -> action queue -> postprocessor -> env_postprocessor -> env.step

As in lerobot remote inference, this rollout owns the action queue and
requests one action chunk from the policy on Terminal A whenever it is empty.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.envs.utils import preprocess_observation
from lerobot.utils.io_utils import write_video

from executorch.examples.qualcomm.robotics.remote_inference.client import (
    RemoteClient,
)


def to_wire(batch: Mapping[str, Any]) -> dict[str, Tensor]:
    """Keep only the observation tensors consumed by the remote policy."""
    return {
        key: value
        for key, value in batch.items()
        if key.startswith("observation.") and isinstance(value, Tensor)
    }


@dataclass(frozen=True)
class EpisodeResult:
    """Result of one completed simulation episode."""

    task: str
    seed: int
    success: bool
    steps: int
    num_plans: int
    terminated: bool
    truncated: bool
    reached_max_steps: bool
    elapsed_seconds: float
    prediction_latencies_seconds: tuple[float, ...]
    video_path: Path | None

    @property
    def mean_prediction_seconds(self) -> float | None:
        if not self.prediction_latencies_seconds:
            return None
        return float(np.mean(self.prediction_latencies_seconds))


class SimulationRollout:
    """Run one synchronous environment-policy closed loop."""

    def __init__(
        self,
        *,
        env: Any,
        env_preprocessor: Any,
        env_postprocessor: Any,
        preprocessor: Any,
        postprocessor: Any,
        client: RemoteClient,
        action_dim: int,
        max_steps: int,
        video_path: Path | None = None,
        video_fps: int = 30,
    ) -> None:
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if video_fps <= 0:
            raise ValueError("video_fps must be positive")

        self._env = env
        self._env_preprocessor = env_preprocessor
        self._env_postprocessor = env_postprocessor
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._client = client
        self._action_dim = action_dim
        self._max_steps = max_steps
        self._video_path = video_path
        self._video_fps = video_fps
        self._actions: deque[Tensor] = deque()

    def run(self, *, seed: int) -> EpisodeResult:
        """Run one simulation episode."""
        self._actions.clear()
        self._preprocessor.reset()
        self._postprocessor.reset()
        self._env_preprocessor.reset()
        self._env_postprocessor.reset()

        observation, _ = self._env.reset(seed=seed)
        task = list(self._env.call("task_description"))[0]
        print(f"[rollout] Episode started seed={seed} task={task!r}", flush=True)

        frames: list[np.ndarray] = []
        latencies: list[float] = []
        success = terminated = truncated = False
        steps = 0
        started_at = time.perf_counter()

        for step_index in range(self._max_steps):
            if self._video_path is not None:
                frames.append(np.asarray(list(self._env.call("render"))[0]))

            if not self._actions:
                batch = preprocess_observation(observation)
                batch["task"] = [task]
                batch = self._preprocessor(self._env_preprocessor(batch))
                action_chunk, latency = self._request_action_chunk(batch)
                self._actions.extend(action_chunk.unbind(dim=1))
                latencies.append(latency)

            action = self._postprocessor(self._actions.popleft())
            action = self._env_postprocessor({"action": action})["action"]

            observation, _, raw_terminated, raw_truncated, info = self._env.step(
                action.cpu().numpy()
            )
            steps = step_index + 1
            success = success or bool(np.any(np.asarray(info.get("is_success", False))))
            terminated = bool(np.all(raw_terminated))
            truncated = bool(np.all(raw_truncated))
            if terminated or truncated:
                break

        elapsed_seconds = time.perf_counter() - started_at
        video_path = self._write_video(frames)
        result = EpisodeResult(
            task=task,
            seed=seed,
            success=success,
            steps=steps,
            num_plans=len(latencies),
            terminated=terminated,
            truncated=truncated,
            reached_max_steps=not (terminated or truncated),
            elapsed_seconds=elapsed_seconds,
            prediction_latencies_seconds=tuple(latencies),
            video_path=video_path,
        )
        print(f"[rollout] Episode completed {result}", flush=True)
        return result

    def _request_action_chunk(self, batch: dict[str, Any]) -> tuple[Tensor, float]:
        """Request one action chunk from the policy on Terminal A."""
        started_at = time.perf_counter()
        response = self._client.exchange(to_wire(batch))
        latency = time.perf_counter() - started_at

        if not isinstance(response, Tensor):
            raise TypeError(
                f"Action chunk must be a torch.Tensor, got {type(response)!r}"
            )
        if response.ndim != 3 or response.shape[0] != 1 or response.shape[1] <= 0:
            raise ValueError(
                "Action chunk must have shape [1, action_steps, action_dimension], "
                f"got {tuple(response.shape)}"
            )
        if response.shape[2] != self._action_dim:
            raise ValueError(
                f"Action width must be {self._action_dim}, got {response.shape[2]}"
            )
        if not response.is_floating_point() or not torch.isfinite(response).all():
            raise ValueError("Action chunk must contain finite floating-point values")

        return response.detach().cpu().contiguous(), latency

    def _write_video(self, frames: list[np.ndarray]) -> Path | None:
        if self._video_path is None:
            return None
        self._video_path.parent.mkdir(parents=True, exist_ok=True)
        write_video(self._video_path, frames, fps=self._video_fps)
        return self._video_path
