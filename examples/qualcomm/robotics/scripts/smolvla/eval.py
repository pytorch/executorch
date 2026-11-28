#!/usr/bin/env python

"""
Run the passive SmolVLA rollout and calibration worker.

Terminal A controls which worker Terminal B executes:

- WorkerType.ROLLOUT
- WorkerType.CALIBRATION
- WorkerType.SHUTDOWN

Each worker uses a new TCP connection.

The rollout worker runs one closed-loop LIBERO episode and writes an MP4.

The calibration worker:

1. Loads LeRobotDataset.
2. Filters observations by task_index.
3. Keeps the first configured number of episodes.
4. Selects one observation every configured interval within each episode.
5. Prepares and sends selected observations to Terminal A.
6. Waits for Terminal A to complete each calibration forward.
"""

from __future__ import annotations

import argparse
import os

from pathlib import Path
from typing import Any

# Must be configured before importing MuJoCo / LIBERO modules.
os.environ.setdefault("MUJOCO_GL", "glfw")

import torch

from torch.utils.data import DataLoader
from torch.utils.data import default_collate
from tqdm import tqdm

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
)
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import (
    make_env,
    make_env_pre_post_processors,
)
from lerobot.policies.factory import (
    make_pre_post_processors,
)

from executorch.examples.qualcomm.robotics.remote_inference.client import (
    RemoteClient,
)
from executorch.examples.qualcomm.robotics.remote_inference.worker import (
    WORKER_COMPLETE,
    WorkerType,
    receive_worker,
)
from executorch.examples.qualcomm.robotics.rollout.rollout import (
    SimulationRollout,
    to_wire,
)
from executorch.examples.qualcomm.robotics.transport.frame_socket import (
    ConnectionConfig,
)
from executorch.examples.qualcomm.robotics.datasets.libero import (
    CalibrationDataset,
)
from executorch.examples.qualcomm.robotics.datasets.utils import (
    CalibrationConfig,
    load_calibration_config,
)

DEFAULT_MODEL_ID = "lerobot/smolvla_libero"
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 5555


OBSERVATION_RENAME_MAP = {
    "observation.images.image":
        "observation.images.camera1",
    "observation.images.image2":
        "observation.images.camera2",
}


LIBERO_CAMERA_CONFIG = {
    "camera_name": (
        "agentview_image,"
        "robot0_eye_in_hand_image"
    ),
    "camera_name_mapping": {
        "agentview_image": "image",
        "robot0_eye_in_hand_image": "image2",
    },
}


DEFAULT_CALIBRATION_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "examples"
    / "libero.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a passive SmolVLA rollout and calibration worker "
            "controlled by Terminal A."
        )
    )

    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
    )
    
    # Connection
    parser.add_argument(
        "--host",
        default=DEFAULT_SERVER_HOST,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_SERVER_PORT,
    )
    parser.add_argument(
        "--connect-timeout-seconds",
        type=float,
        default=600.0,
    )
    parser.add_argument(
        "--connect-retry-interval-seconds",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--read-timeout-seconds",
        type=float,
        default=3000.0,
    )
    parser.add_argument(
        "--write-timeout-seconds",
        type=float,
        default=3000.0,
    )
    
    # Rollout
    parser.add_argument(
        "--rollout-benchmark",
        default="libero_10",
    )

    parser.add_argument(
        "--rollout-task-id",
        type=int,
        default=9,
    )

    parser.add_argument(
        "--rollout-episode-seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--rollout-max-steps",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--rollout-video-fps",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--rollout-output-directory",
        type=Path,
        default=Path("outputs"),
    )

    # Calibration Dataset
    parser.add_argument(
        "--calibration-config",
        type=Path,
        default=DEFAULT_CALIBRATION_CONFIG,
    )

    args = parser.parse_args()

    if args.rollout_task_id < 0:
        parser.error(
            "--rollout-task-id must be nonnegative"
        )

    if args.rollout_max_steps <= 0:
        parser.error(
            "--rollout-max-steps must be positive"
        )

    if args.rollout_video_fps <= 0:
        parser.error(
            "--rollout-video-fps must be positive"
        )

    if args.connect_timeout_seconds <= 0:
        parser.error(
            "--connect-timeout-seconds must be positive"
        )

    if args.connect_retry_interval_seconds <= 0:
        parser.error(
            "--connect-retry-interval-seconds must be positive"
        )

    if args.read_timeout_seconds <= 0:
        parser.error(
            "--read-timeout-seconds must be positive"
        )

    if args.write_timeout_seconds <= 0:
        parser.error(
            "--write-timeout-seconds must be positive"
        )


    return args


def create_remote_client(
    args: argparse.Namespace,
) -> RemoteClient:
    """Create one connection for one worker execution."""

    return RemoteClient(
        ConnectionConfig(
            host=args.host,
            port=args.port,
            retry_timeout_seconds=(
                args.connect_timeout_seconds
            ),
            retry_interval_seconds=(
                args.connect_retry_interval_seconds
            ),
        ),
        read_timeout_seconds=(
            args.read_timeout_seconds
        ),
        write_timeout_seconds=(
            args.write_timeout_seconds
        ),
    )


def load_policy_processors(
    model_id: str,
):
    """Load the saved policy configuration and processors."""

    policy_config = PreTrainedConfig.from_pretrained(model_id)
    policy_config.pretrained_path = model_id
    policy_config.device = "cpu"

    (
        policy_preprocessor,
        policy_postprocessor,
    ) = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=model_id,
        preprocessor_overrides={
            "device_processor": {
                "device": "cpu",
            },
            "rename_observations_processor": {
                "rename_map":
                    OBSERVATION_RENAME_MAP,
            },
        },
    )

    return (
        policy_config,
        policy_preprocessor,
        policy_postprocessor,
    )


def run_calibration_worker(
    *,
    client: RemoteClient,
    calibration_config: CalibrationConfig,
    policy_preprocessor: Any,
) -> None:
    """Send selected observations for remote PTQ calibration."""

    print(
        f"[calibration-worker] Loading dataset "
        f"dataset={calibration_config.dataset_id!r}",
        flush=True,
    )
    
    policy_preprocessor.reset()
    
    
    dataset = LeRobotDataset(
        calibration_config.dataset_id,
        video_backend="pyav",
    )
    calibration_dataset = CalibrationDataset(
        dataset,
        task_id=calibration_config.task_id,
        max_episodes=calibration_config.max_episodes,
        observation_interval=calibration_config.observation_interval,
    )
    observations = DataLoader(
        calibration_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    
    
    accepted = client.exchange(
        {
            "observation_count": len(
                calibration_dataset
            )
        }
    )
    if accepted is not True:
        raise RuntimeError(
            "Terminal A did not accept calibration data"
        )

    for idx, sample in enumerate(
        tqdm(
            observations,
            total=len(observations),
            desc="Calibrate observations",
        ),
        start=1,
    ):  
        
        acknowledged = client.exchange(
            to_wire(policy_preprocessor(sample))
        )

        if acknowledged is not True:
            raise RuntimeError(
                "Terminal A did not acknowledge "
                f"calibration observation "
                f"{idx}"
            )

    print(
        f"[calibration-worker] Completed "
        f"observations={len(calibration_dataset)}",
        flush=True,
    )

    return 


def run_rollout_worker(
    *,
    client: RemoteClient,
    args: argparse.Namespace,
    policy_config: PreTrainedConfig,
    policy_preprocessor: Any,
    policy_postprocessor: Any,
    name: str = "rollout",
) -> None:
    """Run one remote closed-loop rollout."""
    
    # Make Env
    env_cfg = LiberoEnv(
        task=args.rollout_benchmark,
        task_ids=[args.rollout_task_id],
        **LIBERO_CAMERA_CONFIG,
    )
    env = make_env(
        env_cfg,
        n_envs=1,
        use_async_envs=False,
    )
    suite= next(iter(env))
    task_env = env[suite]
    selected_task_id = next(iter(task_env))
    env = task_env[selected_task_id]

    (
        env_preprocessor,
        env_postprocessor,
    ) = make_env_pre_post_processors(
        env_cfg=env_cfg,
        policy_cfg=policy_config,
    )
    
    # Set video output path
    video_path = (
        args.rollout_output_directory
        / (
            f"{args.rollout_benchmark}_"
            f"task_{args.rollout_task_id:02d}_"
            f"seed_{args.rollout_episode_seed:03d}_"
            f"{name}.mp4"
        )
    )

    rollout = SimulationRollout(
        env=env,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=policy_preprocessor,
        postprocessor=policy_postprocessor,
        client=client,
        action_dim=policy_config.action_feature.shape[0],
        max_steps=args.rollout_max_steps,
        video_path=video_path,
        video_fps=args.rollout_video_fps,
    )

    print(
        f"[rollout-worker] Starting "
        f"name={name!r} "
        f"seed={args.rollout_episode_seed}",
        flush=True,
    )

    result = rollout.run(
        seed=args.rollout_episode_seed
    )

    print(
        f"[rollout-worker] Completed "
        f"name={name!r} "
        f"success={result.success} "
        f"steps={result.steps} "
        f"num_plans={result.num_plans} "
        f"terminated={result.terminated} "
        f"truncated={result.truncated} "
        f"reached_max_steps="
        f"{result.reached_max_steps} "
        f"mean_prediction_sec={result.mean_prediction_seconds} "
        f"video={result.video_path}",
        flush=True,
    )

    return 


def main() -> None:
    args = parse_args()

    args.rollout_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load policy processors
    (
        policy_config,
        policy_preprocessor,
        policy_postprocessor,
    ) = load_policy_processors(
        args.model_id
    )

    # Listen for works call
    try:
        while True:
            print(
                f"[worker] Waiting for Terminal A "
                f"host={args.host} "
                f"port={args.port}",
                flush=True,
            )

            client = create_remote_client(args)

            with client:
                worker, name = receive_worker(
                    client
                )

                if worker is WorkerType.SHUTDOWN:
                    print(
                        "[worker] SHUTDOWN requested "
                        "by Terminal A",
                        flush=True,
                    )
                    break

                if worker is WorkerType.ROLLOUT:
                    run_rollout_worker(
                        client=client,
                        args=args,
                        policy_config=policy_config,
                        policy_preprocessor=policy_preprocessor,
                        policy_postprocessor=policy_postprocessor,
                        name=name,
                    )

                    acknowledged = client.exchange(
                        WORKER_COMPLETE
                    )

                    if acknowledged is not True:
                        raise RuntimeError(
                            "Terminal A did not acknowledge "
                            f"rollout completion name={name!r}"
                        )

                    continue

                if worker is WorkerType.CALIBRATION:
                    calibration_config = (
                        load_calibration_config(
                            args.calibration_config
                        )
                    )
                    run_calibration_worker(
                        client=client,
                        calibration_config=calibration_config,
                        policy_preprocessor=(
                            policy_preprocessor
                        ),
                    )

                    acknowledged = client.exchange(
                        WORKER_COMPLETE
                    )

                    if acknowledged is not True:
                        raise RuntimeError(
                            "Terminal A did not acknowledge "
                            "calibration completion"
                        )

                    continue

                raise RuntimeError(
                    f"Unhandled worker type: {worker!r}"
                )

    except Exception as error:
        print(
            f"[worker] Process failed "
            f"error={type(error).__name__}: {error}",
            flush=True,
        )
        raise

    print("[worker] Process completed ", flush=True)


if __name__ == "__main__":
    main()
    
