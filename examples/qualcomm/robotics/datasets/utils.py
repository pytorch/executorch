import yaml
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationConfig:
    dataset_id: str
    task_id: int
    max_episodes: int
    observation_interval: int
    seed: int


def load_calibration_config(
    path: str | Path,
) -> CalibrationConfig:
    """Load calibration configuration from YAML."""

    with open(path, "r") as file:
        config = yaml.safe_load(file)

    return CalibrationConfig(
        dataset_id=config["dataset_id"],
        task_id=config["task_id"],
        max_episodes=config["max_episodes"],
        observation_interval=config[
            "observation_interval"
        ],
        seed=config["seed"],
    )