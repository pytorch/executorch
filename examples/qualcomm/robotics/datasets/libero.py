
TYPES = {
    "spatial": "libero_spatial",
    "object": "libero_object",
    "goal": "libero_goal",
    "10": "libero_10",
    "90": "libero_90",
}
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}
CAMERA_KW = {
    "camera_name": "agentview_image,robot0_eye_in_hand_image",
    "camera_name_mapping": {
        "agentview_image": "image",
        "robot0_eye_in_hand_image": "image2",
    },
}

from tqdm import tqdm
from torch.utils.data import Dataset
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# pip install num2words
# Swap this import per-policy



def select_obs_indices(
    dataset: LeRobotDataset,
    *,
    task_id: int,
    max_episodes: int,
    observation_interval: int,
) -> list:
    """
    Select calibration observation indices from a task.

    Dataset hierarchy:

        Dataset
        └── Task
            └── Episode
                └── Observation

    Selection procedure:

        Select Task
            Keep only observations whose task_index equals task_id.

        Select Episodes
            Group observations by episode_index and keep the first
            max_episodes episodes in dataset order.

        Select Observations
            Within each selected episode, keep one observation every
            observation_interval observations.

    Given:

        Task 2
        ├── Episode 0
        │   ├── Observation 0
        │   ├── Observation 1
        │   ├── Observation 2
        │   ├── Observation 3
        │   └── ...
        │
        ├── Episode 1
        │   ├── Observation 0
        │   ├── Observation 1
        │   ├── Observation 2
        │   ├── Observation 3
        │   └── ...
        │
        └── Episode 2
            └── ...

        task_id=2
        max_episodes=2
        observation_interval=2

    Select Episodes:

        Episode 0
        Episode 1

    Select Observations:

        Episode 0:
            Observation 0
            Observation 2
            Observation 4
            ...

        Episode 1:
            Observation 0
            Observation 2
            Observation 4
            ...

    Result:

        Observation indices from:

            Task 2
            ├── Episode 0
            │   ├── Observation 0
            │   ├── Observation 2
            │   ├── Observation 4
            │   └── ...
            │
            └── Episode 1
                ├── Observation 0
                ├── Observation 2
                ├── Observation 4
                └── ...

    Notes:

        - Dataset order is preserved.
        - Episode order is preserved.
        - Observation order within each episode is preserved.
        - Returned values are dataset indices.
        - Returned indices can be used directly with:

              sample = dataset[index]

          during calibration.

    Returns:
        list[int]: Selected dataset indices.
    """

    if max_episodes <= 0:
        raise ValueError(
            "max_episodes must be positive"
        )
    if observation_interval <= 0:
        raise ValueError(
            "observation_interval must be positive"
        )

    hf_dataset = dataset.hf_dataset
    task_indices = hf_dataset["task_index"]
    episode_indices = hf_dataset["episode_index"]
    selected_episodes = OrderedDict()
    for dataset_index, sample_task_id in tqdm(enumerate(task_indices)):
        if int(sample_task_id) != task_id:
            continue
        episode_index = int(
            episode_indices[dataset_index]
        )
        if episode_index not in selected_episodes:
            if len(selected_episodes) >= max_episodes:
                continue
            selected_episodes[episode_index] = []
        selected_episodes[episode_index].append(
            dataset_index
        )
    if not selected_episodes:
        raise RuntimeError(
            f"No observations found for task_id={task_id}"
        )
    selected_indices = []
    for episode_dataset_indices in selected_episodes.values():
        selected_indices.extend(
            episode_dataset_indices[
                ::observation_interval
            ]
        )
    print(
        f"[calibration-worker] Selected calibration observations "
        f"task_id={task_id} "
        f"episodes={len(selected_episodes)} "
        f"max_episodes={max_episodes} "
        f"observation_interval={observation_interval} "
        f"selected={len(selected_indices)}",
        flush=True,
    )

    return selected_indices



class CalibrationDataset(Dataset):
    """Calibration observations selected from LeRobotDataset."""

    def __init__(
        self,
        dataset: LeRobotDataset,
        *,
        task_id: int,
        max_episodes: int,
        observation_interval: int,
    ) -> None:
        self.dataset = dataset

        self.indices = select_obs_indices(
            dataset,
            task_id=task_id,
            max_episodes=max_episodes,
            observation_interval=observation_interval,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self,
        idx: int,
    ) -> Mapping[str, object]:
        sample = self.dataset[
            self.indices[idx]
        ]

        if not isinstance(sample, Mapping):
            raise TypeError(
                "Dataset sample must be a mapping, "
                f"got {type(sample)!r}"
            )

        return sample
