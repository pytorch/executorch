# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Open Images Person dataset setup and preprocessing."""

import csv
import json
from pathlib import Path
from urllib.parse import urlparse

import torch
from PIL import Image  # type: ignore[import-not-found, import-untyped]
from torch import nn
from torchvision import (  # type: ignore[import-not-found, import-untyped]
    datasets,
    tv_tensors,
)
from torchvision.transforms import v2  # type: ignore[import-not-found, import-untyped]

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = EXAMPLE_DIR / "training" / "datasets"
DATASET_DIR = DATASETS_DIR / "open_images_person"
TRAIN_SAMPLES = 14000
VALIDATION_SAMPLES = 1000
IMAGE_SIZE = 128
PERSON_CATEGORY_ID = 1
Target = dict[str, torch.Tensor]


def exported_image_count(labels_path: Path) -> int:
    """Return the number of images in an exported COCO dataset."""
    if not labels_path.exists():
        return 0
    with labels_path.open(encoding="utf-8") as file:
        return len(json.load(file).get("images", []))


def is_allowed_license(license_url: str) -> bool:
    """Return whether an image is recorded as Creative Commons Attribution."""
    parsed = urlparse(license_url.strip())
    return (
        parsed.scheme in ("http", "https")
        and parsed.netloc.lower().removeprefix("www.") == "creativecommons.org"
        and parsed.path.lower().startswith("/licenses/by/")
    )


def setup_dataset() -> None:
    """Download and export licence-validated Open Images Person splits."""
    import fiftyone as fo  # type: ignore[import-not-found, import-untyped]
    import fiftyone.zoo as foz  # type: ignore[import-not-found, import-untyped]
    from fiftyone import (  # type: ignore[import-not-found, import-untyped]
        ViewField as F,
    )

    for split, max_samples in (
        ("train", TRAIN_SAMPLES),
        ("validation", VALIDATION_SAMPLES),
    ):
        export_dir = DATASET_DIR / split
        labels_path = export_dir / "labels.json"
        if exported_image_count(labels_path) == max_samples:
            print(f"Using existing {split} dataset at {export_dir}.")
            continue
        if labels_path.exists():
            print(
                f"Rebuilding {split} dataset at {export_dir} for "
                f"{max_samples} images."
            )
        fo.config.dataset_zoo_dir = str(DATASETS_DIR / "fiftyone")
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split=split,
            label_types=["detections"],
            classes=["Person"],
            max_samples=max_samples,
            shuffle=True,
            seed=0,
            dataset_name=f"micro-yolo-open-images-v7-{split}",
            drop_existing_dataset=True,
            progress=False,
        )
        metadata_path = (
            DATASETS_DIR
            / "fiftyone"
            / "open-images-v7"
            / split
            / "metadata"
            / "image_ids.csv"
        )
        image_paths = {
            Path(sample.filepath).stem: sample.filepath for sample in dataset
        }
        requested_image_ids = set(image_paths)
        allowed_image_ids = set()
        with metadata_path.open(encoding="utf-8", newline="") as file:
            metadata = csv.DictReader(file)
            if not {"ImageID", "License"} <= set(metadata.fieldnames or ()):
                raise ValueError(f"{metadata_path} is missing image licence metadata.")
            for row in metadata:
                if row["ImageID"] in requested_image_ids and is_allowed_license(
                    row.get("License", "")
                ):
                    allowed_image_ids.add(row["ImageID"])
        if not allowed_image_ids:
            raise ValueError(f"{metadata_path} contains no CC BY images.")
        allowed_image_paths = [image_paths[id] for id in allowed_image_ids]
        dataset.match(F("filepath").is_in(allowed_image_paths)).filter_labels(
            "ground_truth", (F("label") == "Person") & (F("iscrowd") != 1)
        ).export(
            export_dir=str(export_dir),
            dataset_type=fo.types.COCODetectionDataset,
            label_field="ground_truth",
            classes=["Person"],
            overwrite=True,
            progress=False,
        )


class RandomPersonCrop(nn.Module):
    """Place a randomly selected person throughout a cropped image."""

    def __init__(self, probability: float = 1.0, minimum_scale: float = 0.3):
        super().__init__()
        self.probability = probability
        self.minimum_scale = minimum_scale

    @staticmethod
    def _crop_origin(
        box_start: float,
        box_end: float,
        crop_size: int,
        desired_center: float,
        image_size: int,
    ) -> int:
        box_size = box_end - box_start
        if 2 * desired_center * crop_size < box_size:
            origin = box_end - 2 * desired_center * crop_size
        elif 2 * (1 - desired_center) * crop_size < box_size:
            origin = box_start + (1 - 2 * desired_center) * crop_size
        else:
            origin = (box_start + box_end) / 2 - desired_center * crop_size
        return min(max(round(origin), 0), image_size - crop_size)

    @staticmethod
    def _visible_center(
        box_start: float, box_end: float, origin: int, crop_size: int
    ) -> float:
        visible_start = min(max(box_start - origin, 0), crop_size)
        visible_end = min(max(box_end - origin, 0), crop_size)
        return (visible_start + visible_end) / (2 * crop_size)

    def forward(
        self, image: torch.Tensor, boxes: tv_tensors.BoundingBoxes
    ) -> tuple[torch.Tensor, tv_tensors.BoundingBoxes]:
        """Crop an image so one person appears at a sampled position.

        Args:
            image (torch.Tensor): Image to crop.
            boxes (tv_tensors.BoundingBoxes): Person boxes associated with the
                image.

        Returns:
            tuple[torch.Tensor, tv_tensors.BoundingBoxes]: Cropped image and
            boxes.
        """
        if not len(boxes) or torch.rand(()).item() >= self.probability:
            return image, boxes

        height, width = v2.functional.get_size(image)
        anchor_index = torch.randint(len(boxes), ()).item()
        anchor = boxes[anchor_index]
        desired_x = torch.empty(()).uniform_(0.05, 0.95).item()
        desired_y = torch.empty(()).uniform_(0.05, 0.95).item()
        candidates = torch.linspace(1.0, self.minimum_scale, 29).tolist()
        best_crop: tuple[float, int, int, int, int] | None = None
        for crop_scale in candidates:
            crop_height = min(round(height * crop_scale), height)
            crop_width = min(round(width * crop_scale), width)
            left = self._crop_origin(
                float(anchor[0]), float(anchor[2]), crop_width, desired_x, width
            )
            top = self._crop_origin(
                float(anchor[1]), float(anchor[3]), crop_height, desired_y, height
            )
            actual_x = self._visible_center(
                float(anchor[0]), float(anchor[2]), left, crop_width
            )
            actual_y = self._visible_center(
                float(anchor[1]), float(anchor[3]), top, crop_height
            )
            error = max(abs(actual_x - desired_x), abs(actual_y - desired_y))
            if best_crop is None or error < best_crop[0]:
                best_crop = (error, top, left, crop_height, crop_width)
            if error <= 0.025:
                break

        assert best_crop is not None
        _, top, left, crop_height, crop_width = best_crop
        return (
            v2.functional.crop(image, top, left, crop_height, crop_width),
            v2.functional.crop(boxes, top, left, crop_height, crop_width),
        )


def make_image_transform(training: bool) -> v2.Compose:
    """Build the image and bounding-box preprocessing pipeline.

    Args:
        training (bool): Whether to include stochastic training augmentations.

    Returns:
        v2.Compose: Joint image and bounding-box transforms.
    """
    image_transforms: list[torch.nn.Module] = []
    if training:
        image_transforms.extend(
            [
                v2.RandomHorizontalFlip(p=0.5),
                RandomPersonCrop(),
                v2.RandomAffine(10),
                v2.ColorJitter(brightness=0.2, saturation=0.2, hue=0.1),
            ]
        )
    return v2.Compose(
        [
            v2.Lambda(lambda image: image.convert("RGB"), Image.Image),
            v2.ToImage(),
            *image_transforms,
            v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


class MicroYoloDataset(datasets.CocoDetection):
    """COCO person-detection dataset with MicroYolo preprocessing."""

    def __init__(self, split: str, training: bool) -> None:
        """Load a data split with µYOLO image preprocessing."""
        split_dir = DATASET_DIR / split
        super().__init__(split_dir / "data", split_dir / "labels.json")
        self.image_transform = make_image_transform(training)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Target]:
        """Return a transformed image and its detection target."""
        image, annotations = super().__getitem__(index)
        width, height = image.size
        boxes = torch.tensor(
            [
                annotation["bbox"]
                for annotation in annotations
                if annotation["category_id"] == PERSON_CATEGORY_ID
                and not annotation.get("iscrowd", 0)
            ],
            dtype=torch.float32,
        ).reshape(-1, 4)
        if boxes.numel():
            boxes[:, 2:] += boxes[:, :2]
        image, boxes = self.image_transform(
            image,
            tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(height, width)),
        )
        boxes = boxes.as_subclass(torch.Tensor)
        boxes[:, 0::2].clamp_(0, IMAGE_SIZE)
        boxes[:, 1::2].clamp_(0, IMAGE_SIZE)
        boxes = boxes[
            (boxes[:, 2] - boxes[:, 0] >= 12) & (boxes[:, 3] - boxes[:, 1] >= 12)
        ]
        return image, {"boxes": boxes}


def collate(
    batch: list[tuple[torch.Tensor, Target]]
) -> tuple[torch.Tensor, list[Target]]:
    """Stack images while retaining variable-length detection targets."""
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)
