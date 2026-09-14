# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional

import torch

try:
    from .model import MODEL_ID, StableDiffusion3ModelLoader, StableDiffusionComponent
except ImportError:
    from model import MODEL_ID, StableDiffusion3ModelLoader, StableDiffusionComponent
from torch.export import export, save


logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


COMPONENTS = (
    StableDiffusionComponent.TEXT_ENCODER,
    StableDiffusionComponent.TEXT_ENCODER_2,
    StableDiffusionComponent.TEXT_ENCODER_3,
    StableDiffusionComponent.TRANSFORMER,
    StableDiffusionComponent.VAE_DECODER,
)
MEMORY_SENSITIVE_COMPONENTS = {
    StableDiffusionComponent.TRANSFORMER,
    StableDiffusionComponent.VAE_DECODER,
}
LATENT_SIZE_RETRIES = (64, 32, 16, 8)

DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}
NAME_BY_DTYPE: dict[torch.dtype, str] = {
    dtype: name for name, dtype in DTYPE_BY_NAME.items()
}


def parse_dtype(dtype: str) -> torch.dtype:
    """Parse a dtype string into a torch.dtype.

    Supported values are: fp16, bf16, fp32.
    """
    try:
        return DTYPE_BY_NAME[dtype]
    except KeyError as e:
        raise ValueError(f"Unsupported dtype: {dtype}") from e


def dtype_name(dtype: torch.dtype) -> str:
    """Convert a torch dtype to the corresponding CLI dtype name."""
    try:
        return NAME_BY_DTYPE[dtype]
    except KeyError as e:
        raise ValueError(f"Unsupported dtype: {dtype}") from e


def parse_component(component: str) -> StableDiffusionComponent:
    """Parse a component string into a StableDiffusionComponent enum.

    Raises argparse.ArgumentTypeError for unsupported components.
    """
    try:
        return StableDiffusionComponent(component)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Unsupported component: {component}") from e


def build_export_command(
    component: StableDiffusionComponent,
    model_id: str,
    text_encoder_id: Optional[str],
    text_encoder_2_id: Optional[str],
    dtype: torch.dtype,
    max_sequence_length: int,
    latent_size: Optional[int],
    output_dir: Path,
) -> list[str]:
    """Build the command used to export one component in a child process."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model-id",
        model_id,
        "--component",
        component.value,
        "--dtype",
        dtype_name(dtype),
        "--max-sequence-length",
        str(max_sequence_length),
        "--output-dir",
        str(output_dir),
        "--in-process",
    ]
    if latent_size is not None:
        command.extend(["--latent-size", str(latent_size)])
    if text_encoder_id is not None:
        command.extend(["--text-encoder-id", text_encoder_id])
    if text_encoder_2_id is not None:
        command.extend(["--text-encoder-2-id", text_encoder_2_id])
    return command


def is_valid_export_file(path: Path) -> bool:
    """Return true when an existing export is a non-empty readable .pt2 archive."""
    return path.is_file() and path.stat().st_size > 0 and zipfile.is_zipfile(path)


def latent_size_attempts(
    component: StableDiffusionComponent,
    latent_size: Optional[int],
) -> tuple[Optional[int], ...]:
    """Return latent-size attempts for a component export."""
    if component not in MEMORY_SENSITIVE_COMPONENTS:
        return (latent_size,)

    attempts = [latent_size]
    retry_sizes = (
        size
        for size in LATENT_SIZE_RETRIES
        if latent_size is None or size < latent_size
    )
    attempts.extend(retry_sizes)

    deduped_attempts = []
    for attempt in attempts:
        if attempt not in deduped_attempts:
            deduped_attempts.append(attempt)
    return tuple(deduped_attempts)


def remove_invalid_export_file(path: Path) -> None:
    """Remove a stale partial export if it exists and is not a valid .pt2 archive."""
    if path.exists() and not is_valid_export_file(path):
        logger.warning("Removing invalid export: %s", path)
        path.unlink()


def run_component_export_subprocess(
    component: StableDiffusionComponent,
    model_id: str,
    text_encoder_id: Optional[str],
    text_encoder_2_id: Optional[str],
    dtype: torch.dtype,
    max_sequence_length: int,
    latent_size: Optional[int],
    output_dir: Path,
) -> None:
    """Export a component in a child process, retrying smaller latent sizes on OOM."""
    attempted_latent_sizes = []
    last_error = None
    for attempt_latent_size in latent_size_attempts(component, latent_size):
        attempted_latent_sizes.append(attempt_latent_size)
        remove_invalid_export_file(output_dir / f"{component.value}.pt2")
        if attempt_latent_size != latent_size:
            logger.info(
                "Retrying %s with --latent-size %s",
                component.value,
                attempt_latent_size,
            )

        try:
            subprocess.run(
                build_export_command(
                    component=component,
                    model_id=model_id,
                    text_encoder_id=text_encoder_id,
                    text_encoder_2_id=text_encoder_2_id,
                    dtype=dtype,
                    max_sequence_length=max_sequence_length,
                    latent_size=attempt_latent_size,
                    output_dir=output_dir,
                ),
                check=True,
            )
            return
        except subprocess.CalledProcessError as e:
            last_error = e
            if e.returncode >= 0:
                raise
            logger.warning(
                "Exporting %s was killed by signal %s with latent size %s",
                component.value,
                -e.returncode,
                attempt_latent_size,
            )

    attempted = ", ".join(
        "default" if size is None else str(size) for size in attempted_latent_sizes
    )
    raise RuntimeError(
        f"Exporting {component.value} was killed after trying latent sizes: "
        f"{attempted}. Try reducing --max-sequence-length or exporting this "
        "component on a machine with more memory."
    ) from last_error


def export_all_in_subprocesses(
    model_id: str,
    text_encoder_id: Optional[str],
    text_encoder_2_id: Optional[str],
    dtype: torch.dtype,
    max_sequence_length: int,
    latent_size: Optional[int],
    output_dir: Path,
    skip_existing: bool,
) -> list[Path]:
    """Export each SD3 component in a fresh Python process to release memory."""
    output_paths = []
    for component in COMPONENTS:
        output_path = output_dir / f"{component.value}.pt2"
        if skip_existing and is_valid_export_file(output_path):
            logger.info("Skipping %s; %s already exists", component.value, output_path)
            output_paths.append(output_path)
            continue
        if skip_existing and output_path.exists():
            logger.warning(
                "Re-exporting %s because %s is not a valid export",
                component.value,
                output_path,
            )

        logger.info("Exporting %s in a fresh process", component.value)
        run_component_export_subprocess(
            component=component,
            model_id=model_id,
            text_encoder_id=text_encoder_id,
            text_encoder_2_id=text_encoder_2_id,
            dtype=dtype,
            max_sequence_length=max_sequence_length,
            latent_size=latent_size,
            output_dir=output_dir,
        )
        output_paths.append(output_path)

    return output_paths


class SD35LargeExporter:
    """Export wrapped SD3 components, defaulting to Stable Diffusion 3.5 Large."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        text_encoder_id: Optional[str] = None,
        text_encoder_2_id: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        max_sequence_length: int = 256,
        latent_size: Optional[int] = None,
    ):
        self.max_sequence_length = max_sequence_length
        self.latent_size = latent_size
        self.model_loader = StableDiffusion3ModelLoader(
            model_id=model_id,
            text_encoder_id=text_encoder_id,
            text_encoder_2_id=text_encoder_2_id,
            dtype=dtype,
        )

    def load_models(self) -> bool:
        """Load all configured SD3 components."""
        return self.model_loader.load_models()

    def load_component(self, component: StableDiffusionComponent) -> bool:
        """Load only the model component needed for a single export."""
        return self.model_loader.load_models([component])

    def _component_model(self, component: StableDiffusionComponent) -> torch.nn.Module:
        if component == StableDiffusionComponent.TEXT_ENCODER:
            return self.model_loader.get_text_encoder_wrapper()
        if component == StableDiffusionComponent.TEXT_ENCODER_2:
            return self.model_loader.get_text_encoder_2_wrapper()
        if component == StableDiffusionComponent.TEXT_ENCODER_3:
            return self.model_loader.get_text_encoder_3_wrapper()
        if component == StableDiffusionComponent.TRANSFORMER:
            return self.model_loader.get_transformer_wrapper()
        if component == StableDiffusionComponent.VAE_DECODER:
            return self.model_loader.get_vae_decoder_wrapper()
        raise ValueError(f"Unsupported SD3.5 component: {component.value}")

    def export_component(
        self,
        component: StableDiffusionComponent,
        output_dir: Path,
    ) -> Path:
        """Export a single SD3 component to a .pt2 file."""
        dummy_inputs = self.model_loader.get_dummy_inputs(
            max_sequence_length=self.max_sequence_length,
            latent_size=self.latent_size,
        )
        if component not in dummy_inputs:
            raise ValueError(f"No dummy inputs are available for {component.value}")

        model = self._component_model(component).eval()
        component_inputs = dummy_inputs[component]
        logger.info("Exporting %s", component.value)
        exported_program = export(model, component_inputs)

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{component.value}.pt2"
        save(exported_program, output_path)
        logger.info("Saved %s", output_path)
        return output_path

    def export_all(self, output_dir: Path) -> list[Path]:
        """Export all wrapped SD3 components."""
        return [
            self.export_component(component, output_dir) for component in COMPONENTS
        ]


def main() -> None:
    """Parse command line arguments and export the requested SD3 components."""
    parser = argparse.ArgumentParser(
        description=(
            "Export wrapped SD3 components, defaulting to Stable Diffusion 3.5 "
            "Large."
        )
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help="HuggingFace model id to load.",
    )
    parser.add_argument(
        "--text-encoder-id",
        default=None,
        help=(
            "HuggingFace CLIP text encoder repo id. Defaults to the "
            "text_encoder subfolder of --model-id."
        ),
    )
    parser.add_argument(
        "--text-encoder-2-id",
        default=None,
        help=(
            "HuggingFace CLIP text encoder repo id. Defaults to the "
            "text_encoder_2 subfolder of --model-id."
        ),
    )
    parser.add_argument(
        "--component",
        default="all",
        choices=("all", *(component.value for component in COMPONENTS)),
        help=(
            "Component to export: all, text_encoder, text_encoder_2, "
            "text_encoder_3, transformer, or vae_decoder."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("fp16", "bf16", "fp32"),
        default="fp16",
        help="Model dtype used while loading and exporting.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=256,
        help="T5 sequence length used for text_encoder_3 and transformer inputs.",
    )
    parser.add_argument(
        "--latent-size",
        type=int,
        default=None,
        help=(
            "Latent height/width used for transformer and VAE decoder dummy "
            "inputs. Defaults to the transformer sample size, or 128 for a "
            "standalone VAE decoder export."
        ),
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-export components even if the destination .pt2 file already exists.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sd3.5-large-exported"),
        help="Directory where exported .pt2 files are written.",
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    dtype = parse_dtype(args.dtype)

    if args.component == "all":
        export_all_in_subprocesses(
            model_id=args.model_id,
            text_encoder_id=args.text_encoder_id,
            text_encoder_2_id=args.text_encoder_2_id,
            dtype=dtype,
            max_sequence_length=args.max_sequence_length,
            latent_size=args.latent_size,
            output_dir=args.output_dir,
            skip_existing=not args.no_skip_existing,
        )
        return

    component = parse_component(args.component)
    if not args.in_process and component in MEMORY_SENSITIVE_COMPONENTS:
        run_component_export_subprocess(
            component=component,
            model_id=args.model_id,
            text_encoder_id=args.text_encoder_id,
            text_encoder_2_id=args.text_encoder_2_id,
            dtype=dtype,
            max_sequence_length=args.max_sequence_length,
            latent_size=args.latent_size,
            output_dir=args.output_dir,
        )
        return

    exporter = SD35LargeExporter(
        model_id=args.model_id,
        text_encoder_id=args.text_encoder_id,
        text_encoder_2_id=args.text_encoder_2_id,
        dtype=dtype,
        max_sequence_length=args.max_sequence_length,
        latent_size=args.latent_size,
    )
    if not exporter.load_component(component):
        raise RuntimeError("Failed to load SD3.5 Large models")

    exporter.export_component(component, args.output_dir)


if __name__ == "__main__":
    main()
