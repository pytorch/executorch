# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

# mypy: disable-error-code="union-attr,import-not-found"

import argparse
import logging
import os

import datasets  # type: ignore[import-untyped]
import nncf  # type: ignore[import-untyped]

import torch

from executorch.backends.openvino.partitioner import OpenvinoPartitioner
from executorch.backends.openvino.quantizer import (
    OpenVINOQuantizer,
    QuantizationMode,
    quantize_model,
)
from executorch.examples.models.stable_diffusion.model import (  # type: ignore[import-untyped]
    LCMModelLoader,
    StableDiffusionComponent,
)
from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower
from executorch.exir.backend.backend_details import CompileSpec
from torch.export import export
from torchao.quantization.pt2e.quantizer.quantizer import Quantizer
from tqdm import tqdm  # type: ignore[import-untyped]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LCMOpenVINOExporter:
    """Export Latent Consistency Model (LCM) components to OpenVINO PTE files"""

    def __init__(
        self,
        model_id: str = "SimianLuo/LCM_Dreamshaper_v7",
        is_quantization_enabled: bool = False,
        dtype: torch.dtype = torch.float16,
        calibration_dataset_name: str = "google-research-datasets/conceptual_captions",
        calibration_dataset_column: str = "caption",
    ):
        if is_quantization_enabled:
            dtype = torch.float32
        self.is_quantization_enabled = is_quantization_enabled
        self.calibration_dataset_name = calibration_dataset_name
        self.calibration_dataset_column = calibration_dataset_column
        self.model_loader = LCMModelLoader(model_id=model_id, dtype=dtype)

    def load_models(self) -> bool:
        """Load the LCM pipeline and extract components"""
        return self.model_loader.load_models()

    @staticmethod
    def should_quantize_model(sd_model_component: StableDiffusionComponent) -> bool:
        """
        If this is true, then we should quantize activations and weights. Otherwise, only compress the weights.

        :param sd_model_component: the type of model in the stable diffusion pipeline such as Unet, text encoder, VAE etc.
        """
        return sd_model_component == StableDiffusionComponent.UNET

    def get_ov_quantizer(
        self, sd_model_component: StableDiffusionComponent
    ) -> Quantizer:
        quantization_mode = QuantizationMode.INT8WO_ASYM
        if self.should_quantize_model(sd_model_component):
            # Only Unet model will have both weights and activations quantized.
            quantization_mode = QuantizationMode.INT8_TRANSFORMER

        quantizer = OpenVINOQuantizer(mode=quantization_mode)
        return quantizer

    @staticmethod
    def get_unet_calibration_dataset(
        pipeline,
        dataset_name: str,
        dataset_column: str,
        calibration_dataset_size: int = 200,
        num_inference_steps: int = 4,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Collect UNet calibration inputs from prompts."""

        class UNetWrapper(torch.nn.Module):
            def __init__(self, model: torch.nn.Module, config):
                super().__init__()
                self.model = model
                self.config = config
                self.captured_args: list[
                    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                ] = []

            def _pick_correct_arg_or_kwarg(
                self,
                name: str,
                args,
                kwargs,
                idx: int,
            ):
                if name in kwargs and kwargs[name] is not None:
                    return kwargs[name]
                if len(args) > idx:
                    return args[idx]
                raise KeyError(f"Missing required UNet input: {name}")

            def _process_inputs(
                self, *args, **kwargs
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                sample = self._pick_correct_arg_or_kwarg("sample", args, kwargs, 0)
                timestep = self._pick_correct_arg_or_kwarg("timestep", args, kwargs, 1)
                encoder_hidden_states = self._pick_correct_arg_or_kwarg(
                    "encoder_hidden_states", args, kwargs, 2
                )
                timestep = (
                    timestep.unsqueeze(0)
                    if timestep.dim() == 0 and isinstance(timestep, torch.Tensor)
                    else timestep
                )
                processed_args = (
                    sample,
                    timestep,
                    encoder_hidden_states,
                )
                return processed_args

            def forward(self, *args, **kwargs):
                """
                Obtain and pass each input individually to ensure the order is maintained
                and the right values are being passed according to the expected inputs by
                the OpenVINO LCM runner.
                """
                unet_args = self._process_inputs(*args, **kwargs)
                self.captured_args.append(unet_args)
                return self.model(*args, **kwargs)

        calibration_data = []
        dataset = datasets.load_dataset(
            dataset_name,
            split="train",
            trust_remote_code=True,
        ).shuffle(seed=42)
        original_unet = pipeline.unet
        wrapped_unet = UNetWrapper(pipeline.unet, pipeline.unet.config)
        pipeline.unet = wrapped_unet
        # Run inference for data collection
        pbar = tqdm(total=calibration_dataset_size)
        for batch in dataset:
            if dataset_column not in batch:
                raise RuntimeError(
                    f"Column '{dataset_column}' was not found in dataset '{dataset_name}'"
                )
            prompt = batch[dataset_column]
            if not isinstance(prompt, str):
                prompt = str(prompt)
            if len(prompt.split()) > pipeline.tokenizer.model_max_length:
                continue
            # Run the pipeline
            pipeline(
                prompt, num_inference_steps=num_inference_steps, height=512, width=512
            )
            calibration_data.extend(wrapped_unet.captured_args)
            wrapped_unet.captured_args = []
            pbar.update(len(calibration_data) - pbar.n)
            if pbar.n >= calibration_dataset_size:
                break
        pipeline.unet = original_unet
        return calibration_data

    def maybe_quantize_model(
        self,
        model: torch.fx.GraphModule,
        sd_model_component: StableDiffusionComponent,
        is_quantization_enabled: bool,
    ) -> torch.fx.GraphModule:
        """Apply model quantization when enabled."""
        if not is_quantization_enabled:
            return model
        try:
            ov_quantizer = self.get_ov_quantizer(sd_model_component)
            if sd_model_component == StableDiffusionComponent.UNET:
                # Quantize activations for the Unet Model. Other models are weights-only quantized.
                pipeline = self.model_loader.pipeline
                calibration_dataset = self.get_unet_calibration_dataset(
                    pipeline,
                    self.calibration_dataset_name,
                    self.calibration_dataset_column,
                )

                quantized_model = quantize_model(
                    model,
                    mode=QuantizationMode.INT8_TRANSFORMER,
                    calibration_dataset=calibration_dataset,
                    smooth_quant=True,
                )
            else:
                quantized_model = nncf.experimental.torch.fx.compress_pt2e(
                    model, quantizer=ov_quantizer
                )
            return quantized_model
        except Exception as e:
            logger.error(f"Quantization failed for {sd_model_component.value}: {e}")
            import traceback

            traceback.print_exc()
            return model

    def _export_and_maybe_quantize(
        self,
        model: torch.nn.Module,
        dummy_inputs,
        sd_model_component: StableDiffusionComponent,
        is_quantization_enabled: bool,
    ) -> torch.export.ExportedProgram:
        """Export model and optionally quantize before re-export."""
        exported_program = export(model, dummy_inputs)
        exported_program_module = self.maybe_quantize_model(
            exported_program.module(), sd_model_component, is_quantization_enabled
        )
        # Re-export the quantized torch.fx.GraphModule to ExportedProgram
        exported_program = export(exported_program_module, dummy_inputs)
        return exported_program

    def export_text_encoder(self, output_path: str, device: str = "CPU") -> bool:
        """Export CLIP text encoder to PTE file"""
        try:
            logger.info("Exporting text encoder with OpenVINO backend...")

            sd_model_component = StableDiffusionComponent.TEXT_ENCODER

            # Get wrapped model and dummy inputs
            text_encoder_wrapper = self.model_loader.get_text_encoder_wrapper()
            dummy_inputs = self.model_loader.get_dummy_inputs()

            # Export to ATEN graph
            exported_program = self._export_and_maybe_quantize(
                text_encoder_wrapper,
                dummy_inputs[sd_model_component],
                sd_model_component,
                self.is_quantization_enabled,
            )

            # Configure OpenVINO compilation
            compile_spec = [CompileSpec("device", device.encode())]
            partitioner = OpenvinoPartitioner(compile_spec)

            # Lower to edge dialect and apply OpenVINO backend
            edge_manager = to_edge_transform_and_lower(
                exported_program, partitioner=[partitioner]
            )

            # Convert to ExecuTorch program
            executorch_program = edge_manager.to_executorch(
                config=ExecutorchBackendConfig()
            )

            # Save to file
            with open(output_path, "wb") as f:
                f.write(executorch_program.buffer)

            logger.info("✓ Text encoder exported successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to export text encoder: {e}")
            import traceback

            traceback.print_exc()
            return False

    def export_unet(self, output_path: str, device: str = "CPU") -> bool:
        """Export UNet model to PTE file"""
        try:
            logger.info("Exporting UNet model with OpenVINO backend...")
            sd_model_component = StableDiffusionComponent.UNET

            # Get wrapped model and dummy inputs
            unet_wrapper = self.model_loader.get_unet_wrapper()
            dummy_inputs = self.model_loader.get_dummy_inputs()

            # Export to ATEN graph
            exported_program = self._export_and_maybe_quantize(
                unet_wrapper,
                dummy_inputs[sd_model_component],
                sd_model_component,
                self.is_quantization_enabled,
            )

            # Configure OpenVINO compilation
            compile_spec = [CompileSpec("device", device.encode())]
            partitioner = OpenvinoPartitioner(compile_spec)

            # Lower to edge dialect and apply OpenVINO backend
            edge_manager = to_edge_transform_and_lower(
                exported_program, partitioner=[partitioner]
            )

            # Convert to ExecuTorch program
            executorch_program = edge_manager.to_executorch(
                config=ExecutorchBackendConfig()
            )

            # Save to file
            with open(output_path, "wb") as f:
                f.write(executorch_program.buffer)

            logger.info("✓ UNet exported successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to export UNet: {e}")
            import traceback

            traceback.print_exc()
            return False

    def export_vae_decoder(self, output_path: str, device: str = "CPU") -> bool:
        """Export VAE decoder to PTE file"""
        try:
            logger.info("Exporting VAE decoder with OpenVINO backend...")
            sd_model_component = StableDiffusionComponent.VAE_DECODER

            # Get wrapped model and dummy inputs
            vae_decoder = self.model_loader.get_vae_decoder()
            dummy_inputs = self.model_loader.get_dummy_inputs()

            # Export to ATEN graph
            exported_program = self._export_and_maybe_quantize(
                vae_decoder,
                dummy_inputs[sd_model_component],
                sd_model_component,
                self.is_quantization_enabled,
            )

            # Configure OpenVINO compilation
            compile_spec = [CompileSpec("device", device.encode())]
            partitioner = OpenvinoPartitioner(compile_spec)

            # Lower to edge dialect and apply OpenVINO backend
            edge_manager = to_edge_transform_and_lower(
                exported_program, partitioner=[partitioner]
            )

            # Convert to ExecuTorch program
            executorch_program = edge_manager.to_executorch(
                config=ExecutorchBackendConfig()
            )

            # Save to file
            with open(output_path, "wb") as f:
                f.write(executorch_program.buffer)

            logger.info("✓ VAE decoder exported successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to export VAE decoder: {e}")
            import traceback

            traceback.print_exc()
            return False

    def export_all_components(self, output_dir: str, device: str = "CPU") -> bool:
        """Export all LCM components"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Define output paths
        text_encoder_path = os.path.join(output_dir, "text_encoder.pte")
        unet_path = os.path.join(output_dir, "unet.pte")
        vae_decoder_path = os.path.join(output_dir, "vae_decoder.pte")

        # Export each component
        success = True
        success &= self.export_text_encoder(text_encoder_path, device)
        success &= self.export_unet(unet_path, device)
        success &= self.export_vae_decoder(vae_decoder_path, device)

        if success:
            logger.info(f"\n{'='*60}")
            logger.info("✓ All components exported successfully!")
            logger.info(f"Output directory: {output_dir}")
            logger.info(f"{'='*60}")
        else:
            logger.error("Export failed")

        return success


def create_argument_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Export Latent Consistency Model (LCM) components to OpenVINO PTE files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Export LCM_Dreamshaper_v7 (default):
    python export_lcm.py --output_dir ./lcm_models
""",
    )

    parser.add_argument(
        "--model_id",
        type=str,
        default="SimianLuo/LCM_Dreamshaper_v7",
        help="HuggingFace model ID for LCM (default: SimianLuo/LCM_Dreamshaper_v7)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for exported PTE files",
    )

    parser.add_argument(
        "--device",
        choices=["CPU", "GPU", "NPU"],
        default="CPU",
        help="Target OpenVINO device (default: CPU)",
    )

    parser.add_argument(
        "--dtype",
        choices=["fp16", "fp32", "int8"],
        default="fp16",
        help="Model data type. Use int8 to enable PTQ quantization (default: fp16)",
    )

    parser.add_argument(
        "--calibration_dataset_name",
        type=str,
        default="google-research-datasets/conceptual_captions",
        help="HuggingFace dataset name used for UNet calibration when dtype=int8",
    )

    parser.add_argument(
        "--calibration_dataset_column",
        type=str,
        default="caption",
        help="Dataset column name used as prompt text for UNet calibration",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser


def main() -> int:
    """Main execution function"""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("LCM Model Export")
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Device: {args.device} | Dtype: {args.dtype}")
    logger.info("=" * 60)

    # Map dtype string to torch dtype
    is_quantization_enabled = args.dtype == "int8"
    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "int8": torch.float32}
    dtype = dtype_map[args.dtype]

    # Create exporter and load models
    exporter = LCMOpenVINOExporter(
        args.model_id,
        is_quantization_enabled=is_quantization_enabled,
        dtype=dtype,
        calibration_dataset_name=args.calibration_dataset_name,
        calibration_dataset_column=args.calibration_dataset_column,
    )

    if not exporter.load_models():
        logger.error("Failed to load models")
        return 1

    # Export all components
    if not exporter.export_all_components(args.output_dir, args.device):
        return 1

    logger.info("\nTo run inference:")
    logger.info(
        f'  python openvino_lcm.py --models_dir {args.output_dir} --prompt "your prompt" --steps 4'
    )
    return 0


if __name__ == "__main__":
    main()
