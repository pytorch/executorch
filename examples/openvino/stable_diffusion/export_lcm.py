# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

# mypy: disable-error-code="union-attr,import-not-found"

import argparse
import logging
import os

import torch

from executorch.backends.openvino.partitioner import OpenvinoPartitioner
from executorch.examples.models.stable_diffusion.model import (  # type: ignore[import-untyped]
    LCMModelLoader,
)
from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower
from executorch.exir.backend.backend_details import CompileSpec
from torch.export import export

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LCMOpenVINOExporter:
    """Export Latent Consistency Model (LCM) components to OpenVINO PTE files"""

    def __init__(
        self,
        model_id: str = "SimianLuo/LCM_Dreamshaper_v7",
        dtype: torch.dtype = torch.float16,
    ):
        self.model_loader = LCMModelLoader(model_id=model_id, dtype=dtype)

    def load_models(self) -> bool:
        """Load the LCM pipeline and extract components"""
        return self.model_loader.load_models()

    def export_text_encoder(self, output_path: str, device: str = "CPU") -> bool:
        """Export CLIP text encoder to PTE file"""
        try:
            logger.info("Exporting text encoder with OpenVINO backend...")

            # Get wrapped model and dummy inputs
            text_encoder_wrapper = self.model_loader.get_text_encoder_wrapper()
            dummy_inputs = self.model_loader.get_dummy_inputs()

            # Export to ATEN graph
            exported_program = export(
                text_encoder_wrapper, dummy_inputs["text_encoder"]
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

            # Get wrapped model and dummy inputs
            unet_wrapper = self.model_loader.get_unet_wrapper()
            dummy_inputs = self.model_loader.get_dummy_inputs()

            # Export to ATEN graph
            exported_program = export(unet_wrapper, dummy_inputs["unet"])

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

            # Get wrapped model and dummy inputs
            vae_decoder = self.model_loader.get_vae_decoder()
            dummy_inputs = self.model_loader.get_dummy_inputs()

            # Export to ATEN graph
            exported_program = export(vae_decoder, dummy_inputs["vae_decoder"])

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
        choices=["fp16", "fp32"],
        default="fp16",
        help="Model data type (default: fp16)",
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
    dtype_map = {"fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    # Create exporter and load models
    exporter = LCMOpenVINOExporter(args.model_id, dtype=dtype)

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
