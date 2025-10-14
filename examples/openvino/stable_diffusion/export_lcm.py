# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

# mypy: disable-error-code="union-attr,import-not-found"

import argparse
import logging
import os
from typing import Any, Optional

import torch
from torch.export import export

try:
    from diffusers import DiffusionPipeline  # type: ignore[import-not-found]
except ImportError:
    raise ImportError(
        "Please install diffusers and transformers: pip install diffusers transformers"
    )

from executorch.backends.openvino.partitioner import OpenvinoPartitioner
from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower
from executorch.exir.backend.backend_details import CompileSpec

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LCMExporter:
    """Export Latent Consistency Model (LCM) components to OpenVINO-optimized PTE files"""

    def __init__(
        self,
        model_id: str = "SimianLuo/LCM_Dreamshaper_v7",
        dtype: torch.dtype = torch.float16,
    ):
        self.model_id = model_id
        self.dtype = dtype
        self.pipeline: Optional[DiffusionPipeline] = None
        self.text_encoder: Any = None
        self.unet: Any = None
        self.vae: Any = None
        self.tokenizer: Any = None

    def load_models(self) -> bool:
        """Load the LCM pipeline and extract components"""
        try:
            logger.info(f"Loading LCM pipeline: {self.model_id} (dtype: {self.dtype})")
            self.pipeline = DiffusionPipeline.from_pretrained(
                self.model_id, torch_dtype=self.dtype, use_safetensors=True
            )

            # Extract individual components
            self.text_encoder = self.pipeline.text_encoder
            self.unet = self.pipeline.unet
            self.vae = self.pipeline.vae
            self.tokenizer = self.pipeline.tokenizer

            # Set models to evaluation mode
            self.text_encoder.eval()
            self.unet.eval()
            self.vae.eval()

            logger.info("Successfully loaded all LCM model components")
            return True

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            import traceback

            traceback.print_exc()
            return False

    def export_text_encoder(self, output_path: str, device: str = "CPU") -> bool:
        """Export CLIP text encoder to PTE file"""
        try:
            logger.info("Exporting text encoder...")

            # Create wrapper to extract last_hidden_state from CLIP output
            class TextEncoderWrapper(torch.nn.Module):
                def __init__(self, text_encoder):
                    super().__init__()
                    self.text_encoder = text_encoder

                def forward(self, input_ids):
                    # Call text encoder and extract last_hidden_state
                    output = self.text_encoder(input_ids, return_dict=True)
                    return output.last_hidden_state

            text_encoder_wrapper = TextEncoderWrapper(self.text_encoder)
            text_encoder_wrapper.eval()

            # Create dummy input for text encoder
            dummy_input_ids = torch.ones(1, 77, dtype=torch.long)

            # Export to ATEN graph
            exported_program = export(text_encoder_wrapper, (dummy_input_ids,))

            # Configure OpenVINO compilation
            compile_spec = [CompileSpec("device", device.encode())]
            partitioner = OpenvinoPartitioner(compile_spec)

            # Lower to edge dialect and apply OpenVINO backend
            edge_manager = to_edge_transform_and_lower(
                exported_program, partitioner=[partitioner]
            )

            # Convert to ExecutorTorch program
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
            logger.info("Exporting UNet model...")

            # Create a wrapper to extract the sample tensor from UNet2DConditionOutput
            class UNetWrapper(torch.nn.Module):
                def __init__(self, unet):
                    super().__init__()
                    self.unet = unet

                def forward(self, latents, timestep, encoder_hidden_states):
                    # Call UNet and extract sample from the output
                    output = self.unet(
                        latents, timestep, encoder_hidden_states, return_dict=True
                    )
                    return output.sample

            unet_wrapper = UNetWrapper(self.unet)
            unet_wrapper.eval()

            # Create dummy inputs for UNet
            batch_size = 1
            latent_channels = 4
            latent_height = 64
            latent_width = 64

            # Get the correct text embedding dimension from the UNet config
            text_embed_dim = self.unet.config.cross_attention_dim
            text_seq_len = 77

            dummy_latents = torch.randn(
                batch_size,
                latent_channels,
                latent_height,
                latent_width,
                dtype=self.dtype,
            )
            dummy_timestep = torch.tensor([981])  # Random timestep
            dummy_encoder_hidden_states = torch.randn(
                batch_size, text_seq_len, text_embed_dim, dtype=self.dtype
            )

            # Export to ATEN graph
            exported_program = export(
                unet_wrapper,
                (dummy_latents, dummy_timestep, dummy_encoder_hidden_states),
            )

            # Configure OpenVINO compilation
            compile_spec = [CompileSpec("device", device.encode())]
            partitioner = OpenvinoPartitioner(compile_spec)

            # Lower to edge dialect and apply OpenVINO backend
            edge_manager = to_edge_transform_and_lower(
                exported_program, partitioner=[partitioner]
            )

            # Convert to ExecutorTorch program
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
            logger.info("Exporting VAE decoder...")

            # Create wrapper for VAE decoder only
            class VAEDecoder(torch.nn.Module):
                def __init__(self, vae):
                    super().__init__()
                    self.vae = vae

                def forward(self, latents):
                    # Scale latents
                    latents = latents / self.vae.config.scaling_factor
                    # Decode
                    image = self.vae.decode(latents).sample
                    # Scale to [0, 1]
                    image = (image / 2 + 0.5).clamp(0, 1)
                    return image

            vae_decoder = VAEDecoder(self.vae)
            vae_decoder.eval()

            # Create dummy input for VAE decoder
            dummy_latents = torch.randn(1, 4, 64, 64, dtype=self.dtype)

            # Export to ATEN graph
            exported_program = export(vae_decoder, (dummy_latents,))

            # Configure OpenVINO compilation
            compile_spec = [CompileSpec("device", device.encode())]
            partitioner = OpenvinoPartitioner(compile_spec)

            # Lower to edge dialect and apply OpenVINO backend
            edge_manager = to_edge_transform_and_lower(
                exported_program, partitioner=[partitioner]
            )

            # Convert to ExecutorTorch program
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
    exporter = LCMExporter(args.model_id, dtype=dtype)

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
