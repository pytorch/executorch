# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Host-side runner for experimental multi-method diffusion programs."""

import argparse
import inspect
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from executorch.exir._warnings import experimental

from .metadata import DiffusionMetadata, METADATA_METHODS
from .pipeline_specs import DiffusionPipelineSpec, PIPELINE_SPECS

ConditioningTensors = tuple[torch.Tensor, ...]
PromptConditioning = tuple[ConditioningTensors, ConditioningTensors | None]


def _execute(method: Any, inputs: Sequence[torch.Tensor]) -> tuple[Any, ...]:
    outputs = (
        method.execute(list(inputs)) if hasattr(method, "execute") else method(*inputs)
    )
    if isinstance(outputs, torch.Tensor):
        return (outputs,)
    if not isinstance(outputs, (tuple, list)):
        return (outputs,)
    return tuple(
        output.clone() if isinstance(output, torch.Tensor) else output
        for output in outputs
    )


def _cast_floating_tensors(
    tensors: ConditioningTensors, dtype: torch.dtype
) -> ConditioningTensors:
    return tuple(
        tensor.to(dtype) if tensor.is_floating_point() else tensor for tensor in tensors
    )


def _preprocess_image(
    image: torch.Tensor,
    height: int,
    width: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    expected = (1, 3, height, width)
    if tuple(image.shape) != expected:
        raise ValueError(f"image shape must be {expected}, got {tuple(image.shape)}")
    if not image.is_floating_point():
        raise ValueError("image must have a floating-point dtype")
    if not bool(torch.all((image >= 0) & (image <= 1))):
        raise ValueError("image values must be in [0, 1]")
    return image.to(dtype)


@experimental("This API is experimental and may change without notice.")
class DiffusionRunner:
    """Runs static batch-one diffusion programs with injected host assets."""

    def __init__(self, program: Any, tokenizers: Sequence[Any], scheduler: Any) -> None:
        self.program = program
        self.tokenizers = tuple(tokenizers)
        self.scheduler = scheduler
        self.metadata = self._read_metadata()
        try:
            self.pipeline_spec: DiffusionPipelineSpec = PIPELINE_SPECS[
                self.metadata.pipeline
            ]
        except KeyError as error:
            raise ValueError(
                f"unsupported diffusion pipeline {self.metadata.pipeline!r}"
            ) from error
        self._validate_assets()

    @classmethod
    def from_huggingface(
        cls,
        program: Any,
        model_id: str,
        revision: str | None = None,
        scheduler_name: str | None = None,
    ) -> "DiffusionRunner":
        try:
            import diffusers
            from transformers import AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "The Hugging Face factory requires diffusers and transformers"
            ) from error
        metadata = cls._read_program_metadata(program)
        try:
            pipeline_spec = PIPELINE_SPECS[metadata.pipeline]
        except KeyError as error:
            raise ValueError(
                f"unsupported diffusion pipeline {metadata.pipeline!r}"
            ) from error
        tokenizers = [
            AutoTokenizer.from_pretrained(model_id, subfolder=slot, revision=revision)
            for slot in pipeline_spec.tokenizer_slots
        ]
        scheduler_class = getattr(diffusers, scheduler_name or metadata.scheduler)
        scheduler = scheduler_class.from_pretrained(
            model_id, subfolder="scheduler", revision=revision
        )
        return cls(program, tokenizers, scheduler)

    def _method(self, name: str):
        return (
            self.program.load_method(name)
            if hasattr(self.program, "load_method")
            else self.program[name]
        )

    @classmethod
    def _read_program_metadata(cls, program: Any) -> DiffusionMetadata:
        def method(name: str):
            return (
                program.load_method(name)
                if hasattr(program, "load_method")
                else program[name]
            )

        values = {}
        for key, method_name in METADATA_METHODS.items():
            output = _execute(method(method_name), ())[0]
            values[key] = (
                output.item()
                if key != "max_sequence_lengths"
                and isinstance(output, torch.Tensor)
                and output.numel() == 1
                else output
            )
        return DiffusionMetadata.from_constant_method_values(values)

    def _read_metadata(self) -> DiffusionMetadata:
        return self._read_program_metadata(self.program)

    def _validate_assets(self) -> None:
        if len(self.tokenizers) != len(self.metadata.max_sequence_lengths):
            raise ValueError(
                f"{self.metadata.pipeline} requires "
                f"{len(self.metadata.max_sequence_lengths)} tokenizer(s)"
            )
        if len(self.tokenizers) != len(self.pipeline_spec.text_encoder_input_names):
            raise ValueError(
                f"{self.metadata.pipeline} requires text encoder input "
                f"declarations for {len(self.tokenizers)} tokenizer(s)"
            )
        scheduler_config = getattr(self.scheduler, "config", None)
        prediction_type = getattr(scheduler_config, "prediction_type", None)
        if (
            self.metadata.scheduler_prediction_type is not None
            and prediction_type != self.metadata.scheduler_prediction_type
        ):
            raise ValueError(
                f"scheduler prediction_type {prediction_type!r} does not match "
                f"exported prediction_type "
                f"{self.metadata.scheduler_prediction_type!r}"
            )
        for index, (tokenizer, length) in enumerate(
            zip(self.tokenizers, self.metadata.max_sequence_lengths)
        ):
            maximum = getattr(tokenizer, "model_max_length", length)
            if maximum < length:
                raise ValueError(
                    f"tokenizer {index} supports {maximum} tokens, expected {length}"
                )

    def _tokenize(self, prompt: str) -> tuple[tuple[torch.Tensor, ...], ...]:
        encoder_inputs = []
        for tokenizer, length, input_names in zip(
            self.tokenizers,
            self.metadata.max_sequence_lengths,
            self.pipeline_spec.text_encoder_input_names,
        ):
            encoding = tokenizer(
                prompt,
                padding="max_length",
                max_length=length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = getattr(encoding, "input_ids", None)
            if input_ids is None and isinstance(encoding, dict):
                input_ids = encoding.get("input_ids")
            if not isinstance(input_ids, torch.Tensor):
                raise ValueError("tokenizer output 'input_ids' must be a tensor")
            if tuple(input_ids.shape) != (1, length):
                raise ValueError(
                    f"tokenizer returned {tuple(input_ids.shape)}, "
                    f"expected (1, {length})"
                )
            inputs = []
            for name in input_names:
                value = getattr(encoding, name, None)
                if value is None and isinstance(encoding, dict):
                    value = encoding.get(name)
                if not isinstance(value, torch.Tensor):
                    raise ValueError(f"tokenizer output {name!r} must be a tensor")
                inputs.append(value)
            encoder_inputs.append(tuple(inputs))
        return tuple(encoder_inputs)

    def _encode_prompt(self, prompt: str) -> tuple[torch.Tensor, ...]:
        encoder_outputs = tuple(
            _execute(self._method(f"text_encoder_{index}"), inputs)
            for index, inputs in enumerate(self._tokenize(prompt))
        )
        return self.pipeline_spec.assemble_conditioning(encoder_outputs)

    def prepare_conditioning(
        self,
        prompt: str,
        negative_prompt: str | None,
        use_cfg: bool,
    ) -> PromptConditioning:
        positive = self._encode_prompt(prompt)
        if not use_cfg:
            negative = None
        elif negative_prompt is None:
            if self.metadata.zero_negative_prompt:
                negative = tuple(torch.zeros_like(value) for value in positive)
            else:
                negative = self._encode_prompt("")
        else:
            negative = self._encode_prompt(negative_prompt)
        return positive, negative

    def _resolve_dimensions(
        self, height: int | None, width: int | None
    ) -> tuple[int, int]:
        for name, value in (("height", height), ("width", width)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        output_height = self.metadata.height if height is None else height
        output_width = self.metadata.width if width is None else width
        if not self.metadata.dynamic_shapes and (
            output_height != self.metadata.height or output_width != self.metadata.width
        ):
            raise ValueError(
                f"program was exported with static size "
                f"{self.metadata.height}x{self.metadata.width}, but requested "
                f"{output_height}x{output_width}; use the exported height and "
                f"width or re-export with dynamic shapes"
            )
        if self.metadata.dynamic_shapes and not (
            self.metadata.min_size <= output_height <= self.metadata.max_size
            and self.metadata.min_size <= output_width <= self.metadata.max_size
        ):
            raise ValueError(
                f"height and width must be between {self.metadata.min_size} "
                f"and {self.metadata.max_size}"
            )
        if (
            output_height % self.metadata.vae_scale_factor
            or output_width % self.metadata.vae_scale_factor
        ):
            raise ValueError(
                f"height and width must be divisible by "
                f"{self.metadata.vae_scale_factor}"
            )
        if self.metadata.dynamic_shapes and (
            output_height % self.metadata.dynamic_spatial_multiple
            or output_width % self.metadata.dynamic_spatial_multiple
        ):
            raise ValueError(
                f"height and width must be divisible by "
                f"{self.metadata.dynamic_spatial_multiple} for this dynamic program"
            )
        return output_height, output_width

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        steps: int | None = None,
        cfg_scale: float | None = None,
        guidance_scale: float | None = None,
        seed: int = 0,
        image: torch.Tensor | None = None,
        strength: float = 0.8,
        height: int | None = None,
        width: int | None = None,
    ) -> torch.Tensor:
        steps = self.metadata.default_inference_steps if steps is None else steps
        cfg_scale = self.metadata.default_cfg_scale if cfg_scale is None else cfg_scale
        guidance_scale = (
            self.metadata.default_guidance_scale
            if guidance_scale is None
            else guidance_scale
        )
        if steps <= 0:
            raise ValueError("steps must be positive")
        if not math.isfinite(cfg_scale) or cfg_scale < 0:
            raise ValueError("cfg_scale must be finite and non-negative")
        if cfg_scale > 1 and not self.metadata.supports_cfg:
            raise ValueError("this program does not support CFG")
        if guidance_scale is not None and (
            not math.isfinite(guidance_scale) or guidance_scale < 0
        ):
            raise ValueError("guidance_scale must be finite and non-negative")
        if guidance_scale is not None and not self.metadata.accepts_guidance_input:
            raise ValueError("this program does not accept a guidance input")
        if not 0 < strength <= 1:
            raise ValueError("strength must be in (0, 1]")
        if negative_prompt is not None and not self.metadata.supports_negative_prompt:
            raise ValueError("this program does not support negative prompts")
        output_height, output_width = self._resolve_dimensions(height, width)
        use_cfg = self.metadata.supports_cfg and self.pipeline_spec.should_use_cfg(
            cfg_scale, negative_prompt
        )
        positive, negative = self.prepare_conditioning(prompt, negative_prompt, use_cfg)
        generator = torch.Generator().manual_seed(seed)

        denoise_metadata = self.program.metadata("denoise")
        latent_meta = denoise_metadata.input_tensor_meta(0)
        exported_latent_shape = tuple(latent_meta.sizes())
        timesteps = self.pipeline_spec.prepare_schedule(
            self.scheduler,
            steps,
            height=output_height,
            width=output_width,
        )
        dtype = getattr(torch, self.metadata.denoiser_dtype)
        vae_dtype = getattr(torch, self.metadata.vae_dtype)
        if image is None:
            latents = self.pipeline_spec.create_latents(
                exported_latent_shape,
                height=output_height,
                width=output_width,
                vae_scale_factor=self.metadata.vae_scale_factor,
                generator=generator,
                dtype=dtype,
                scheduler=self.scheduler,
            )
        else:
            image = _preprocess_image(image, output_height, output_width, vae_dtype)
            mean, std = _execute(self._method("encode"), (image,))
            image_latents = self.pipeline_spec.prepare_encoded_latents(
                mean,
                std,
                height=output_height,
                width=output_width,
                generator=generator,
            )
            latents, timesteps = self.pipeline_spec.prepare_img2img_latents(
                self.scheduler,
                image_latents,
                timesteps,
                steps=steps,
                strength=strength,
                generator=generator,
                dtype=dtype,
            )

        denoise = self._method("denoise")
        positive = _cast_floating_tensors(positive, dtype)
        if use_cfg:
            if negative is None:
                raise ValueError("CFG requires negative conditioning")
            negative = _cast_floating_tensors(negative, dtype)
            conditionings = (negative, positive)
        else:
            conditionings = (positive,)
        step_kwargs = (
            {"generator": generator}
            if "generator" in inspect.signature(self.scheduler.step).parameters
            else {}
        )
        for step_index, timestep in enumerate(timesteps, start=1):
            print(f"Step {step_index}/{len(timesteps)}")
            model_input = self.pipeline_spec.prepare_model_input(
                self.scheduler, latents, timestep
            )
            timestep_input = timestep.reshape(1).to(torch.float32)
            predictions = tuple(
                _execute(
                    denoise,
                    self.pipeline_spec.prepare_denoiser_inputs(
                        model_input,
                        timestep_input,
                        conditioning,
                        height=output_height,
                        width=output_width,
                        dtype=dtype,
                        guidance_scale=guidance_scale,
                    ),
                )[0]
                for conditioning in conditionings
            )
            prediction = self.pipeline_spec.combine_predictions(
                predictions, cfg_scale=cfg_scale
            )
            latents = self.scheduler.step(
                prediction,
                timestep,
                latents,
                return_dict=False,
                **step_kwargs,
            )[0]
        decode_latents = self.pipeline_spec.prepare_decode_latents(
            latents,
            height=output_height,
            width=output_width,
            vae_scale_factor=self.metadata.vae_scale_factor,
        )
        return _execute(self._method("decode"), (decode_latents,))[0]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a diffusion PTE")
    parser.add_argument("program", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt")
    parser.add_argument("--output", type=Path, default=Path("output.png"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--cfg-scale", type=float)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--revision")
    parser.add_argument("--scheduler")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    args = parser.parse_args(argv)

    from executorch.runtime import Runtime, Verification
    from PIL import Image

    program = Runtime.get().load_program(
        str(args.program), verification=Verification.Minimal
    )
    runner = DiffusionRunner.from_huggingface(
        program,
        args.model_id,
        revision=args.revision,
        scheduler_name=args.scheduler,
    )
    image = runner.generate(
        args.prompt,
        negative_prompt=args.negative_prompt,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        height=args.height,
        width=args.width,
    )
    pixels = image[0].permute(1, 2, 0).float().clamp(0, 1).mul(255).byte().cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
