import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline


class StableDiffusion:
    def __init__(self, seed=42):
        self.model_id: str = "stabilityai/stable-diffusion-2-1-base"
        self.generator = torch.manual_seed(seed)
        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            self.model_id, subfolder="scheduler"
        )

        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id, scheduler=self.scheduler, torch_dtype=torch.float32
        )
        self.pipe = self.pipe.to("cpu")

    def __call__(self, prompt, height, width, num_time_steps):
        image = self.pipe(
            prompt, height, width, num_time_steps, generator=self.generator
        ).images[0]
        return image
