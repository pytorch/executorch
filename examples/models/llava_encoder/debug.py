from export import export_image_encoder, export_text_model, export_token_embedding

from model import LlavaModel
import torch


llava_model = LlavaModel()
llava = llava_model.get_eager_model()

llava = llava.to(torch.float32)

prompt_before_image, resized, prompt_after_image = llava_model.get_example_inputs()

embeddings = llava.prefill_embedding(prompt_before_image, resized, prompt_after_image)

llava_text_model = llava.text_model

ref = llava_text_model(embeddings, torch.tensor([0]))
