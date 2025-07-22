from transformers import VoxtralForConditionalGeneration, AutoProcessor
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
repo_id = "mistralai/Voxtral-Mini-3B-2507"

processor = AutoProcessor.from_pretrained(repo_id)
model = VoxtralForConditionalGeneration.from_pretrained(repo_id, torch_dtype=torch.bfloat16, device_map=device)

conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "url": "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/dude_where_is_my_car.wav",
            },
            # {"type": "text", "text": "What can you tell me about this audio?"},
        ],
    }
]

inputs = processor.apply_chat_template(conversation)
inputs = inputs.to(device, dtype=torch.bfloat16)

class VoxtralEncoderForExecuTorch(nn.Module):
    def __init__(self, audio_encoder: nn.Module, mm_projector: nn.Module, intermediate_size: int):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.mm_projector = mm_projector
        self.intermediate_size = intermediate_size

    def forward(self, input_features: torch.FloatTensor):
        audio_outputs = self.audio_encoder(input_features)
        audio_hidden_states = audio_outputs.last_hidden_state
        audio_hidden_states = audio_hidden_states.reshape(-1, self.intermediate_size)
        audio_embeds = self.mm_projector(audio_hidden_states)

        # TODO: add the below two lines after confirming equality with eager.
        # audio_token_mask = input_ids == self.config.audio_token_id
        # inputs_embeds[audio_token_mask] = audio_embeds
        
        return audio_embeds

voxtral_encoder = VoxtralEncoderForExecuTorch(
    model.audio_tower,
    model.multi_modal_projector,
    model.config.audio_config.intermediate_size,
)
expected_seq_length = model.audio_tower.config.max_source_positions * model.audio_tower.conv1.stride[0] * model.audio_tower.conv2.stride[0]  # From https://github.com/huggingface/transformers/blob/main/src/transformers/models/voxtral/modeling_voxtral.py#L342, should be equal to 3000.
sample_encoder_inputs = (torch.rand(1, 128, expected_seq_length),)  # Shape of input_features from sample Voxtral audio input from voxtral.md, but with batch size = 1 (representing < 30 seconds of audio). See https://github.com/huggingface/transformers/blob/fbeaf96f9e2291c21277ac658a33ea8752728bf3/src/transformers/models/voxtral/processing_voxtral.py#L91 for more info.
dynamic_shapes = {
    "input_features": {0: torch.export.Dim.STATIC, 1: torch.export.Dim.STATIC, 2: torch.export.Dim.STATIC}
}

ep = torch.export.export(
    voxtral_encoder,
    args=sample_encoder_inputs,
    dynamic_shapes=dynamic_shapes,
    strict=True,
)

eager_output = model.get_audio_embeds(sample_encoder_inputs[0])
ep_output = ep.module()(*sample_encoder_inputs)
torch.allclose(eager_output, ep_output)
        
# outputs = model.generate(**inputs, max_new_tokens=500)
# decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

# print("\nGenerated response:")
# print("=" * 80)
# print(decoded_outputs[0])
# print("=" * 80)
