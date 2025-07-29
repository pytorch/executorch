from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.devtools import BundledProgram, generate_etrecord, Inspector
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
from executorch.exir.program import EdgeProgramManager
from executorch.extension.pybindings.portable_lib import _load_for_executorch, _load_for_executorch_from_buffer

import pandas as pd
from transformers import VoxtralForConditionalGeneration, AutoProcessor
import torch
from torch import nn

import os


WORKING_DIR = "/Users/jackzhxng/Documents/voxtral"


class VoxtralEncoderForExecuTorch(nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.audio_encoder = model.audio_tower
        self.mm_projector = model.multi_modal_projector
        self.intermediate_size = model.config.audio_config.intermediate_size
        self.audio_token_id = model.config.audio_token_id

    def forward(
        self,
        input_features: torch.FloatTensor,
    ):
        audio_outputs = self.audio_encoder(input_features)
        audio_hidden_states = audio_outputs.last_hidden_state
        audio_hidden_states = audio_hidden_states.reshape(-1, self.intermediate_size)
        audio_embeds = self.mm_projector(audio_hidden_states)

        return audio_embeds

device = "cuda" if torch.cuda.is_available() else "cpu"
repo_id = "mistralai/Voxtral-Mini-3B-2507"

processor = AutoProcessor.from_pretrained(repo_id)
model = VoxtralForConditionalGeneration.from_pretrained(repo_id, device_map=device)
voxtral_encoder = VoxtralEncoderForExecuTorch(model)

conversation = [
    {
        "role": "user",
        "content": [
            # {
            #     "type": "audio",
            #     "url": "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/dude_where_is_my_car.wav",
            # },
            {"type": "text", "text": "What can you tell me about this audio?"},
        ],
    }
]
inputs = processor.apply_chat_template(conversation)
# outputs = model.generate(**inputs, max_new_tokens=500)

expected_seq_length = model.audio_tower.config.max_source_positions * model.audio_tower.conv1.stride[0] * model.audio_tower.conv2.stride[0]  # From https://github.com/huggingface/transformers/blob/main/src/transformers/models/voxtral/modeling_voxtral.py#L342, should be equal to 3000.
sample_input_features = torch.rand(3, 128, expected_seq_length, dtype=torch.float32)  # Shape of input_features from sample Voxtral audio input from voxtral.md, but with batch size = 1 (representing < 30 seconds of audio). See https://github.com/huggingface/transformers/blob/fbeaf96f9e2291c21277ac658a33ea8752728bf3/src/transformers/models/voxtral/processing_voxtral.py#L91 for more info.
encoder_input_kwargs = {
    "input_features": sample_input_features,  # (bsz, features, seq_len)
}

max_audio_len = 150  # In s, should be a multiple of 30.
max_seq_len = 2048
dynamic_shapes = {
    "input_features": {
        0: torch.export.Dim("enc_batch_size_dim", min=0, max=max_audio_len//30),
    },
}

with torch.no_grad():
    ep = torch.export.export(
        voxtral_encoder,
        args=(),
        kwargs=encoder_input_kwargs,
        dynamic_shapes=dynamic_shapes,
        strict=True,
    )

et_prog = to_edge_transform_and_lower(
    ep,
    partitioner=[XnnpackPartitioner()],
    compile_config=EdgeCompileConfig(
        _check_ir_validity=False,
        _skip_dim_order=True,
    ),
    # constant_methods=metadata,
    # transform_passes=[RemovePaddingIdxEmbeddingPass()],
).to_executorch()

# Generate etrecord using synthetically created edge program manager which just wraps ExportedProgram.
edge_manager_for_etrecord = EdgeProgramManager(ep)
inputs = [
    (sample_input_features,),
]
method_test_suites = [
    MethodTestSuite(
        method_name="forward",
        test_cases=[
            MethodTestCase(inputs=input, expected_outputs=model.get_audio_embeds(sample_input_features))
            for input in inputs
        ],
    )
]
bundled_program = BundledProgram(et_prog, method_test_suites)
etrecord_path = os.path.join(WORKING_DIR, "etrecord.bin")
generate_etrecord(etrecord_path, edge_manager_for_etrecord, bundled_program)

et_mod = _load_for_executorch_from_buffer(
    et_prog.buffer,
    enable_etdump=True,
    debug_buffer_size=1024 * 1024 * 1024,
)

# Run eager for baseline results
eager_output = model.get_audio_embeds(sample_input_features)

# Run exported program.
ep_output = ep.module().forward(sample_input_features)

# Run executorch program while also generating etdump for debugging.
etdump_path = os.path.join(WORKING_DIR, "etdump.etdp")
debug_buffer_path = os.path.join(WORKING_DIR, "debug_buffer.bin")
et_output = et_mod.run_method("forward", (sample_input_features,))
et_mod.write_etdump_result_to_file(etdump_path, debug_buffer_path)

def print_debug_graph(df: pd.DataFrame):
    max_gap = max([val for sublist in df['gap'] for val in sublist])

    df['aot_intermediate_output'] = df['aot_intermediate_output'].astype(str).str[:24] + '...'
    df['runtime_intermediate_output'] = df['runtime_intermediate_output'].astype(str).str[:24] + '...'
    def format_scientific(x, digits=3):
        return f"{x:.{digits}e}"
    df['gap'] = df['gap'].apply(lambda lst: [format_scientific(val, 3) for val in lst])
    with pd.option_context(
        "display.width",
        1000000,
        "display.max_columns",
        None,
        "display.colheader_justify",
        "left",
    ):
        print(df)

    print("Max gap:", max_gap)

inspector = Inspector(
    etdump_path=etdump_path,
    etrecord=etrecord_path,
    debug_buffer_path=debug_buffer_path,
)
pd.set_option("display.width", 100000)
pd.set_option("display.max_columns", None)
df = inspector.calculate_numeric_gap("MSE")
print_debug_graph(df)

breakpoint()

torch.allclose(eager_output, ep_output)
torch.allclose(ep_output, et_output)
