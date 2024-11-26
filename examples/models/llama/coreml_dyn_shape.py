# Copied from https://apple.github.io/coremltools/docs-guides/source/stateful-models.html#example-toy-attention-model-with-stateful-kv-cache

import torch
import torch.nn as nn


class SimpleAttention(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        Q = self.query(x)  # (batch_size, seq_len, embed_size)
        K = self.key(x)  # (batch_size, seq_len, embed_size)
        V = self.value(x)  # (batch_size, seq_len, embed_size)
        return torch.nn.functional.scaled_dot_product_attention(Q, K, V)


class ToyModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = SimpleAttention(embed_size)
        self.fc = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        embedded = self.embedding(x)
        attention_output = self.attention(embedded)
        return self.fc(attention_output)


vocab_size = 32000
embed_size = 1024
batch_size = 1
seq_len = 5
max_seq_len = 1024
num_iterations = 100

import coremltools as ct
import numpy as np

torch_model = ToyModel(vocab_size, embed_size)
torch_model.eval()
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
torch_output = torch_model(input_ids).detach().numpy()
traced_model = torch.jit.trace(torch_model, [input_ids])

############################################################################################################
# Using
#   query_length = ct.RangeDim(lower_bound=1, upper_bound=max_seq_len, default=1)
# leads to mlpackage file that crashes on phone.  Changing it to static
#   query_length = 1
# leads to mlpackage file that runs on phone.
############################################################################################################
query_length = ct.RangeDim(lower_bound=1, upper_bound=max_seq_len, default=1)
# query_length = 1


inputs = [
    ct.TensorType(shape=(batch_size, query_length), dtype=np.int32, name="input_ids")
]
outputs = [ct.TensorType(dtype=np.float16, name="output")]

converted_model = ct.convert(
    traced_model,
    inputs=inputs,
    outputs=outputs,
    minimum_deployment_target=ct.target.iOS18,
    compute_units=ct.ComputeUnit.CPU_AND_GPU,
)

converted_model.save("/Users/scroy/Desktop/coreml.mlpackage")
