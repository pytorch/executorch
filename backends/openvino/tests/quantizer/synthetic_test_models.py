import torch


class SimpleTransformer(torch.nn.Module):
    def __init__(self, vocab_size=100, hidden_size=64, num_layers=2):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, tokens, input_pos):
        x = self.embed(tokens)

        for layer in self.layers:
            x = torch.relu(layer(x))

        logits = self.lm_head(x)

        return logits
