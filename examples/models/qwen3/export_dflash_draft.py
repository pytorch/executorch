import argparse
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from executorch.backends.mlx.examples.llm.dflash_draft_model import DFlashDraftModel, load_dflash_config


def load_draft_model(draft_id: str, target_state_dict: dict) -> DFlashDraftModel:
    path = Path(snapshot_download(draft_id, allow_patterns=["*.safetensors", "*.json"]))
    config = load_dflash_config(path)
    model = DFlashDraftModel(config)

    draft_weights = {}
    for f in path.glob("*.safetensors"):
        draft_weights.update(load_file(str(f)))

    missing, unexpected = model.load_state_dict(draft_weights, strict=False)
    assert not unexpected, f"Unexpected draft checkpoint keys: {unexpected}"
    still_missing = [k for k in missing if not k.startswith(("embed_tokens.", "lm_head."))]
    assert not still_missing, f"Missing draft checkpoint keys: {still_missing}"

    model.embed_tokens.weight.data.copy_(target_state_dict["model.embed_tokens.weight"])
    lm_head_key = "lm_head.weight" if "lm_head.weight" in target_state_dict else "model.embed_tokens.weight"
    model.lm_head.weight.data.copy_(target_state_dict[lm_head_key])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft-model", default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--output", default="qwen3_4b_dflash_draft.pte")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--ctx-len", type=int, default=8)
    args = parser.parse_args()

    target = AutoModelForCausalLM.from_pretrained(args.target_model, dtype="auto")
    model = load_draft_model(args.draft_model, target.state_dict())
    model.eval()
    model = model.float()
    del target

    block_size, ctx_len = args.block_size, args.ctx_len
    hidden_size = model.fc.in_features
    tokens = torch.randint(0, 1000, (1, block_size), dtype=torch.long)
    target_hidden = torch.randn(1, ctx_len, hidden_size)
    position_ids = torch.arange(ctx_len + block_size).unsqueeze(0)

    exported = torch.export.export(model, (tokens, target_hidden, position_ids))

    from executorch.exir import to_edge_transform_and_lower
    from executorch.backends.mlx.partitioner import MLXPartitioner

    edge = to_edge_transform_and_lower(exported, partitioner=[MLXPartitioner()])
    et_program = edge.to_executorch()

    with open(args.output, "wb") as f:
        f.write(et_program.buffer)
    print(f"Saved draft model to: {args.output}")


if __name__ == "__main__":
    main()
