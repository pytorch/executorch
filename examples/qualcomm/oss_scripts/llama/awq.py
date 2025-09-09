import functools
from collections import defaultdict

import torch
import torch.nn as nn
from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import LlamaModel
from tqdm import tqdm


def fake_quantize_tensor(w, n_bit=4):
    # symmetric quantization
    max_val = w.abs().amax(dim=1, keepdim=True)
    max_val = max_val.clamp(min=1e-5)
    max_int = 2 ** (n_bit - 1) - 1
    min_int = -(2 ** (n_bit - 1) - 1)
    scales = max_val / max_int

    w = (torch.clamp(torch.round(w / scales), min_int, max_int)) * scales
    return w


def compute_and_apply_scale(prev_op, layers, inp, block, **kwargs):
    x = inp
    # w: co, ci
    # x: n, ci
    with torch.no_grad():
        org_out = block(x, **kwargs)
        if isinstance(org_out, tuple):
            org_out = org_out[0]

    x_max = x.abs().view(-1, x.shape[-1]).mean(0)

    best_error = float("inf")
    best_scales = None

    n_grid = 20

    org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
    for ratio in range(n_grid):
        ratio = ratio * 1 / n_grid
        scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
        scales = scales / (scales.max() * scales.min()).sqrt()
        for layer in layers:
            layer.weight.mul_(scales.view(1, -1))
            layer.weight.data = fake_quantize_tensor(layer.weight.data).detach() / (
                scales.view(1, -1)
            )

        out = block(x, **kwargs)
        if isinstance(out, tuple):
            out = out[0]

        loss = (org_out - out).float().pow(2).mean().item()  # float prevents overflow
        is_best = loss < best_error
        if is_best:
            best_error = loss
            best_scales = scales
        block.load_state_dict(org_sd)

    best_scales = best_scales.view(-1)

    # Apply scales to previous
    scales = best_scales.detach()
    if isinstance(prev_op, nn.Linear):
        prev_op.weight[-scales.size(0) :].div_(scales.view(-1, 1))
        if prev_op.bias is not None:
            prev_op.bias.div_(scales.view(-1))
    elif isinstance(prev_op, nn.RMSNorm):
        prev_op.weight.div_(scales)
        if hasattr(prev_op, "bias") and prev_op.bias is not None:
            prev_op.bias.div_(scales)
    # Apply scales to layers
    for layer in layers:
        layer.weight.mul_(scales.view(1, -1))


def apply_awq(model: LlamaModel, w_bit=4):
    """Implement AWQ from scratch... but in a way that is applicable to the Qualcomm LlamaModel definition."""
    with torch.no_grad():
        tokens, atten_mask = model.get_example_inputs(use_kv_cache=False)
        # tokens = get_dataset() will add this later...
        hidden_states = model.tok_embeddings(tokens)

        for i in tqdm(range(len(model.layers)), desc="Running AWQ..."):
            """Solving AWQ layer by layer. In each decoder layer, we apply scales at four points:
            attention_norm -- [q, k, v]
                         v -- o     *not implemented yet
                  ffn_norm -- [gate, up]
                        up -- down
            """
            decoder_layer = model.layers[i]
            named_linears = {
                name: m
                for name, m in decoder_layer.named_modules()
                if isinstance(m, nn.Linear)
            }

            def cache_input_hook(module, x, y, name, feat_dict):
                x = x[0]
                x = x.detach()
                feat_dict[name].append(x)

            input_feat = defaultdict(list)
            handles = []
            for name in named_linears:
                handles.append(
                    named_linears[name].register_forward_hook(
                        functools.partial(
                            cache_input_hook, name=name, feat_dict=input_feat
                        )
                    )
                )
            # get output as next layer's input
            kwargs = {
                "freqs_cos": model.freqs_cos,
                "freqs_sin": model.freqs_sin,
                "atten_mask": atten_mask,
                "k_caches": None,
                "v_caches": None,
            }
            hidden_states, _, _ = decoder_layer(hidden_states, **kwargs)
            for h in handles:
                h.remove()
            input_feat = {
                k: torch.cat(v, dim=0) for k, v in input_feat.items()
            }  # multi-input?

            compute_and_apply_scale(
                prev_op=decoder_layer.attention_norm,
                layers=[
                    decoder_layer.attention.wq,
                    decoder_layer.attention.wk,
                    decoder_layer.attention.wv,
                ],
                inp=input_feat["attention.wq"],
                block=decoder_layer.attention,
                **kwargs
            )
            # Need to add v--o. Technical difficulty due to GQA. But apparently this has the least impact anyway
            compute_and_apply_scale(
                prev_op=decoder_layer.ffn_norm,
                layers=[decoder_layer.feed_forward.w1, decoder_layer.feed_forward.w3],
                inp=input_feat["feed_forward.w1"],
                block=decoder_layer.feed_forward,
            )
            compute_and_apply_scale(
                prev_op=decoder_layer.feed_forward.w3,
                layers=[decoder_layer.feed_forward.w2],
                inp=input_feat["feed_forward.w2"],
                block=decoder_layer.feed_forward.w2,
            )
