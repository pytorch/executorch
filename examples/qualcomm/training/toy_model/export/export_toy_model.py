# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export a 4-layer toy model as a joint (forward+backward) QNN/HTP training graph.

For weight_sharing=false (default): compiles forward and forward_backward into
separate .pte files (<name>_fwd.pte and <name>_fwd_bwd.pte). The forward .pte
uses the functional_call trick so weights are USER_INPUTs -- trained weights can
be fed at inference time without re-exporting.

For weight_sharing=true: compiles both methods into a single .pte (one shared
compile context). Requires real Hexagon/HTP hardware.

Methods:
  - "forward": inference graph. Weights are USER_INPUTs via functional_call --
    feed trained_model.safetensors at inference time, no re-export needed.
  - "forward_backward": the joint graph. Weights are plain USER_INPUTs, fed
    in and read back out (as gradients) by the host training loop each step.

No checkpointing, no quantization, no pretrained weights -- everything here
is randomly initialized and self-contained.
"""

import argparse
import json
import logging
import os

import torch
import torch.nn as nn

from config import load_config

from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    get_soc_to_chipset_map,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.extension.export_util.utils import save_pte_program
from joint_graph import JointGraphCreator
from model import ToyModel, ToyModelWithLoss
from safetensors.torch import save_file


def _make_fwd_wrapper(toy_model: ToyModel) -> nn.Module:
    """Return an nn.Module whose forward(w1, w2, w3, w4, x) runs toy_model via
    functional_call so the exported graph has weights as USER_INPUTs, not baked
    constants. The wrapper holds no parameters of its own."""
    param_names = list(dict(toy_model.named_parameters()).keys())

    class _FwdWrapper(nn.Module):
        def forward(self, *args):
            weights = args[: len(param_names)]
            x = args[len(param_names)]
            params = dict(zip(param_names, weights))
            return torch.func.functional_call(toy_model, params, (x,))

    return _FwdWrapper()


def _compile_and_save(
    fwd_wrapper,
    joint_graph,
    fwd_inputs,
    joint_inputs,
    soc_model,
    output_folder,
    use_weight_sharing,
    pte_name,
):
    backend_options = generate_htp_compiler_spec(
        use_fp16=True,
        use_weight_sharing=use_weight_sharing,
    )
    compile_specs = generate_qnn_executorch_compiler_spec(
        soc_model=get_soc_to_chipset_map()[soc_model],
        backend_options=backend_options,
    )

    if use_weight_sharing:
        # Weight-sharing requires both methods in one compile call.
        edge_program = to_edge_transform_and_lower_to_qnn(
            module={"forward": fwd_wrapper, "forward_backward": joint_graph},
            inputs={"forward": fwd_inputs, "forward_backward": joint_inputs},
            compiler_specs=compile_specs,
        )
        save_pte_program(edge_program.to_executorch(), pte_name, output_folder)
        return {"pte_name": f"{pte_name}.pte"}
    else:
        # Compile forward and forward_backward into separate .pte files so
        # each can be loaded independently on-device.
        fwd_name = f"{pte_name}_fwd"
        fwd_bwd_name = f"{pte_name}_fwd_bwd"

        fwd_program = to_edge_transform_and_lower_to_qnn(
            module={"forward": fwd_wrapper},
            inputs={"forward": fwd_inputs},
            compiler_specs=compile_specs,
        )
        save_pte_program(fwd_program.to_executorch(), fwd_name, output_folder)

        fwd_bwd_program = to_edge_transform_and_lower_to_qnn(
            module={"forward_backward": joint_graph},
            inputs={"forward_backward": joint_inputs},
            compiler_specs=compile_specs,
        )
        save_pte_program(fwd_bwd_program.to_executorch(), fwd_bwd_name, output_folder)
        return {
            "pte_name": f"{fwd_bwd_name}.pte",
            "fwd_pte_name": f"{fwd_name}.pte",
        }


def export_toy_model(
    toy_model, x, target, soc_model, output_folder, weight_sharing, pte_name
):
    train_module = ToyModelWithLoss(toy_model)
    creator = JointGraphCreator(train_module, (x, target))
    joint_graph = creator.create()

    placeholder_names = {
        n.name for n in joint_graph.graph.nodes if n.op == "placeholder"
    }
    params_by_name = dict(train_module.named_parameters())
    param_names = [
        name
        for i, name in enumerate(creator.joint_desc.params_spec or [])
        if f"primals_{i + 1}" in placeholder_names
    ]
    weights = [params_by_name[name] for name in param_names]

    # aot_export_joint_with_descriptors's returned GraphModule takes exactly
    # two positional args -- forward(primals, tangents) -- each a flat list,
    # not one positional arg per primals_N/tangents_N placeholder node. This
    # grouped shape is what torch.export.export (called internally by
    # to_edge_transform_and_lower_to_qnn) must be given to retrace it.
    tangent = torch.ones([1])
    joint_inputs = ((*weights, x, target), (tangent,))

    # functional_call wrapper: same ToyModel computation, weights as USER_INPUTs.
    fwd_wrapper = _make_fwd_wrapper(toy_model)
    fwd_inputs = (*weights, x)

    pte_info = _compile_and_save(
        fwd_wrapper,
        joint_graph,
        fwd_inputs,
        joint_inputs,
        soc_model,
        output_folder,
        weight_sharing,
        pte_name,
    )

    save_file(
        {name: w.contiguous() for name, w in zip(param_names, weights)},
        os.path.join(output_folder, "init_weights.safetensors"),
    )

    grad_names = [f"grad_{name}" for name in param_names]
    manifest = {
        "param_names": param_names,
        "inputs": [*param_names, "x", "target", "tangent"],
        "outputs": ["loss", "pred", *grad_names],
        "x_shape": list(x.shape),
        "target_shape": list(target.shape),
        "weight_sharing": weight_sharing,
        **pte_info,
    }
    with open(os.path.join(output_folder, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    logging.basicConfig(level=logging.INFO)
    toy_model_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(
        description="Export a 4-layer toy model as joint QNN/HTP training .pte's",
    )
    parser.add_argument(
        "--config", type=str, default=os.path.join(toy_model_dir, "config.json")
    )
    config_args, _ = parser.parse_known_args()
    config = load_config(config_args.config)

    parser.add_argument(
        "--soc_model", type=str, default=config.get("soc_model", "SM8850")
    )
    parser.add_argument(
        "-f", "--output_folder", type=str, default=config.get("output_dir", "")
    )
    parser.add_argument("--seed", type=int, default=config.get("seed"))
    parser.add_argument("--batch_size", type=int, default=config.get("batch_size", 4))
    parser.add_argument("--in_dim", type=int, default=config.get("in_dim", 8))
    parser.add_argument("--hidden_dim", type=int, default=config.get("hidden_dim", 32))
    parser.add_argument("--out_dim", type=int, default=config.get("out_dim", 4))
    parser.add_argument(
        "--weight_sharing",
        action=argparse.BooleanOptionalAction,
        default=config.get("weight_sharing", False),
    )
    parser.add_argument("--name", type=str, default=config.get("name", "toy_model"))
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    output_folder = args.output_folder
    if output_folder:
        if not os.path.isabs(output_folder):
            output_folder = os.path.join(toy_model_dir, output_folder)
        output_folder = os.path.join(
            output_folder, "ws" if args.weight_sharing else "no_ws"
        )
        os.makedirs(output_folder, exist_ok=True)

    toy_model = ToyModel(
        in_dim=args.in_dim, hidden_dim=args.hidden_dim, out_dim=args.out_dim
    )
    x = torch.randn(args.batch_size, args.in_dim)
    target = torch.randn(args.batch_size, args.out_dim)

    export_toy_model(
        toy_model,
        x,
        target,
        args.soc_model,
        output_folder,
        args.weight_sharing,
        args.name,
    )


if __name__ == "__main__":
    main()
