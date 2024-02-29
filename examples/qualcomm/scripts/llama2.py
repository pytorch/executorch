import argparse
import os
from pathlib import Path

import torch
from executorch.examples.models.model_factory import EagerModelFactory
from executorch.examples.qualcomm.scripts.utils import build_executorch_binary


class ConvAsLinear(torch.nn.Module):
    """
    A Module such that we can use a Conv1d module instead of Linear module.
    """

    def __init__(self, linear_module):
        super().__init__()
        in_features = linear_module.in_features
        out_features = linear_module.out_features
        # Create a conv1d module based on the linear module
        self.conv1d_module = torch.nn.Conv1d(in_features, out_features, kernel_size=1)
        # Copy the weights and bias over
        self.conv1d_module.weight.data = linear_module.weight.data.unsqueeze(-1)
        if linear_module.bias is not None:
            self.conv1d_module.bias.data = linear_module.bias.data
        else:
            self.conv1d_module.bias = None

    def forward(self, arg):
        # Permute the input to match the conv1d
        reshape_input = arg.permute(
            0, 2, 1
        )  # [batch_size, in_features, sequence_length]
        output_conv1d = self.conv1d_module(reshape_input)
        # Permute the output before returning the output from conv1d
        reshape_output = output_conv1d.permute(
            0, 2, 1
        )  # [batch_size, sequence_length, out_features]
        return reshape_output


def swap_linear_module_with_conv1d_inplace(
    model: torch.nn.Module,
):
    """
    Swap all Linear modules with Conv1d(kernel_size=1) modules in-place.
    """
    if type(model) is torch.nn.Linear:
        model = ConvAsLinear(model)

    for name, module in model.named_children():
        module = swap_linear_module_with_conv1d_inplace(module)
        setattr(model, name, module)
    return model


def main(checkpoint_path: str, params_path: str):
    # QNN_SDK_ROOT might also be an argument, but it is used in various places.
    # So maybe it's fine to just use the environment.
    if "QNN_SDK_ROOT" not in os.environ:
        raise RuntimeError("Environment variable QNN_SDK_ROOT must be set")
    print(f"QNN_SDK_ROOT={os.getenv('QNN_SDK_ROOT')}")

    if "LD_LIBRARY_PATH" not in os.environ:
        print(
            "[Warning] LD_LIBRARY_PATH is not set. If errors like libQnnHtp.so "
            "not found happen, please follow setup.md to set environment."
        )
    else:
        print(f"LD_LIBRARY_PATH={os.getenv('LD_LIBRARY_PATH')}")

    model, example_inputs, _ = EagerModelFactory.create_model(
        "llama2",
        "Llama2Model",
        checkpoint=checkpoint_path,
        params=params_path,
        use_kv_cache=False,#args.use_kv_cache,
        fairseq2=False,#args.fairseq2,
    )
    print("linear model")
    print(model)
    model = swap_linear_module_with_conv1d_inplace(model)

    print("conv model")
    print(model)
    out_conv_model = model(*example_inputs)
    print("conv model output: ", out_conv_model)

    pte_filename = "llama2_qnn"
    build_executorch_binary(
       model,
       example_inputs,
       soc_model="SM8450",
       file_name=f"artifact/{pte_filename}",
       dataset=[example_inputs],
       skip_node_op_set={
           "aten.view_copy.default",
           "aten.squeeze_copy.dims",
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ckpt_dir = f"{Path(__file__).absolute().parents[2].as_posix()}"

    parser.add_argument(
        "-c",
        "--checkpoint",
        default=f"{ckpt_dir}/models/llama2/params/demo_rand_params.pth",
        help="checkpoint path",
    )
    parser.add_argument(
        "-p",
        "--params",
        default=f"{ckpt_dir}/models/llama2/params/demo_config.json",
        help="config.json",
    )
    args = parser.parse_args()
    main(args.checkpoint, args.params)
