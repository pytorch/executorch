# Copyright 2024-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script to compile the model for the NXP Neutron NPU

import argparse
import io
import logging
from collections import defaultdict

import executorch.extension.pybindings.portable_lib
import executorch.kernels.quantized  # noqa F401

import torch
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from executorch.backends.nxp.edge_passes.neutron_edge_pass_manager import (
    NeutronEdgePassManager,
)
from executorch.backends.nxp.edge_passes.remove_additional_quantize_dequantize_nodes_pass import (
    RemoveAdditionalQDQClustersPass,
)
from executorch.backends.nxp.edge_passes.remove_io_quant_ops_pass import (
    RemoveIOQuantOpsPass,
)
from executorch.backends.nxp.neutron_partitioner import NeutronPartitioner
from executorch.backends.nxp.nxp_backend import generate_neutron_compile_spec
from executorch.backends.nxp.quantizer.neutron_quantizer import NeutronQuantizer
from executorch.backends.nxp.quantizer.utils import post_training_quantize
from executorch.devtools.visualization.visualization_utils import (
    visualize_with_clusters,
)
from executorch.examples.models import MODEL_NAME_TO_MODEL
from executorch.examples.models.model_factory import EagerModelFactory
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.extension.export_util import save_pte_program
from torch.export import export

from .experimental.cifar_net.cifar_net import CifarNet, test_cifarnet_model
from .models.mobilenet_v2 import MobilenetV2

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def print_ops_in_edge_program(edge_program):
    """Find all ops used in the `edge_program` and print them out along with their occurrence counts."""

    ops_and_counts = defaultdict(
        lambda: 0
    )  # Mapping ops to the numer of times they are used.
    for node in edge_program.graph.nodes:
        if "call" not in node.op:
            continue  # `placeholder` or `output`. (not an operator)

        if hasattr(node.target, "_schema"):
            # Regular op.
            # noinspection PyProtectedMember
            op = node.target._schema.schema.name
        else:
            # Builtin function.
            op = str(node.target)

        ops_and_counts[op] += 1

    # Sort the ops based on how many times they are used in the model.
    ops_and_counts = sorted(ops_and_counts.items(), key=lambda x: x[1], reverse=True)

    # Print the ops and use counts.
    for op, count in ops_and_counts:
        print(f"{op: <50} {count}x")


def get_model_and_inputs_from_name(model_name: str):
    """Given the name of an example pytorch model, return it, example inputs and calibration inputs (can be None)

    Raises RuntimeError if there is no example model corresponding to the given name.
    """

    calibration_inputs = None
    # Case 1: Model is defined in this file
    if model_name in models.keys():
        m = models[model_name]()
        model = m.get_eager_model()
        example_inputs = m.get_example_inputs()
        calibration_inputs = m.get_calibration_inputs(64)
    # Case 2: Model is defined in executorch/examples/models/
    elif model_name in MODEL_NAME_TO_MODEL.keys():
        logging.warning(
            "Using a model from examples/models not all of these are currently supported"
        )
        model, example_inputs, _, _ = EagerModelFactory.create_model(
            *MODEL_NAME_TO_MODEL[model_name]
        )
    else:
        raise RuntimeError(
            f"Model '{model_name}' is not a valid name. Use --help for a list of available models."
        )

    return model, example_inputs, calibration_inputs


models = {
    "cifar10": CifarNet,
    "mobilenetv2": MobilenetV2,
}


if __name__ == "__main__":  # noqa C901
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help=f"Provide model name. Valid ones: {set(models.keys())}",
    )
    parser.add_argument(
        "-d",
        "--delegate",
        action="store_true",
        required=False,
        default=False,
        help="Flag for producing eIQ NeutronBackend delegated model",
    )
    parser.add_argument(
        "--target",
        required=False,
        default="imxrt700",
        help="Platform for running the delegated model",
    )
    parser.add_argument(
        "-c",
        "--neutron_converter_flavor",
        required=False,
        default="SDK_25_09",
        help="Flavor of installed neutron-converter module. Neutron-converter module named "
        "'neutron_converter_SDK_25_09' has flavor 'SDK_25_09'.",
    )
    parser.add_argument(
        "-q",
        "--quantize",
        action="store_true",
        required=False,
        default=False,
        help="Produce a quantized model",
    )
    parser.add_argument(
        "-s",
        "--so_library",
        required=False,
        default=None,
        help="Path to custome kernel library",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Set the logging level to debug."
    )
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        required=False,
        default=False,
        help="Test the selected model and print the accuracy between 0 and 1.",
    )
    parser.add_argument(
        "-r",
        "--remove-quant-io-ops",
        action="store_true",
        required=False,
        default=False,
        help="Remove I/O De/Quantize nodes. Model will start to accept quantized "
        "inputs and produce quantized outputs.",
    )
    parser.add_argument(
        "--operators_not_to_delegate",
        required=False,
        default=[],
        type=str,
        nargs="*",
        help="List of operators not to delegate. E.g., --operators_not_to_delegate aten::convolution aten::mm",
    )
    parser.add_argument(
        "--visualize",
        choices=["show", "store"],
        help="Visualize the lowered program. `show` launches a browser tab with the visualization. `store` stores the "
        "visualization in a json file for later inspection. See `docs/source/visualize-with-clusters.md` for details.",
    )

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format=FORMAT, force=True)

    neutron_target_spec = NeutronTargetSpec(
        target=args.target, neutron_converter_flavor=args.neutron_converter_flavor
    )

    # 1. pick model from one of the supported lists
    model, example_inputs, calibration_inputs = get_model_and_inputs_from_name(
        args.model_name
    )
    model = model.eval()

    # 2. Export the model to ATEN
    exported_program = torch.export.export(model, example_inputs, strict=True)

    module = exported_program.module()

    # 3. Quantize if required
    if args.quantize:
        if calibration_inputs is None:
            logging.warning(
                "No calibration inputs available, using the example inputs instead"
            )
            calibration_inputs = example_inputs
        quantizer = NeutronQuantizer(neutron_target_spec)
        module = post_training_quantize(module, calibration_inputs, quantizer)

    if args.so_library is not None:
        logging.debug(f"Loading libraries: {args.so_library}")
        torch.ops.load_library(args.so_library)

    if args.test:
        match args.model_name:
            case "cifar10":
                accuracy = test_cifarnet_model(module)

            case _:
                raise NotImplementedError(
                    f"Testing of model `{args.model_name}` is not yet supported."
                )

        quantized_str = "quantized " if args.quantize else ""
        print(f"\nAccuracy of the {quantized_str}`{args.model_name}`: {accuracy}\n")

    # 4. Transform and lower

    compile_spec = generate_neutron_compile_spec(
        args.target,
        operators_not_to_delegate=args.operators_not_to_delegate,
        neutron_converter_flavor=args.neutron_converter_flavor,
    )
    partitioners = (
        [NeutronPartitioner(compile_spec, neutron_target_spec)] if args.delegate else []
    )

    edge_program_manager = to_edge_transform_and_lower(
        export(module, example_inputs, strict=True),
        transform_passes=NeutronEdgePassManager(),
        partitioner=partitioners,
        compile_config=EdgeCompileConfig(),
    )

    if args.remove_quant_io_ops:
        edge_program_manager = edge_program_manager.transform(
            [RemoveIOQuantOpsPass(edge_program_manager=edge_program_manager)]
        )

    edge_program_manager = edge_program_manager.transform(
        NeutronEdgePassManager([RemoveAdditionalQDQClustersPass()])
    )

    logging.debug(f"Lowered graph:\n{edge_program_manager.exported_program().graph}")

    # 5. Export to ExecuTorch program
    try:
        exec_prog = edge_program_manager.to_executorch(
            config=ExecutorchBackendConfig(extract_delegate_segments=False)
        )
    except RuntimeError as e:
        if "Missing out variants" in str(e.args[0]):
            raise RuntimeError(
                e.args[0]
                + ".\nThis likely due to an external so library not being loaded. Supply a path to it with the "
                "--so_library flag."
            ).with_traceback(e.__traceback__) from None
        else:
            raise e

    def executorch_program_to_str(ep, verbose=False):
        f = io.StringIO()
        ep.dump_executorch_program(out=f, verbose=verbose)
        return f.getvalue()

    logging.debug(f"Executorch program:\n{executorch_program_to_str(exec_prog)}")

    # 6. Serialize to *.pte
    model_name = f"{args.model_name}" + (
        "_nxp_delegate" if args.delegate is True else ""
    )
    save_pte_program(exec_prog, model_name)

    # 7. Optionally visualize the model.
    if args.visualize == "show":
        visualize_with_clusters(exec_prog.exported_program())
    elif args.visualize == "store":
        file_name = f"{args.model_name}-visualization.json"
        logging.info(
            f"Saved the graph visualization in `{file_name}`. It can be opened using the ModelExplorer."
        )
        visualize_with_clusters(exec_prog.exported_program(), file_name)
