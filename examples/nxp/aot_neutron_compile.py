# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script to compile the model for the NXP Neutron NPU

import argparse
import dataclasses
import io
import logging
import os
from collections.abc import Callable, Iterable
from functools import partial

import executorch.extension.pybindings.portable_lib  # noqa F401
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
from executorch.backends.nxp.nxp_backend import (
    core_aten_ops_exception_list,
    default_preserve_ops,
    generate_neutron_compile_spec,
)
from executorch.backends.nxp.quantizer.neutron_quantizer import NeutronQuantizer
from executorch.backends.nxp.quantizer.utils import calibrate_and_quantize
from executorch.devtools.visualization.visualization_utils import (
    visualize_with_clusters,
)
from executorch.examples.models import MODEL_NAME_TO_MODEL
from executorch.examples.models.model_factory import EagerModelFactory

from executorch.examples.nxp.experimental.cifar_net.cifar_net import (
    CifarNet,
    train_cifarnet_model,
    verify_cifarnet_model,
)
from executorch.examples.nxp.models.mlperf_tiny.anomaly_detection.mlperf_tiny_anomaly_detection import (
    MLPerfTinyAnomalyDetection,
)
from executorch.examples.nxp.models.mlperf_tiny.image_classification.mlperf_tiny_image_classification import (
    MLPerfTinyImageClassification,
)
from executorch.examples.nxp.models.mlperf_tiny.keyword_spotting.mlperf_tiny_keyword_spotting import (
    MLPerfTinyKeywordSpotting,
)
from executorch.examples.nxp.models.mlperf_tiny.visual_wake_words.mlperf_tiny_visual_wake_words import (
    MLPerfTinyVisualWakeWords,
)
from executorch.examples.nxp.models.mobilenet_v2 import MobileNetV2
from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    ExecutorchBackendConfig,
    ExecutorchProgramManager,
    to_edge_transform_and_lower,
)
from executorch.extension.export_util import save_pte_program

LOG_FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ExampleModel:
    """
    Attributes:
        model: Eager PyTorch model.
        example_inputs: Inputs used for `torch.export`.
        calibration_inputs: Inputs for post-training quantization. Example inputs are used if None.
        train_fn: `train_fn(module, num_epochs=..., channels_last=...)` used for QAT. QAT is unsupported if None.
        evaluate_fn: `evaluate_fn(module) -> accuracy` used by `--test`. Testing is unsupported if None.
    """

    model: torch.nn.Module
    example_inputs: tuple[torch.Tensor, ...]
    calibration_inputs: Iterable | None = None
    train_fn: Callable[..., torch.nn.Module] | None = None
    evaluate_fn: Callable[[torch.nn.Module], float] | None = None


ModelLoader = Callable[[argparse.Namespace], ExampleModel]


def _load_cifarnet(args: argparse.Namespace) -> ExampleModel:
    if args.use_random_dataset:
        raise NotImplementedError(
            f"Random dataset for model `{args.model_name}` is not implemented."
        )

    cifar_net = CifarNet()

    def _train_cifarnet(
        module: torch.nn.Module, num_epochs: int, channels_last: bool
    ) -> torch.nn.Module:
        return train_cifarnet_model(module, num_epochs=num_epochs)

    return ExampleModel(
        model=cifar_net.get_eager_model(),
        example_inputs=cifar_net.get_example_inputs(),
        calibration_inputs=cifar_net.get_calibration_inputs(
            args.calibration_batch_size
        ),
        train_fn=_train_cifarnet,
        evaluate_fn=verify_cifarnet_model,
    )


def _load_nxp_delegable_model(
    model_cls: type, args: argparse.Namespace
) -> ExampleModel:
    """Load an example model that is integrated into NXP's ExecuTorch backend and is fully delegable."""
    instance = model_cls(
        dataset_path=args.dataset_path,
        use_random_dataset=args.use_random_dataset,
        num_samples=args.num_random_samples,
        balanced_dataset=False,
    )
    return ExampleModel(
        model=instance.get_eager_model(),
        example_inputs=instance.get_example_inputs(),
        calibration_inputs=instance.get_calibration_inputs(args.calibration_batch_size),
        train_fn=getattr(instance, "train_model_fn", None),
    )


def _load_executorch_example_model(args: argparse.Namespace) -> ExampleModel:
    """Load a generic model from `executorch/examples/models/`."""
    logging.warning(
        "Using a model from examples/models, not all of these are currently supported."
    )
    model, example_inputs, _, _ = EagerModelFactory.create_model(
        *MODEL_NAME_TO_MODEL[args.model_name]
    )
    return ExampleModel(model=model, example_inputs=example_inputs)


# To add a new NXP delegable example model, register its loader here.
NXP_MODELS: dict[str, ModelLoader] = {
    "cifar10": _load_cifarnet,
    "mobilenetv2": partial(_load_nxp_delegable_model, MobileNetV2),
    "mlperf_tiny_anomaly_detection": partial(
        _load_nxp_delegable_model, MLPerfTinyAnomalyDetection
    ),
    "mlperf_tiny_image_classification": partial(
        _load_nxp_delegable_model, MLPerfTinyImageClassification
    ),
    "mlperf_tiny_keyword_spotting": partial(
        _load_nxp_delegable_model, MLPerfTinyKeywordSpotting
    ),
    "mlperf_tiny_visual_wake_words": partial(
        _load_nxp_delegable_model, MLPerfTinyVisualWakeWords
    ),
}


def load_example_model(args: argparse.Namespace) -> ExampleModel:
    """Load the model selected by `args.model_name` from the NXP or generic ExecuTorch examples."""
    if args.model_name in NXP_MODELS:
        return NXP_MODELS[args.model_name](args)

    if args.model_name in MODEL_NAME_TO_MODEL:
        return _load_executorch_example_model(args)

    raise RuntimeError(
        f"Model '{args.model_name}' is not a valid name. Use --help for a list of available models."
    )


def check_model_capabilities(example: ExampleModel, args: argparse.Namespace):
    """Fail early if the requested features are not supported by the selected model."""
    if args.use_qat and example.train_fn is None:
        raise ValueError(f"QAT training is not supported for model '{args.model_name}'")

    if args.test and example.evaluate_fn is None:
        raise NotImplementedError(
            f"Testing of model `{args.model_name}` is not yet supported."
        )


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def apply_dim_order(example: ExampleModel, channels_last: bool) -> ExampleModel:
    """Optionally convert the model and example inputs to the channels last dim order."""
    if not channels_last:
        print(
            "HINT: Converting your model to channels last may significantly improve inference speed. You can use the "
            "flag `--use_channels_last_dim_order`. See `docs/source/backends/nxp/nxp-dim-order.md` for more information."
        )
        return example

    # The dim order of the example inputs defines the dim order of the intermediate tensors in the model.
    # The channels last memory format is defined only for 4D tensors, other inputs keep their dim order.
    example_inputs = []
    for idx, example_input in enumerate(example.example_inputs):
        if isinstance(example_input, torch.Tensor) and example_input.dim() == 4:
            example_input = example_input.to(memory_format=torch.channels_last)
        else:
            logging.warning(
                f"Example input {idx} is not a 4D tensor, keeping its original dim order."
            )
        example_inputs.append(example_input)

    return dataclasses.replace(
        example,
        model=example.model.to(memory_format=torch.channels_last),
        example_inputs=tuple(example_inputs),
    )


def _quantize_qat(
    module: torch.nn.Module,
    quantizer: NeutronQuantizer,
    train_fn: Callable[..., torch.nn.Module],
    num_epochs: int,
    channels_last: bool,
) -> torch.nn.Module:
    # The observers are calibrated during the training, so no calibration inputs are needed.
    return calibrate_and_quantize(
        module,
        calibration_inputs=[],
        quantizer=quantizer,
        is_qat=True,
        train_fn=partial(train_fn, num_epochs=num_epochs, channels_last=channels_last),
    )


def _quantize_ptq(
    module: torch.nn.Module, quantizer: NeutronQuantizer, example: ExampleModel
) -> torch.nn.Module:
    calibration_inputs = example.calibration_inputs
    if calibration_inputs is None:
        logging.warning(
            "No calibration inputs available, using the example inputs instead"
        )
        calibration_inputs = [example.example_inputs]

    return calibrate_and_quantize(module, calibration_inputs, quantizer)


def quantize(
    module: torch.nn.Module,
    example: ExampleModel,
    target_spec: NeutronTargetSpec,
    args: argparse.Namespace,
) -> torch.nn.Module:
    """Quantize the exported module using either QAT or post-training quantization."""
    quantizer = NeutronQuantizer(target_spec, is_qat=args.use_qat)

    if args.use_qat:
        print(
            f"Starting {args.qat_num_epochs} epochs of QAT training with {args.model_name} model..."
        )
        return _quantize_qat(
            module,
            quantizer,
            example.train_fn,
            args.qat_num_epochs,
            args.use_channels_last_dim_order,
        )

    return _quantize_ptq(module, quantizer, example)


def evaluate(module: torch.nn.Module, example: ExampleModel, args: argparse.Namespace):
    """Compute and print the accuracy of the (optionally quantized) module."""
    accuracy = example.evaluate_fn(module)
    quantized_str = "quantized " if args.quantize else ""
    print(f"\nAccuracy of the {quantized_str}`{args.model_name}`: {accuracy}\n")


def _create_partitioners(
    module: torch.nn.Module,
    target_spec: NeutronTargetSpec,
    args: argparse.Namespace,
) -> list[NeutronPartitioner]:
    if not args.delegate:
        return []

    compile_spec = generate_neutron_compile_spec(
        args.target,
        operators_not_to_delegate=args.operators_not_to_delegate,
        fetch_constants_to_sram=args.fetch_constants_to_sram,
        dump_kernel_selection_code=args.dump_kernel_selection_code,
        use_profiling=args.use_profiling,
    )
    return [
        NeutronPartitioner(
            compile_spec,
            target_spec,
            post_quantization_state_dict=module.state_dict(),
            preserve_ops=default_preserve_ops,
        )
    ]


def lower_to_edge(
    module: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    target_spec: NeutronTargetSpec,
    args: argparse.Namespace,
) -> EdgeProgramManager:
    """Export the module to Edge dialect, optionally delegating to Neutron, and run the NXP edge passes."""
    edge_program_manager = to_edge_transform_and_lower(
        torch.export.export(module, example_inputs, strict=True),
        transform_passes=NeutronEdgePassManager(),
        generate_etrecord=args.use_profiling,
        partitioner=_create_partitioners(module, target_spec, args),
        compile_config=EdgeCompileConfig(
            _core_aten_ops_exception_list=core_aten_ops_exception_list,
        ),
    )

    if args.remove_quant_io_ops:
        edge_program_manager = edge_program_manager.transform(
            [RemoveIOQuantOpsPass(edge_program_manager=edge_program_manager)]
        )

    edge_program_manager = edge_program_manager.transform(
        NeutronEdgePassManager([RemoveAdditionalQDQClustersPass()])
    )

    logging.debug(f"Lowered graph:\n{edge_program_manager.exported_program().graph}")
    return edge_program_manager


def to_executorch(edge_program_manager: EdgeProgramManager) -> ExecutorchProgramManager:
    """Convert the Edge program to an ExecuTorch program."""
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
        raise

    logging.debug(f"Executorch program:\n{_executorch_program_to_str(exec_prog)}")
    return exec_prog


def _executorch_program_to_str(
    exec_prog: ExecutorchProgramManager, verbose: bool = False
) -> str:
    f = io.StringIO()
    exec_prog.dump_executorch_program(out=f, verbose=verbose)
    return f.getvalue()


# ---------------------------------------------------------------------------
# Output artifacts
# ---------------------------------------------------------------------------


def _output_name(args: argparse.Namespace) -> str:
    name = args.model_name
    if args.delegate:
        name += "_nxp_delegate"
    if args.use_profiling:
        name += "_profile"
    return name


def _save_etrecord(
    exec_prog: ExecutorchProgramManager, model_name: str, etrecord_dir: str
):
    etrecord_path = os.path.join(etrecord_dir, f"{model_name}_etrecord.bin")
    os.makedirs(etrecord_dir, exist_ok=True)
    exec_prog.get_etrecord().save(etrecord_path)

    logging.info(
        "The model was converted with profiling enabled. The time spent generating the profiling dump is traced as the "
        "final delegate operation and can be ignored, as no dump is produced for non-profilable models."
    )
    logging.info(f"The ETRecord for the model was saved to {etrecord_path}.")


def _visualize(exec_prog: ExecutorchProgramManager, mode: str, model_name: str):
    if mode == "show":
        visualize_with_clusters(exec_prog.exported_program())
    elif mode == "store":
        file_name = f"{model_name}-visualization.json"
        logging.info(
            f"Saved the graph visualization in `{file_name}`. It can be opened using the ModelExplorer."
        )
        visualize_with_clusters(exec_prog.exported_program(), file_name)


def save_artifacts(exec_prog: ExecutorchProgramManager, args: argparse.Namespace):
    """Store the `.pte` program and the optional ETRecord and visualization."""
    if args.use_profiling:
        _save_etrecord(exec_prog, args.model_name, args.etrecord_dir)

    save_pte_program(exec_prog, _output_name(args))

    if args.visualize is not None:
        _visualize(exec_prog, args.visualize, args.model_name)


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------


def _get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile a PyTorch model to an ExecuTorch program for the NXP eIQ Neutron NPU."
    )

    model_group = parser.add_argument_group("Model and data")
    model_group.add_argument(
        "-m",
        "--model_name",
        required=True,
        metavar="MODEL",
        choices=list(NXP_MODELS) + list(MODEL_NAME_TO_MODEL),
        help=f"Name of the model to compile. NXP models: {sorted(NXP_MODELS)}. NXP models can be fully delegated to Neutron NPU.",
    )
    model_group.add_argument(
        "-dst",
        "--dataset_path",
        default=None,
        help="Path to custom dataset archive (.pt, .xz) to use for calibration.",
    )
    model_group.add_argument(
        "--use_random_dataset",
        default=False,
        action="store_true",
        help="The calibration and testing datasets will be generated randomly instead of being downloaded.",
    )
    model_group.add_argument(
        "--num_random_samples",
        default=None,
        type=int,
        help="Number of random samples to generate, required when `--use_random_dataset` flag is set.",
    )
    model_group.add_argument(
        "--use_channels_last_dim_order",
        default=False,
        action="store_true",
        help="The model (including the Neutron backend) will use the channels last dim order, which can result in "
        "faster inference. The inputs must also be provided in the channels last dim order.",
    )

    quant_group = parser.add_argument_group("Quantization")
    quant_group.add_argument(
        "-q",
        "--quantize",
        default=False,
        action="store_true",
        help="Produce a quantized model",
    )
    quant_group.add_argument(
        "--use_qat",
        default=False,
        action="store_true",
        help="Use QAT mode for quantization (performs `--qat_num_epochs` QAT training epochs)",
    )
    quant_group.add_argument(
        "--qat_num_epochs",
        default=2,
        type=int,
        help="Number of QAT training epochs, used with `--use_qat`.",
    )
    quant_group.add_argument(
        "--calibration_batch_size",
        default=64,
        type=int,
        help="Batch size of the calibration data used for post-training quantization.",
    )

    lowering_group = parser.add_argument_group("Lowering")
    lowering_group.add_argument(
        "-d",
        "--delegate",
        action="store_true",
        default=False,
        help="Flag for producing eIQ NeutronBackend delegated model",
    )
    lowering_group.add_argument(
        "--target",
        default="imxrt700",
        help="Platform for running the delegated model",
    )
    lowering_group.add_argument(
        "--operators_not_to_delegate",
        default=[],
        type=str,
        nargs="*",
        help="List of operators not to delegate. E.g., --operators_not_to_delegate aten::convolution aten::mm",
    )
    lowering_group.add_argument(
        "-r",
        "--remove-quant-io-ops",
        action="store_true",
        default=False,
        help="Remove I/O De/Quantize nodes. Model will start to accept quantized "
        "inputs and produce quantized outputs.",
    )
    lowering_group.add_argument(
        "--fetch_constants_to_sram",
        action="store_true",
        default=False,
        help="This feature allows running models which do not fit into SRAM by offloading them to an external memory.",
    )
    lowering_group.add_argument(
        "--dump_kernel_selection_code",
        action="store_true",
        default=False,
        help="During compilation to Neutron microcode by Neutron Compiler, a kernel selection file will be dumped in "
        "the working directory. This file can be used for reduction of Neutron Firmware size in the built app. "
        "See `docs/source/backends/nxp/nxp-kernel-selection.md` for details.",
    )
    lowering_group.add_argument(
        "-s",
        "--so_library",
        default=None,
        help="Path to custom kernel library",
    )

    debug_group = parser.add_argument_group("Testing, profiling and debugging")
    debug_group.add_argument(
        "-t",
        "--test",
        action="store_true",
        default=False,
        help="Test the selected model and print the accuracy between 0 and 1.",
    )
    debug_group.add_argument(
        "--use_profiling",
        action="store_true",
        default=False,
        help="Enable profiling for eIQ Neutron NPU delegated model",
    )
    debug_group.add_argument(
        "--etrecord_dir",
        default="etrecord",
        help="Directory where the ETRecord is stored, used with `--use_profiling`.",
    )
    debug_group.add_argument(
        "--visualize",
        choices=["show", "store"],
        help="Visualize the lowered program. `show` launches a browser tab with the visualization. `store` stores the "
        "visualization in a json file for later inspection. See `docs/source/visualize-with-clusters.md` for details.",
    )
    debug_group.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Set the logging level to debug.",
    )

    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _get_arg_parser()
    args = parser.parse_args(argv)

    if args.use_qat and not args.quantize:
        parser.error("`--use_qat` requires `--quantize`.")

    if args.use_random_dataset and args.num_random_samples is None:
        parser.error("`--use_random_dataset` requires `--num_random_samples`.")

    if args.qat_num_epochs < 1:
        parser.error("`--qat_num_epochs` must be a positive integer.")

    if args.calibration_batch_size < 1:
        parser.error("`--calibration_batch_size` must be a positive integer.")

    return args


def run(args: argparse.Namespace):
    """Run the end-to-end AoT compilation flow for the Neutron backend."""
    if args.so_library is not None:
        logging.debug(f"Loading libraries: {args.so_library}")
        torch.ops.load_library(args.so_library)

    target_spec = NeutronTargetSpec(target=args.target)

    # 1. Load the model and check that it supports the requested features.
    example = load_example_model(args)
    check_model_capabilities(example, args)
    example.model.eval()
    example = apply_dim_order(example, args.use_channels_last_dim_order)

    # 2. Export the model to ATen.
    module = torch.export.export(
        example.model, example.example_inputs, strict=True
    ).module()

    # 3. Quantize and optionally test.
    if args.quantize:
        module = quantize(module, example, target_spec, args)

    if args.test:
        evaluate(module, example, args)

    # 4. Transform, lower and convert to an ExecuTorch program.
    edge_program_manager = lower_to_edge(
        module, example.example_inputs, target_spec, args
    )
    exec_prog = to_executorch(edge_program_manager)

    # 5. Serialize to *.pte and store the optional artifacts.
    save_artifacts(exec_prog, args)


def main(argv: list[str] | None = None):
    """Command line entry point. Configures the root logger and runs the flow."""
    args = _parse_args(argv)

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT, force=True)

    run(args)


if __name__ == "__main__":
    main()
