# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import inspect
import logging
import os.path
import shutil
import subprocess
from os import mkdir

import numpy as np
import torch
from pytest_mock import MockerFixture
from torch.export import ExportedProgram

from executorch.backends.nxp.tests_models.dataset_creator import RandomDatasetCreator
from executorch.backends.nxp.tests_models.graph_verifier import GraphVerifier
from executorch.backends.nxp.tests_models.model_input_spec import ModelInputSpec
from executorch.backends.nxp.tests_models.model_output_comparator import AllCloseOutputComparator
from executorch.backends.nxp.tests_models.utils import to_quantized_executorch_program, save_pte_program
from executorch.backends.nxp.backend.edge_helper import is_channels_last_dim_order
from executorch.backends.nxp.backend.ir.converter.conversion import translator
from executorch.backends.nxp.neutron_partitioner import NeutronPartitioner
from executorch.devtools.visualization.visualization_utils import visualize_with_clusters

from executorch.backends.nxp.tests_models.config_importer import test_config
from executorch.backends.nxp.tests_models.outputs_dir_importer import outputs_dir

logger = logging.getLogger(__name__)

OUTPUTS_DIR = outputs_dir.OUTPUTS_DIR
NSYS_PATH = test_config.NSYS_PATH
NSYS_CONFIG_PATH = test_config.NSYS_CONFIG_PATH
NSYS_FIRMWARE_PATH = test_config.NSYS_FIRMWARE_PATH
NEUTRON_TEST_PATH = test_config.NEUTRON_TEST_PATH

def _run_delegated_executorch_program(model, test_dir, test_name, calibration_dataset_dir, testing_dataset_dir,
                                      input_spec, dlg_model_verifier, npu_results_dir, mocker,
                                      use_qat: bool = False) -> ExportedProgram:
    if len(input_spec) == 1:
        # Single input, use --dataset
        dataset_cli = "--dataset"
        dataset_or_inputs = testing_dataset_dir
    else:
        # Multiple input, use --inputs with subdirectories
        dataset_cli = "--inputs"
        dataset_or_inputs = ",".join(sorted([os.path.join(testing_dataset_dir, d) for d in os.listdir(testing_dataset_dir)])
        )

    # Run nxp_executor_runner with program delegated to NPU
    delegated_model_path = os.path.abspath(os.path.join(test_dir, f'{test_name}_delegated.pte'))

    delegated_cmd = f"{NEUTRON_TEST_PATH} --model {delegated_model_path} {dataset_cli} {dataset_or_inputs} \
        --output {npu_results_dir} --firmware {NSYS_FIRMWARE_PATH} --nsys {NSYS_PATH} --nsys_config {NSYS_CONFIG_PATH}"
    try:
        if mocker:
            method = getattr(NeutronPartitioner, "partition")

            def wrapper(*args, **kwargs):
                result = method(*args, **kwargs)
                visualize_with_clusters(result.tagged_exported_program,
                                        os.path.join(test_dir, test_name + "_partitioned.json"),
                                        False)
                return result

            wrapped = functools.update_wrapper(wrapper, method)
            mocker.patch.object(NeutronPartitioner, "partition", side_effect=wrapped, autospec=True)
        delegated_program = to_quantized_executorch_program(
            model, input_spec, calibration_dataset_dir, delegate_to_npu=True, use_qat=use_qat
        )
    except RuntimeError as e:
        if "Model converted with neutron-converter has" in str(e):
            dlg_model_verifier.check_num_delegated_nodes(e.args[1])
        raise

    exported_program = delegated_program.exported_program()
    nodes = list(exported_program.graph.nodes)
    assert any([node.name.startswith("executorch_call_delegate") for node in
                nodes]), "No delegated parts found in program delegated to NPU!"
    dlg_model_verifier.verify_graph(exported_program.graph)

    save_pte_program(delegated_program, test_name + "_delegated", test_dir)
    execute_cmd(delegated_cmd)

    return exported_program


def _run_non_delegated_executorch_program(model, test_dir, test_name, calibration_dataset_dir, testing_dataset_dir,
                                          input_spec,
                                          cpu_results_dir) -> ExportedProgram:
    if len(input_spec) == 1:
        # Single input, use --dataset
        dataset_cli = "--dataset"
        dataset_or_inputs = testing_dataset_dir
    else:
        # Multiple input, use --inputs with subdirectories
        dataset_cli = "--inputs"
        dataset_or_inputs = ",".join(sorted([os.path.join(testing_dataset_dir, d) for d in os.listdir(testing_dataset_dir)])
        )

    # Run program via nxp_executor_runner on CPU
    non_delegated_model_path = os.path.abspath(os.path.join(test_dir, f'{test_name}_non_delegated.pte'))

    non_delegated_cmd = f"{NEUTRON_TEST_PATH} --model {non_delegated_model_path} {dataset_cli} {dataset_or_inputs} \
        --output {cpu_results_dir} --firmware {NSYS_FIRMWARE_PATH} --nsys {NSYS_PATH} --nsys_config {NSYS_CONFIG_PATH}"

    non_delegated_program = to_quantized_executorch_program(
        model, input_spec, calibration_dataset_dir, delegate_to_npu=False
    )

    nodes = list(non_delegated_program.exported_program().graph.nodes)
    assert all([not node.name.startswith("executorch_call_delegate") for node in
                nodes]), "Delegated parts found in program executed on CPU!"

    save_pte_program(non_delegated_program, test_name + "_non_delegated", test_dir)
    execute_cmd(non_delegated_cmd)

    return non_delegated_program.exported_program()


def read_prepared_samples(dataset_dir: str, input_spec: list[ModelInputSpec]) -> list[
    tuple[np.ndarray, ...]]:
    """Read numpy arrays generated by a `DatasetCreator`.

    :param dataset_dir: Directory containing the generated samples
    :param input_spec: List of ModelInputSpec defining the shape and type of each input

    :return:  List of tuples, where each tuple contains numpy arrays for one sample
    """
    all_samples = []

    # Multi-input: samples are in numbered subdirectories
    if len(input_spec) > 1:
        sample_dirs = sorted([d for d in os.listdir(dataset_dir)
                              if os.path.isdir(os.path.join(dataset_dir, d))])

        for sample_name in sample_dirs:
            sample_dir = os.path.join(dataset_dir, sample_name)
            current_samples = []

            for spec_idx, spec in enumerate(input_spec):
                bin_file_path = os.path.join(sample_dir, f"{str(spec_idx).zfill(2)}.bin")
                sample_vector = np.fromfile(bin_file_path, dtype=spec.type).reshape(spec.shape)
                current_samples.append(sample_vector)

            all_samples.append(tuple(current_samples))

    # Single-input: binary files are directly in dataset_dir
    else:
        bin_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.bin')])

        for bin_file in bin_files:
            bin_file_path = os.path.join(dataset_dir, bin_file)
            sample_vector = np.fromfile(bin_file_path, dtype=input_spec[0].type).reshape(input_spec[0].shape)
            all_samples.append((sample_vector,))

    return all_samples


def store_results(results: list[tuple[np.ndarray, ...]], output_dir: str, reference_dir: str):
    """Store a list of output arrays in the directory structure matching the reference directory.

    :param results: List of tuples, where each tuple contains numpy arrays (outputs for one sample)
    :param output_dir: Directory where results will be stored

    Directory structure created matches reference_dir:
        output_dir/
        ├── sample_0/
        │   ├── 0000.bin
        │   └── 0001.bin
        ├── some_other_sample/
        │   ├── 0000.bin
        │   └── 0001.bin
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get subdirectories from reference directory
    sample_dirs = sorted([d for d in os.listdir(reference_dir)
                          if os.path.isdir(os.path.join(reference_dir, d))])

    assert len(sample_dirs) == len(results), \
        f"Number of samples ({len(results)}) must match number of subdirectories in reference_dir ({len(sample_dirs)})"

    for sample_idx, (sample_name, sample_outputs) in enumerate(zip(sample_dirs, results)):
        sample_dir = os.path.join(output_dir, sample_name)
        os.makedirs(sample_dir, exist_ok=True)

        # Store each output tensor
        for output_idx, output_array in enumerate(sample_outputs):
            bin_file_name = f"{str(output_idx).zfill(4)}.bin"
            bin_file_path = os.path.join(sample_dir, bin_file_name)
            output_array.tofile(bin_file_path)


def _run_pytorch_program(model, testing_dataset_dir, input_spec: list[ModelInputSpec],
                         output_spec: list[torch.Tensor], cpu_results_dir, npu_results_dir):
    all_outputs = []

    for input_samples in read_prepared_samples(testing_dataset_dir, input_spec):
        current_input_samples = []
        for spec, sample in zip(input_spec, input_samples, strict=True):
            match spec.dim_order:
                case torch.contiguous_format:
                    # Use the data as is, just turn it into a PyTorch tensor.
                    sample = torch.tensor(sample)

                case torch.channels_last:
                    # The tensor data was stored by the DatasetCreator as channels last (NHWC), but it was now
                    #  incorrectly parsed as contiguous/channels first (NCHW). Transpose it to channels last to preserve
                    #  the semantics.
                    channels_last_shape = translator.dims_to_channels_last(list(spec.shape))
                    sample = np.moveaxis(sample.reshape(channels_last_shape), -1, 1)
                    sample = torch.tensor(sample).to(memory_format=torch.channels_last)

                case _:
                    raise ValueError(f"Unsupported dim_order: {spec.dim_order}")

            current_input_samples.append(sample)

        # Run the model.
        output = model(*current_input_samples)
        if isinstance(output, torch.Tensor):
            output = (output,)

        current_outputs = []

        for o, o_spec in zip(output, output_spec, strict=True):
            dim_order = list(o_spec.dim_order())  # ExecuTorch dim order.
            rank = len(o_spec.shape)
            if dim_order == list(range(rank)):  # Contiguous dim order.
                current_outputs.append(o.detach().numpy())

            elif is_channels_last_dim_order(dim_order):  # Channels last dim order.
                # The NPU variant outputs channels last (NHWC). We need to convert the CPU output to match.
                o = o.detach().numpy().reshape(o_spec.shape)
                current_outputs.append(np.moveaxis(o, 1, -1))

            else:
                raise ValueError(f"Unsupported dim_order: {o_spec.dim_order}")

        all_outputs.append(current_outputs)

    # Store all the results.
    store_results(all_outputs, cpu_results_dir, npu_results_dir)


def convert_run_compare(
        model: torch.nn.Module, input_spec: list[ModelInputSpec] | tuple,
        dlg_model_verifier: GraphVerifier,
        dataset_creator=RandomDatasetCreator(),
        output_comparator=AllCloseOutputComparator(),
        mocker: MockerFixture = None,
        run_cpu_version_in_pytorch: bool = False,
        use_qat: bool = False,
):
    """
    Run provided program twice with neutron-test and check if results correspond. At first,
    non-delegated program is executed (all nodes run on CPU), followed by delegated one
    (some nodes run on Neutron NPU).

    :param model: Executed PyTorch model.
    :param input_spec: Model input specification. Can be either tuple - single float32 input model - or list
        of ModelInputSpec.
    :param dataset_creator: Creator that should fill provided `dataset_dir` with model input samples.
    :param output_comparator: Comparator of results produced by NPU and CPU runs of the program.
    :param dlg_model_verifier: Graph verifier instance.
    :param run_cpu_version_in_pytorch: If True, runs CPU version in float32 PyTorch instead of quantized ExecuTorch.
    :param mocker: Mocker instance used by visualizer.
    :param use_qat: If True, applies quantization-aware training before conversion (without the QAT training).
    """
    assert os.path.exists(NSYS_PATH)
    assert os.path.exists(NSYS_CONFIG_PATH)
    assert os.path.exists(NSYS_FIRMWARE_PATH)

    test_name = _get_caller_name()
    test_dir = os.path.join(OUTPUTS_DIR, test_name)

    shutil.rmtree(test_dir, ignore_errors=True)
    mkdir(test_dir)

    dataset_dir = os.path.join(test_dir, "dataset")
    mkdir(dataset_dir)
    if isinstance(input_spec, tuple):
        input_spec = [ModelInputSpec(input_spec)]

    calibration_dataset_dir, testing_dataset_dir = dataset_creator.generate_samples(dataset_dir, input_spec)

    cpu_results_dir = os.path.join(test_dir, "results_cpu")
    npu_results_dir = os.path.join(test_dir, "results_npu")

    delegated_program = _run_delegated_executorch_program(
        model, test_dir, test_name, calibration_dataset_dir, testing_dataset_dir, input_spec, dlg_model_verifier,
        npu_results_dir, mocker, use_qat=use_qat
    )

    output_spec = _get_program_output_spec(delegated_program)

    if run_cpu_version_in_pytorch:
        _run_pytorch_program(model, testing_dataset_dir, input_spec, output_spec, cpu_results_dir, npu_results_dir)
    else:
        _run_non_delegated_executorch_program(
            model, test_dir, test_name, calibration_dataset_dir, testing_dataset_dir, input_spec, cpu_results_dir
        )

    output_tensor_spec = _get_program_output_spec(delegated_program)

    npu_results_dir = os.path.join(test_dir, "results_npu")
    cpu_results_dir = os.path.join(test_dir, "results_cpu")
    output_comparator.compare_results(cpu_results_dir, npu_results_dir, output_tensor_spec)


def _get_caller_name():
    for idx, frame in enumerate(inspect.stack()):
        if frame.function == "convert_run_compare":
            # Look one index above to get caller
            return inspect.stack()[idx + 1].function


def execute_cmd(cmd, cwd='.'):
    env = {
        "LD_LIBRARY_PATH": NSYS_PATH.parent
    }

    with subprocess.Popen(cmd, cwd=cwd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env) as process:
        cmd_out, cmd_err = process.communicate()
        cmd_out_decoded = cmd_out.strip().decode('utf-8', errors='replace')
        cmd_error_decoded = cmd_err.strip().decode('utf-8', errors='replace')

        for line in cmd_out_decoded.split("\n"):
            logger.info(line)

        for line in cmd_error_decoded.split("\n"):
            if line:
                logger.warning(line)

        return_code = process.returncode
        if return_code != 0:
            logger.fatal(cmd_error_decoded)
            raise Exception('Error running command: "%s", rc: %d' % (cmd, return_code))

    return cmd_out_decoded, cmd_error_decoded, return_code


def _get_program_output_spec(exported_program) -> list[torch.Tensor]:
    """
    Get output tensor specification for provided program.

    :param exported_program: Exported program.
    :return: List of output PyTorch tensors.
    """
    nodes = list(exported_program.graph.nodes)
    # TODO robert: since version 0.5 the user_outputs are not updated after delegation.
    # Hence bellow code does not works
    # Remove/update if the feature/bug if confirmed.

    # program_outputs = exported_program.graph_signature.user_outputs
    #
    # output_tensors_spec = []
    #
    # for node in nodes:
    #     if node.name in program_outputs:
    #         output_tensors_spec.append(node.meta["val"])
    #
    # assert len(output_tensors_spec) == len(program_outputs)

    output_tensors_spec = list(exported_program.graph.output_node().meta["val"])

    return output_tensors_spec
