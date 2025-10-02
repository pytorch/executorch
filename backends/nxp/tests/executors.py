# Copyright 2023-2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from typing import Callable, Dict, Union

import numpy
import numpy as np
import torch

from executorch.backends.nxp.backend.edge_program_converter import (
    EdgeProgramToIRConverter,
)
from executorch.backends.nxp.backend.ir import logger
from executorch.backends.nxp.backend.ir.conversion_config import ConversionConfig
from executorch.backends.nxp.backend.ir.converter.conversion.translator import (
    create_channels_first_to_channels_last_permutation,
    create_channels_last_to_channels_first_permutation,
)
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    NodeConverter,
    Target,
)
from torch.export import ExportedProgram
from torch.fx import Node
from torch.fx.graph import Graph


# If executed on i.MX platform, there is no tensorflow module. And typically the intention is to use the tflite python
# interpreter available in tflite_runtime
try:
    import tensorflow.lite as tflite
except ModuleNotFoundError:
    import tflite_runtime.interpreter as tflite


class EdgeProgramExecutor:

    def __init__(self, edge_program: ExportedProgram):
        self.edge_program = edge_program

    def inference(
        self, input_data: Union[numpy.ndarray, Dict[int, numpy.ndarray]]
    ) -> Union[numpy.ndarray, Dict[str, numpy.ndarray]]:

        if isinstance(input_data, numpy.ndarray):
            program_inputs = [torch.from_numpy(input_data)]
        else:
            program_inputs = [
                torch.from_numpy(in_data) for in_data in input_data.values()
            ]

        output = self.edge_program.module()(*program_inputs)

        if isinstance(output, torch.Tensor):
            return output.detach().numpy()
        elif isinstance(output, tuple) and len(output) == 1:
            return output[0].detach().numpy()
        elif isinstance(output, tuple):
            output_names = self.edge_program.graph_signature.user_outputs

            return {
                name: tensor.detach().numpy()
                for (name, tensor) in zip(output_names, output)
            }

        raise RuntimeError(
            "Edge program inference with multiple outputs not implemented"
        )


class TFLiteExecutor:
    _interpreter: tflite.Interpreter

    def __init__(
        self,
        model_path: str = None,
        model_content=None,
        save_model=False,
        saved_model_name="model.tflite",
        delegate_path=None,
        num_threads=None,
        op_resolver_type=tflite.experimental.OpResolverType.AUTO,
    ):
        """
        Construct TFLiteExecutor used to quickly run inference on TFLite model.
        Exactly one of "model_path" and "model_content" must be specified.

        :param model_path: Path to executed TFLite model.
        :param model_content: Path to byte representation of TFLite model.
        :param save_model: If true and model was provided through "model_content",
            model is saved to storage with name "saved_model_name".
        :param saved_model_name: Model name used when model stored to storage. Default
            value is "model.tflite".
        :param delegate_path: External delegate to be used for the TFLiteExecutor, see
            https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter for details. Default value is None.
        :param num_threads: number of threads to be used by the TFLiteExecutor, see
            https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter for details. Default value is None.
        :param op_resolver_type: Op kernels to be used by the TFLiteExecutor, see
            https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter for details. Default value is
            tflite.experimental.OpResolverType.AUTO.
        """
        assert model_path is not None or model_content is not None
        assert model_path is None or model_content is None

        if delegate_path is not None:
            delegate = [tflite.load_delegate(delegate_path)]
        else:
            delegate = None

        if save_model:
            with open(saved_model_name, "wb") as f:
                f.write(model_content)

        if model_path is not None:
            self._interpreter = tflite.Interpreter(
                model_path=model_path,
                experimental_delegates=delegate,
                num_threads=num_threads,
                experimental_op_resolver_type=op_resolver_type,
            )
        else:
            self._interpreter = tflite.Interpreter(
                model_content=model_content,
                experimental_delegates=delegate,
                num_threads=num_threads,
                experimental_op_resolver_type=op_resolver_type,
            )

        self._interpreter.allocate_tensors()

    def inference(
        self, input_data: Union[numpy.ndarray, Dict[int, numpy.ndarray]]
    ) -> Union[numpy.ndarray, Dict[str, numpy.ndarray]]:
        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()

        if isinstance(input_data, numpy.ndarray):
            self._interpreter.set_tensor(input_details[0]["index"], input_data)
        elif isinstance(input_data, Dict):
            if len(input_data) != len(input_details):
                logger.w(
                    f"Number of model inputs: '{len(input_details)}', and provided input data: '{len(input_data)}'"
                    f" is not the same. Using first {len(input_details)} inputs tensors."
                )
            for index in range(len(input_details)):
                self._interpreter.set_tensor(
                    input_details[index]["index"], input_data[index]
                )

        self._interpreter.allocate_tensors()
        self._interpreter.invoke()

        output_data = {}

        for output_detail in output_details:
            output_data[output_detail["name"]] = self._interpreter.get_tensor(
                output_detail["index"]
            )

        # Flatten output if there is only one value in output dictionary
        if len(output_data) == 1:
            return np.asarray(next(iter(output_data.values())))
        else:
            return output_data

    def get_output_details(self, index):
        return self._interpreter.get_output_details()[index]


def compare_output_arrays(
    tfl_output: np.ndarray,
    edge_output: np.ndarray,
    output_name: str,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
):
    """Assert that the provided numpy arrays are equal.

    :param tfl_output: Numpy array holding the output of the TFLite model.
    :param edge_output: Numpy array holding the output of the ExportedProgram.
    :param output_name: Common name of the above arrays.
    :param rtol: Relative tolerance.
    :param atol: Absolute tolerance.
    """
    if tfl_output.dtype.char == edge_output.dtype.char == "O":
        # String types fail in the following checks. Cast them to float32 before comparison.
        tfl_output = tfl_output.astype(np.float32)
        edge_output = edge_output.astype(np.float32)

    if tfl_output.dtype != np.bool_ and tfl_output.size != 0:
        logger.d(
            f"Maximum output difference of the `{output_name}`tensor: {np.max(np.abs(tfl_output - edge_output))}"
        )

    assert tfl_output.shape == edge_output.shape, "Output shapes don't match!"

    if (max_diff := np.abs(np.max(tfl_output - edge_output))) > 0.0:
        logger.w(
            f"Maximum absolute difference of the tensor '{output_name}': '{max_diff}'"
        )

    assert np.allclose(
        tfl_output, edge_output, rtol=rtol, atol=atol, equal_nan=True
    ), f"Output values of the `{output_name}` tensor don't match!"


class TFLiteIOPreprocess:

    def preprocess(self, data: np.ndarray | dict[int, numpy.ndarray]):
        return data


class ToChannelFirstPreprocess(TFLiteIOPreprocess):
    def __init__(self, dim_0_reduced: bool | dict[int, bool] = False):
        self.dim_0_reduced = dim_0_reduced

    def preprocess(self, data: np.ndarray | dict[int, np.ndarray]):
        def get_channel_first_permutation(tensor, dim_0_reduced):
            tensor_rank = len(tensor.shape)
            perm = create_channels_last_to_channels_first_permutation(tensor_rank)
            if dim_0_reduced and tensor_rank > 1:
                perm[0], perm[1] = perm[1], perm[0]
            return perm

        transpose_fn = lambda x, rank: np.transpose(  # noqa E731
            x, get_channel_first_permutation(x, rank)
        )
        if isinstance(data, np.ndarray) and isinstance(self.dim_0_reduced, bool):
            preprocessed_data = transpose_fn(data, self.dim_0_reduced)

        elif isinstance(data, dict) and isinstance(self.dim_0_reduced, bool):
            preprocessed_data = {
                k: transpose_fn(v, self.dim_0_reduced) for k, v in data.items()
            }

        elif isinstance(data, dict) and isinstance(self.dim_0_reduced, dict):
            preprocessed_data = {
                k: transpose_fn(v, self.dim_0_reduced[k]) for k, v in data.items()
            }

        else:
            raise ValueError(
                "Invalid combination of inputs. Data can be either np.ndarray or dict. If original number "
                "of dimension is used, it can be only int for np.ndarray data or dict of ints for dict "
                "data with same keys."
            )
        return preprocessed_data


class ToChannelLastPreprocess(TFLiteIOPreprocess):
    def preprocess(self, data: np.ndarray | dict[int, np.ndarray]):
        def get_channel_last_permutation(tensor):
            return create_channels_first_to_channels_last_permutation(len(tensor.shape))

        transpose_fn = lambda x: np.transpose(  # noqa E731
            x, get_channel_last_permutation(x)
        )
        if isinstance(data, np.ndarray):
            preprocessed_data = transpose_fn(data)
        else:
            preprocessed_data = {k: transpose_fn(v) for k, v in data.items()}
        return preprocessed_data


class ToNHWCPreprocess(TFLiteIOPreprocess):

    def preprocess(self, data: np.ndarray | dict[int, numpy.ndarray]):
        warnings.warn(
            "Method is deprecated. Use ToChannelFirstPreprocess/ToChannelLastPreprocess instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        transpose_fn = lambda x: np.transpose(x, [0, 2, 3, 1])  # noqa E731
        if isinstance(data, np.ndarray):
            preprocessed_data = transpose_fn(data)
        else:
            preprocessed_data = {k: transpose_fn(v) for k, v in data.items()}
        return preprocessed_data


class ToNCHWPreprocess(TFLiteIOPreprocess):

    def preprocess(self, data: np.ndarray | dict[int, numpy.ndarray]):
        warnings.warn(
            "Method is deprecated. Use ToChannelFirstPreprocess/ToChannelLastPreprocess instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        transpose_fn = lambda x: np.transpose(x, [0, 3, 1, 2])  # noqa E731
        if isinstance(data, np.ndarray):
            preprocessed_data = transpose_fn(data)
        else:
            preprocessed_data = {k: transpose_fn(v) for k, v in data.items()}
        return preprocessed_data


def convert_run_compare(
    edge_program: ExportedProgram,
    input_data,
    rtol=1.0e-5,
    atol=1.0e-8,
    save_models=False,
    tfl_model: (bytes, dict) = None,
    tflite_input_preprocess: TFLiteIOPreprocess = TFLiteIOPreprocess(),  # noqa B008
    tflite_output_preprocess: TFLiteIOPreprocess = TFLiteIOPreprocess(),  # noqa B008
    conversion_config: ConversionConfig = ConversionConfig(),  # noqa B008
    tflite_op_resolver_type=tflite.experimental.OpResolverType.AUTO,
) -> (TFLiteExecutor, EdgeProgramExecutor):

    if tfl_model is None:
        tfl_model, _ = EdgeProgramToIRConverter().convert_program(
            edge_program, conversion_config
        )

    edge_program_executor = EdgeProgramExecutor(edge_program)
    edge_program_output = edge_program_executor.inference(input_data)

    tflite_input_data = tflite_input_preprocess.preprocess(input_data)
    tflite_executor = TFLiteExecutor(
        model_content=tfl_model,
        save_model=save_models,
        op_resolver_type=tflite_op_resolver_type,
    )
    tflite_output = tflite_executor.inference(tflite_input_data)
    tflite_output = tflite_output_preprocess.preprocess(tflite_output)

    if isinstance(tflite_output, dict) and isinstance(edge_program_output, dict):
        if (
            len(
                set(tflite_output.keys()).symmetric_difference(
                    set(edge_program_output.keys())
                )
            )
            == 0
        ):
            # Both TFLite and ExportedProgram output dictionaries have the same keys.
            for output_name, tflite_out in tflite_output.items():
                compare_output_arrays(
                    tflite_out,
                    edge_program_output[output_name],
                    output_name,
                    rtol,
                    atol,
                )

        else:
            logger.e(
                logger.Code.INTERNAL_ERROR,
                "Original program and converted TFLite models have different outputs.",
            )

    elif isinstance(tflite_output, np.ndarray) and isinstance(
        edge_program_output, np.ndarray
    ):
        compare_output_arrays(
            tflite_output, edge_program_output, "main output", rtol, atol
        )

    else:
        # This can happen for example, if the TFLite model does not have some outputs, which are in exported program.
        logger.e(
            logger.Code.NOT_IMPLEMENTED,
            "Original ExportedProgram and converted TFLite models have different"
            " number of outputs. Testing is not implemented for this case.",
        )

    return tflite_executor, edge_program_executor


def graph_contains_any_of_ops(graph: Graph, ops: list) -> bool:
    return any(node.target in ops for node in graph.nodes)


target_support_check_function = Callable[[Node, Target], bool]


class OverrideTargetSupportCheck:

    def __init__(
        self,
        converter_class: type[NodeConverter],
        *,
        new_target_support_check: target_support_check_function,
    ):
        self._converter_class = converter_class
        self.new_target_support_check = new_target_support_check
        self.old_target_support_check = converter_class._is_supported_on_target

    def __enter__(self):
        self._converter_class._is_supported_on_target = self.new_target_support_check

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._converter_class._is_supported_on_target = self.old_target_support_check
