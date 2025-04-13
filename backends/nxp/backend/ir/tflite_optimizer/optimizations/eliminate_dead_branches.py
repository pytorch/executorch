# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.nxp.backend.ir import logger
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model
from executorch.backends.nxp.backend.ir.tflite_optimizer.optimizations.base_optimization import BaseOptimization


class EliminateDeadBranches(BaseOptimization):

    def __call__(self) -> bool:
        _, output_to_ops = self._create_tensor_to_operator_dictionaries()

        output_names = [tensor.name for tensor in self._builder.get_sub_graph().outputs.tmp_outputs]

        tensor_names_to_process = set(output_names)
        tensors_to_keep = set()
        ops_to_keep = set()
        processed_ops = set()

        # Iterate from output tensors to inputs and mark all visited nodes & tensors
        while len(tensor_names_to_process) != 0:
            tensor = tensor_names_to_process.pop()
            tensors_to_keep.add(tensor)

            if tensor not in output_to_ops:
                # Input tensor or already processed
                continue

            op: tflite_model.Operator = output_to_ops[tensor]

            if op in processed_ops:
                continue

            # Append all inputs and outputs to next processing. Outputs of nodes aren't
            # necessarily outputs of the model but must be preserved.
            for tensor in op.tmp_inputs + op.tmp_outputs:
                tensor_names_to_process.add(tensor.name)

            ops_to_keep.add(op)
            processed_ops.add(op)

        if not self._conversion_config.allow_inputs_stripping:
            # Keep all inputs (even if they are not used) when prohibited by user
            tensors_to_keep.update([tensor.name for tensor in self._builder.get_sub_graph().inputs.tmp_inputs])

        # Remove unused ops
        ops = self._builder.get_operators().vector
        i, removed_ops_count = 0, 0
        while i < len(ops):
            if ops[i] in ops_to_keep:
                i += 1
            else:
                removed_ops_count += 1
                del ops[i]

        # Remove unused tensors
        tensors = self._builder.get_tensors().vector
        i = 0
        while i < len(tensors):
            if tensors[i].name in tensors_to_keep:
                i += 1
            else:
                del tensors[i]

        if removed_ops_count != 0:
            logger.i(f"Dead branch elimination optimization removed {removed_ops_count} unused ops from the graph.")

        return removed_ops_count != 0
