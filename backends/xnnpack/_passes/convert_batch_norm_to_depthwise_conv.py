# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import Optional

import torch
from executorch.backends.transforms.utils import (
    create_constant_placeholder,
    delete_constant_placeholder,
)
from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass
from executorch.backends.xnnpack.utils.utils import (
    get_param_tensor,
    get_tensor_name,
    is_param_node,
)
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult
from torch.export.graph_signature import InputKind


class ConvertBatchNormToDepthwiseConvPass(XNNPACKPass):
    """
    Converts standalone batch norm operations to depthwise convolutions.
    This allows XNNPACK to handle batch norm operations that cannot be fused
    with preceding convolutions.
    
    BatchNorm formula: y = (x - mean) / sqrt(var + eps) * weight + bias
    This can be represented as a 1x1 depthwise convolution with:
    - conv_weight = weight / sqrt(var + eps)  
    - conv_bias = bias - mean * weight / sqrt(var + eps)
    """

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        constant_placeholders_to_delete = set()
        nodes_to_convert = []
        
        # First pass: identify standalone batch norm nodes
        for node in graph.nodes:
            if (
                node.target != exir_ops.edge.aten._native_batch_norm_legit_no_training.default
                and node.target != exir_ops.edge.aten.native_batch_norm.default
            ):
                continue
                
            # Check if this batch norm can be fused with a preceding conv
            # If so, skip it - the fusion pass will handle it
            if self._can_be_fused_with_conv(node):
                continue
                
            # Check if this is a valid standalone batch norm to convert
            if self._can_convert_to_depthwise_conv(node):
                nodes_to_convert.append(node)
        
        # Second pass: convert the identified nodes
        for bn_node in nodes_to_convert:
            conv_node = self._convert_batch_norm_to_depthwise_conv(
                graph_module, bn_node, constant_placeholders_to_delete
            )
            if conv_node is not None:
                # Replace all uses of batch norm getitem(0) with the conv node
                for user in list(bn_node.users):
                    if user.target == operator.getitem and user.args[1] == 0:
                        user.replace_all_uses_with(conv_node)
                        graph.erase_node(user)
                
                # Remove the batch norm node
                graph.erase_node(bn_node)
        
        # Clean up unused constant placeholders
        if constant_placeholders_to_delete:
            graph_module.graph.eliminate_dead_code()
            for node in constant_placeholders_to_delete:
                if node is not None and len(node.users) == 0:
                    delete_constant_placeholder(self.exported_program, node)
        
        graph_module.recompile()
        # Regenerate metadata and shape information
        graph_module = super().call(graph_module).graph_module
        
        return PassResult(graph_module, True)

    def _can_be_fused_with_conv(self, bn_node: torch.fx.Node) -> bool:
        """Check if this batch norm can be fused with a preceding convolution."""
        # Import here to avoid circular dependency
        from executorch.backends.xnnpack._passes.fuse_batch_norm_with_conv import (
            FuseBatchNormWithConvPass,
        )
        
        input_node = bn_node.all_input_nodes[0]
        
        # Check if input is a conv with single user (this batch norm)
        if (
            input_node.target == exir_ops.edge.aten.convolution.default
            and len(input_node.users) == 1
        ):
            return FuseBatchNormWithConvPass.can_fuse(
                input_node, bn_node, self.exported_program
            )
        
        return False

    def _can_convert_to_depthwise_conv(self, bn_node: torch.fx.Node) -> bool:
        """Check if this batch norm can be converted to depthwise conv."""
        
        # All users must be getitem ops accessing the first element (output tensor)
        for user in bn_node.users:
            if user.target != operator.getitem or user.args[1] != 0:
                return False
        
        # Check that we have the required parameters
        if len(bn_node.args) < 5:
            return False
            
        # Weight, bias, running_mean, running_var must be parameters
        param_nodes = bn_node.args[1:5]  # weight, bias, running_mean, running_var
        
        for param_node in param_nodes:
            if not isinstance(param_node, torch.fx.Node):
                return False
            if not is_param_node(self.exported_program, param_node):
                return False
        
        return True

    def _convert_batch_norm_to_depthwise_conv(
        self,
        graph_module: torch.fx.GraphModule,
        bn_node: torch.fx.Node,
        constant_placeholders_to_delete: set,
    ) -> Optional[torch.fx.Node]:
        """Convert a batch norm node to a depthwise convolution."""
        
        # Extract batch norm parameters
        input_tensor = bn_node.args[0]
        
        # Cast args to Node types for parameter access
        bn_weight_node = bn_node.args[1] if isinstance(bn_node.args[1], torch.fx.Node) else None
        bn_bias_node = bn_node.args[2] if isinstance(bn_node.args[2], torch.fx.Node) else None
        running_mean_node = bn_node.args[3] if isinstance(bn_node.args[3], torch.fx.Node) else None
        running_var_node = bn_node.args[4] if isinstance(bn_node.args[4], torch.fx.Node) else None
        
        if any(node is None for node in [bn_weight_node, bn_bias_node, running_mean_node, running_var_node]):
            return None
        
        # These are guaranteed to be non-None now
        assert bn_weight_node is not None
        assert bn_bias_node is not None
        assert running_mean_node is not None
        assert running_var_node is not None
        
        bn_weight = get_param_tensor(self.exported_program, bn_weight_node)
        bn_bias = get_param_tensor(self.exported_program, bn_bias_node)
        running_mean = get_param_tensor(self.exported_program, running_mean_node)
        running_var = get_param_tensor(self.exported_program, running_var_node)
        
        # Get epsilon value
        if str(bn_node.target).endswith("native_batch_norm.default"):
            eps = bn_node.args[7] if len(bn_node.args) > 7 else 1e-5
        else:  # _native_batch_norm_legit_no_training
            eps = bn_node.args[6] if len(bn_node.args) > 6 else 1e-5
            
        # Ensure eps is a float
        if not isinstance(eps, (int, float)):
            eps = 1e-5

        if any(param is None for param in [bn_weight, bn_bias, running_mean, running_var]):
            return None
            
        # Ensure all parameters are tensors
        assert isinstance(bn_weight, torch.Tensor)
        assert isinstance(bn_bias, torch.Tensor)
        assert isinstance(running_mean, torch.Tensor)
        assert isinstance(running_var, torch.Tensor)

        # Calculate depthwise conv parameters
        # BatchNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
        # Depthwise Conv: y = x * conv_weight + conv_bias
        # Therefore: conv_weight = weight / sqrt(var + eps)
        #           conv_bias = bias - mean * weight / sqrt(var + eps)
        
        inv_std = torch.rsqrt(running_var + eps)
        conv_weight_1d = bn_weight * inv_std
        conv_bias_1d = bn_bias - running_mean * conv_weight_1d
        
        # Reshape for depthwise conv: [C] -> [C, 1, 1, 1] for 2D conv
        # Assuming 4D input tensor [N, C, H, W]
        num_channels = conv_weight_1d.shape[0]
        conv_weight = conv_weight_1d.view(num_channels, 1, 1, 1)
        conv_bias = conv_bias_1d
        
        # Create parameter names
        bn_weight_name = get_tensor_name(self.exported_program, bn_weight_node)
        conv_weight_name = (bn_weight_name + "_as_depthwise_conv_weight").replace(".", "_")
        conv_bias_name = (bn_weight_name + "_as_depthwise_conv_bias").replace(".", "_")
        
        # Create new parameter nodes
        graph = graph_module.graph
        with graph.inserting_before(bn_node):
            conv_weight_node = create_constant_placeholder(
                exp_program=self.exported_program,
                graph=graph,
                kind=InputKind.PARAMETER,
                name=conv_weight_name,
                data=conv_weight,
            )
            
            conv_bias_node = create_constant_placeholder(
                exp_program=self.exported_program,
                graph=graph,
                kind=InputKind.PARAMETER,  
                name=conv_bias_name,
                data=conv_bias,
            )
            
            # Create depthwise convolution node
            # Args: input, weight, bias, stride, padding, dilation, transposed, output_padding, groups
            conv_args = (
                input_tensor,           # input
                conv_weight_node,       # weight  
                conv_bias_node,         # bias
                [1, 1],                 # stride
                [0, 0],                 # padding
                [1, 1],                 # dilation
                False,                  # transposed
                [0, 0],                 # output_padding
                num_channels,           # groups (depthwise = groups = in_channels)
            )
            
            conv_node = graph.create_node(
                "call_function",
                exir_ops.edge.aten.convolution.default,
                args=conv_args,
            )
        
        # Mark old parameters for deletion
        constant_placeholders_to_delete.update(bn_node.args[1:5])
        
        return conv_node

    @staticmethod
    def can_convert_standalone_batch_norm(
        bn_node: torch.fx.Node, program: ExportedProgram
    ) -> bool:
        """
        Static method to check if a standalone batch norm can be converted.
        Used by the partitioner configuration.
        """
        # All users must be getitem ops accessing the first element
        for user in bn_node.users:
            if user.target != operator.getitem or user.args[1] != 0:
                return False
        
        # Check that we have required parameters
        if len(bn_node.args) < 5:
            return False
            
        # Weight, bias, running_mean, running_var must be parameters
        param_nodes = bn_node.args[1:5]
        
        for param_node in param_nodes:
            if not isinstance(param_node, torch.fx.Node):
                return False
            if not is_param_node(program, param_node):
                return False
        
        return True
