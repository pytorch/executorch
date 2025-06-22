#!/usr/bin/env python3

import sys
import os

# Add the executorch src to path
sys.path.insert(0, '/Users/x/Desktop/executorch/src')

# Simple test to verify the pass can be imported and instantiated
def test_import():
    try:
        # Direct import test
        import torch
        from executorch.backends.xnnpack._passes.convert_batch_norm_to_depthwise_conv import (
            ConvertBatchNormToDepthwiseConvPass,
        )
        
        print("✓ Successfully imported ConvertBatchNormToDepthwiseConvPass")
        
        # Create a dummy exported program
        class DummyModule(torch.nn.Module):
            def forward(self, x):
                return x
        
        # Try to create the pass instance
        dummy_module = DummyModule()
        example_args = (torch.randn(1, 2, 4, 4),)
        exported_program = torch.export.export(dummy_module, example_args)
        
        pass_instance = ConvertBatchNormToDepthwiseConvPass(exported_program)
        print("✓ Successfully created pass instance")
        
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

def test_static_method():
    try:
        # Test the static method
        import torch
        from executorch.backends.xnnpack._passes.convert_batch_norm_to_depthwise_conv import (
            ConvertBatchNormToDepthwiseConvPass,
        )
        
        class TestBN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(2)
                
            def forward(self, x):
                return self.bn(x)
        
        model = TestBN().eval()
        example_args = (torch.randn(1, 2, 4, 4),)
        exported_program = torch.export.export(model, example_args)
        
        # Find the batch norm node
        bn_node = None
        for node in exported_program.graph.nodes:
            if 'batch_norm' in str(node.target):
                bn_node = node
                break
        
        if bn_node:
            result = ConvertBatchNormToDepthwiseConvPass.can_convert_standalone_batch_norm(bn_node, exported_program)
            print(f"✓ Static method test completed. Can convert: {result}")
        else:
            print("✗ No batch norm node found in graph")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Static method test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing ConvertBatchNormToDepthwiseConvPass...")
    
    success1 = test_import()
    success2 = test_static_method()
    
    if success1 and success2:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
