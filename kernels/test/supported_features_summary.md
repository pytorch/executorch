# Supported features table
| feature|aten|optimized|portable|quantized|custom_kernel_example |
| ---|---|---|---|---|--- |
| **namespace global**|---|---|---|---|--- |
| is_aten|True|False|False|False|False |
| output_resize|True|False|False|False|False |
| **namespace op_gelu**|---|---|---|---|--- |
| dtype_double|True|False|True|True|True |
| **namespace op_log_softmax**|---|---|---|---|--- |
| dtype_double|True|False|True|True|True |

# Source
All of supported features are defined in fbcode/executorch/kernels/test/supported_features.yaml.

Each kernel can have its own overrides, which are defined in
('aten', 'fbcode/executorch/kernels/test/supported_features_def_aten.yaml')
('optimized', 'fbcode/executorch/kernels/optimized/test/supported_features_def.yaml')
('portable', 'fbcode/executorch/kernels/portable/test/supported_features_def.yaml')
('quantized', 'fbcode/executorch/kernels/quantized/test/supported_features_def.yaml')
('custom_kernel_example', 'fbcode/executorch/kernels/test/custom_kernel_example/supported_features_def.yaml')
