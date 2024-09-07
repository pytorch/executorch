load("//third-party:glob_defs.bzl", "subdir_glob")
load("//backends/xnnpack/third-party/XNNPACK/gen:microkernels.bzl", "prod_srcs_for_arch")

def prod_srcs_for_arch_wrapper(arch):
    prod_srcs = prod_srcs_for_arch(arch)
    wrapped_srcs = []
    for src in prod_srcs:
        wrapped_srcs.append("XNNPACK/" + src)
    
    return wrapped_srcs

def get_xnnpack_headers():
    src_headers = subdir_glob([
        ("XNNPACK/src", "**/*.h"),
    ])
    fixed_headers = {}
    for k, v in src_headers.items():
        new_key = k
        if not k.startswith("xnnpack") and not k.startswith("configs"):
            new_key = "src/" + k
        fixed_headers[new_key] = v
    include_headers = subdir_glob([
        ("XNNPACK/include", "*.h"),
    ])
    
    return fixed_headers | include_headers

# Manually Update this from 
OPERATOR_SRCS = [
    "XNNPACK/src/operator-delete.c",
    "XNNPACK/src/operators/argmax-pooling-nhwc.c",
    "XNNPACK/src/operators/average-pooling-nhwc.c",
    "XNNPACK/src/operators/batch-matrix-multiply-nc.c",
    "XNNPACK/src/operators/binary-elementwise-nd.c",
    "XNNPACK/src/operators/channel-shuffle-nc.c",
    "XNNPACK/src/operators/constant-pad-nd.c",
    "XNNPACK/src/operators/convolution-nchw.c",
    "XNNPACK/src/operators/convolution-nhwc.c",
    "XNNPACK/src/operators/deconvolution-nhwc.c",
    "XNNPACK/src/operators/dynamic-fully-connected-nc.c",
    "XNNPACK/src/operators/fully-connected-nc.c",
    "XNNPACK/src/operators/global-average-pooling-ncw.c",
    "XNNPACK/src/operators/global-average-pooling-nwc.c",
    "XNNPACK/src/operators/lut-elementwise-nc.c",
    "XNNPACK/src/operators/max-pooling-nhwc.c",
    "XNNPACK/src/operators/prelu-nc.c",
    "XNNPACK/src/operators/reduce-nd.c",
    "XNNPACK/src/operators/resize-bilinear-nchw.c",
    "XNNPACK/src/operators/resize-bilinear-nhwc.c",
    "XNNPACK/src/operators/rope-nthc.c",
    "XNNPACK/src/operators/scaled-dot-product-attention-nhtc.c",
    "XNNPACK/src/operators/slice-nd.c",
    "XNNPACK/src/operators/softmax-nc.c",
    "XNNPACK/src/operators/transpose-nd.c",
    "XNNPACK/src/operators/unary-elementwise-nc.c",
    "XNNPACK/src/operators/unpooling-nhwc.c",
]

LOGGING_SRCS = [
    "XNNPACK/src/enums/allocation-type.c",
    "XNNPACK/src/enums/datatype-strings.c",
    "XNNPACK/src/enums/microkernel-type.c",
    "XNNPACK/src/enums/node-type.c",
    "XNNPACK/src/enums/operator-type.c",
    "XNNPACK/src/log.c",
]

XNNPACK_SRCS = [
    "XNNPACK/src/configs/argmaxpool-config.c",
    "XNNPACK/src/configs/avgpool-config.c",
    "XNNPACK/src/configs/binary-elementwise-config.c",
    "XNNPACK/src/configs/cmul-config.c",
    "XNNPACK/src/configs/conv-hwc2chw-config.c",
    "XNNPACK/src/configs/dwconv-config.c",
    "XNNPACK/src/configs/dwconv2d-chw-config.c",
    "XNNPACK/src/configs/experiments-config.c",
    "XNNPACK/src/configs/gavgpool-config.c",
    "XNNPACK/src/configs/gavgpool-cw-config.c",
    "XNNPACK/src/configs/gemm-config.c",
    "XNNPACK/src/configs/ibilinear-chw-config.c",
    "XNNPACK/src/configs/ibilinear-config.c",
    "XNNPACK/src/configs/lut32norm-config.c",
    "XNNPACK/src/configs/maxpool-config.c",
    "XNNPACK/src/configs/pavgpool-config.c",
    "XNNPACK/src/configs/prelu-config.c",
    "XNNPACK/src/configs/raddstoreexpminusmax-config.c",
    "XNNPACK/src/configs/reduce-config.c",
    "XNNPACK/src/configs/rmax-config.c",
    "XNNPACK/src/configs/spmm-config.c",
    "XNNPACK/src/configs/transpose-config.c",
    "XNNPACK/src/configs/unary-elementwise-config.c",
    "XNNPACK/src/configs/unpool-config.c",
    "XNNPACK/src/configs/vmulcaddc-config.c",
    "XNNPACK/src/configs/xx-fill-config.c",
    "XNNPACK/src/configs/xx-pad-config.c",
    "XNNPACK/src/configs/x8-lut-config.c",
    "XNNPACK/src/configs/zip-config.c",
    "XNNPACK/src/init.c",
    "XNNPACK/src/params.c",
]

SUBGRAPH_SRCS = [
    "XNNPACK/src/memory-planner.c",
    "XNNPACK/src/runtime.c",
    "XNNPACK/src/subgraph.c",
    "XNNPACK/src/subgraph/abs.c",
    "XNNPACK/src/subgraph/add2.c",
    "XNNPACK/src/subgraph/argmax-pooling-2d.c",
    "XNNPACK/src/subgraph/average-pooling-2d.c",
    "XNNPACK/src/subgraph/bankers-rounding.c",
    "XNNPACK/src/subgraph/batch-matrix-multiply.c",
    "XNNPACK/src/subgraph/ceiling.c",
    "XNNPACK/src/subgraph/clamp.c",
    "XNNPACK/src/subgraph/concatenate.c",
    "XNNPACK/src/subgraph/convert.c",
    "XNNPACK/src/subgraph/convolution-2d.c",
    "XNNPACK/src/subgraph/copy.c",
    "XNNPACK/src/subgraph/copysign.c",
    "XNNPACK/src/subgraph/deconvolution-2d.c",
    "XNNPACK/src/subgraph/depth-to-space-2d.c",
    "XNNPACK/src/subgraph/depthwise-convolution-2d.c",
    "XNNPACK/src/subgraph/divide.c",
    "XNNPACK/src/subgraph/elu.c",
    "XNNPACK/src/subgraph/even-split.c",
    "XNNPACK/src/subgraph/exp.c",
    "XNNPACK/src/subgraph/floor.c",
    "XNNPACK/src/subgraph/fully-connected-sparse.c",
    "XNNPACK/src/subgraph/fully-connected.c",
    "XNNPACK/src/subgraph/gelu.c",
    "XNNPACK/src/subgraph/global-average-pooling.c",
    "XNNPACK/src/subgraph/global-sum-pooling.c",
    "XNNPACK/src/subgraph/hardswish.c",
    "XNNPACK/src/subgraph/leaky-relu.c",
    "XNNPACK/src/subgraph/log.c",
    "XNNPACK/src/subgraph/max-pooling-2d.c",
    "XNNPACK/src/subgraph/maximum2.c",
    "XNNPACK/src/subgraph/minimum2.c",
    "XNNPACK/src/subgraph/multiply2.c",
    "XNNPACK/src/subgraph/negate.c",
    "XNNPACK/src/subgraph/prelu.c",
    "XNNPACK/src/subgraph/reciprocal-square-root.c",
    "XNNPACK/src/subgraph/reshape-helpers.c",
    "XNNPACK/src/subgraph/scaled-dot-product-attention.c",
    "XNNPACK/src/subgraph/sigmoid.c",
    "XNNPACK/src/subgraph/softmax.c",
    "XNNPACK/src/subgraph/space-to-depth-2d.c",
    "XNNPACK/src/subgraph/square-root.c",
    "XNNPACK/src/subgraph/square.c",
    "XNNPACK/src/subgraph/squared-difference.c",
    "XNNPACK/src/subgraph/static-constant-pad.c",
    "XNNPACK/src/subgraph/static-mean.c",
    "XNNPACK/src/subgraph/static-resize-bilinear-2d.c",
    "XNNPACK/src/subgraph/static-slice.c",
    "XNNPACK/src/subgraph/static-transpose.c",
    "XNNPACK/src/subgraph/subtract.c",
    "XNNPACK/src/subgraph/tanh.c",
    "XNNPACK/src/subgraph/unpooling-2d.c",
    "XNNPACK/src/subgraph/validation.c",
    "XNNPACK/src/tensor.c",
]

TABLE_SRCS = [
    "XNNPACK/src/tables/exp2-k-over-64.c",
    "XNNPACK/src/tables/exp2-k-over-2048.c",
    "XNNPACK/src/tables/exp2minus-k-over-4.c",
    "XNNPACK/src/tables/exp2minus-k-over-8.c",
    "XNNPACK/src/tables/exp2minus-k-over-16.c",
    "XNNPACK/src/tables/exp2minus-k-over-32.c",
    "XNNPACK/src/tables/exp2minus-k-over-64.c",
    "XNNPACK/src/tables/exp2minus-k-over-2048.c",
    "XNNPACK/src/tables/vlog.c",
]
