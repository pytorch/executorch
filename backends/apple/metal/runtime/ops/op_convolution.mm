/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/apple/metal/runtime/ops/common.h>

namespace executorch {
namespace backends {
namespace metal {

namespace {

// How a 4D convolution operand is laid out in memory. Inductor's layout
// optimization hands conv2d channels-last input and weight, and expects a
// channels-last output back.
enum class ConvMemoryFormat { Contiguous, ChannelsLast };

// Mirrors c10's is_channels_last_strides_2d, which is what PyTorch's
// suggest_memory_format() runs on a 4D tensor.
bool suggests_channels_last(const Tensor& tensor) {
  const auto sizes = tensor.sizes();
  const auto strides = tensor.strides();
  if (strides[1] == 0) {
    return false;
  }
  int64_t min = 0;
  for (int d : {1, 3, 2, 0}) {
    if (sizes[d] == 0 || strides[d] < min) {
      return false;
    }
    // Tells N111 (contiguous) apart from NC11 stored channels-last.
    if (d == 0 && min == strides[1]) {
      return false;
    }
    min = strides[d];
    if (sizes[d] > 1) {
      min *= sizes[d];
    }
  }
  return true;
}

// Classifies a 4D tensor from its strides. A dimension of size 1 does not
// change where the elements are, so it is ignored when checking that the data
// is dense in one of the two layouts. A tensor with such a dimension can fit
// both; its elements read the same either way, but the layout still decides
// the layout of the output, and the generated wrapper expects the one PyTorch
// would pick. Returns false for any other stride pattern.
bool get_conv_memory_format(const Tensor& tensor, ConvMemoryFormat* format) {
  const auto sizes = tensor.sizes();
  const auto strides = tensor.strides();
  const int64_t d1 = sizes[1], d2 = sizes[2], d3 = sizes[3];
  const int64_t contiguous[4] = {d1 * d2 * d3, d2 * d3, d3, 1};
  const int64_t channels_last[4] = {d2 * d3 * d1, 1, d3 * d1, d1};

  bool is_contiguous = true;
  bool is_channels_last = true;
  for (int i = 0; i < 4; i++) {
    if (sizes[i] == 1) {
      continue;
    }
    is_contiguous = is_contiguous && strides[i] == contiguous[i];
    is_channels_last = is_channels_last && strides[i] == channels_last[i];
  }

  if (is_contiguous && is_channels_last) {
    *format = suggests_channels_last(tensor) ? ConvMemoryFormat::ChannelsLast
                                             : ConvMemoryFormat::Contiguous;
  } else if (is_contiguous) {
    *format = ConvMemoryFormat::Contiguous;
  } else if (is_channels_last) {
    *format = ConvMemoryFormat::ChannelsLast;
  } else {
    return false;
  }
  return true;
}

// Reorders a 4D graph tensor from `N, d1, d2, C` to `N, C, d1, d2`.
MPSGraphTensor* channels_last_to_first(MPSGraph* graph, MPSGraphTensor* tensor) {
  tensor = [graph transposeTensor:tensor dimension:1 withDimension:3 name:nil];
  return [graph transposeTensor:tensor dimension:2 withDimension:3 name:nil];
}

// Reorders a 4D graph tensor from `N, C, d1, d2` to `N, d1, d2, C`.
MPSGraphTensor* channels_first_to_last(MPSGraph* graph, MPSGraphTensor* tensor) {
  tensor = [graph transposeTensor:tensor dimension:1 withDimension:2 name:nil];
  return [graph transposeTensor:tensor dimension:2 withDimension:3 name:nil];
}

} // namespace

extern "C" {

AOTITorchError aoti_torch_mps_convolution(
    AOTITensorHandle input,
    AOTITensorHandle weight,
    AOTITensorHandle* bias,
    const int64_t* stride,
    int64_t stride_len_,
    const int64_t* padding,
    int64_t padding_len_,
    const int64_t* dilation,
    int64_t dilation_len_,
    int32_t transposed,
    const int64_t* output_padding,
    int64_t output_padding_len_,
    int64_t groups,
    AOTITensorHandle* ret0) {
  ET_LOG(Debug, "aoti_torch_mps_convolution: Starting with input=%p, weight=%p, bias=%p, groups=%lld, transposed=%d",
         input, weight, bias, groups, transposed);

  if (!input || !weight || !ret0) {
    ET_LOG(Error, "aoti_torch_mps_convolution: null required handles (input, weight, or ret0)");
    return Error::InvalidArgument;
  }

  @autoreleasepool {
    try {
      // Convert AOTITensorHandle to ExecuTorch tensors
      auto input_tensor = reinterpret_cast<Tensor*>(input);
      auto weight_tensor = reinterpret_cast<Tensor*>(weight);

      // bias can be null for convolutions without bias
      Tensor* bias_tensor = nullptr;
      if (bias && *bias) {
        bias_tensor = reinterpret_cast<Tensor*>(*bias);
        ET_LOG(Debug, "aoti_torch_mps_convolution: Has bias tensor");
      } else {
        ET_LOG(Debug, "aoti_torch_mps_convolution: No bias tensor");
      }

      ET_LOG(Debug, "aoti_torch_mps_convolution: Converted tensor handles to ET tensors");

      // Log tensor shapes for debugging
      ET_LOG(Debug, "aoti_torch_mps_convolution: input shape: [%d, %d, %d, %d]",
             input_tensor->dim() > 0 ? (int)input_tensor->sizes()[0] : 0,
             input_tensor->dim() > 1 ? (int)input_tensor->sizes()[1] : 0,
             input_tensor->dim() > 2 ? (int)input_tensor->sizes()[2] : 0,
             input_tensor->dim() > 3 ? (int)input_tensor->sizes()[3] : 0);

      ET_LOG(Debug, "aoti_torch_mps_convolution: weight shape: [%d, %d, %d, %d]",
             weight_tensor->dim() > 0 ? (int)weight_tensor->sizes()[0] : 0,
             weight_tensor->dim() > 1 ? (int)weight_tensor->sizes()[1] : 0,
             weight_tensor->dim() > 2 ? (int)weight_tensor->sizes()[2] : 0,
             weight_tensor->dim() > 3 ? (int)weight_tensor->sizes()[3] : 0);

      // Log convolution parameters
      if (stride && stride_len_ >= 2) {
        ET_LOG(Debug, "aoti_torch_mps_convolution: stride: [%lld, %lld]", stride[0], stride[1]);
      }
      if (padding && padding_len_ >= 2) {
        ET_LOG(Debug, "aoti_torch_mps_convolution: padding: [%lld, %lld]", padding[0], padding[1]);
      }
      if (dilation && dilation_len_ >= 2) {
        ET_LOG(Debug, "aoti_torch_mps_convolution: dilation: [%lld, %lld]", dilation[0], dilation[1]);
      }
      if (output_padding && output_padding_len_ >= 2) {
        ET_LOG(Debug, "aoti_torch_mps_convolution: output_padding: [%lld, %lld]", output_padding[0], output_padding[1]);
      }

      // Support conv1d and conv2d by inspecting weight rank.
      // conv1d: weight dims = [C_out, C_in, K]
      // conv2d: weight dims = [C_out, C_in, Kh, Kw]
      bool is_conv1d = (weight_tensor->dim() == 3);

      // Accept input ranks:
      // conv1d: 2D (C,W) or 3D (N,C,W)
      // conv2d: 3D (C,H,W) or 4D (N,C,H,W)
      bool has_batch_dim = false;
      bool is_input_4d = false;
      int64_t N = 1, C_in = 0, H_in = 1, W_in = 0;
      if (is_conv1d) {
        if (input_tensor->dim() == 2) {
          // (C, W)
          has_batch_dim = false;
          C_in = input_tensor->sizes()[0];
          W_in = input_tensor->sizes()[1];
          H_in = 1;
        } else if (input_tensor->dim() == 3) {
          // (N, C, W)
          has_batch_dim = true;
          N = input_tensor->sizes()[0];
          C_in = input_tensor->sizes()[1];
          W_in = input_tensor->sizes()[2];
          H_in = 1;
        } else {
          ET_LOG(Error, "aoti_torch_mps_convolution: conv1d expects 2D or 3D input, got %d", (int)input_tensor->dim());
          return Error::InvalidArgument;
        }
      } else {
        is_input_4d = (input_tensor->dim() == 4);
        if (is_input_4d) {
          // (N, C, H, W)
          has_batch_dim = true;
          N = input_tensor->sizes()[0];
          C_in = input_tensor->sizes()[1];
          H_in = input_tensor->sizes()[2];
          W_in = input_tensor->sizes()[3];
        } else if (input_tensor->dim() == 3) {
          // (C, H, W)
          has_batch_dim = false;
          N = 1;
          C_in = input_tensor->sizes()[0];
          H_in = input_tensor->sizes()[1];
          W_in = input_tensor->sizes()[2];
        } else {
          ET_LOG(Error, "aoti_torch_mps_convolution: conv2d expects 3D or 4D input, got %d", (int)input_tensor->dim());
          return Error::InvalidArgument;
        }
      }

      // Inductor picks channels-last for conv2d graphs, so the operands cannot
      // be assumed to be contiguous. As in ATen, the output is channels-last
      // when either operand is.
      ConvMemoryFormat input_format = ConvMemoryFormat::Contiguous;
      ConvMemoryFormat weight_format = ConvMemoryFormat::Contiguous;
      if (!is_conv1d && is_input_4d) {
        if (!get_conv_memory_format(*input_tensor, &input_format)) {
          ET_LOG(Error, "aoti_torch_mps_convolution: input must be contiguous or channels-last, got strides [%d, %d, %d, %d]",
                 (int)input_tensor->strides()[0], (int)input_tensor->strides()[1],
                 (int)input_tensor->strides()[2], (int)input_tensor->strides()[3]);
          return Error::InvalidArgument;
        }
        if (!get_conv_memory_format(*weight_tensor, &weight_format)) {
          ET_LOG(Error, "aoti_torch_mps_convolution: weight must be contiguous or channels-last, got strides [%d, %d, %d, %d]",
                 (int)weight_tensor->strides()[0], (int)weight_tensor->strides()[1],
                 (int)weight_tensor->strides()[2], (int)weight_tensor->strides()[3]);
          return Error::InvalidArgument;
        }
      }
      const bool input_channels_last = input_format == ConvMemoryFormat::ChannelsLast;
      const bool weight_channels_last = weight_format == ConvMemoryFormat::ChannelsLast;
      const bool output_channels_last = input_channels_last || weight_channels_last;

      // Get weight dimensions
      int64_t C_out = weight_tensor->sizes()[0];  // output channels
      int64_t kernel_h = is_conv1d ? 1 : weight_tensor->sizes()[2];  // kernel height
      int64_t kernel_w = is_conv1d ? weight_tensor->sizes()[2] : weight_tensor->sizes()[3];  // kernel width

      // Calculate output spatial dimensions
      int64_t stride_h = is_conv1d ? 1 : (stride && stride_len_ > 0 ? stride[0] : 1);
      int64_t stride_w = is_conv1d ? (stride && stride_len_ > 0 ? stride[0] : 1)
                                   : (stride && stride_len_ > 1 ? stride[1] : 1);
      int64_t pad_h = is_conv1d ? 0 : (padding && padding_len_ > 0 ? padding[0] : 0);
      int64_t pad_w = is_conv1d ? (padding && padding_len_ > 0 ? padding[0] : 0)
                                : (padding && padding_len_ > 1 ? padding[1] : 0);
      int64_t dil_h = is_conv1d ? 1 : (dilation && dilation_len_ > 0 ? dilation[0] : 1);
      int64_t dil_w = is_conv1d ? (dilation && dilation_len_ > 0 ? dilation[0] : 1)
                                : (dilation && dilation_len_ > 1 ? dilation[1] : 1);

      int64_t H_out, W_out;
      if (transposed) {
        // For transposed convolution, output size calculation is different
        int64_t output_pad_h = is_conv1d ? 0 : (output_padding && output_padding_len_ > 0 ? output_padding[0] : 0);
        int64_t output_pad_w = is_conv1d ? (output_padding && output_padding_len_ > 0 ? output_padding[0] : 0)
                                         : (output_padding && output_padding_len_ > 1 ? output_padding[1] : 0);
        H_out = is_conv1d ? 1 : ((H_in - 1) * stride_h - 2 * pad_h + dil_h * (kernel_h - 1) + output_pad_h + 1);
        W_out = (W_in - 1) * stride_w - 2 * pad_w + dil_w * (kernel_w - 1) + output_pad_w + 1;
      } else {
        // Regular convolution output size calculation
        H_out = is_conv1d ? 1 : ((H_in + 2 * pad_h - dil_h * (kernel_h - 1) - 1) / stride_h + 1);
        W_out = (W_in + 2 * pad_w - dil_w * (kernel_w - 1) - 1) / stride_w + 1;
      }

      if (!is_conv1d && is_input_4d) {
        ET_LOG(Debug, "aoti_torch_mps_convolution: Calculated 4D output shape: [%lld, %lld, %lld, %lld]", N, C_out, H_out, W_out);
      } else if (!is_conv1d) {
        ET_LOG(Debug, "aoti_torch_mps_convolution: Calculated 3D output shape: [%lld, %lld, %lld]", C_out, H_out, W_out);
      } else if (is_conv1d && has_batch_dim) {
        ET_LOG(Debug, "aoti_torch_mps_convolution: Calculated 3D (1D conv) output shape: [%lld, %lld, %lld]", N, C_out, W_out);
      } else {
        ET_LOG(Debug, "aoti_torch_mps_convolution: Calculated 2D (1D conv) output shape: [%lld, %lld]", C_out, W_out);
      }

      // Validate output dimensions are positive
      if (N <= 0 || C_out <= 0 || H_out <= 0 || W_out <= 0) {
        ET_LOG(Error, "aoti_torch_mps_convolution: Invalid output dimensions N=%lld, C_out=%lld, H_out=%lld, W_out=%lld",
               N, C_out, H_out, W_out);
        return Error::InvalidArgument;
      }

      // Use the same dispatch pattern as other MPS operations for consistent synchronization
      ETMetalStream* stream = getCurrentMetalStream();
      if (!stream) {
        ET_LOG(Error, "aoti_torch_mps_convolution: Failed to get current Metal stream");
        return Error::Internal;
      }

      // Get Metal device
      id<MTLDevice> device = get_metal_device();
      if (!device) {
        ET_LOG(Error, "aoti_torch_mps_convolution: Failed to get Metal device");
        throw std::runtime_error("Failed to get Metal device");
      }

      // End any existing kernel coalescing to ensure a clean state for MPS
      stream->endKernelCoalescing();

      // Ensure stream is ready; command buffer handled internally by stream helpers

      // Determine data type and element size
      int32_t dtype = static_cast<int32_t>(input_tensor->scalar_type());
      MPSDataType mps_dtype;
      size_t element_size;

      if (dtype == static_cast<int32_t>(SupportedDTypes::FLOAT32)) {
        mps_dtype = MPSDataTypeFloat32;
        element_size = sizeof(float);
      } else if (dtype == static_cast<int32_t>(SupportedDTypes::BFLOAT16)) {
        mps_dtype = MPSDataTypeBFloat16;
        element_size = sizeof(uint16_t);  // bfloat16 is 16 bits
      } else {
        ET_LOG(Error, "aoti_torch_mps_convolution: Unsupported data type: %d", dtype);
        throw std::runtime_error("Unsupported data type for convolution");
      }

      ET_LOG(Debug, "aoti_torch_mps_convolution: mps_dtype=%d, element_size=%zu", mps_dtype, element_size);
      // Get weight's input channel dimension from the weight tensor (not from input)
      // For grouped convolutions, weight shape is [C_out, C_in/groups, kH, kW]
      int64_t weight_C_in = weight_tensor->sizes()[1];  // This handles grouped convs correctly

      // Define tensor shapes for placeholders (needed for both cache hit and miss)
      // These describe the buffers as they sit in memory, so a channels-last
      // operand is declared NHWC / OHWI and reordered inside the graph.
      NSArray<NSNumber*>* inputShape = input_channels_last
          ? @[@(N), @(H_in), @(W_in), @(C_in)]
          : @[@(N), @(C_in), @(H_in), @(W_in)];
      NSArray<NSNumber*>* weightShape = weight_channels_last
          ? @[@(C_out), @(kernel_h), @(kernel_w), @(weight_C_in)]
          : @[@(C_out), @(weight_C_in), @(kernel_h), @(kernel_w)];

      // Create cache key for this convolution
      GraphCacheKey cache_key;
      cache_key.op_name = "conv";
      // The layouts and the bias change the graph's topology, not just its
      // shapes, and a transposed convolution's output padding changes its
      // descriptor, so they all have to be part of the key.
      const int64_t key_output_pad_h = transposed && output_padding && output_padding_len_ > 0 ? output_padding[0] : 0;
      const int64_t key_output_pad_w = transposed && output_padding && output_padding_len_ > 1 ? output_padding[1] : 0;
      cache_key.shape_params = {N, C_in, H_in, W_in, C_out, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups,
                                input_channels_last, weight_channels_last, bias_tensor != nullptr,
                                key_output_pad_h, key_output_pad_w};
      cache_key.dtype = dtype;
      cache_key.transpose_flag = (transposed != 0);

      // Check if we have a cached graph
      MPSGraph* mpsGraph = nullptr;
      MPSGraphTensor* convOutput = nil;
      MPSGraphTensor* finalOutput = nil;
      MPSGraphTensor* inputPlaceholder = nil;
      MPSGraphTensor* weightPlaceholder = nil;
      MPSGraphTensor* biasPlaceholder = nil;
      bool has_bias = (bias_tensor != nullptr);

      auto cache_it = graph_cache.find(cache_key);
      if (cache_it != graph_cache.end()) {
        // Cache hit - reuse compiled graph and tensor references
        CachedGraph& cached = cache_it->second;
        mpsGraph = cached.graph;
        inputPlaceholder = cached.input1;
        weightPlaceholder = cached.input2;
        biasPlaceholder = cached.input3;  // May be nil if no bias
        finalOutput = cached.output;

        cache_stats.hits++;
        cache_stats.logStats();
        ET_LOG(Debug, "aoti_torch_mps_convolution: Using cached MPSGraph (cache hit, %zu total hits)", cache_stats.hits);

      } else {
        // Cache miss - create and compile new graph
        mpsGraph = [MPSGraph new];
        cache_stats.misses++;
        cache_stats.logStats();
        ET_LOG(Debug, "aoti_torch_mps_convolution: Created new MPSGraph instance (cache miss, %zu total misses)", cache_stats.misses);

        ET_LOG(Debug, "aoti_torch_mps_convolution: Creating placeholders with shapes input:[%d,%d,%d,%d] weight:[%d,%d,%d,%d]",
                (int)N, (int)C_in, (int)H_in, (int)W_in,
                (int)C_out, (int)C_in, (int)kernel_h, (int)kernel_w);

        // Create placeholders for input tensors
        inputPlaceholder = [mpsGraph placeholderWithShape:inputShape
                                                  dataType:mps_dtype
                                                      name:@"input"];
        weightPlaceholder = [mpsGraph placeholderWithShape:weightShape
                                                    dataType:mps_dtype
                                                        name:@"weight"];

        ET_LOG(Debug, "aoti_torch_mps_convolution: Created input and weight placeholders");

        // The convolution below is described as NCHW / OIHW.
        MPSGraphTensor* convInput = input_channels_last
            ? channels_last_to_first(mpsGraph, inputPlaceholder)
            : inputPlaceholder;
        MPSGraphTensor* convWeight = weight_channels_last
            ? channels_last_to_first(mpsGraph, weightPlaceholder)
            : weightPlaceholder;

        // Create convolution descriptor
        MPSGraphConvolution2DOpDescriptor* convDesc = [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:stride_w
                                                                                                        strideInY:stride_h
                                                                                                      dilationRateInX:dil_w
                                                                                                      dilationRateInY:dil_h
                                                                                                          groups:groups
                                                                                                          paddingLeft:pad_w
                                                                                                        paddingRight:pad_w
                                                                                                          paddingTop:pad_h
                                                                                                        paddingBottom:pad_h
                                                                                                          paddingStyle:MPSGraphPaddingStyleExplicit
                                                                                                          dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                                                                                                      weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

        ET_LOG(Debug, "aoti_torch_mps_convolution: Created convolution descriptor with stride=[%lld,%lld], padding=[%lld,%lld], dilation=[%lld,%lld], groups=%lld",
                stride_w, stride_h, pad_w, pad_h, dil_w, dil_h, groups);

        // Perform convolution using MPSGraph
        if (transposed) {
          ET_LOG(Debug, "aoti_torch_mps_convolution: Using transposed convolution");
          // For transposed convolution, we need to handle output padding
          int64_t output_pad_h = output_padding && output_padding_len_ > 0 ? output_padding[0] : 0;
          int64_t output_pad_w = output_padding && output_padding_len_ > 1 ? output_padding[1] : 0;

          // For transposed convolution, we need to adjust the padding calculation
          // In transposed convolution, the effective padding is typically negative
          // and we use output_padding to control the final output size
          int64_t transposed_pad_h = pad_h - output_pad_h;
          int64_t transposed_pad_w = pad_w - output_pad_w;

          // Create transposed convolution descriptor with adjusted padding
          MPSGraphConvolution2DOpDescriptor* transposedConvDesc = [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:stride_w
                                                                                                                    strideInY:stride_h
                                                                                                              dilationRateInX:dil_w
                                                                                                              dilationRateInY:dil_h
                                                                                                                      groups:groups
                                                                                                                paddingLeft:transposed_pad_w
                                                                                                                paddingRight:transposed_pad_w
                                                                                                                  paddingTop:transposed_pad_h
                                                                                                              paddingBottom:transposed_pad_h
                                                                                                                paddingStyle:MPSGraphPaddingStyleExplicit
                                                                                                                dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                                                                                                            weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

          convOutput = [mpsGraph convolution2DWithSourceTensor:convInput
                                                    weightsTensor:convWeight
                                                        descriptor:transposedConvDesc
                                                              name:@"transposed_convolution"];
        } else {
          ET_LOG(Debug, "aoti_torch_mps_convolution: Using regular convolution");
          convOutput = [mpsGraph convolution2DWithSourceTensor:convInput
                                                    weightsTensor:convWeight
                                                        descriptor:convDesc
                                                              name:@"convolution"];
        }

        ET_LOG(Debug, "aoti_torch_mps_convolution: Successfully created convolution tensor");

        // Handle bias if provided
        if (bias_tensor) {
          ET_LOG(Debug, "aoti_torch_mps_convolution: Adding bias to convolution output");

          // Create bias placeholder
          NSArray<NSNumber*>* biasShape = @[@(C_out)];
          biasPlaceholder = [mpsGraph placeholderWithShape:biasShape
                                                    dataType:mps_dtype
                                                        name:@"bias"];

          // Add bias to convolution output. MPSGraph broadcasts from the
          // trailing dimension, so the rank-1 bias has to be shaped to line up
          // with the channels of the NCHW result rather than with its width.
          MPSGraphTensor* biasPerChannel = [mpsGraph reshapeTensor:biasPlaceholder
                                                         withShape:@[@1, @(C_out), @1, @1]
                                                              name:@"bias_per_channel"];
          finalOutput = [mpsGraph additionWithPrimaryTensor:convOutput
                                            secondaryTensor:biasPerChannel
                                                        name:@"add_bias"];

          ET_LOG(Debug, "aoti_torch_mps_convolution: Added bias placeholder to graph");
        } else {
          finalOutput = convOutput;
        }

        if (output_channels_last) {
          finalOutput = channels_first_to_last(mpsGraph, finalOutput);
        }

        // Cache the compiled graph and tensor references for reuse
        CachedGraph cached_graph;
        cached_graph.graph = mpsGraph;
        cached_graph.input1 = inputPlaceholder;
        cached_graph.input2 = weightPlaceholder;
        cached_graph.input3 = biasPlaceholder;  // May be nil if no bias
        cached_graph.output = finalOutput;
        graph_cache[cache_key] = cached_graph;

        ET_LOG(Debug, "aoti_torch_mps_convolution: Cached compiled MPSGraph for future reuse");
      }  // End of cache miss block

      // Create feeds dictionary for graph execution
      NSMutableDictionary* feeds = [NSMutableDictionary dictionary];

      // Get Metal buffers from tensors
      bool settle_aliases = false;
      id<MTLBuffer> input_buffer = get_mtl_buffer(input_tensor, "aoti_torch_mps_convolution", "input", &settle_aliases);
      id<MTLBuffer> weight_buffer = get_mtl_buffer(weight_tensor, "aoti_torch_mps_convolution", "weight", &settle_aliases);

      ET_LOG(Debug, "aoti_torch_mps_convolution: Using existing Metal buffers - input=%p, weight=%p",
              input_buffer, weight_buffer);

      // Create MPSGraphTensorData objects for input tensors
      MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:input_buffer
                                                                                shape:inputShape
                                                                            dataType:mps_dtype];
      MPSGraphTensorData* weightData = [[MPSGraphTensorData alloc] initWithMTLBuffer:weight_buffer
                                                                                shape:weightShape
                                                                            dataType:mps_dtype];

      feeds[inputPlaceholder] = inputData;
      feeds[weightPlaceholder] = weightData;

      MPSGraphTensorData* biasData = nil;

      // Add bias data to feeds if provided
      if (bias_tensor && biasPlaceholder) {
        id<MTLBuffer> bias_buffer = get_mtl_buffer(bias_tensor, "aoti_torch_mps_convolution", "bias", &settle_aliases);

        NSArray<NSNumber*>* biasShape = @[@(C_out)];
        biasData = [[MPSGraphTensorData alloc] initWithMTLBuffer:bias_buffer
                                                           shape:biasShape
                                                        dataType:mps_dtype];

        feeds[biasPlaceholder] = biasData;
        ET_LOG(Debug, "aoti_torch_mps_convolution: Added bias tensor to feeds");
      }

      ET_LOG(Debug, "aoti_torch_mps_convolution: Created feeds dictionary");

      // Create Metal buffer for output tensor
      size_t output_size_bytes = N * C_out * H_out * W_out * element_size;
      void* output_contents_ptr = nullptr;
      id<MTLBuffer> output_buffer = allocate_mtl_buffer(&output_contents_ptr, output_size_bytes);

      // Create results dictionary (MPSGraph output is 4D)
      NSArray<NSNumber*>* outputShape = output_channels_last
          ? @[@(N), @(H_out), @(W_out), @(C_out)]
          : @[@(N), @(C_out), @(H_out), @(W_out)];
      MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:output_buffer
                                                                                shape:outputShape
                                                                              dataType:mps_dtype];

      NSDictionary* results = @{finalOutput: outputData};
      ET_LOG(Debug, "aoti_torch_mps_convolution: Created results dictionary");

      // Execute the MPSGraph
      ET_LOG(Debug, "aoti_torch_mps_convolution: Executing MPSGraph");

      @try {
        // Use stream helper to encode and synchronize correctly
        stream->executeMPSGraph(mpsGraph, feeds, results, SyncType::COMMIT, settle_aliases);
      } @catch (NSException *exception) {
        ET_LOG(Error, "aoti_torch_mps_convolution: NSException caught during executeMPSGraph: %s - %s",
              [[exception name] UTF8String], [[exception reason] UTF8String]);
        throw std::runtime_error("MPSGraph execution failed with NSException");
      } @catch (...) {
        ET_LOG(Error, "aoti_torch_mps_convolution: MPSGraph execution failed");
        throw std::runtime_error("MPSGraph execution failed");
      }

      ET_LOG(Debug, "aoti_torch_mps_convolution: MPSGraph execution completed successfully");

      // Create output tensor handle on device (MPS) that points to GPU buffer
      std::vector<int64_t> output_sizes_int64;
      std::vector<int64_t> output_strides;
      if (!is_conv1d && is_input_4d) {
        output_sizes_int64 = {N, C_out, H_out, W_out};
        if (output_channels_last) {
          // Channels-last (NHWC) strides
          output_strides = {
              H_out * W_out * C_out,
              1,
              W_out * C_out,
              C_out
          };
        } else {
          // Contiguous NCHW strides
          output_strides = {
              C_out * H_out * W_out,
              H_out * W_out,
              W_out,
              1
          };
        }
      } else if (!is_conv1d) {
        output_sizes_int64 = {C_out, H_out, W_out};
        // Contiguous CHW strides
        output_strides = {
            H_out * W_out,
            W_out,
            1
        };
      } else if (is_conv1d && has_batch_dim) {
        output_sizes_int64 = {N, C_out, W_out};
        // Contiguous NCW strides
        output_strides = {
            C_out * W_out,
            W_out,
            1
        };
      } else {
        output_sizes_int64 = {C_out, W_out};
        // Contiguous CW strides
        output_strides = {
            W_out,
            1
        };
      }

      // Use the GPU buffer contents pointer directly for the tensor storage
      void* tensor_data = output_contents_ptr;

      AOTITensorHandle output_tensor_handle = nullptr;

      AOTITorchError create_result = aoti_torch_create_tensor_from_blob_v2(
          tensor_data,
          static_cast<int64_t>(output_sizes_int64.size()),  // ndim
          output_sizes_int64.data(),
          output_strides.data(),
          0,  // storage_offset
          dtype,  // dtype
          13,  // device_type (MPS)
          0,  // device_index
          &output_tensor_handle,
          0,  // layout (strided)
          nullptr,  // opaque_metadata
          0   // opaque_metadata_size
      );

      if (create_result != Error::Ok || !output_tensor_handle) {
        ET_LOG(Error, "aoti_torch_mps_convolution: Failed to create output tensor, error code: %d", static_cast<int>(create_result));
        aoti_torch_mps_free(tensor_data);  // Free the allocated GPU memory on failure
        throw std::runtime_error("Failed to create output tensor");
      }

      // Verify the tensor was created with the correct size
      auto* et_tensor = reinterpret_cast<Tensor*>(output_tensor_handle);
      size_t actual_numel = et_tensor->numel();
      size_t expected_numel = static_cast<size_t>(N * C_out * H_out * W_out);

      if (actual_numel != expected_numel) {
        ET_LOG(Error, "aoti_torch_mps_convolution: Tensor size mismatch. Expected %zu, got %zu", expected_numel, actual_numel);
        aoti_torch_mps_free(tensor_data);  // Free the allocated GPU memory on failure
        throw std::runtime_error("Tensor size mismatch");
      }

      // Store the tensor handle - mark that we own the memory since we manually allocated it
      *ret0 = output_tensor_handle;
      // Mark that we own the memory for these tensors
      // Note: memory_to_n_tensor is managed automatically in aoti_torch_create_tensor_from_blob_v2
      // The function sets it to NOT_OWN, but we need to change it to 1 since we allocated it
      extern std::unordered_map<void*, int32_t> memory_to_n_tensor;
      memory_to_n_tensor[tensor_data] = 1;

      [inputData release];
      [weightData release];
      if (biasData) [biasData release];
      [outputData release];

      ET_LOG(Debug, "aoti_torch_mps_convolution: Created output tensor with %zu elements using MPSGraph", actual_numel);

      ET_LOG(Debug, "aoti_torch_mps_convolution: Executed successfully");
      return Error::Ok;

    } catch (const std::exception& e) {
      ET_LOG(Error, "aoti_torch_mps_convolution exception: %s", e.what());
      return Error::Internal;
    } catch (...) {
      ET_LOG(Error, "aoti_torch_mps_convolution: unknown exception");
      return Error::Internal;
    }
  }
}


} // extern "C"

} // namespace metal
} // namespace backends
} // namespace executorch
