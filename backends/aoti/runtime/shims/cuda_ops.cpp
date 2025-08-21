/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cuda_ops.h"
#include "memory.h"
#include "tensor_attribute.h"
#include <iostream>
#include <cudnn.h>
#include <cublas_v2.h>

namespace executorch {
namespace backends {
namespace aoti {

using executorch::runtime::Error;
using executorch::runtime::etensor::Tensor;

// Global cuDNN handle
static cudnnHandle_t cudnn_handle = nullptr;

// Initialize cuDNN handle
static void init_cudnn() {
    if (cudnn_handle == nullptr) {
        cudnnCreate(&cudnn_handle);
    }
}

extern "C" {

AOTITorchError aoti_torch_cuda_addmm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat1,
    AtenTensorHandle mat2,
    double beta,
    double alpha) {
    
    std::cout << "aoti_torch_cuda_addmm_out called with beta=" << beta << ", alpha=" << alpha << std::endl;
    
    // Get tensor dimensions
    auto mat1_sizes = mat1->sizes();
    auto mat2_sizes = mat2->sizes();
    auto self_sizes = self->sizes();
    auto out_sizes = out->sizes();
    
    // mat1: [M, K], mat2: [K, N], result: [M, N]
    int64_t M = mat1_sizes[0];
    int64_t K = mat1_sizes[1];
    int64_t N = mat2_sizes[1];
    
    std::cout << "ADDMM: mat1[" << M << "," << K << "] @ mat2[" << K << "," << N << "] -> out[" << M << "," << N << "]" << std::endl;
    
    // Use cuBLAS for matrix multiplication
    cublasHandle_t cublas_handle;
    cublasStatus_t cublas_status = cublasCreate(&cublas_handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Failed to create cuBLAS handle" << std::endl;
        return Error::Internal;
    }
    
    // Set cuBLAS to use tensor op math for better performance
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
    
    const float f_alpha = static_cast<float>(alpha);
    const float f_beta = static_cast<float>(beta);
    
    // Perform: out = beta * self + alpha * (mat1 @ mat2)
    // First: out = beta * self (copy self to out and scale)
    if (beta != 0.0) {
        Error copy_err = aoti_torch_copy_(out, self, 0);
        if (copy_err != Error::Ok) {
            cublasDestroy(cublas_handle);
            return copy_err;
        }
        
        // Scale by beta if not 1.0
        if (beta != 1.0) {
            cublas_status = cublasSscal(cublas_handle, M * N, &f_beta, 
                                       static_cast<float*>(out->mutable_data_ptr()), 1);
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "cuBLAS scale failed" << std::endl;
                cublasDestroy(cublas_handle);
                return Error::Internal;
            }
        }
    } else {
        // Zero out the output tensor
        cudaMemset(out->mutable_data_ptr(), 0, M * N * sizeof(float));
    }
    
    // Then: out += alpha * (mat1 @ mat2)
    // cuBLAS uses column-major, so we compute: C = alpha * A^T * B^T + beta * C
    // Which gives us: out = alpha * mat1 @ mat2 + beta * out
    const float gemm_beta = 1.0f; // Since we already handled the beta scaling above
    
    cublas_status = cublasSgemm(
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose for column-major interpretation
        N, M, K,                   // Dimensions swapped for column-major
        &f_alpha,                  // alpha
        static_cast<const float*>(mat2->data_ptr()), N,  // B matrix (mat2)
        static_cast<const float*>(mat1->data_ptr()), K,  // A matrix (mat1)  
        &gemm_beta,                // beta (1.0 since we pre-scaled)
        static_cast<float*>(out->mutable_data_ptr()), N  // C matrix (out)
    );
    
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS GEMM failed: " << cublas_status << std::endl;
        cublasDestroy(cublas_handle);
        return Error::Internal;
    }
    
    cublasDestroy(cublas_handle);
    
    std::cout << "aoti_torch_cuda_addmm_out completed successfully" << std::endl;
    return Error::Ok;
}

AOTITorchError aoti_torch_cuda_convolution(
    AtenTensorHandle input,
    AtenTensorHandle weight,
    AtenTensorHandle* bias,
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
    AtenTensorHandle* ret0) {
    
    std::cout << "aoti_torch_cuda_convolution called" << std::endl;
    
    init_cudnn();
    
    // Get input dimensions
    auto input_sizes = input->sizes();
    auto weight_sizes = weight->sizes();
    
    int batch_size = input_sizes[0];
    int input_channels = input_sizes[1];
    int input_height = input_sizes[2];
    int input_width = input_sizes[3];
    
    int output_channels = weight_sizes[0];
    int kernel_height = weight_sizes[2];
    int kernel_width = weight_sizes[3];
    
    // Calculate output dimensions
    int output_height = (input_height + 2 * padding[0] - dilation[0] * (kernel_height - 1) - 1) / stride[0] + 1;
    int output_width = (input_width + 2 * padding[1] - dilation[1] * (kernel_width - 1) - 1) / stride[1] + 1;
    
    std::cout << "Conv2d: input[" << batch_size << "," << input_channels << "," << input_height << "," << input_width << "]"
              << " -> output[" << batch_size << "," << output_channels << "," << output_height << "," << output_width << "]" << std::endl;
    
    // Create output tensor
    std::vector<int64_t> output_sizes = {batch_size, output_channels, output_height, output_width};
    
    AOTITensorHandle output_handle;
    Error create_err = aoti_torch_empty_strided(
        output_sizes.size(),
        output_sizes.data(),
        nullptr, // use default strides
        6, // float32 dtype
        1, // cuda device
        0, // device index
        &output_handle);
    
    if (create_err != Error::Ok) {
        std::cerr << "Failed to create output tensor for convolution" << std::endl;
        return create_err;
    }
    
    // Setup cuDNN descriptors
    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t weight_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnCreateTensorDescriptor(&bias_desc);
    cudnnCreateFilterDescriptor(&weight_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);
    
    // Set tensor descriptors
    cudnnSetTensorNdDescriptor(input_desc, CUDNN_DATA_FLOAT, 4, 
                               (int*)input_sizes.data(), 
                               (int*)input->strides().data());
    
    cudnnSetTensorNdDescriptor(output_desc, CUDNN_DATA_FLOAT, 4,
                               (int*)output_sizes.data(),
                               (int*)output_handle->strides().data());
    
    cudnnSetFilterNdDescriptor(weight_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4,
                               (int*)weight_sizes.data());
    
    // Set convolution descriptor
    cudnnSetConvolutionNdDescriptor(conv_desc, 2,
                                    (int*)padding, (int*)stride, (int*)dilation,
                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    
    if (groups > 1) {
        cudnnSetConvolutionGroupCount(conv_desc, groups);
    }
    
    // Find best convolution algorithm
    cudnnConvolutionFwdAlgo_t algo;
    cudnnGetConvolutionForwardAlgorithm(cudnn_handle, input_desc, weight_desc, conv_desc, output_desc,
                                        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);
    
    // Get workspace size
    size_t workspace_size;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle, input_desc, weight_desc, conv_desc, output_desc, algo, &workspace_size);
    
    // Allocate workspace
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }
    
    // Perform convolution
    const float alpha = 1.0f, beta = 0.0f;
    cudnnStatus_t conv_status = cudnnConvolutionForward(
        cudnn_handle,
        &alpha,
        input_desc, input->data_ptr(),
        weight_desc, weight->data_ptr(),
        conv_desc, algo,
        workspace, workspace_size,
        &beta,
        output_desc, output_handle->mutable_data_ptr());
    
    if (conv_status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "cuDNN convolution failed: " << cudnnGetErrorString(conv_status) << std::endl;
        if (workspace) cudaFree(workspace);
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroyTensorDescriptor(bias_desc);
        cudnnDestroyFilterDescriptor(weight_desc);
        cudnnDestroyConvolutionDescriptor(conv_desc);
        aoti_torch_delete_tensor_object(output_handle);
        return Error::Internal;
    }
    
    // Add bias if present
    if (bias && *bias) {
        auto bias_sizes = (*bias)->sizes();
        cudnnSetTensorNdDescriptor(bias_desc, CUDNN_DATA_FLOAT, 4,
                                   (int*)bias_sizes.data(),
                                   (int*)(*bias)->strides().data());
        
        cudnnAddTensor(cudnn_handle, &alpha, bias_desc, (*bias)->data_ptr(),
                       &alpha, output_desc, output_handle->mutable_data_ptr());
    }
    
    // Cleanup
    if (workspace) cudaFree(workspace);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyFilterDescriptor(weight_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    
    *ret0 = output_handle;
    
    std::cout << "aoti_torch_cuda_convolution completed successfully" << std::endl;
    return Error::Ok;
}

} // extern "C"

} // namespace aoti
} // namespace backends
} // namespace executorch