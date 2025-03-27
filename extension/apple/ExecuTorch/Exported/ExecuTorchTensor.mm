/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecuTorchTensor.h"

#import "ExecuTorchError.h"
#import "ExecuTorchUtils.h"

#import <executorch/extension/tensor/tensor.h>

using namespace executorch::aten;
using namespace executorch::extension;
using namespace executorch::runtime;

NSInteger ExecuTorchSizeOfDataType(ExecuTorchDataType dataType) {
  return elementSize(static_cast<ScalarType>(dataType));
}

NSInteger ExecuTorchElementCountOfShape(NSArray<NSNumber *> *shape) {
  NSInteger count = 1;
  for (NSNumber *dimension in shape) {
    count *= dimension.integerValue;
  }
  return count;
}

@implementation ExecuTorchTensor {
  TensorPtr _tensor;
  NSArray<NSNumber *> *_shape;
  NSArray<NSNumber *> *_strides;
  NSArray<NSNumber *> *_dimensionOrder;
}

- (instancetype)initWithNativeInstance:(void *)nativeInstance {
  ET_CHECK(nativeInstance);
  if (self = [super init]) {
    _tensor = std::move(*reinterpret_cast<TensorPtr *>(nativeInstance));
    ET_CHECK(_tensor);
  }
  return self;
}

- (instancetype)initWithTensor:(ExecuTorchTensor *)otherTensor {
  ET_CHECK(otherTensor);
  auto tensor = make_tensor_ptr(
    **reinterpret_cast<TensorPtr *>(otherTensor.nativeInstance)
  );
  return [self initWithNativeInstance:&tensor];
}

- (instancetype)copy {
  return [self copyWithZone:nil];
}

- (instancetype)copyWithZone:(nullable NSZone *)zone {
  auto tensor = clone_tensor_ptr(_tensor);
  return [[ExecuTorchTensor allocWithZone:zone] initWithNativeInstance:&tensor];
}

- (void *)nativeInstance {
  return &_tensor;
}

- (ExecuTorchDataType)dataType {
  return static_cast<ExecuTorchDataType>(_tensor->scalar_type());
}

- (NSArray<NSNumber *> *)shape {
  if (!_shape) {
    _shape = utils::toNSArray(_tensor->sizes());
  }
  return _shape;
}

- (NSArray<NSNumber *> *)dimensionOrder {
  if (!_dimensionOrder) {
    _dimensionOrder = utils::toNSArray(_tensor->dim_order());
  }
  return _dimensionOrder;
}

- (NSArray<NSNumber *> *)strides {
  if (!_strides) {
    _strides = utils::toNSArray(_tensor->strides());
  }
  return _strides;
}

- (ExecuTorchShapeDynamism)shapeDynamism {
  return static_cast<ExecuTorchShapeDynamism>(_tensor->shape_dynamism());
}

- (NSInteger)count {
  return _tensor->numel();
}

- (void)bytesWithHandler:(void (^)(const void *pointer, NSInteger count, ExecuTorchDataType type))handler {
  ET_CHECK(handler);
  handler(_tensor->unsafeGetTensorImpl()->data(), self.count, self.dataType);
}

- (void)mutableBytesWithHandler:(void (^)(void *pointer, NSInteger count, ExecuTorchDataType dataType))handler {
  ET_CHECK(handler);
  handler(_tensor->unsafeGetTensorImpl()->mutable_data(), self.count, self.dataType);
}

- (BOOL)resizeToShape:(NSArray<NSNumber *> *)shape
                error:(NSError **)error {
  const auto resizeError = resize_tensor_ptr(
    _tensor, utils::toVector<SizesType>(shape)
  );
  if (resizeError != Error::Ok) {
    if (error) {
      *error = [NSError errorWithDomain:ExecuTorchErrorDomain
                                   code:(NSInteger)resizeError
                               userInfo:nil];
    }
    return NO;
  }
  _shape = nil;
  _strides = nil;
  _dimensionOrder = nil;
  return YES;
}

@end

@implementation ExecuTorchTensor (BytesNoCopy)

- (instancetype)initWithBytesNoCopy:(void *)pointer
                              shape:(NSArray<NSNumber *> *)shape
                            strides:(NSArray<NSNumber *> *)strides
                     dimensionOrder:(NSArray<NSNumber *> *)dimensionOrder
                           dataType:(ExecuTorchDataType)dataType
                      shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism {
  ET_CHECK(pointer);
  auto tensor = make_tensor_ptr(
    utils::toVector<SizesType>(shape),
    pointer,
    utils::toVector<DimOrderType>(dimensionOrder),
    utils::toVector<StridesType>(strides),
    static_cast<ScalarType>(dataType),
    static_cast<TensorShapeDynamism>(shapeDynamism)
  );
  return [self initWithNativeInstance:&tensor];
}

- (instancetype)initWithBytesNoCopy:(void *)pointer
                              shape:(NSArray<NSNumber *> *)shape
                            strides:(NSArray<NSNumber *> *)strides
                     dimensionOrder:(NSArray<NSNumber *> *)dimensionOrder
                           dataType:(ExecuTorchDataType)dataType {
  return [self initWithBytesNoCopy:pointer
                             shape:shape
                           strides:strides
                    dimensionOrder:dimensionOrder
                          dataType:dataType
                     shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

- (instancetype)initWithBytesNoCopy:(void *)pointer
                              shape:(NSArray<NSNumber *> *)shape
                           dataType:(ExecuTorchDataType)dataType
                      shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism {
  return [self initWithBytesNoCopy:pointer
                             shape:shape
                           strides:@[]
                    dimensionOrder:@[]
                          dataType:dataType
                     shapeDynamism:shapeDynamism];
}

- (instancetype)initWithBytesNoCopy:(void *)pointer
                              shape:(NSArray<NSNumber *> *)shape
                           dataType:(ExecuTorchDataType)dataType {
  return [self initWithBytesNoCopy:pointer
                             shape:shape
                           strides:@[]
                    dimensionOrder:@[]
                          dataType:dataType
                     shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

@end

@implementation ExecuTorchTensor (Bytes)

- (instancetype)initWithBytes:(const void *)pointer
                        shape:(NSArray<NSNumber *> *)shape
                      strides:(NSArray<NSNumber *> *)strides
               dimensionOrder:(NSArray<NSNumber *> *)dimensionOrder
                     dataType:(ExecuTorchDataType)dataType
                shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism {
  ET_CHECK(pointer);
  const auto size = ExecuTorchElementCountOfShape(shape) * ExecuTorchSizeOfDataType(dataType);
  std::vector<uint8_t> data(static_cast<const uint8_t *>(pointer),
                            static_cast<const uint8_t *>(pointer) + size);
  auto tensor = make_tensor_ptr(
    utils::toVector<SizesType>(shape),
    std::move(data),
    utils::toVector<DimOrderType>(dimensionOrder),
    utils::toVector<StridesType>(strides),
    static_cast<ScalarType>(dataType),
    static_cast<TensorShapeDynamism>(shapeDynamism)
  );
  return [self initWithNativeInstance:&tensor];
}

- (instancetype)initWithBytes:(const void *)pointer
                        shape:(NSArray<NSNumber *> *)shape
                      strides:(NSArray<NSNumber *> *)strides
               dimensionOrder:(NSArray<NSNumber *> *)dimensionOrder
                     dataType:(ExecuTorchDataType)dataType {
  return [self initWithBytes:pointer
                       shape:shape
                     strides:strides
              dimensionOrder:dimensionOrder
                    dataType:dataType
               shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

- (instancetype)initWithBytes:(const void *)pointer
                        shape:(NSArray<NSNumber *> *)shape
                     dataType:(ExecuTorchDataType)dataType
                shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism {
  return [self initWithBytes:pointer
                       shape:shape
                     strides:@[]
              dimensionOrder:@[]
                    dataType:dataType
               shapeDynamism:shapeDynamism];
}

- (instancetype)initWithBytes:(const void *)pointer
                        shape:(NSArray<NSNumber *> *)shape
                     dataType:(ExecuTorchDataType)dataType {
  return [self initWithBytes:pointer
                       shape:shape
                     strides:@[]
              dimensionOrder:@[]
                    dataType:dataType
               shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

@end
