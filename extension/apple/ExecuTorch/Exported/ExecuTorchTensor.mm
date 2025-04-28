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
  NSData *_data;
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

- (BOOL)isEqualToTensor:(nullable ExecuTorchTensor *)other {
  if (!other) {
    return NO;
  }
  const auto *data = _tensor->unsafeGetTensorImpl()->data();
  const auto *otherData = other->_tensor->unsafeGetTensorImpl()->data();
  const auto size = self.count * ExecuTorchSizeOfDataType(self.dataType);
  return self.dataType == other.dataType &&
         self.count == other.count &&
         [self.shape isEqual:other.shape] &&
         [self.dimensionOrder isEqual:other.dimensionOrder] &&
         [self.strides isEqual:other.strides] &&
         self.shapeDynamism == other.shapeDynamism &&
         (data && otherData ? std::memcmp(data, otherData, size) == 0 : data == otherData);
}

- (BOOL)isEqual:(nullable id)other {
  if (self == other) {
    return YES;
  }
  if (![other isKindOfClass:[ExecuTorchTensor class]]) {
    return NO;
  }
  return [self isEqualToTensor:(ExecuTorchTensor *)other];
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

@implementation ExecuTorchTensor (Data)

- (instancetype)initWithData:(NSData *)data
                       shape:(NSArray<NSNumber *> *)shape
                     strides:(NSArray<NSNumber *> *)strides
              dimensionOrder:(NSArray<NSNumber *> *)dimensionOrder
                    dataType:(ExecuTorchDataType)dataType
               shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism {
  ET_CHECK_MSG(data.length >= ExecuTorchElementCountOfShape(shape) * ExecuTorchSizeOfDataType(dataType),
               "Data length is too small");
  self = [self initWithBytesNoCopy:(void *)data.bytes
                             shape:shape
                           strides:strides
                    dimensionOrder:dimensionOrder
                          dataType:dataType
                     shapeDynamism:shapeDynamism];
  if (self) {
    _data = data;
  }
  return self;
}

- (instancetype)initWithData:(NSData *)data
                       shape:(NSArray<NSNumber *> *)shape
                     strides:(NSArray<NSNumber *> *)strides
              dimensionOrder:(NSArray<NSNumber *> *)dimensionOrder
                    dataType:(ExecuTorchDataType)dataType {
  return [self initWithData:data
                      shape:shape
                    strides:strides
             dimensionOrder:dimensionOrder
                   dataType:dataType
              shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

- (instancetype)initWithData:(NSData *)data
                       shape:(NSArray<NSNumber *> *)shape
                    dataType:(ExecuTorchDataType)dataType
               shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism {
  return [self initWithData:data
                      shape:shape
                    strides:@[]
             dimensionOrder:@[]
                   dataType:dataType
              shapeDynamism:shapeDynamism];
}

- (instancetype)initWithData:(NSData *)data
                       shape:(NSArray<NSNumber *> *)shape
                    dataType:(ExecuTorchDataType)dataType {
  return [self initWithData:data
                      shape:shape
                    strides:@[]
             dimensionOrder:@[]
                   dataType:dataType
              shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

@end

@implementation ExecuTorchTensor (Scalars)

- (instancetype)initWithScalars:(NSArray<NSNumber *> *)scalars
                          shape:(NSArray<NSNumber *> *)shape
                        strides:(NSArray<NSNumber *> *)strides
                 dimensionOrder:(NSArray<NSNumber *> *)dimensionOrder
                       dataType:(ExecuTorchDataType)dataType
                  shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism {
  const NSInteger count = scalars.count;
  ET_CHECK_MSG(count == ExecuTorchElementCountOfShape(shape),
               "Number of scalars does not match the shape");
  std::vector<uint8_t> data;
  data.resize(count * ExecuTorchSizeOfDataType(dataType));
  for (NSUInteger index = 0; index < count; ++index) {
    ET_SWITCH_REALHBBF16_AND_UINT_TYPES(
      static_cast<ScalarType>(dataType), nil, "initWithScalars", CTYPE, [&] {
        reinterpret_cast<CTYPE *>(data.data())[index] = utils::extractValue<CTYPE>(scalars[index]);
      }
    );
  }
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

- (instancetype)initWithScalars:(NSArray<NSNumber *> *)scalars
                          shape:(NSArray<NSNumber *> *)shape
                        strides:(NSArray<NSNumber *> *)strides
                 dimensionOrder:(NSArray<NSNumber *> *)dimensionOrder
                       dataType:(ExecuTorchDataType)dataType {
  return [self initWithScalars:scalars
                         shape:shape
                       strides:strides
                dimensionOrder:dimensionOrder
                      dataType:dataType
                 shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

- (instancetype)initWithScalars:(NSArray<NSNumber *> *)scalars
                          shape:(NSArray<NSNumber *> *)shape
                       dataType:(ExecuTorchDataType)dataType
                  shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism {
  return [self initWithScalars:scalars
                         shape:shape
                       strides:@[]
                dimensionOrder:@[]
                      dataType:dataType
                 shapeDynamism:shapeDynamism];
}

- (instancetype)initWithScalars:(NSArray<NSNumber *> *)scalars
                          shape:(NSArray<NSNumber *> *)shape
                       dataType:(ExecuTorchDataType)dataType {
  return [self initWithScalars:scalars
                         shape:shape
                       strides:@[]
                dimensionOrder:@[]
                      dataType:dataType
                 shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

- (instancetype)initWithScalars:(NSArray<NSNumber *> *)scalars
                       dataType:(ExecuTorchDataType)dataType
                  shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism {
  return [self initWithScalars:scalars
                         shape:@[@(scalars.count)]
                       strides:@[]
                dimensionOrder:@[]
                      dataType:dataType
                 shapeDynamism:shapeDynamism];
}

- (instancetype)initWithScalars:(NSArray<NSNumber *> *)scalars
                       dataType:(ExecuTorchDataType)dataType {
  return [self initWithScalars:scalars
                         shape:@[@(scalars.count)]
                       strides:@[]
                dimensionOrder:@[]
                      dataType:dataType
                 shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

- (instancetype)initWithScalars:(NSArray<NSNumber *> *)scalars
                          shape:(NSArray<NSNumber *> *)shape
                  shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism {
  return [self initWithScalars:scalars
                         shape:shape
                       strides:@[]
                dimensionOrder:@[]
                      dataType:static_cast<ExecuTorchDataType>(utils::deduceType(scalars.firstObject))
                 shapeDynamism:shapeDynamism];
}

- (instancetype)initWithScalars:(NSArray<NSNumber *> *)scalars
                          shape:(NSArray<NSNumber *> *)shape {
  return [self initWithScalars:scalars
                         shape:shape
                       strides:@[]
                dimensionOrder:@[]
                      dataType:static_cast<ExecuTorchDataType>(utils::deduceType(scalars.firstObject))
                 shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

- (instancetype)initWithScalars:(NSArray<NSNumber *> *)scalars {
  return [self initWithScalars:scalars
                         shape:@[@(scalars.count)]
                       strides:@[]
                dimensionOrder:@[]
                      dataType:static_cast<ExecuTorchDataType>(utils::deduceType(scalars.firstObject))
                 shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

@end

@implementation ExecuTorchTensor (Scalar)

- (instancetype)initWithScalar:(NSNumber *)scalar
                      dataType:(ExecuTorchDataType)dataType {
  return [self initWithScalars:@[scalar]
                         shape:@[]
                       strides:@[]
                dimensionOrder:@[]
                      dataType:dataType
                 shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

- (instancetype)initWithScalar:(NSNumber *)scalar {
  return [self initWithScalars:@[scalar]
                         shape:@[]
                       strides:@[]
                dimensionOrder:@[]
                      dataType:static_cast<ExecuTorchDataType>(utils::deduceType(scalar))
                 shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

- (instancetype)initWithByte:(uint8_t)scalar {
  return [self initWithBytes:&scalar
                       shape:@[]
                     strides:@[]
              dimensionOrder:@[]
                    dataType:ExecuTorchDataTypeByte
               shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

- (instancetype)initWithChar:(int8_t)scalar {
  return [self initWithBytes:&scalar
                       shape:@[]
                     strides:@[]
              dimensionOrder:@[]
                    dataType:ExecuTorchDataTypeChar
               shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

- (instancetype)initWithShort:(int16_t)scalar {
  return [self initWithBytes:&scalar
                       shape:@[]
                     strides:@[]
              dimensionOrder:@[]
                    dataType:ExecuTorchDataTypeShort
               shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

- (instancetype)initWithInt:(int32_t)scalar {
  return [self initWithBytes:&scalar
                       shape:@[]
                     strides:@[]
              dimensionOrder:@[]
                    dataType:ExecuTorchDataTypeInt
               shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

- (instancetype)initWithLong:(int64_t)scalar {
  return [self initWithBytes:&scalar
                       shape:@[]
                     strides:@[]
              dimensionOrder:@[]
                    dataType:ExecuTorchDataTypeLong
               shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

- (instancetype)initWithFloat:(float)scalar {
  return [self initWithBytes:&scalar
                       shape:@[]
                     strides:@[]
              dimensionOrder:@[]
                    dataType:ExecuTorchDataTypeFloat
               shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

- (instancetype)initWithDouble:(double)scalar {
  return [self initWithBytes:&scalar
                       shape:@[]
                     strides:@[]
              dimensionOrder:@[]
                    dataType:ExecuTorchDataTypeDouble
               shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

- (instancetype)initWithBool:(BOOL)scalar {
  return [self initWithBytes:&scalar
                       shape:@[]
                     strides:@[]
              dimensionOrder:@[]
                    dataType:ExecuTorchDataTypeBool
               shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

- (instancetype)initWithUInt16:(uint16_t)scalar {
  return [self initWithBytes:&scalar
                       shape:@[]
                     strides:@[]
              dimensionOrder:@[]
                    dataType:ExecuTorchDataTypeUInt16
               shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

- (instancetype)initWithUInt32:(uint32_t)scalar {
  return [self initWithBytes:&scalar
                       shape:@[]
                     strides:@[]
              dimensionOrder:@[]
                    dataType:ExecuTorchDataTypeUInt32
               shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

- (instancetype)initWithUInt64:(uint64_t)scalar {
  return [self initWithBytes:&scalar
                       shape:@[]
                     strides:@[]
              dimensionOrder:@[]
                    dataType:ExecuTorchDataTypeUInt64
               shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

- (instancetype)initWithInteger:(NSInteger)scalar {
  return [self initWithBytes:&scalar
                       shape:@[]
                     strides:@[]
              dimensionOrder:@[]
                    dataType:(sizeof(scalar) == 8 ? ExecuTorchDataTypeLong : ExecuTorchDataTypeInt)
               shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

- (instancetype)initWithUnsignedInteger:(NSUInteger)scalar {
  return [self initWithBytes:&scalar
                       shape:@[]
                     strides:@[]
              dimensionOrder:@[]
                    dataType:(sizeof(scalar) == 8 ? ExecuTorchDataTypeUInt64 : ExecuTorchDataTypeUInt32)
               shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

@end
