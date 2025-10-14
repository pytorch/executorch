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

static inline NSString *dataTypeDescription(ExecuTorchDataType dataType) {
  switch (dataType) {
    case ExecuTorchDataTypeByte:
      return @"byte";
    case ExecuTorchDataTypeChar:
      return @"char";
    case ExecuTorchDataTypeShort:
      return @"short";
    case ExecuTorchDataTypeInt:
      return @"int";
    case ExecuTorchDataTypeLong:
      return @"long";
    case ExecuTorchDataTypeHalf:
      return @"half";
    case ExecuTorchDataTypeFloat:
      return @"float";
    case ExecuTorchDataTypeDouble:
      return @"double";
    case ExecuTorchDataTypeComplexHalf:
      return @"complexHalf";
    case ExecuTorchDataTypeComplexFloat:
      return @"complexFloat";
    case ExecuTorchDataTypeComplexDouble:
      return @"complexDouble";
    case ExecuTorchDataTypeBool:
      return @"bool";
    case ExecuTorchDataTypeQInt8:
      return @"qint8";
    case ExecuTorchDataTypeQUInt8:
      return @"quint8";
    case ExecuTorchDataTypeQInt32:
      return @"qint32";
    case ExecuTorchDataTypeBFloat16:
      return @"bfloat16";
    case ExecuTorchDataTypeQUInt4x2:
      return @"quint4x2";
    case ExecuTorchDataTypeQUInt2x4:
      return @"quint2x4";
    case ExecuTorchDataTypeBits1x8:
      return @"bits1x8";
    case ExecuTorchDataTypeBits2x4:
      return @"bits2x4";
    case ExecuTorchDataTypeBits4x2:
      return @"bits4x2";
    case ExecuTorchDataTypeBits8:
      return @"bits8";
    case ExecuTorchDataTypeBits16:
      return @"bits16";
    case ExecuTorchDataTypeFloat8_e5m2:
      return @"float8_e5m2";
    case ExecuTorchDataTypeFloat8_e4m3fn:
      return @"float8_e4m3fn";
    case ExecuTorchDataTypeFloat8_e5m2fnuz:
      return @"float8_e5m2fnuz";
    case ExecuTorchDataTypeFloat8_e4m3fnuz:
      return @"float8_e4m3fnuz";
    case ExecuTorchDataTypeUInt16:
      return @"uint16";
    case ExecuTorchDataTypeUInt32:
      return @"uint32";
    case ExecuTorchDataTypeUInt64:
      return @"uint64";
    default:
      return @"undefined";
  }
}

static inline NSString *shapeDynamismDescription(ExecuTorchShapeDynamism dynamism) {
  switch (dynamism) {
    case ExecuTorchShapeDynamismStatic:
      return @"static";
    case ExecuTorchShapeDynamismDynamicBound:
      return @"dynamicBound";
    case ExecuTorchShapeDynamismDynamicUnbound:
      return @"dynamicUnbound";
    default:
      return @"undefined";
  }
}

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
    *reinterpret_cast<TensorPtr *>(otherTensor.nativeInstance)
  );
  self = [self initWithNativeInstance:&tensor];
  if (self) {
    _data = otherTensor->_data;
  }
  return self;
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

- (void)bytesWithHandler:(NS_NOESCAPE void (^)(const void *pointer, NSInteger count, ExecuTorchDataType type))handler {
  ET_CHECK(handler);
  handler(_tensor->unsafeGetTensorImpl()->data(), self.count, self.dataType);
}

- (void)mutableBytesWithHandler:(NS_NOESCAPE void (^)(void *pointer, NSInteger count, ExecuTorchDataType dataType))handler {
  ET_CHECK(handler);
  handler(_tensor->unsafeGetTensorImpl()->mutable_data(), self.count, self.dataType);
}

- (BOOL)resizeToShape:(NSArray<NSNumber *> *)shape
                error:(NSError **)error {
  const auto errorCode = resize_tensor_ptr(_tensor, utils::toVector<SizesType>(shape));
  if (errorCode != Error::Ok) {
    if (error) {
      *error = ExecuTorchErrorWithCode((ExecuTorchErrorCode)errorCode);
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

- (NSString *)description {
  std::ostringstream os;
  os << "Tensor {";
  os << "\n  dataType: " << dataTypeDescription(static_cast<ExecuTorchDataType>(_tensor->scalar_type())).UTF8String << ",";
  os << "\n  shape: [";
  const auto& sizes = _tensor->sizes();
  for (size_t index = 0; index < sizes.size(); ++index) {
    if (index > 0) {
      os << ",";
    }
    os << sizes[index];
  }
  os << "],";
  os << "\n  strides: [";
  const auto& strides = _tensor->strides();
  for (size_t index = 0; index < strides.size(); ++index) {
    if (index > 0) {
      os << ",";
    }
    os << strides[index];
  }
  os << "],";
  os << "\n  dimensionOrder: [";
  const auto& dim_order = _tensor->dim_order();
  for (size_t index = 0; index < dim_order.size(); ++index) {
    if (index > 0) {
      os << ",";
    }
    os << static_cast<int>(dim_order[index]);
  }
  os << "],";
  os << "\n  shapeDynamism: " << shapeDynamismDescription(static_cast<ExecuTorchShapeDynamism>(_tensor->shape_dynamism())).UTF8String << ",";
  auto const count = _tensor->numel();
  os << "\n  count: " << count << ",";
  os << "\n  scalars: [";
  // Create a minimal context for error handling in ET_SWITCH
  struct {
    [[noreturn]] void fail(torch::executor::Error /* error */) {
      ET_CHECK_MSG(false, "Unsupported dtype in description");
    }
  } ctx;
  ET_SWITCH_REALHBBF16_AND_UINT_TYPES(
    static_cast<ScalarType>(_tensor->scalar_type()),
    ctx,
    "description",
    CTYPE,
    [&] {
      auto const *pointer = reinterpret_cast<const CTYPE*>(_tensor->unsafeGetTensorImpl()->data());
      auto const countToPrint = std::min(count, (ssize_t)100);
      for (size_t index = 0; index < countToPrint; ++index) {
        if (index > 0) {
          os << ",";
        }
        if constexpr (std::is_same_v<CTYPE, int8_t> ||
                      std::is_same_v<CTYPE, uint8_t>) {
          os << static_cast<int>(pointer[index]);
        } else {
          os << pointer[index];
        }
      }
      if (count > countToPrint) {
        os << ",...";
      }
    }
  );
  os << "]";
  os << "\n}";
  return @(os.str().c_str());
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
  // Create a minimal context for error handling in ET_SWITCH
  struct {
    [[noreturn]] void fail(torch::executor::Error /* error */) {
      ET_CHECK_MSG(false, "Unsupported dtype in initWithScalars");
    }
  } ctx;
  for (NSUInteger index = 0; index < count; ++index) {
    ET_SWITCH_REALHBBF16_AND_UINT_TYPES(
      static_cast<ScalarType>(dataType), ctx, "initWithScalars", CTYPE, [&] {
        reinterpret_cast<CTYPE *>(data.data())[index] = utils::toType<CTYPE>(scalars[index]);
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

@implementation ExecuTorchTensor (Empty)

+ (instancetype)emptyTensorWithShape:(NSArray<NSNumber *> *)shape
                             strides:(NSArray<NSNumber *> *)strides
                            dataType:(ExecuTorchDataType)dataType
                       shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism {
  auto tensor = empty_strided(
    utils::toVector<SizesType>(shape),
    utils::toVector<StridesType>(strides),
    static_cast<ScalarType>(dataType),
    static_cast<TensorShapeDynamism>(shapeDynamism)
  );
  return [[self alloc] initWithNativeInstance:&tensor];
}

+ (instancetype)emptyTensorWithShape:(NSArray<NSNumber *> *)shape
                            dataType:(ExecuTorchDataType)dataType
                       shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism {
  return [self emptyTensorWithShape:shape
                            strides:@[]
                           dataType:dataType
                      shapeDynamism:shapeDynamism];
}

+ (instancetype)emptyTensorWithShape:(NSArray<NSNumber *> *)shape
                            dataType:(ExecuTorchDataType)dataType {
  return [self emptyTensorWithShape:shape
                            strides:@[]
                           dataType:dataType
                      shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

+ (instancetype)emptyTensorLikeTensor:(ExecuTorchTensor *)tensor
                             dataType:(ExecuTorchDataType)dataType
                        shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism {
  return [self emptyTensorWithShape:tensor.shape
                            strides:tensor.strides
                           dataType:dataType
                      shapeDynamism:shapeDynamism];
}

+ (instancetype)emptyTensorLikeTensor:(ExecuTorchTensor *)tensor
                             dataType:(ExecuTorchDataType)dataType {
  return [self emptyTensorWithShape:tensor.shape
                            strides:tensor.strides
                           dataType:dataType
                      shapeDynamism:tensor.shapeDynamism];
}

+ (instancetype)emptyTensorLikeTensor:(ExecuTorchTensor *)tensor {
  return [self emptyTensorWithShape:tensor.shape
                            strides:tensor.strides
                           dataType:tensor.dataType
                      shapeDynamism:tensor.shapeDynamism];
}

@end

@implementation ExecuTorchTensor (Full)

+ (instancetype)fullTensorWithShape:(NSArray<NSNumber *> *)shape
                             scalar:(NSNumber *)scalar
                            strides:(NSArray<NSNumber *> *)strides
                           dataType:(ExecuTorchDataType)dataType
                      shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism {
  Scalar fillValue;
  // Create a minimal context for error handling in ET_SWITCH
  struct {
    [[noreturn]] void fail(torch::executor::Error /* error */) {
      ET_CHECK_MSG(false, "Unsupported dtype in fullTensor");
    }
  } ctx;
  ET_SWITCH_REALHBBF16_AND_UINT_TYPES(
    static_cast<ScalarType>(dataType), ctx, "fullTensor", CTYPE, [&] {
      fillValue = utils::toType<CTYPE>(scalar);
    }
  );
  auto tensor = full_strided(
    utils::toVector<SizesType>(shape),
    utils::toVector<StridesType>(strides),
    fillValue,
    static_cast<ScalarType>(dataType),
    static_cast<TensorShapeDynamism>(shapeDynamism)
  );
  return [[self alloc] initWithNativeInstance:&tensor];
}

+ (instancetype)fullTensorWithShape:(NSArray<NSNumber *> *)shape
                             scalar:(NSNumber *)scalar
                           dataType:(ExecuTorchDataType)dataType
                      shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism {
  return [self fullTensorWithShape:shape
                            scalar:scalar
                           strides:@[]
                          dataType:dataType
                     shapeDynamism:shapeDynamism];
}

+ (instancetype)fullTensorWithShape:(NSArray<NSNumber *> *)shape
                             scalar:(NSNumber *)scalar
                           dataType:(ExecuTorchDataType)dataType {
  return [self fullTensorWithShape:shape
                            scalar:scalar
                           strides:@[]
                          dataType:dataType
                     shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

+ (instancetype)fullTensorLikeTensor:(ExecuTorchTensor *)tensor
                              scalar:(NSNumber *)scalar
                            dataType:(ExecuTorchDataType)dataType
                       shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism {
  return [self fullTensorWithShape:tensor.shape
                            scalar:scalar
                           strides:tensor.strides
                          dataType:dataType
                     shapeDynamism:shapeDynamism];
}

+ (instancetype)fullTensorLikeTensor:(ExecuTorchTensor *)tensor
                              scalar:(NSNumber *)scalar
                            dataType:(ExecuTorchDataType)dataType {
  return [self fullTensorWithShape:tensor.shape
                            scalar:scalar
                           strides:tensor.strides
                          dataType:dataType
                     shapeDynamism:tensor.shapeDynamism];
}

+ (instancetype)fullTensorLikeTensor:(ExecuTorchTensor *)tensor
                              scalar:(NSNumber *)scalar {
  return [self fullTensorWithShape:tensor.shape
                            scalar:scalar
                           strides:tensor.strides
                          dataType:tensor.dataType
                     shapeDynamism:tensor.shapeDynamism];
}

@end

@implementation ExecuTorchTensor (Ones)

+ (instancetype)onesTensorWithShape:(NSArray<NSNumber *> *)shape
                           dataType:(ExecuTorchDataType)dataType
                      shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism {
  return [self fullTensorWithShape:shape
                            scalar:@(1)
                           strides:@[]
                          dataType:dataType
                     shapeDynamism:shapeDynamism];
}

+ (instancetype)onesTensorWithShape:(NSArray<NSNumber *> *)shape
                           dataType:(ExecuTorchDataType)dataType {
  return [self fullTensorWithShape:shape
                            scalar:@(1)
                           strides:@[]
                          dataType:dataType
                     shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

+ (instancetype)onesTensorLikeTensor:(ExecuTorchTensor *)tensor
                            dataType:(ExecuTorchDataType)dataType
                       shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism {
  return [self fullTensorWithShape:tensor.shape
                            scalar:@(1)
                           strides:tensor.strides
                          dataType:dataType
                     shapeDynamism:shapeDynamism];
}

+ (instancetype)onesTensorLikeTensor:(ExecuTorchTensor *)tensor
                            dataType:(ExecuTorchDataType)dataType {
  return [self fullTensorWithShape:tensor.shape
                            scalar:@(1)
                           strides:tensor.strides
                          dataType:dataType
                     shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

+ (instancetype)onesTensorLikeTensor:(ExecuTorchTensor *)tensor {
  return [self fullTensorWithShape:tensor.shape
                            scalar:@(1)
                           strides:tensor.strides
                          dataType:tensor.dataType
                     shapeDynamism:tensor.shapeDynamism];
}

@end

@implementation ExecuTorchTensor (Zeros)

+ (instancetype)zerosTensorWithShape:(NSArray<NSNumber *> *)shape
                            dataType:(ExecuTorchDataType)dataType
                       shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism {
  return [self fullTensorWithShape:shape
                            scalar:@(0)
                           strides:@[]
                          dataType:dataType
                     shapeDynamism:shapeDynamism];
}

+ (instancetype)zerosTensorWithShape:(NSArray<NSNumber *> *)shape
                            dataType:(ExecuTorchDataType)dataType {
  return [self fullTensorWithShape:shape
                            scalar:@(0)
                           strides:@[]
                          dataType:dataType
                     shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

+ (instancetype)zerosTensorLikeTensor:(ExecuTorchTensor *)tensor
                             dataType:(ExecuTorchDataType)dataType
                        shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism {
  return [self fullTensorWithShape:tensor.shape
                            scalar:@(0)
                           strides:tensor.strides
                          dataType:dataType
                     shapeDynamism:shapeDynamism];
}

+ (instancetype)zerosTensorLikeTensor:(ExecuTorchTensor *)tensor
                             dataType:(ExecuTorchDataType)dataType {
  return [self fullTensorWithShape:tensor.shape
                            scalar:@(0)
                           strides:tensor.strides
                          dataType:dataType
                     shapeDynamism:tensor.shapeDynamism];
}

+ (instancetype)zerosTensorLikeTensor:(ExecuTorchTensor *)tensor {
  return [self fullTensorWithShape:tensor.shape
                            scalar:@(0)
                           strides:tensor.strides
                          dataType:tensor.dataType
                     shapeDynamism:tensor.shapeDynamism];
}

@end

@implementation ExecuTorchTensor (Random)

+ (instancetype)randomTensorWithShape:(NSArray<NSNumber *> *)shape
                              strides:(NSArray<NSNumber *> *)strides
                             dataType:(ExecuTorchDataType)dataType
                        shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism {
  auto tensor = rand_strided(
    utils::toVector<SizesType>(shape),
    utils::toVector<StridesType>(strides),
    static_cast<ScalarType>(dataType),
    static_cast<TensorShapeDynamism>(shapeDynamism)
  );
  return [[self alloc] initWithNativeInstance:&tensor];
}

+ (instancetype)randomTensorWithShape:(NSArray<NSNumber *> *)shape
                             dataType:(ExecuTorchDataType)dataType
                        shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism {
  return [self randomTensorWithShape:shape
                             strides:@[]
                            dataType:dataType
                       shapeDynamism:shapeDynamism];
}

+ (instancetype)randomTensorWithShape:(NSArray<NSNumber *> *)shape
                             dataType:(ExecuTorchDataType)dataType {
  return [self randomTensorWithShape:shape
                             strides:@[]
                            dataType:dataType
                       shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

+ (instancetype)randomTensorLikeTensor:(ExecuTorchTensor *)tensor
                              dataType:(ExecuTorchDataType)dataType
                         shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism {
  return [self randomTensorWithShape:tensor.shape
                             strides:tensor.strides
                            dataType:dataType
                       shapeDynamism:shapeDynamism];
}

+ (instancetype)randomTensorLikeTensor:(ExecuTorchTensor *)tensor
                              dataType:(ExecuTorchDataType)dataType {
  return [self randomTensorWithShape:tensor.shape
                             strides:tensor.strides
                            dataType:dataType
                       shapeDynamism:tensor.shapeDynamism];
}

+ (instancetype)randomTensorLikeTensor:(ExecuTorchTensor *)tensor {
  return [self randomTensorWithShape:tensor.shape
                             strides:tensor.strides
                            dataType:tensor.dataType
                       shapeDynamism:tensor.shapeDynamism];
}

@end

@implementation ExecuTorchTensor (RandomNormal)

+ (instancetype)randomNormalTensorWithShape:(NSArray<NSNumber *> *)shape
                                    strides:(NSArray<NSNumber *> *)strides
                                   dataType:(ExecuTorchDataType)dataType
                              shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism {
  auto tensor = randn_strided(
    utils::toVector<SizesType>(shape),
    utils::toVector<StridesType>(strides),
    static_cast<ScalarType>(dataType),
    static_cast<TensorShapeDynamism>(shapeDynamism)
  );
  return [[self alloc] initWithNativeInstance:&tensor];
}

+ (instancetype)randomNormalTensorWithShape:(NSArray<NSNumber *> *)shape
                                   dataType:(ExecuTorchDataType)dataType
                              shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism {
  return [self randomNormalTensorWithShape:shape
                                   strides:@[]
                                  dataType:dataType
                             shapeDynamism:shapeDynamism];
}

+ (instancetype)randomNormalTensorWithShape:(NSArray<NSNumber *> *)shape
                                   dataType:(ExecuTorchDataType)dataType {
  return [self randomNormalTensorWithShape:shape
                                   strides:@[]
                                  dataType:dataType
                             shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

+ (instancetype)randomNormalTensorLikeTensor:(ExecuTorchTensor *)tensor
                                    dataType:(ExecuTorchDataType)dataType
                               shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism {
  return [self randomNormalTensorWithShape:tensor.shape
                                   strides:tensor.strides
                                  dataType:dataType
                             shapeDynamism:shapeDynamism];
}

+ (instancetype)randomNormalTensorLikeTensor:(ExecuTorchTensor *)tensor
                                    dataType:(ExecuTorchDataType)dataType {
  return [self randomNormalTensorWithShape:tensor.shape
                                   strides:tensor.strides
                                  dataType:dataType
                             shapeDynamism:tensor.shapeDynamism];
}

+ (instancetype)randomNormalTensorLikeTensor:(ExecuTorchTensor *)tensor {
  return [self randomNormalTensorWithShape:tensor.shape
                                   strides:tensor.strides
                                  dataType:tensor.dataType
                             shapeDynamism:tensor.shapeDynamism];
}

@end

@implementation ExecuTorchTensor (RandomInteger)

+ (instancetype)randomIntegerTensorWithLow:(NSInteger)low
                                      high:(NSInteger)high
                                     shape:(NSArray<NSNumber *> *)shape
                                   strides:(NSArray<NSNumber *> *)strides
                                  dataType:(ExecuTorchDataType)dataType
                             shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism {
  auto tensor = randint_strided(
    low,
    high,
    utils::toVector<SizesType>(shape),
    utils::toVector<StridesType>(strides),
    static_cast<ScalarType>(dataType),
    static_cast<TensorShapeDynamism>(shapeDynamism)
  );
  return [[self alloc] initWithNativeInstance:&tensor];
}

+ (instancetype)randomIntegerTensorWithLow:(NSInteger)low
                                      high:(NSInteger)high
                                     shape:(NSArray<NSNumber *> *)shape
                                  dataType:(ExecuTorchDataType)dataType
                             shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism {
  return [self randomIntegerTensorWithLow:low
                                     high:high
                                    shape:shape
                                  strides:@[]
                                 dataType:dataType
                            shapeDynamism:shapeDynamism];
}

+ (instancetype)randomIntegerTensorWithLow:(NSInteger)low
                                      high:(NSInteger)high
                                     shape:(NSArray<NSNumber *> *)shape
                                  dataType:(ExecuTorchDataType)dataType {
  return [self randomIntegerTensorWithLow:low
                                     high:high
                                    shape:shape
                                  strides:@[]
                                 dataType:dataType
                            shapeDynamism:ExecuTorchShapeDynamismDynamicBound];
}

+ (instancetype)randomIntegerTensorLikeTensor:(ExecuTorchTensor *)tensor
                                          low:(NSInteger)low
                                         high:(NSInteger)high
                                     dataType:(ExecuTorchDataType)dataType
                                shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism {
  return [self randomIntegerTensorWithLow:low
                                     high:high
                                    shape:tensor.shape
                                  strides:tensor.strides
                                 dataType:dataType
                            shapeDynamism:shapeDynamism];
}

+ (instancetype)randomIntegerTensorLikeTensor:(ExecuTorchTensor *)tensor
                                          low:(NSInteger)low
                                         high:(NSInteger)high
                                     dataType:(ExecuTorchDataType)dataType {
  return [self randomIntegerTensorWithLow:low
                                     high:high
                                    shape:tensor.shape
                                  strides:tensor.strides
                                 dataType:dataType
                            shapeDynamism:tensor.shapeDynamism];
}

+ (instancetype)randomIntegerTensorLikeTensor:(ExecuTorchTensor *)tensor
                                          low:(NSInteger)low
                                         high:(NSInteger)high {
  return [self randomIntegerTensorWithLow:low
                                     high:high
                                    shape:tensor.shape
                                  strides:tensor.strides
                                 dataType:tensor.dataType
                            shapeDynamism:tensor.shapeDynamism];
}

@end
