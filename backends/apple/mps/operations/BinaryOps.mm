//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include "utils/MPSGraphInterface.h"

#include "BinaryOps.h"

namespace mps {
using namespace torch;

PyMPSGraphTensor*
MPSGraphModule::binaryOpTensor(
  MPSGraphTensor* primaryTensor,
  MPSGraphTensor* secondaryTensor,
  const std::string& op_name,
  std::function<MPSGraphTensor*(MPSGraphTensor*, MPSGraphTensor*)> binaryOpFunction) {
  MPSDataType mpsInputDataType = [primaryTensor dataType];
  MPSDataType mpsOtherDataType = [secondaryTensor dataType];

  ScalarType inputDataType = getScalarType(mpsInputDataType);
  ScalarType otherDataType = getScalarType(mpsOtherDataType);

  MPSGraphTensor* primaryCastTensor = primaryTensor;
  MPSGraphTensor* secondaryCastTensor = secondaryTensor;
  ScalarType common_dtype = c10::promoteTypes(inputDataType, otherDataType);
  if (inputDataType != common_dtype) {
    primaryCastTensor = castMPSTensor(mpsGraph, primaryTensor, common_dtype);
  }
  if (otherDataType != common_dtype) {
    secondaryCastTensor = castMPSTensor(mpsGraph, secondaryTensor, common_dtype);
  }

  return binaryOpFunction(primaryCastTensor, secondaryCastTensor);
}

PyMPSGraphTensor*
MPSGraphModule::additionWithTensor(MPSGraphTensor* primaryTensor,
                                   MPSGraphTensor* secondaryTensor,
                                   Scalar alpha) {
  MPSGraphTensor* primaryCastTensor = primaryTensor;
  MPSGraphTensor* secondaryCastTensor = secondaryTensor;
  auto _alpha = alpha.isFloatingPoint() ? alpha.to<float>() : alpha.to<int>();

  ScalarType primaryDataType = getScalarType(primaryCastTensor.dataType);
  ScalarType secondaryDataType = getScalarType(secondaryCastTensor.dataType);

  MPSDataType commonDataType = getMPSDataType(c10::promoteTypes(primaryDataType, secondaryDataType));

  if(primaryCastTensor.dataType != commonDataType) {
    primaryCastTensor = [mpsGraph castTensor:primaryCastTensor
                                      toType:commonDataType
                                        name:nil];
  }

  if(secondaryCastTensor.dataType != commonDataType) {
    secondaryCastTensor = [mpsGraph castTensor:secondaryCastTensor
                                      toType:commonDataType
                                        name:nil];
  }

  if(_alpha!=1.0) {
    MPSGraphTensor* alphaTensor = [mpsGraph constantWithScalar:_alpha
                                                         shape:@[@1]
                                                      dataType:primaryCastTensor.dataType];
    secondaryCastTensor = [mpsGraph multiplicationWithPrimaryTensor:secondaryCastTensor
                                             secondaryTensor:alphaTensor
                                                        name:nil];
  }
  MPSGraphTensor* resultTensor = [mpsGraph additionWithPrimaryTensor:primaryCastTensor
                                                     secondaryTensor:secondaryCastTensor
                                                                name:nil];
  return resultTensor;
}

PyMPSGraphTensor*
MPSGraphModule::subtractionWithTensor(MPSGraphTensor* primaryTensor,
                                   MPSGraphTensor* secondaryTensor,
                                   Scalar alpha) {
  MPSGraphTensor* primaryCastTensor = primaryTensor;
  MPSGraphTensor* secondaryCastTensor = secondaryTensor;
  auto _alpha = alpha.isFloatingPoint() ? alpha.to<float>() : alpha.to<int>();

  ScalarType primaryDataType = getScalarType(primaryCastTensor.dataType);
  ScalarType secondaryDataType = getScalarType(secondaryCastTensor.dataType);

  MPSDataType commonDataType = getMPSDataType(c10::promoteTypes(primaryDataType, secondaryDataType));

  if(primaryCastTensor.dataType != commonDataType) {
    primaryCastTensor = [mpsGraph castTensor:primaryCastTensor
                                      toType:commonDataType
                                        name:nil];
  }

  if(secondaryCastTensor.dataType != commonDataType) {
    secondaryCastTensor = [mpsGraph castTensor:secondaryCastTensor
                                      toType:commonDataType
                                        name:nil];
  }

  if(_alpha!=1.0) {
    MPSGraphTensor* alphaTensor = [mpsGraph constantWithScalar:_alpha
                                                         shape:@[@1]
                                                      dataType:primaryCastTensor.dataType];
    secondaryCastTensor = [mpsGraph multiplicationWithPrimaryTensor:secondaryCastTensor
                                             secondaryTensor:alphaTensor
                                                        name:nil];
  }
  MPSGraphTensor* resultTensor = [mpsGraph subtractionWithPrimaryTensor:primaryCastTensor
                                                     secondaryTensor:secondaryCastTensor
                                                                name:nil];
  return resultTensor;
}

PyMPSGraphTensor*
MPSGraphModule::multiplicationWithScalar(MPSGraphTensor* inputTensor, Scalar scalar) {
  auto value = scalar.isFloatingPoint() ? scalar.to<float>() : scalar.to<int>();
  MPSGraphTensor* constantTensor = [mpsGraph constantWithScalar:value
                                                          shape:@[@1]
                                                       dataType:inputTensor.dataType];
  MPSGraphTensor* resultTensor = [mpsGraph multiplicationWithPrimaryTensor:inputTensor
                                                     secondaryTensor:constantTensor
                                                                name:nil];
  return resultTensor;
}

MPSGraphTensor*
MPSGraphModule::trunc_tensor(MPSGraphTensor* inputTensor) {
  // Rounding is a no-op for integral types, and also a reasonable workaround
  // For MPSGraph bug on Apple Silicon, that throws `Function floorOp_i64 was not found in the library`
  // See https://github.com/pytorch/pytorch/issues/84995
  bool isFloatInput = ([inputTensor dataType] & MPSDataTypeFloatBit) != 0;
  if (!isFloatInput) {
    return inputTensor;
  }

  return [mpsGraph truncateWithTensor:inputTensor
                                  name:nil];
};

PyMPSGraphTensor*
MPSGraphModule::div_mode_template(
  MPSGraphTensor* primaryTensor,
  MPSGraphTensor* secondaryTensor,
  c10::optional<c10::string_view> rounding_mode,
  const string& op_name) {
  MPSDataType mpsInputDataType = [primaryTensor dataType];
  MPSDataType mpsOtherDataType = [secondaryTensor dataType];

  ScalarType inputDataType = getScalarType(mpsInputDataType);
  ScalarType otherDataType = getScalarType(mpsOtherDataType);

  if(rounding_mode.has_value() && *rounding_mode == "trunc"){
    TORCH_CHECK(inputDataType != ScalarType::Half,
                "MPS: does not support trunc_divide op with float16 input");
  }

  auto divOpFunc = [&](MPSGraphTensor* primaryCastTensor,
                      MPSGraphTensor* secondaryCastTensor) -> MPSGraphTensor* {
    bool isFloatInput = ([primaryCastTensor dataType] & MPSDataTypeFloatBit) != 0;
    if(!isFloatInput && rounding_mode.has_value() && (*rounding_mode == "floor" || *rounding_mode == "trunc")) {
      primaryCastTensor = [mpsGraph castTensor:primaryCastTensor
                                        toType:MPSDataTypeFloat32
                                          name:@"primaryCastTensor"];
      secondaryCastTensor = [mpsGraph castTensor:secondaryCastTensor
                                          toType:MPSDataTypeFloat32
                                            name:@"secondaryCastTensor"];
    }
    MPSGraphTensor* divTensor =  [mpsGraph divisionWithPrimaryTensor:primaryCastTensor
                                                      secondaryTensor:secondaryCastTensor
                                                                name:nil];

    // Rounding is a no-op for integral types, and also a reasonable workaround
    // For MPSGraph bug on Apple Silicon, that throws `Function floorOp_i64 was not found in the library`
    // See https://github.com/pytorch/pytorch/issues/84995
    bool isFloatOutput = ([divTensor dataType] & MPSDataTypeFloatBit) != 0;
    if (!rounding_mode.has_value() || !isFloatOutput) {
      return divTensor;
    } else if (*rounding_mode == "trunc") {
      auto truncTensor =  trunc_tensor(divTensor);
      if (op_name == "fmod_mps_out") {
        auto mulTensor = [mpsGraph multiplicationWithPrimaryTensor:truncTensor
                                                    secondaryTensor:secondaryCastTensor
                                                              name:nil];
        return [mpsGraph subtractionWithPrimaryTensor:primaryCastTensor
                                      secondaryTensor:mulTensor
                                                  name:nil];
      }
      return truncTensor;
    } else if (*rounding_mode == "floor") {
      MPSGraphTensor* floorTensor = [mpsGraph floorWithTensor:divTensor name:nil];
      if (op_name == "remainder_out_mps") {
        auto mulTensor = [mpsGraph multiplicationWithPrimaryTensor:floorTensor
                                                    secondaryTensor:secondaryCastTensor
                                                              name:nil];
        return [mpsGraph subtractionWithPrimaryTensor:primaryCastTensor
                                      secondaryTensor:mulTensor
                                                  name:nil];
      }
      return floorTensor;
    } else {
      assert(0 && "Invalid rounding mode\n");
    }
    return nullptr;
  };
  return binaryOpTensor(primaryTensor, secondaryTensor, op_name, divOpFunc);
}

PyMPSGraphTensor*
MPSGraphModule::binaryOpWithScalar(MPSGraphTensor *inputTensor, Scalar scalar,
  const std::string &op_name,
  std::function<MPSGraphTensor*(MPSGraphTensor*, MPSGraphTensor*)> binaryOpFunction) {
  auto value = scalar.isFloatingPoint() ? scalar.to<float>() : scalar.to<int>();
  MPSGraphTensor* constantTensor = [mpsGraph constantWithScalar:value
                                                          shape:@[@1]
                                                       dataType:inputTensor.dataType];
  return binaryOpTensor(inputTensor, constantTensor, op_name, binaryOpFunction);
}

} // namespace mps

