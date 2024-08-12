/*
 * Copyright (c) 2024 MediaTek Inc.
 *
 * Licensed under the BSD License (the "License"); you may not use this file
 * except in compliance with the License. See the license file in the root
 * directory of this source tree for more details.
 */

#include "NeuronExecutor.h"
#include "NeuronLog.h"
#include "api/NeuronAdapter.h"

#include <string>
#include <vector>

#define RESTORE_DLA_EXTENSION_OPERAND_TYPE 0x0100
#define RESTORE_DLA_EXTENSION_OPERATION_TYPE 0x0000
#define RESTORE_DLA_EXTENSION_NAME "com.mediatek.compiled_network"

namespace torch {
namespace executor {
namespace neuron {

NeuronExecutor::NeuronExecutor(){};

int NeuronExecutor::LoadFromCompiledNetwork(
    const void* buffer,
    size_t size,
    int inputCount,
    int outputCount,
    std::string& runtimeOption) {
  NeuronModel* model = nullptr;
  NeuronCompilation* compilation = nullptr;
  NeuronExecution* execution = nullptr;

  std::vector<NeuronOperandType> mInputOperand;
  std::vector<NeuronOperandType> mOutputOperand;
  // ---------------------------Model------------------------------------
  int err = NEURON_NO_ERROR;
  err |= NeuronModel_create(&model);
  CHECK_NO_ERROR(err);

  mModel = std::unique_ptr<NeuronModel, NeuronDeleter>(model);

  std::vector<uint32_t> input_op_number;
  // fake input, the real outputs are loaded by compiled network.
  NeuronOperandType fakeInputOperandType{
      .type = NEURON_TENSOR_FLOAT32,
      .dimensionCount = 0,
      .scale = 0.0f,
      .zeroPoint = 0,
  };

  for (int i = 0; i < inputCount; i++) {
    mInputOperand.push_back(fakeInputOperandType);
  }
  for (int i = 0; i < mInputOperand.size(); i++) {
    err |= NeuronModel_addOperand(model, &mInputOperand[i]);
    input_op_number.emplace_back(i);
  }

  int32_t operandType = 0;
  const uint16_t network_operand_restore_data =
      RESTORE_DLA_EXTENSION_OPERAND_TYPE;
  const char* extensionRestoreCompiledNetwork = RESTORE_DLA_EXTENSION_NAME;
  err |= NeuronModel_getExtensionOperandType(
      model,
      extensionRestoreCompiledNetwork,
      network_operand_restore_data,
      &operandType);
  CHECK_NO_ERROR(err);

  NeuronOperandType extenOperandType{
      .type = operandType,
      .dimensionCount = 0,
      .scale = 0.0f,
      .zeroPoint = 0,
  };

  err |= NeuronModel_addOperand(model, &extenOperandType);
  CHECK_NO_ERROR(err);
  input_op_number.emplace_back(input_op_number.size());

  // fake output, the real outputs are loaded by compiled network.
  NeuronOperandType fakeOutputOperandType{
      .type = NEURON_TENSOR_FLOAT32,
      .dimensionCount = 0,
      .scale = 0.0f,
      .zeroPoint = 0,
  };

  for (int i = 0; i < outputCount; i++) {
    mOutputOperand.push_back(fakeOutputOperandType);
  }

  std::vector<uint32_t> output_op_number;
  for (int i = 0; i < mOutputOperand.size(); i++) {
    err |= NeuronModel_addOperand(model, &mOutputOperand[i]);
    output_op_number.emplace_back(i + input_op_number.size());
  }

  CHECK_NO_ERROR(err);

  err |=
      NeuronModel_setOperandValue(model, input_op_number.back(), buffer, size);

  int32_t operationType = 0;
  const uint16_t network_operation_type_restore =
      RESTORE_DLA_EXTENSION_OPERATION_TYPE;
  err |= NeuronModel_getExtensionOperationType(
      model,
      extensionRestoreCompiledNetwork,
      network_operation_type_restore,
      &operationType);

  CHECK_NO_ERROR(err);

  // Add extension operation
  err |= NeuronModel_addOperation(
      model,
      (NeuronOperationType)operationType,
      input_op_number.size(),
      input_op_number.data(),
      output_op_number.size(),
      output_op_number.data());

  CHECK_NO_ERROR(err);

  // Identify input and output
  err |= NeuronModel_identifyInputsAndOutputs(
      model,
      input_op_number.size() - 1,
      input_op_number.data(),
      output_op_number.size(),
      output_op_number.data());

  CHECK_NO_ERROR(err);

  err |= NeuronModel_finish(model);
  CHECK_NO_ERROR(err);
  // ---------------------------Compilation------------------------------------
  // err = NeuronCompilation_e(model, &compilation) != NEURON_NO_ERROR;
  err = NeuronCompilation_createWithOptions(
      model, &compilation, runtimeOption.c_str());
  CHECK_NO_ERROR(err);

  mCompilation = std::unique_ptr<NeuronCompilation, NeuronDeleter>(compilation);

  err |=
      NeuronCompilation_setPreference(compilation, NEURON_PREFER_TURBO_BOOST);
  err |= NeuronCompilation_setPriority(compilation, NEURON_PRIORITY_HIGH);
  CHECK_NO_ERROR(err);

  err = NeuronCompilation_finish(compilation);
  CHECK_NO_ERROR(err);

  // ---------------------------Execution------------------------------------
  // Create Neuron executor instance.
  err = NeuronExecution_create(compilation, &execution);
  CHECK_NO_ERROR(err);
  mExecution = std::unique_ptr<NeuronExecution, NeuronDeleter>(execution);

  return NEURON_NO_ERROR;
}

} // namespace neuron
} // namespace executor
} // namespace torch
