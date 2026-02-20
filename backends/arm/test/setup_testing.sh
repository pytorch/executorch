#!/usr/bin/env bash
# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eu

script_dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
et_root_dir=$(realpath "${script_dir}/../../..")
build_executor_runner=${et_root_dir}/backends/arm/scripts/build_executor_runner.sh
build_root_test_dir=${et_root_dir}/arm_test/arm_semihosting_executor_runner
extraflags="-DET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE=83886080"

# By default tests with an elf without any portable_ops
# If you supply use_portable_ops=True when creating the ArmTester()
# you will instead test with some portable ops compiled in, see list below.

#--target --system_config --memory_mode should match the ArmTester used setup see backends/arm/test/common.py

${build_executor_runner} --pte=semihosting --target=ethos-u55-128 --system_config=Ethos_U55_High_End_Embedded --memory_mode=Shared_Sram --output="${build_root_test_dir}_corstone-300" --extra_build_flags=${extraflags}
${build_executor_runner} --pte=semihosting --target=ethos-u85-128 --system_config=Ethos_U85_SYS_DRAM_Mid --memory_mode=Dedicated_Sram_384KB --output="${build_root_test_dir}_corstone-320" --extra_build_flags=${extraflags}

# List of portable ops used by testing, this is mainly used to test models in the flow
# test setup to make sure models that are not fully delegated can still be tested and run OK
# To use this you can set use_portable_ops=True when creating ArmTester()

portable_ops_list_u55="aten::permute_copy.out,aten::convolution.out,aten::relu.out,aten::_native_batch_norm_legit_no_training.out,aten::as_strided_copy.out,aten::mean.out,aten::squeeze_copy.dims,dim_order_ops::_clone_dim_order.out"
portable_ops_list_u85="aten::permute_copy.out,aten::convolution.out,aten::relu.out,aten::_native_batch_norm_legit_no_training.out,aten::as_strided_copy.out,aten::mean.out,aten::full_like.out,aten::bmm.out,aten::scalar_tensor.out,aten::index.Tensor_out,aten::where.self_out,dim_order_ops::_to_dim_order_copy.out"

${build_executor_runner} --pte=semihosting --target=ethos-u55-128 --system_config=Ethos_U55_High_End_Embedded --memory_mode=Shared_Sram --select_ops_list="${portable_ops_list_u55}" --output="${build_root_test_dir}_portable-ops_corstone-300" --extra_build_flags=${extraflags}
${build_executor_runner} --pte=semihosting --target=ethos-u85-128 --system_config=Ethos_U85_SYS_DRAM_Mid --memory_mode=Dedicated_Sram_384KB --select_ops_list="${portable_ops_list_u85}" --output="${build_root_test_dir}_portable-ops_corstone-320" --extra_build_flags=${extraflags}
