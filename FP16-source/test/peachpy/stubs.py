from peachpy import *
from peachpy.x86_64 import *

import fp16.avx, fp16.avx2


arg_fp16 = Argument(ptr(const_uint16_t), name="fp16")
arg_fp32 = Argument(ptr(uint32_t), name="fp32")

with Function("fp16_alt_xmm_to_fp32_ymm_peachpy__avx2", (arg_fp16, arg_fp32), target=uarch.default + isa.avx2):

    reg_fp16 = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_fp16, arg_fp16)

    reg_fp32 = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_fp32, arg_fp32)

    xmm_fp16 = XMMRegister()
    VMOVUPS(xmm_fp16, [reg_fp16])
    ymm_fp32 = fp16.avx2.fp16_alt_xmm_to_fp32_ymm(xmm_fp16)
    VMOVUPS([reg_fp32], ymm_fp32)

    RETURN()

with Function("fp16_alt_xmm_to_fp32_xmm_peachpy__avx", (arg_fp16, arg_fp32), target=uarch.default + isa.avx):

    reg_fp16 = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_fp16, arg_fp16)

    reg_fp32 = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_fp32, arg_fp32)

    xmm_fp16 = XMMRegister()
    VMOVUPS(xmm_fp16, [reg_fp16])
    xmm_fp32 = fp16.avx.fp16_alt_xmm_to_fp32_xmm(xmm_fp16)
    VMOVUPS([reg_fp32], xmm_fp32)

    RETURN()
