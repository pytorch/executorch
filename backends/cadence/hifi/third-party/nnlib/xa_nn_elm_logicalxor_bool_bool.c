#include "xa_nnlib_common.h"

WORD32 xa_nn_elm_logicalxor_boolxbool_bool(WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_inp1,
                    const   WORD8 * __restrict__ p_inp2,
                            WORD32  num_elm)
{
  /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD8), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);

    ae_int24x2 *pin1 = (ae_int24x2 *)p_inp1;
    ae_int24x2 *pin2 = (ae_int24x2 *)p_inp2;
    ae_int24x2 *pout = (ae_int24x2 *)p_out;
    int i;
    int N = num_elm;
    /* Following line divides N by 6. Much faster than compiler implementation. Works for N<32768. */ 
    /* unsigned int Nby6 = (N*10923)>>16;*/
    /* Following works for all int32 N */
    int Nby6 =  AE_MOVAD32_H(AE_MOVINT32X2_FROMINT64(AE_MUL32_LL(N, 0x2AAAAAAB)));
    int remainder_start = 6*Nby6;

    ae_valign align_src_in1, align_src_in2, align_dst;
    align_src_in1 = AE_LA64_PP(pin1);
    align_src_in2 = AE_LA64_PP(pin2);
    align_dst    = AE_ZALIGN64();

/* Loop is unrolled by 6, to use LA24X2/SA24X2 */
    for(i=0; i < Nby6; i++){
        ae_int24x2 vi1, vi2, vo;
        AE_LA24X2_IP(vi1, align_src_in1, pin1);
        AE_LA24X2_IP(vi2, align_src_in2, pin2);
        vo = AE_XOR24(vi1, vi2);
        AE_SA24X2_IP(vo, align_dst, pout);
    }
    AE_SA64POS_FP(align_dst, pout);

    /* Remainder loop */
    #pragma no_unroll
    for(i=remainder_start; i < N; i++){
        p_out[i] = p_inp1[i] ^ p_inp2[i];
    }

    return 0;
}