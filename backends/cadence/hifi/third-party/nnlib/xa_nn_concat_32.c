#include "xa_type_def.h"
#include "xa_nn_common.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_common_macros.h"
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_common.h"

WORD32 xa_nn_concat_32_32(WORD32 * __restrict__ p_out
                        ,const WORD32 *const p_out_shape
                        ,const WORD32 **pp_inps
                        ,const WORD32 *const *pp_inps_shape
                        ,WORD32 num_out_dims
                        ,WORD32 num_inp
                        ,WORD32 num_inp_dims
                        ,WORD32 axis)
{
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(pp_inps, -1);
  XA_NNLIB_ARG_CHK_PTR(pp_inps_shape, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(pp_inps, sizeof(WORD32 *), -1);
  XA_NNLIB_ARG_CHK_ALIGN(pp_inps_shape, sizeof(WORD32 *), -1);
  //Validate Arguments
  XA_NNLIB_ARG_CHK_COND((num_out_dims <= 0 || num_out_dims > 6), -1);
  XA_NNLIB_ARG_CHK_COND((num_inp <= 0 || num_inp > 10), -1);
  XA_NNLIB_ARG_CHK_COND((num_inp_dims != num_out_dims), -1);
  XA_NNLIB_ARG_CHK_COND((axis < -num_out_dims || axis >= num_out_dims), -1);

  int i = 0, j = 0;
  for(i = 0; i < num_out_dims; i++)
  {
    XA_NNLIB_ARG_CHK_COND((p_out_shape[i] <= 0), -1);
  }

  if(axis < 0)
    axis = num_out_dims + axis;

  WORD32 concat_size = 0;
  for (i = 0; i < num_inp; i++)
  {
    XA_NNLIB_ARG_CHK_PTR(pp_inps[i], -1);
    XA_NNLIB_ARG_CHK_PTR(pp_inps_shape[i], -1);
    XA_NNLIB_ARG_CHK_ALIGN(pp_inps_shape[i], sizeof(WORD32), -1);
#pragma loop_count min=1
    for(j = 0; j < num_out_dims; j++)
    {
      XA_NNLIB_ARG_CHK_COND((pp_inps_shape[i][j] != p_out_shape[j] && j != axis), -1);
    }
    
    XA_NNLIB_ARG_CHK_COND((pp_inps_shape[i][axis] <= 0), -1);
    concat_size += pp_inps_shape[i][axis];
  }

  XA_NNLIB_ARG_CHK_COND((p_out_shape[axis] != concat_size), -1);

  //Calculate outer and inner size for axis
  WORD32 outer_size = 1;
#pragma no_simd
  for(int i = 0; i < axis; i++)
  {
    outer_size *= p_out_shape[i];
  }

  WORD32 base_inner_size = 1;
#pragma no_simd
  for(int i = axis + 1; i < num_out_dims; i++)
  {
    base_inner_size *= p_out_shape[i];
  }

  WORD32 *ptmp_out = p_out;
  for(int i = 0; i < num_inp; i++)
  {
    const WORD32 copy_size = pp_inps_shape[i][axis] * base_inner_size;
    WORD32 *output_ptr = ptmp_out;
    const WORD32* input_ptr = pp_inps[i];

    if(((copy_size & 1) == 0) && (((concat_size * base_inner_size) & 1) == 0)
      && (((unsigned)input_ptr & 1) == 0) && (((unsigned)output_ptr & 1) == 0))
    {
      if(copy_size <= 8)
      {
        const ae_f32 *pae_inp = (const ae_f32 *)input_ptr;
        for(int k = 0; k < outer_size; k++)
        {
          ae_f32 *pae_out = (ae_f32 *)output_ptr;
#pragma concurrent
#pragma no_simd
          for(int ic = 0; ic < copy_size; ic++)
          {
            *pae_out++ = *pae_inp++;
          }
          output_ptr += concat_size * base_inner_size;
        }
      }
      else
      {
        for(int k = 0; k < outer_size; k++)
        {
          const ae_int32x2 *pae_inp = (const ae_int32x2 *)input_ptr;
          ae_int32x2 *pae_out = (ae_int32x2 *)output_ptr;
          ae_valign inp_a, out_a;
          inp_a = AE_LA64_PP(pae_inp);
          out_a = AE_ZALIGN64();
          for(int ic = 0; ic < (copy_size >> 1); ic++)
          {
            ae_int32x2 d0;
            AE_LA32X2_IP(d0, inp_a, pae_inp);
            AE_SA32X2_IP(d0, out_a, pae_out);
          }
          AE_SA64POS_FP(out_a, pae_out);
          const ae_f32 *puae_inp = (const ae_f32 *)pae_inp;
          ae_f32 *puae_out = (ae_f32 *)pae_out;
#pragma concurrent
          for(int ic = 0; ic < (copy_size & 1); ic++)
          {
            puae_out[copy_size - 1] = puae_inp[copy_size - 1];
          }
          input_ptr += copy_size;
          output_ptr += concat_size * base_inner_size;
        }
      }
    }
    else
    {
      if(copy_size <= 6)
      {
        for(int k = 0; k < outer_size; k++)
        {
#pragma concurrent
#pragma no_unroll
          for(int ic = 0; ic < copy_size; ic++)
          {
            output_ptr[ic] = *input_ptr++;
          }
          output_ptr += concat_size * base_inner_size;
        }
      }
      else
      {
        for(int k = 0; k < outer_size; k++)
        {
          const ae_int32x2 *pae_inp = (const ae_int32x2 *)input_ptr;
          ae_int32x2 *pae_out = (ae_int32x2 *)output_ptr;
          ae_valign inp_a, out_a;
          inp_a = AE_LA64_PP(pae_inp);
          out_a = AE_ZALIGN64();

#pragma concurrent
          for(int ic = 0; ic < copy_size >> 1; ic++)
          {
            ae_int32x2 d0;
            AE_LA32X2_IP(d0, inp_a, pae_inp);
            AE_SA32X2_IP(d0, out_a, pae_out);
          }
          AE_SA64POS_FP(out_a, pae_out);
          
          for(int ic = 0; ic < (copy_size & 1); ic++)
          {
            output_ptr[copy_size - 1] = input_ptr[copy_size - 1];
          }
          input_ptr += copy_size;
          output_ptr += concat_size * base_inner_size;
        }
      }
    }
    ptmp_out += copy_size;
  }
  return 0;
}