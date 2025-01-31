/*******************************************************************************
* Copyright (c) 2018-2024 Cadence Design Systems, Inc.
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to use this Software with Cadence processor cores only and
* not with any other processors and platforms, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

******************************************************************************/
#include "xa_nnlib_common.h"

#include <string.h>

/*
 * Currently only supports upto 5D input tensors.
 * 1/2/3/4 D input tensors will be scaled up to 5D.
 * For example, 2x3 -> 1x1x1x2x3.
 */

WORD32 xa_nn_transpose_8_8(WORD8 * __restrict__ p_out
                    ,const WORD32 *const p_out_shape
                    ,const WORD8 * __restrict__ p_inp
                    ,const WORD32 *const p_inp_shape
                    ,const WORD32 * __restrict__ p_permute_vec
                    ,WORD32 num_out_dims
                    ,WORD32 num_inp_dims)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_permute_vec, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp_shape, -1);

  /* Invalid input checks */
  XA_NNLIB_ARG_CHK_COND(((num_inp_dims <= 0) || (num_inp_dims > 5)), -1);
  XA_NNLIB_ARG_CHK_COND((num_out_dims != num_inp_dims), -1);

  int itr = 0;
  for(itr=0; itr < num_inp_dims; itr++)
  {
    XA_NNLIB_ARG_CHK_COND((p_inp_shape[itr] <= 0), -1);
  }
  for(itr=0; itr < num_out_dims; itr++)
  {
    XA_NNLIB_ARG_CHK_COND((p_out_shape[itr] <= 0), -1);
  }

  /* Output shape provided must be correct based on input
   * shape and permute values */
  for(itr=0; itr < num_out_dims; itr++)
  {
    int output_dim = p_out_shape[itr];
    int expected_dim = p_inp_shape[p_permute_vec[itr]];
    XA_NNLIB_ARG_CHK_COND((output_dim != expected_dim), -1);
  }

  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_permute_vec, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp_shape, sizeof(WORD32), -1);

  /* Shift all dim with 1 in the outer part */
  int eff_output_shape[5];
  int eff_permute_vec[5];

  for(int i = 0; i < num_out_dims; i++)
  {
    eff_output_shape[i] = p_out_shape[i];
    eff_permute_vec[i] = p_permute_vec[i];
  }
  
  int one_i=num_out_dims-1, non_one_i=num_out_dims-1;
  while(one_i > 0 && non_one_i >=0){
    while(one_i > 0 && eff_output_shape[one_i]!=1){
      one_i--;
    }
    non_one_i = one_i;
    while(non_one_i >= 0 && eff_output_shape[non_one_i]==1)
    {
      non_one_i--;
    }
    if(one_i > 0 && non_one_i >=0){
      int temp;
      /*swap output_shape*/
      {
        temp = eff_output_shape[one_i];
        eff_output_shape[one_i] = eff_output_shape[non_one_i];
        eff_output_shape[non_one_i] = temp;
      }
      /*swap permute_vec*/
      {
        temp = eff_permute_vec[one_i];
        eff_permute_vec[one_i] = eff_permute_vec[non_one_i];
        eff_permute_vec[non_one_i] = temp;
      }
      
    }
  }


  /* Promoting lesser dim tensors to 5D tensors. 
   * Also updating the permute_vec and shapes as needed for optimization */
  int p_5D_inp_shape[5] = {1, 1, 1, 1, 1};
  int p_5D_out_shape[5] = {1, 1, 1, 1, 1};
  int p_5D_permute_vec[5] = {0, 1, 2, 3, 4};
  
  /* Check if any inner inp dimension is same in the output */
  int last_dim_same = 1, last_n_same_dim = 0;
  itr = num_inp_dims - 1;
  while(itr >= 0)
  {
    last_n_same_dim = (last_dim_same && (eff_permute_vec[itr] == itr)) ? (last_n_same_dim + 1) : last_n_same_dim;
    last_dim_same = (eff_permute_vec[itr] == itr) ? last_dim_same & 1 : last_dim_same & 0;
    itr--;
  }
  
  int dims_added = 5 - num_inp_dims;
  itr = num_inp_dims - 1;
  int same_count = last_n_same_dim;
  int count = 4;
  while(itr >= 0)
  {
    p_5D_inp_shape[count] = (same_count > 0) ? p_5D_inp_shape[count]*p_inp_shape[itr] : p_inp_shape[itr];
    p_5D_out_shape[count] = (same_count > 0) ? p_5D_out_shape[count]*eff_output_shape[itr] : eff_output_shape[itr];
    same_count--;
    itr--;
    count = (same_count > 0) ? count : count - 1;
  }
  
  itr = num_inp_dims - 1;
  same_count = (last_n_same_dim) ? num_inp_dims - (last_n_same_dim - 1) : 0;
  count = 4;
  while(itr >= 0)
  {
    p_5D_permute_vec[count] = (same_count > 0) ? eff_permute_vec[itr-(last_n_same_dim - 1)] + dims_added + last_n_same_dim - 1 : eff_permute_vec[itr] + dims_added;
    same_count--;
    itr--;
    count--;
  }
  
  int out_dim0, out_dim1, out_dim2, out_dim3, out_dim4;
  int inp_dim1, inp_dim2, inp_dim3, inp_dim4;
  int inp_stride[5];

  out_dim0 = p_5D_out_shape[0]; 
  out_dim1 = p_5D_out_shape[1]; 
  out_dim2 = p_5D_out_shape[2]; 
  out_dim3 = p_5D_out_shape[3];
  out_dim4 = p_5D_out_shape[4];

  inp_dim1 = p_5D_inp_shape[1]; 
  inp_dim2 = p_5D_inp_shape[2]; 
  inp_dim3 = p_5D_inp_shape[3];
  inp_dim4 = p_5D_inp_shape[4];

  inp_stride[0] = inp_dim1*inp_dim2*inp_dim3*inp_dim4;
  inp_stride[1] = inp_dim2*inp_dim3*inp_dim4;
  inp_stride[2] = inp_dim3*inp_dim4;
  inp_stride[3] = inp_dim4;
  inp_stride[4] = 1;

  if(last_n_same_dim)
  {
    int itr0, itr1, itr2, itr3;
    WORD8 *p_inp0 = (WORD8*)p_inp;
    for(itr0 = 0; itr0 < out_dim0; itr0++)
    {
      WORD8 *p_inp1 = p_inp0+(itr0*inp_stride[p_5D_permute_vec[0]]);
#pragma loop_count min=1
      for(itr1 = 0; itr1 < out_dim1; itr1++)
      {
        WORD8 *p_inp2 = p_inp1+(itr1*inp_stride[p_5D_permute_vec[1]]);
#pragma loop_count min=1
        for(itr2 = 0; itr2 < out_dim2; itr2++)
        {
          WORD8 *p_inp3 = p_inp2+(itr2*inp_stride[p_5D_permute_vec[2]]);
#pragma loop_count min=1
          for(itr3 = 0; itr3 < out_dim3; itr3++, p_out+=out_dim4)
          {
            WORD8 *p_inp4 = p_inp3+(itr3*inp_stride[p_5D_permute_vec[3]]);
            memcpy(p_out, p_inp4, out_dim4);
          }
        }
      }
    }
  }
  else
  {
    int itr0, itr1, itr2, itr3, itr4;
    WORD8 *p_inp0 = (WORD8*)p_inp;
    for(itr0 = 0; itr0 < out_dim0; itr0++)
    {
      WORD8 *p_inp1 = p_inp0+(itr0*inp_stride[p_5D_permute_vec[0]]);
      for(itr1 = 0; itr1 < out_dim1; itr1++)
      {
        WORD8 *p_inp2 = p_inp1+(itr1*inp_stride[p_5D_permute_vec[1]]);
        for(itr2 = 0; itr2 < out_dim2; itr2++)
        {
          WORD8 *p_inp3 = p_inp2+(itr2*inp_stride[p_5D_permute_vec[2]]);
          for(itr3 = 0; itr3 < out_dim3; itr3++)
          {
            WORD8 *p_inp4 = p_inp3+(itr3*inp_stride[p_5D_permute_vec[3]]);
            for(itr4 = 0; itr4 < out_dim4; itr4++)
            {
              WORD8 d0 = *(p_inp4);
              p_inp4 += inp_stride[p_5D_permute_vec[4]];
              *p_out++ = d0;

            }
          }
        }
      }
    }
  }

  return 0;
}
