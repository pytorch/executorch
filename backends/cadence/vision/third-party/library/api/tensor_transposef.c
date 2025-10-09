#include "api.h"
#include "common.h"

/*
 * Currently only supports upto 5D input tensors.
 * 1/2/3/4 D input tensors will be scaled up to 5D.
 * For example, 2x3 -> 1x1x1x2x3.
 */

void tensor_transposef(float32_t *restrict ptr_out
    ,const int *const ptr_out_shape
    ,const float32_t *restrict ptr_inp
    ,const int *const ptr_inp_shape
    ,const int *restrict ptr_permute_vec
    ,int num_out_dims
    ,int num_inp_dims)
{

  /* Shift all dim with 1 in the outer part */
  int eff_output_shape[5];
  int eff_permute_vec[5];

  for (int i = 0; i < num_out_dims; i++){
    eff_output_shape[i] = ptr_out_shape[i];
    eff_permute_vec[i] = ptr_permute_vec[i];
  }

  int one_i = num_out_dims - 1, non_one_i = num_out_dims - 1;
  while (one_i > 0 && non_one_i >= 0){
    while (one_i > 0 && eff_output_shape[one_i] != 1){
      one_i--;
    }
    non_one_i = one_i;
    while (non_one_i >= 0 && eff_output_shape[non_one_i]==1){
      non_one_i--;
    }
    if (one_i > 0 && non_one_i >= 0){
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
  int ptr_5D_inp_shape[5] = {1, 1, 1, 1, 1};
  int ptr_5D_out_shape[5] = {1, 1, 1, 1, 1};
  int ptr_5D_permute_vec[5] = {0, 1, 2, 3, 4};

  /* Check if any inner inp dimension is same in the output */
  int last_dim_same = 1, last_n_same_dim = 0;
  int itr = num_inp_dims - 1;
  while(itr >= 0){
    last_n_same_dim = (last_dim_same && (eff_permute_vec[itr] == itr)) ? (last_n_same_dim + 1) : last_n_same_dim;
    last_dim_same = (eff_permute_vec[itr] == itr) ? last_dim_same & 1 : last_dim_same & 0;
    itr--;
  }

  int dims_added = 5 - num_inp_dims;
  itr = num_inp_dims - 1;
  int same_count = last_n_same_dim;
  int count = 4;
  while(itr >= 0){
    ptr_5D_inp_shape[count] = (same_count > 0) ? ptr_5D_inp_shape[count] * ptr_inp_shape[itr] : ptr_inp_shape[itr];
    ptr_5D_out_shape[count] = (same_count > 0) ? ptr_5D_out_shape[count] * eff_output_shape[itr] : eff_output_shape[itr];
    same_count--;
    itr--;
    count = (same_count > 0) ? count : count - 1;
  }

  itr = num_inp_dims - 1;
  same_count = (last_n_same_dim) ? num_inp_dims - (last_n_same_dim - 1) : 0;
  count = 4;
  while(itr >= 0){
    ptr_5D_permute_vec[count] = (same_count > 0) ? eff_permute_vec[itr-(last_n_same_dim - 1)] + dims_added + last_n_same_dim - 1 : eff_permute_vec[itr] + dims_added;
    same_count--;
    itr--;
    count--;
  }

  int out_dim0, out_dim1, out_dim2, out_dim3, out_dim4;
  int inp_dim1, inp_dim2, inp_dim3, inp_dim4;
  int inp_stride[5];

  out_dim0 = ptr_5D_out_shape[0];
  out_dim1 = ptr_5D_out_shape[1];
  out_dim2 = ptr_5D_out_shape[2];
  out_dim3 = ptr_5D_out_shape[3];
  out_dim4 = ptr_5D_out_shape[4];

  inp_dim1 = ptr_5D_inp_shape[1];
  inp_dim2 = ptr_5D_inp_shape[2];
  inp_dim3 = ptr_5D_inp_shape[3];
  inp_dim4 = ptr_5D_inp_shape[4];

  inp_stride[0] = inp_dim1 * inp_dim2 * inp_dim3 * inp_dim4;
  inp_stride[1] = inp_dim2 * inp_dim3 * inp_dim4;
  inp_stride[2] = inp_dim3 * inp_dim4;
  inp_stride[3] = inp_dim4;
  inp_stride[4] = 1;

  if (last_n_same_dim){
    int itr0, itr1, itr2, itr3, itr4;
    float32_t *ptr_inp0 = (float32_t *)ptr_inp;
    for (itr0 = 0; itr0 < out_dim0; itr0++){
      float32_t *ptr_inp1 = ptr_inp0 + (itr0 * inp_stride[ptr_5D_permute_vec[0]]);
#pragma looptr_count min=1
      for (itr1 = 0; itr1 < out_dim1; itr1++){
        float32_t *ptr_inp2 = ptr_inp1 + (itr1 * inp_stride[ptr_5D_permute_vec[1]]);
#pragma looptr_count min=1
        for (itr2 = 0; itr2 < out_dim2; itr2++){
          float32_t *ptr_inp3 = ptr_inp2 + (itr2 * inp_stride[ptr_5D_permute_vec[2]]);
#pragma looptr_count min=1
          for (itr3 = 0; itr3 < out_dim3; itr3++, ptr_out += out_dim4){
            float32_t *ptr_inp4 = ptr_inp3 + (itr3 * inp_stride[ptr_5D_permute_vec[3]]);
            xb_vecN_2xf32 *restrict pae_i = (xb_vecN_2xf32 *)(ptr_inp4);
            xb_vecN_2xf32 *restrict pae_o = (xb_vecN_2xf32 *)(ptr_out);
            valign a_inp = IVP_LAN_2XF32_PP(pae_i);
            valign a_out = IVP_ZALIGN();
            xb_vecN_2xf32 d0;
            for(itr4 = 0; itr4 < (out_dim4 >> (LOG2_IVP_SIMD_WIDTH - 1)); itr4++){
              IVP_LAN_2XF32_IP(d0, a_inp, pae_i);
              IVP_SAN_2XF32_IP(d0, a_out, pae_o);
            }
            IVP_SAPOSN_2XF32_FP(a_out, pae_o);
            float32_t *restrict puae_i = (float32_t *)(pae_i);
            float32_t *restrict puae_o = (float32_t *)(pae_o);
#pragma looptr_count max = 17
            for(itr4 = 0; itr4 < (out_dim4 & (IVP_SIMD_WIDTH / 2 - 1)); itr4++){
              puae_o[itr4] = puae_i[itr4];
            }
          }
        }
      }
    }
  }
  else{
    int itr0, itr1, itr2, itr3, itr4;
    float32_t *ptr_inp0 = (float32_t *)ptr_inp;
    for(itr0 = 0; itr0 < out_dim0; itr0++){
      float32_t *ptr_inp1 = ptr_inp0 + (itr0 * inp_stride[ptr_5D_permute_vec[0]]);
      for(itr1 = 0; itr1 < out_dim1; itr1++){
        float32_t *ptr_inp2 = ptr_inp1 + (itr1 * inp_stride[ptr_5D_permute_vec[1]]);
        for(itr2 = 0; itr2 < out_dim2; itr2++){
          float32_t *ptr_inp3 = ptr_inp2 + (itr2 * inp_stride[ptr_5D_permute_vec[2]]);
          for(itr3 = 0; itr3 < out_dim3; itr3++){
            float32_t *ptr_inp4 = ptr_inp3 + (itr3 * inp_stride[ptr_5D_permute_vec[3]]);
            for(itr4 = 0; itr4 < out_dim4; itr4++){
              *ptr_out++ = *ptr_inp4;
              ptr_inp4 = ptr_inp4 + inp_stride[ptr_5D_permute_vec[4]];
            }
          }
        }
      }
    }
  }
}
