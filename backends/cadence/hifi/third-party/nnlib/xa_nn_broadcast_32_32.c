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
/*
 * xa_nn_broadcast_32_32.c
 */

#include "xa_nnlib_common.h"
//#include "xa_nn_basic_state.h"

#include<string.h>
#include<stdbool.h>

#include "stdio.h"

/*
 * This file is sourced from ../hifi5/xa_nn_broadcast_8_8.c
 */

#define NUMDIMS_MAX 8

typedef struct bcast_expansion_struct_{
    size_t load_num_elem;
    int    replicate_loadedElm_times;
    int    repeat_operation;
} bcast_expansion_rule ;

WORD32* broadcast_node_32(bcast_expansion_rule *steps, unsigned int step_id,
        WORD32 *dst, WORD32 *src);

void *xa_nn_memcpy(void * dest1,const void *src1, size_t n1)
{
  char *dest = (char *)dest1;
  char *src = (char *)src1;
  int n = (int)n1;
  ae_int16x4 * __restrict d_align_addr, * __restrict s_align_addr;
  int i;
  void *orig_dest = dest;

  if (n < 32) {
    return memcpy(dest, src, n);
  }

  if ( !(((int) dest) %8) && !(((int) src) %8)) { // 64-bit aligned
    s_align_addr = (ae_int16x4 *) src;
    d_align_addr = (ae_int16x4 *) dest;
    for (i=0; i<n>>3; i++) {
        d_align_addr[i] = s_align_addr[i];
    }

    for (i=(n&~7); i<n; i++) {
      dest[i] = src[i];
    }
    return orig_dest;
  }

  if ( (((int) dest) %2) || (((int) src) %2)) { // 16-bit aligned
    if ( (((int) dest) %2) && (((int) src) %2)) { // 16-bit aligned
      *dest++ = *src++;
       n--;
    } else {
      #if 0
      return memcpy(dest, src, n);
      #else
        ae_int32x2 *pOut = (ae_int32x2 *)dest;
        ae_int32x2 *pInp = (ae_int32x2 *)src;
        ae_valign alignIn, alignOut;
        alignIn = AE_LA64_PP(pInp);
        alignOut = AE_ZALIGN64();
        ae_int24x2 d0;
        int Nby6 =  AE_MOVAD32_H(AE_MOVINT32X2_FROMINT64(AE_MUL32_LL(n, 0x2AAAAAAB)));
        int remainder_start = 6*Nby6;

        for(i=0;i<Nby6;i++)
        {
          AE_LA24X2_IP(d0, alignIn, pInp);
          AE_SA24X2_IP(d0, alignOut, pOut);
        }
        AE_SA64POS_FP(alignOut, pOut);
        /* remainder loop */
        for(i=remainder_start; i < n; i++){
          dest[i] = src[i];
      }
      return orig_dest;
      #endif
    }
  }
  int n2 = n/2;
  ae_valign d_align = AE_ZALIGN64();
  d_align_addr = (ae_int16x4 *) dest;
  s_align_addr = (ae_int16x4 *) src;
  ae_valign s_align = AE_LA64_PP(s_align_addr);
  ae_int16x4 t,t2;
  for (i=0; i<n2>>3; i++) {
      AE_LA16X4_IP(t, s_align, s_align_addr);
      AE_LA16X4_IP(t2, s_align, s_align_addr);
      AE_SA16X4_IP(t, d_align, d_align_addr);
      AE_SA16X4_IP(t2, d_align, d_align_addr);
  }
  AE_SA64POS_FP(d_align, d_align_addr);
  ae_int16 *s_src = (ae_int16 *) src;
  ae_int16 *s_dest = (ae_int16 *) dest;
  for (i=8*i; i<n2; i++) {
    s_dest[i] = s_src[i];
  }
  if (n % 2) {
    dest[n-1] = src[n-1];
  }
  return orig_dest;
} /* xa_nn_memcpy */

WORD32 xa_nn_broadcast_32_32( WORD32* __restrict__ p_out,      /* pointer to write broadcasted output data to */
        const int *const out_shape,         /* output shape resulting after broadcast */

        WORD32* __restrict__ p_in,    /* pointer to unextended input data */
        const int * const in_shape,         /* input shape */
        int num_dims)
{

    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(out_shape, -1);
    XA_NNLIB_ARG_CHK_PTR(p_in, -1);
    XA_NNLIB_ARG_CHK_PTR(in_shape, -1);

    /* IO shape pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(in_shape, sizeof(WORD32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(out_shape, sizeof(WORD32), -1);

    /* Check if number of dims is valid */
    XA_NNLIB_ARG_CHK_COND(num_dims<=0 || num_dims>8, -1);

    int i = 0;

    /* Check for valid IO shapes */
    for(i=0; i<num_dims; i++){
        XA_NNLIB_CHK_COND(in_shape[i]<=0, -1);
        XA_NNLIB_CHK_COND(out_shape[i]<=0, -1);
    }

    /* Check if input shape can be broadcasted to requested output shape */
    for(i=0; i<num_dims; i++){
        if(in_shape[i] != out_shape[i]){
            /* in_shape is either same as out_shape or 1 */
            XA_NNLIB_CHK_COND( in_shape[i] != 1, -1);
        }
    }

    /* bcast_expansion_steps contains a sequence to steps execute for a broadcast op */
    bcast_expansion_rule bcast_expansion_steps[NUMDIMS_MAX] = {{0}};

    int k=0;
    int dim=0;
    const void *res=0;

    int num_elem_load = 1;
    int num_copy_times = 1;
    int num_repeat = 1;

    dim = num_dims-1;
    while(dim>=0){

        /* Find the sub-matrix size */
        while(in_shape[dim] != 1 && dim>=0){
            num_elem_load *= out_shape[dim];
            dim--;
        }

        /* Find the number of times this sub-matrix needs to be copied */
        num_copy_times = 1;
        while(in_shape[dim] == 1 && dim>=0){
            num_copy_times *= out_shape[dim];
            dim--;
        }

        /* Find the number of times the above copy needs to be repeated */
        num_repeat = 1;
        while(in_shape[dim] != 1 && dim>=0){
            num_repeat *= 1 * out_shape[dim];
            dim--;
        }

        bcast_expansion_steps[k].load_num_elem  = num_elem_load;
        bcast_expansion_steps[k].replicate_loadedElm_times = num_copy_times;
        bcast_expansion_steps[k].repeat_operation = num_repeat;
        k++;

        num_elem_load = num_elem_load * num_copy_times * num_repeat;
    }

    res = broadcast_node_32(bcast_expansion_steps, num_dims-1,
            p_out, p_in);
    (void)res; /* Unused return value */

    return 0;
}

WORD32* broadcast_node_32(bcast_expansion_rule *steps, unsigned int step_id,
        WORD32 *dst, WORD32 *src) {
    int step_itr=0, rep_itr=0;
    int i=0, j=0, k=0;
    bcast_expansion_rule *step = NULL;

    // ignore steps that are null
    while(steps[step_id].repeat_operation == 0 && step_id>0){
        step_id--;
    }

    // step is now the parent node for this iteration
    step = &steps[step_id];
    size_t numLoadedElm = step->load_num_elem;

    WORD32 *cp_dst = dst;
    WORD32 *cp_src = src;
    WORD32 *cp_src_temp=NULL;
    WORD32 *cp_dst_temp=NULL;

    if(numLoadedElm>32){
        if(step_id > 0){
            for(step_itr=0; step_itr<step->repeat_operation; step_itr++){
                src = broadcast_node_32(steps, step_id-1, dst, src);
                cp_src = dst;
                cp_dst = dst + numLoadedElm;
                for(rep_itr=1; rep_itr<step->replicate_loadedElm_times; rep_itr++){
                    xa_nn_memcpy(cp_dst, cp_src, 4 * numLoadedElm);
                    cp_dst += numLoadedElm;
                }
                dst = cp_dst;
            }
            return src;
        } else {
            if(numLoadedElm == 1){
                for(j=0; j<step->repeat_operation; j++){
//                    memset((void*)cp_dst, (void*)cp_src, 4 * step->replicate_loadedElm_times);
                	for(i = 0; i < step->replicate_loadedElm_times; i++)
                		cp_dst[i] = cp_src[0];
                    cp_dst += step->replicate_loadedElm_times;
                    cp_src++;
                }
            } else {
                for(j=0; j<step->repeat_operation; j++){
                    for(i=0; i<step->replicate_loadedElm_times; i++){
                        xa_nn_memcpy(cp_dst, cp_src, 4 * numLoadedElm);
                        cp_dst += numLoadedElm;
                    }
                    cp_src += numLoadedElm;
                }
            }
            return cp_src;
        }
    }
    else{
        if(step_id > 0){
            for(step_itr=0; step_itr<step->repeat_operation; step_itr++){
                src = broadcast_node_32(steps, step_id-1, dst, src);
                cp_src = dst;
                cp_dst = dst + numLoadedElm;
                for(rep_itr=1; rep_itr<step->replicate_loadedElm_times; rep_itr++){
                    for(k=0; k<(int)numLoadedElm; k++){
                        cp_src_temp = cp_src;
                        cp_dst_temp = cp_dst;
                        cp_dst_temp[k] = cp_src_temp[k];
                    }
                    cp_dst += numLoadedElm;
                }
                dst = cp_dst;
            }
            return src;
        } else {
            if(numLoadedElm == 1){
                for(j=0; j<step->repeat_operation; j++){
//                    memset((void*)cp_dst, *(WORD32 *)cp_src, 4 * step->replicate_loadedElm_times);
                	for(i = 0; i < step->replicate_loadedElm_times; i++)
                		cp_dst[i] = cp_src[0];
                    cp_dst += step->replicate_loadedElm_times;
                    cp_src++;
                }
            } else {
                for(j=0; j < step->repeat_operation; j++){
                    for(i=0; i < step->replicate_loadedElm_times; i++){
                        for(k=0; k<(int)(numLoadedElm); k++){
                            cp_src_temp = cp_src;
                            cp_dst_temp = cp_dst;
                            cp_dst_temp[k] = cp_src_temp[k];

                        }
                        cp_dst += numLoadedElm;
                    }
                    cp_src += numLoadedElm;
                }
            }
            return cp_src;
        }
    }
}
