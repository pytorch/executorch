#include "xa_nnlib_common.h"
#include <string.h>
//#include "xa_nn_basic_state.h"
#include "xa_nnlib_common_macros.h"

#define ALIGNMENT_8   8

#define ALIGN_PTR(x, bytes)     ((((unsigned)(x))+(bytes-1))&(~(bytes-1)))

static void vecmean16_inpx3(const xtfloatx2 *p_src1, const xtfloat* p_src2, const xtfloat* p_src3, xtfloatx2 *p_dst, int N){
  int i = 0;
  ae_valign align_src1, align_dst;
  ae_valign align_src2, align_src3;
  align_src1 = AE_LA64_PP(p_src1);
  align_src2 = AE_LA64_PP(p_src2);
  align_src3 = AE_LA64_PP(p_src3);
  align_dst = AE_ZALIGN64();

  for(i=0; i < (N >> 2); i++)
  {
    xtfloatx2 j1_h, j1_l, j2_h, j2_l;

    xtfloatx2 wout1, wout2;
    XT_LASX2IP(wout1, align_src1, p_src1);
    XT_LASX2IP(wout2, align_src1, p_src1);
    
    XT_LASX2IP(j1_h, align_src2, (xtfloatx2 *)p_src2);
    XT_LASX2IP(j1_l, align_src2, (xtfloatx2 *)p_src2);
    XT_LASX2IP(j2_h, align_src3, (xtfloatx2 *)p_src3);
    XT_LASX2IP(j2_l, align_src3, (xtfloatx2 *)p_src3);  

    j1_h = XT_ADD_SX2(j1_h, j2_h);
    j1_l = XT_ADD_SX2(j1_l, j2_l);
    wout1 = XT_ADD_SX2(wout1, j1_h);
    wout2 = XT_ADD_SX2(wout2, j1_l);

    XT_SASX2IP(wout1, align_dst, p_dst);
    XT_SASX2IP(wout2, align_dst, p_dst);
  }
  AE_SA64POS_FP(align_dst, p_dst); // finalize the stream

  //Remainder Loop
  for(i=0; i < (N & 3); i++)
  {
    xtfloat j1, j2;
    xtfloat wout1;
    XT_LSXP(wout1, (xtfloat *)p_src1, sizeof(xtfloat));
    j1 = (xtfloat) *(p_src2 + i);
    j2 = (xtfloat) *(p_src3 + i);
    
    j1 = XT_ADD_S(j1, j2);
    wout1 = XT_ADD_S(wout1, j1);
    XT_SSXP(wout1, (xtfloat *)p_dst, sizeof(xtfloat));
  }
}

static void vecmean16_inpx2(const xtfloatx2 *p_src1, const xtfloat* p_src2, xtfloatx2 *p_dst, int N){
  ae_valign align_src1, align_dst;
  ae_valign align_src2;
  align_src1 = AE_LA64_PP(p_src1);
  align_src2 = AE_LA64_PP(p_src2);
  align_dst = AE_ZALIGN64();

  int i = 0;
  for(i=0; i < (N >> 2); i++)
  {
    xtfloatx2 j1, j2;
    xtfloatx2 wout1, wout2;
    XT_LASX2IP(wout1, align_src1, p_src1);
    XT_LASX2IP(wout2, align_src1, p_src1);

    XT_LASX2IP(j1, align_src2, (xtfloatx2 *)p_src2);
    XT_LASX2IP(j2, align_src2, (xtfloatx2 *)p_src2);
    
    wout1 = XT_ADD_SX2(wout1, j1);
    wout2 = XT_ADD_SX2(wout2, j2);

    XT_SASX2IP(wout1, align_dst, p_dst);
    XT_SASX2IP(wout2, align_dst, p_dst);
  }
  AE_SA64POS_FP(align_dst, p_dst); // finalize the stream

  //Remainder Loop
  for(i=0; i < (N & 3); i++)
  {
    xtfloat j1;
    xtfloat wout1;
    XT_LSXP(wout1, (xtfloat *)p_src1, sizeof(xtfloat));
    j1 = (xtfloat) *(p_src2 + i);
    wout1 = XT_ADD_S(wout1, j1);
    XT_SSXP(wout1, (xtfloat *)p_dst, sizeof(xtfloat));
  }
}

static void vecmean32_inpx3(const xtfloatx2* p_src1, const xtfloatx2* p_wsrc2, const xtfloatx2* p_wsrc3, xtfloatx2 *p_dst, int N){
  ae_valign align_src1, align_src2, align_src3, align_dst;
  align_src1 = AE_LA64_PP(p_src1);
  align_src2 = AE_LA64_PP(p_wsrc2);
  align_src3 = AE_LA64_PP(p_wsrc3);
  align_dst = AE_ZALIGN64();

  int i = 0;
  for(i=0; i < (N >> 2); i++)
  {
    xtfloatx2 j1, j2, j3, j4;
    xtfloatx2 wj1, wj2;
    xtfloatx2 wout1, wout2;
    XT_LASX2IP(wout1, align_src1, p_src1);
    XT_LASX2IP(wout2, align_src1, p_src1);
    XT_LASX2IP(j1, align_src2, p_wsrc2);
    XT_LASX2IP(j2, align_src3, p_wsrc3);
    XT_LASX2IP(j3, align_src2, p_wsrc2);
    XT_LASX2IP(j4, align_src3, p_wsrc3);
    
    wj1 = XT_ADD_SX2(j1, j2);
    wj2 = XT_ADD_SX2(j3, j4);
    wout1 = XT_ADD_SX2(wout1, wj1);
    wout2 = XT_ADD_SX2(wout2, wj2);
    XT_SASX2IP(wout1, align_dst, p_dst);
    XT_SASX2IP(wout2, align_dst, p_dst);
  }
  AE_SA64POS_FP(align_dst, p_dst); // finalize the stream

  //Remainder Loop
  for(i=0; i < (N & 3); i++)
  {
    xtfloat j1, j2;
    xtfloat wj1;
    xtfloat wout1;
    XT_LSXP(wout1, (xtfloat *)p_src1, 4);
    XT_LSXP(j1, (xtfloat *)p_wsrc2, 4);
    XT_LSXP(j2, (xtfloat *)p_wsrc3, 4);
    wj1 = XT_ADD_S(j1, j2);
    wout1 = XT_ADD_S(wout1, wj1);
    XT_SSXP(wout1, (xtfloat *)p_dst, sizeof(xtfloat));
  }
}

static void vecmean32_inpx2(const xtfloatx2* p_src1, const xtfloatx2* p_wsrc2, xtfloatx2 *p_dst, int N){
  ae_valign align_src1, align_src2, align_dst;
  align_src1 = AE_LA64_PP(p_src1);
  align_src2 = AE_LA64_PP(p_wsrc2);
  align_dst = AE_ZALIGN64();

  int i = 0;
  for(i=0; i < (N >> 2); i++)
  {
    xtfloatx2 j1, j2;
    xtfloatx2 wout1, wout2;
    XT_LASX2IP(wout1, align_src1, p_src1);
    XT_LASX2IP(wout2, align_src1, p_src1);
    XT_LASX2IP(j1, align_src2, p_wsrc2);
    XT_LASX2IP(j2, align_src2, p_wsrc2);
    wout1 = XT_ADD_SX2(wout1, j1);
    wout2 = XT_ADD_SX2(wout2, j2);
    XT_SASX2IP(wout1, align_dst, p_dst);
    XT_SASX2IP(wout2, align_dst, p_dst);
  }
  AE_SA64POS_FP(align_dst, p_dst); // finalize the stream

  //Remainder Loop
  for(i=0; i < (N & 3); i++)
  {
    xtfloat j1;
    xtfloat wout1;
    XT_LSXP(wout1, (xtfloat *)p_src1, 4);
    XT_LSXP(j1, (xtfloat *)p_wsrc2, 4);
    wout1 = XT_ADD_S(wout1, j1);
    XT_SSXP(wout1, (xtfloat *)p_dst, sizeof(WORD32));
  }
}

static inline void xa_nn_reduce_sum_4D_f32_f32(const FLOAT32 * __restrict__ p_inp
                                                       ,const WORD32 *const p_4D_inp_shape
                                                       ,const WORD32 * __restrict__ p_axis_data
                                                       ,WORD32 num_inp_dims
                                                       ,WORD32 num_axis_dims
                                                       ,pVOID p_scratch_in)
{
  xtfloat *p_in = (xtfloat *)(p_inp);
  xtfloat *p_scratch = (xtfloat *)(p_scratch_in);

  int temp_inp_n = p_4D_inp_shape[0]; 
  int temp_inp_h = p_4D_inp_shape[1]; 
  int temp_inp_w = p_4D_inp_shape[2]; 
  int temp_inp_c = p_4D_inp_shape[3];

  int itr_axis = 0, itr_n = 0, itr_h = 0, itr_w = 0, itr_c = 0;
  xtfloat *p_src2, *p_src3;
  xtfloatx2 *p_src1;
  xtfloatx2 * p_dst;
  ae_valign align_src2;

  int axis_dims_count = num_axis_dims;
  if(axis_dims_count)
  {
    switch(p_axis_data[itr_axis])
    {
      case 0: {
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        for(itr_n=0; itr_n < (temp_inp_n & ~(2 - 1)); itr_n += 2)
        {
          p_src1 = (xtfloatx2 *)p_scratch;
          p_src2 = p_in + itr_n * plane_size;
          p_src3 = p_in + (itr_n + 1) * plane_size;
          p_dst  = (xtfloatx2 *)p_scratch;
          vecmean16_inpx3(p_src1, p_src2, p_src3, p_dst, plane_size);
        }

        if(temp_inp_n & 1)
        {
          p_src1 = (xtfloatx2 *)p_scratch;
          p_src2 = (p_in + itr_n * plane_size);
          p_dst  = (xtfloatx2 *)p_scratch;
          vecmean16_inpx2(p_src1, p_src2, p_dst, plane_size);
        }
        temp_inp_n = 1;  
        }break;
      case 1: {     
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int wc_plane_size = temp_inp_w * temp_inp_c;
        for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
        {
          p_src1 = (xtfloatx2 *)(p_scratch + (itr_n * wc_plane_size)); 
          for(itr_h=0; itr_h < (temp_inp_h & ~(2 - 1)); itr_h += 2)
          {
            p_src2 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size);
            p_src3 = p_in + (itr_n * plane_size) + ((itr_h + 1) * wc_plane_size);
            p_dst = (xtfloatx2 *)(p_scratch + (itr_n * wc_plane_size));
            vecmean16_inpx3(p_src1, p_src2, p_src3, p_dst, wc_plane_size);
            p_src1 = (xtfloatx2 *)(p_scratch + (itr_n * wc_plane_size));
          }

          if(temp_inp_h & 1)
          {
            p_src2 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size);
            p_dst = (xtfloatx2 *)(p_scratch + (itr_n * wc_plane_size));
            vecmean16_inpx2(p_src1, p_src2, p_dst, wc_plane_size);
          }
        }
        temp_inp_h = 1;
        }break;
      case 2:{                    
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int wc_plane_size = temp_inp_w * temp_inp_c;
        int hc_plane_size = temp_inp_h * temp_inp_c;

        for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
        {
          for(itr_h=0; itr_h < (temp_inp_h); itr_h++)
          {
            p_src1 = (xtfloatx2 *)(p_scratch + (((itr_n * hc_plane_size) + itr_h * temp_inp_c))); 
            for(itr_w=0; itr_w < (temp_inp_w & ~(2 - 1)); itr_w += 2)
            {
              p_src2 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c);
              p_src3 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + ((itr_w + 1) * temp_inp_c);
              p_dst = (xtfloatx2 *)(p_scratch + (itr_n * hc_plane_size) + itr_h * temp_inp_c);
              vecmean16_inpx3(p_src1, p_src2, p_src3, p_dst, temp_inp_c);
              p_src1 = (xtfloatx2 *)(p_scratch + (itr_n * hc_plane_size) + (itr_h * temp_inp_c));
            }

            if(temp_inp_w & 1)
            {
              p_src2 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c);
              p_dst = (xtfloatx2 *)(p_scratch + (itr_n * hc_plane_size) + itr_h * temp_inp_c);
              vecmean16_inpx2(p_src1, p_src2, p_dst, temp_inp_c);
            }
          }
          }
        temp_inp_w = 1;
        }break;
      case 3: {                  
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int wc_plane_size = temp_inp_w * temp_inp_c;
        int hw_plane_size = temp_inp_h * temp_inp_w;
        int rem_c = (temp_inp_c & 7); 

        for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
        {
          for(itr_h=0; itr_h < (temp_inp_h); itr_h++)
          {
            for(itr_w=0; itr_w < (temp_inp_w); itr_w++)
            {
              p_src1 = (xtfloatx2 *)(p_scratch + (((itr_n * hw_plane_size) + (itr_h * temp_inp_w) + itr_w)));
              p_src2 = p_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c);
              p_dst = (xtfloatx2 *)(p_scratch + (itr_n * hw_plane_size) + (itr_h * temp_inp_w) + itr_w);
              align_src2 = AE_LA64_PP(p_src2);

              for(itr_c=0; itr_c < (temp_inp_c >> 3); itr_c++)
              {
                xtfloatx2 j11, j12, j21, j22, i1;
                i1 = XT_LSX((xtfloat *)p_src1, 0);
                XT_LASX2IP(j11, align_src2, (xtfloatx2 *)p_src2);
                XT_LASX2IP(j12, align_src2, (xtfloatx2 *)p_src2);
                XT_LASX2IP(j21, align_src2, (xtfloatx2 *)p_src2);
                XT_LASX2IP(j22, align_src2, (xtfloatx2 *)p_src2);
                
                j11 = XT_ADD_SX2(j11, j12);
                j21 = XT_ADD_SX2(j21, j22);
                
                xtfloatx2 t1 = XT_SEL32_HH_SX2(j11, j11);
                xtfloatx2 t2 = XT_SEL32_HH_SX2(j21, j21);
                
                j11 = XT_ADD_SX2(j11, t1);
                j21 = XT_ADD_SX2(j21, t2);
                
                j11 = XT_ADD_SX2(j11, j21);
                i1 = XT_ADD_SX2(i1, j11);
                
                XT_SSX(i1, (xtfloat *)p_dst, 0);
                
                p_src1 = p_dst;
              }
              //Remainder Loop
              for(itr_c=0; itr_c < rem_c ; itr_c++)
              {
                xtfloat j1;
                xtfloat i1;
                i1 = XT_LSX((xtfloat *)p_src1, 0);
                j1 = *p_src2++;
                
                i1 = XT_ADD_S(i1, j1);
                XT_SSX(i1, (xtfloat *)p_dst, 0);
              }
            }
          }
        }
        temp_inp_c = 1;
        }break;
      default:
        break;
    }

    axis_dims_count--;
    itr_axis++;
  }

  while(axis_dims_count)
  {
    ae_valign align_src;
    xtfloat *p_scr_in = p_scratch;
    xtfloatx2 *p_wsrc2, *p_wsrc3;
    switch(p_axis_data[itr_axis])
    {
      case 0: {              
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        for(itr_n=1; itr_n < ((temp_inp_n -1) & ~(2 - 1)); itr_n += 2)
        {
          p_src1 = (xtfloatx2 *)p_scratch;
          p_wsrc2 = (xtfloatx2 *)(p_scr_in + itr_n * plane_size);
          p_wsrc3 = (xtfloatx2 *)(p_scr_in + (itr_n + 1) * plane_size);
          p_dst  = (xtfloatx2 *)p_scratch;
          vecmean32_inpx3(p_src1, p_wsrc2, p_wsrc3, p_dst, plane_size);
        }

        if((temp_inp_n - 1) & 1)
        {
          p_src1 = (xtfloatx2 *)p_scratch;
          p_wsrc2 = (xtfloatx2 *)(p_scr_in + itr_n * plane_size);
          p_dst  =  (xtfloatx2 *)p_scratch;
          vecmean32_inpx2(p_src1, p_wsrc2, p_dst, plane_size);
        }
        temp_inp_n = 1;
        }break;
      case 1: {            
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int wc_plane_size = temp_inp_w * temp_inp_c;
        for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
        {
          p_src1 = (xtfloatx2 *)(p_scratch + + (itr_n * plane_size));
          for(itr_h = 1; itr_h < ((temp_inp_h - 1) & ~(2 - 1)); itr_h += 2)
          {
            p_wsrc2 = (xtfloatx2 *)(p_scr_in + (itr_n * plane_size) + (itr_h * wc_plane_size));
            p_wsrc3 = (xtfloatx2 *)(p_scr_in + (itr_n * plane_size) + ((itr_h + 1) * wc_plane_size));
            p_dst = (xtfloatx2 *)(p_scratch + (itr_n * wc_plane_size));
            vecmean32_inpx3(p_src1, p_wsrc2, p_wsrc3, p_dst, wc_plane_size);
            p_src1 = (xtfloatx2 *)(p_scratch + (itr_n * wc_plane_size));
          }

          if((temp_inp_h - 1) & 1)
          {
            p_wsrc2 = (xtfloatx2 *)(p_scr_in + (itr_n * plane_size) + (itr_h * wc_plane_size));
            p_dst = (xtfloatx2 *)(p_scratch + (itr_n * wc_plane_size));
            vecmean32_inpx2(p_src1, p_wsrc2, p_dst, plane_size);
          }
        }
        temp_inp_h = 1;
        }break;
      case 2:{                
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int wc_plane_size = temp_inp_w * temp_inp_c;
        int hc_plane_size = temp_inp_h * temp_inp_c;
        for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
        {
          for(itr_h=0; itr_h < (temp_inp_h); itr_h++)
          {
            p_src1 = (xtfloatx2 *)(p_scratch + ((itr_n * plane_size) + (itr_h * wc_plane_size)));
            for(itr_w = 1; itr_w < ((temp_inp_w - 1) & ~(2 - 1)); itr_w += 2)
            {
              p_wsrc2 = (xtfloatx2 *)(p_scr_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c));
              p_wsrc3 = (xtfloatx2 *)(p_scr_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + ((itr_w + 1) * temp_inp_c));
              p_dst = (xtfloatx2 *)(p_scratch + (itr_n * hc_plane_size) + itr_h * temp_inp_c);
              vecmean32_inpx3(p_src1, p_wsrc2, p_wsrc3, p_dst, temp_inp_c);
              p_src1 = (xtfloatx2 *)(p_scratch + (itr_n * hc_plane_size) + (itr_h * temp_inp_c));
            }

            if((temp_inp_w - 1) & 1)
            {
              p_wsrc2 = (xtfloatx2 *)(p_scr_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c));
              p_dst = (xtfloatx2 *)(p_scratch + (itr_n * hc_plane_size) + itr_h * temp_inp_c);
              vecmean32_inpx2(p_src1, p_wsrc2, p_dst, temp_inp_c);
            }
          }
        }
        temp_inp_w = 1;
        }break;
      case 3: {              
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int wc_plane_size = temp_inp_w * temp_inp_c;
        int hw_plane_size = temp_inp_h * temp_inp_w;
        int rem_c = ((temp_inp_c) & 3); 
        for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
        {
          for(itr_h=0; itr_h < (temp_inp_h); itr_h++)
          {
            for(itr_w=0; itr_w < (temp_inp_w); itr_w++)
            {
              p_wsrc2 = (xtfloatx2 *)(p_scr_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c));
              p_dst = (xtfloatx2 *)(p_scratch + (itr_n * hw_plane_size) + (itr_h * temp_inp_w) + itr_w);
              align_src = AE_LA64_PP(p_wsrc2);
              xtfloatx2 i1 = XT_AE_MOVXTFLOATX2_FROMF32X2(AE_MOVDA32(0));
              for(itr_c = 0; itr_c < (temp_inp_c >> 2); itr_c++)
              {
                xtfloatx2 j1, j2;
                XT_LASX2IP(j1, align_src, p_wsrc2);
                XT_LASX2IP(j2, align_src, p_wsrc2);
                
                xtfloatx2 t1 = XT_SEL32_HH_SX2(j1, j1);
                xtfloatx2 t2 = XT_SEL32_HH_SX2(j2, j2);
                
                j1 = XT_ADD_SX2(t1, j1);
                j2 = XT_ADD_SX2(t2, j2);
                
                i1 = XT_ADD_SX2(i1, j1);
                i1 = XT_ADD_SX2(i1, j2);
              }

              //Remainder Loop
              for(itr_c=0; itr_c < rem_c; itr_c++)
              {
                xtfloat j1;
                XT_LSXP(j1, (xtfloat *)p_wsrc2, sizeof(xtfloat)); 
                i1 = XT_ADD_S(i1, j1);
              }
              XT_SSX(i1, (xtfloat *)p_dst, 0);
            }
          }
        }
        temp_inp_c = 1;
        }break;
      default:
        break;
    }
    axis_dims_count--;
    itr_axis++;
  }
}

WORD32 xa_nn_reduce_mean_4D_f32_f32(
                                    FLOAT32 * __restrict__ p_out,
                                    const WORD32 *const p_out_shape,
                                    const FLOAT32 * __restrict__ p_inp,
                                    const WORD32 *const p_inp_shape,
                                    const WORD32 * __restrict__ p_axis,
                                    WORD32 num_out_dims,
                                    WORD32 num_inp_dims,
                                    WORD32 num_axis_dims,
                                    void * __restrict__ p_scratch_in)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_axis, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp_shape, -1);

  /* Invalid input checks */
  XA_NNLIB_ARG_CHK_COND(((num_inp_dims <= 0) || (num_inp_dims > 4)), -1);
  XA_NNLIB_ARG_CHK_COND(((num_out_dims <= 0) || (num_out_dims > 4)), -1);
  XA_NNLIB_ARG_CHK_COND(((num_axis_dims < 0) || (num_axis_dims > 4)), -1);

  int axis_itr = 0, inp_itr = 0, out_itr = 0;
  int num_elm_in_axis = 1;
  int current, past = -1;
  for(axis_itr=0; axis_itr < num_axis_dims; axis_itr++)
  {
    current = p_axis[axis_itr];
    XA_NNLIB_ARG_CHK_COND(((current < 0) || (current > (num_inp_dims - 1))), -1);
    XA_NNLIB_ARG_CHK_COND((p_inp_shape[current] > 1024), -1);

    /* Avoid calculation in case of repeated axis dims*/
    if(current != past)
    {
      num_elm_in_axis *= p_inp_shape[current];
      past = current;
    }
  }

  for(inp_itr=0; inp_itr < num_inp_dims; inp_itr++)
  {
    XA_NNLIB_ARG_CHK_COND((p_inp_shape[inp_itr] <= 0), -1);
  }

  int out_length = 1;
  for(out_itr=0; out_itr < num_out_dims; out_itr++)
  {
    XA_NNLIB_ARG_CHK_COND((p_out_shape[out_itr] <= 0), -1);
    out_length *= p_out_shape[out_itr];
  }

  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_axis, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp_shape, sizeof(WORD32), -1);

  FLOAT32 *p_in = (FLOAT32 *)(p_inp);
  WORD32 *p_scratch = (WORD32 *)(ALIGN_PTR(p_scratch_in, ALIGNMENT_8));

  // Changing order of axis data so that reduce max will be first computed
  // across largest inp shape dim in axis. This is required to
  // minimize the scratch usage.
  int inp_length = 1, p_axis_data[4] = {0}, inp_shape_max;
  if(num_axis_dims)
  {
    inp_shape_max = p_inp_shape[p_axis[0]];
    axis_itr = 1;
    int max_axis_itr = 0;
    int temp_p_axis_0 = p_axis[0];
    for(axis_itr = 0; axis_itr < num_axis_dims; axis_itr++)
    {
      p_axis_data[axis_itr] = p_axis[axis_itr];
    }
    for(axis_itr = 1; axis_itr < num_axis_dims; axis_itr++)
    {
      if(p_inp_shape[p_axis[axis_itr]] > inp_shape_max)
      {
        inp_shape_max = p_inp_shape[p_axis[axis_itr]];
        max_axis_itr = axis_itr;
      }
    }
    p_axis_data[0] = p_axis_data[max_axis_itr];
    p_axis_data[max_axis_itr] = temp_p_axis_0;

    inp_itr = 0;
    for(inp_itr=0; inp_itr < num_inp_dims; inp_itr++)
    {
      inp_length *= p_inp_shape[inp_itr];
    }

    memset(p_scratch, 0, ((inp_length / inp_shape_max) * sizeof(WORD32))); //TODO: Alternate approach for memset?
  }

  // Promoting lesser dim tensors to 4D tensors. Also modifying axis
  // data accordingly.
  int p_4D_inp_shape[4] = {1, 1, 1, 1};
  int itr = num_inp_dims - 1;
  int count = 3;
  while(itr >= 0)
  {
    p_4D_inp_shape[count] = p_inp_shape[itr];
    itr--;
    count--;
  }
  for(itr = 0; itr < num_axis_dims; itr++)
  {
    p_axis_data[itr] = p_axis_data[itr] + (4 - num_inp_dims);
  }
  ae_valign align_out = AE_ZALIGN64();

  if(num_axis_dims)
  {
    if(num_elm_in_axis > 1)
    { 
      xa_nn_reduce_sum_4D_f32_f32(p_in,
                                  p_4D_inp_shape,
                                  p_axis_data,
                                  num_inp_dims,
                                  num_axis_dims,
                                  p_scratch);
      itr = 0;
      xtfloatx2 *p_src1 = (xtfloatx2 *)(p_scratch);
      
      float div = 1;
      
      for(int i = 0; i < num_axis_dims; i++)
      {
        div = div * (float)p_4D_inp_shape[p_axis_data[i]];
      }
      
      float mul = 1 / div;
    
      xtfloatx2 multiplier = XT_LSX((xtfloat *)&mul, 0);

      for(itr = 0; itr < (out_length >> 3); itr++)
      {
        xtfloatx2 temp1, temp2, temp3, temp4;

        temp2 = XT_LSX2X(p_src1, 8);
        temp3 = XT_LSX2X(p_src1, 16);
        temp4 = XT_LSX2X(p_src1, 24);
        XT_LSX2XP(temp1, p_src1, 32);
        
        temp1 = XT_MUL_SX2(temp1, multiplier);
        temp2 = XT_MUL_SX2(temp2, multiplier);
        temp3 = XT_MUL_SX2(temp3, multiplier);
        temp4 = XT_MUL_SX2(temp4, multiplier);
        
        XT_SASX2IP(temp1, align_out, (xtfloatx2 *)p_out);
        XT_SASX2IP(temp2, align_out, (xtfloatx2 *)p_out);
        XT_SASX2IP(temp3, align_out, (xtfloatx2 *)p_out);
        XT_SASX2IP(temp4, align_out, (xtfloatx2 *)p_out);     
      }
      AE_SA64POS_FP(align_out, p_out);

      for(itr = 0; itr < (out_length & 7); itr++)
      {
        xtfloat temp1;
        XT_LSXP(temp1, (xtfloat *)p_src1, 4);
        temp1 = XT_MUL_S(temp1, multiplier);
        XT_SSXP(temp1, (xtfloat *)p_out, 4);
      }
    }
    else
    {

      memcpy(p_out, p_inp, inp_length * sizeof(FLOAT32));
    }
  }
  else
  {
    memcpy(p_out, p_inp, inp_length * sizeof(FLOAT32)); 
  }

  return 0;
}
