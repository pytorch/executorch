/*
 * This implementation is extracted from PyTorch:
 *   Repo: github.com/pytorch/pytorch
 *   File: torch/lib/TH/THHalf.c
 *   Commit ID: 92481b59d31199df57420d4b14912348cc780d1d
 * Functions are made "static inline" for performance
 */

/* Copyright 1993-2014 NVIDIA Corporation.  All rights reserved. */

// Host functions for converting between FP32 and FP16 formats

static inline void TH_halfbits2float(unsigned short* src, float* res)
{
    unsigned h = *src;
    unsigned sign = ((h >> 15) & 1);
    unsigned exponent = ((h >> 10) & 0x1f);
    unsigned mantissa = ((h & 0x3ff) << 13);

    if (exponent == 0x1f) {  /* NaN or Inf */
        mantissa = (mantissa ? (sign = 0, 0x7fffff) : 0);
        exponent = 0xff;
    } else if (!exponent) {  /* Denorm or Zero */
        if (mantissa) {
            unsigned int msb;
            exponent = 0x71;
            do {
                msb = (mantissa & 0x400000);
                mantissa <<= 1;  /* normalize */
                --exponent;
            } while (!msb);
            mantissa &= 0x7fffff;  /* 1.mantissa is implicit */
        }
    } else {
        exponent += 0x70;
    }

    *(unsigned*)res = ((sign << 31) | (exponent << 23) | mantissa);
}

static inline void TH_float2halfbits(float* src, unsigned short* dest)
{
    unsigned x = *(unsigned*)src;
    unsigned u = (x & 0x7fffffff), remainder, shift, lsb, lsb_s1, lsb_m1;
    unsigned sign, exponent, mantissa;

    // Get rid of +NaN/-NaN case first.
    if (u > 0x7f800000) {
      *dest = 0x7fffU;
      return ;
    }
  
    sign = ((x >> 16) & 0x8000);
  
    // Get rid of +Inf/-Inf, +0/-0.
    if (u > 0x477fefff) {
      *dest = sign | 0x7c00U;
      return; 
    }
    if (u < 0x33000001) {
      *dest = (sign | 0x0000);
      return;
    }

    exponent = ((u >> 23) & 0xff);
    mantissa = (u & 0x7fffff);

    if (exponent > 0x70) {
        shift = 13;
        exponent -= 0x70;
    } else {
        shift = 0x7e - exponent;
        exponent = 0;
        mantissa |= 0x800000;
    }
    lsb = (1 << shift);
    lsb_s1 = (lsb >> 1);
    lsb_m1 = (lsb - 1);
  
    // Round to nearest even.
    remainder = (mantissa & lsb_m1);
    mantissa >>= shift;
    if (remainder > lsb_s1 || (remainder == lsb_s1 && (mantissa & 0x1))) {
        ++mantissa;
        if (!(mantissa & 0x3ff)) {
            ++exponent;
            mantissa = 0;
        }
    }  

    *dest = (sign | (exponent << 10) | mantissa);  
}
