/*
 * Dump output tensor data after each operator for layer-by-layer comparison.
 * Include with: #include <dump_tensor.h>
 *
 * Output format:
 *   LAYER_DUMP : <op> : <numel> : dtype=<d> : first=[v0,v1,...] : sum=<s> : min=<lo> : max=<hi>
 *
 * Compare generic vs optimized:
 *   grep LAYER_DUMP generic.log > gen_dump.txt
 *   grep LAYER_DUMP opt.log     > opt_dump.txt
 *   diff gen_dump.txt opt_dump.txt
 */
#pragma once

#include <cstdio>
#include <cstdint>
#include <cfloat>

/* ScalarType values: Byte=0, Char=1, Short=2, Int=3, Long=4, Half=5, Float=6 */

#define _DUMP_N 16  /* number of leading values to print */

#define DUMP_TENSOR(name, tensor) do { \
    const auto _dn = (tensor).numel(); \
    const int _dt = (int)(tensor).scalar_type(); \
    printf("LAYER_DUMP : %s : %d : dtype=%d", #name, (int)_dn, _dt); \
    if (_dt == 6) { /* Float */ \
        const float* _dp = (tensor).const_data_ptr<float>(); \
        int _k = (int)_dn < _DUMP_N ? (int)_dn : _DUMP_N; \
        printf(" : first=["); \
        for (int _i = 0; _i < _k; _i++) printf("%s%.6f", _i?",":"", _dp[_i]); \
        printf("]"); \
        double _sum = 0; float _lo = _dp[0], _hi = _dp[0]; \
        for (int _i = 0; _i < (int)_dn; _i++) { \
            _sum += _dp[_i]; \
            if (_dp[_i] < _lo) _lo = _dp[_i]; \
            if (_dp[_i] > _hi) _hi = _dp[_i]; \
        } \
        printf(" : sum=%.4f : min=%.6f : max=%.6f", _sum, _lo, _hi); \
    } else if (_dt == 1) { /* Char / int8 */ \
        const int8_t* _dp = (tensor).const_data_ptr<int8_t>(); \
        int _k = (int)_dn < _DUMP_N ? (int)_dn : _DUMP_N; \
        printf(" : first=["); \
        for (int _i = 0; _i < _k; _i++) printf("%s%d", _i?",":"", (int)_dp[_i]); \
        printf("]"); \
        int64_t _sum = 0; int _lo = _dp[0], _hi = _dp[0]; \
        for (int _i = 0; _i < (int)_dn; _i++) { \
            _sum += _dp[_i]; \
            if (_dp[_i] < _lo) _lo = _dp[_i]; \
            if (_dp[_i] > _hi) _hi = _dp[_i]; \
        } \
        printf(" : sum=%lld : min=%d : max=%d", (long long)_sum, _lo, _hi); \
    } else if (_dt == 0) { /* Byte / uint8 */ \
        const uint8_t* _dp = (tensor).const_data_ptr<uint8_t>(); \
        int _k = (int)_dn < _DUMP_N ? (int)_dn : _DUMP_N; \
        printf(" : first=["); \
        for (int _i = 0; _i < _k; _i++) printf("%s%u", _i?",":"", (unsigned)_dp[_i]); \
        printf("]"); \
        int64_t _sum = 0; int _lo = _dp[0], _hi = _dp[0]; \
        for (int _i = 0; _i < (int)_dn; _i++) { \
            _sum += _dp[_i]; \
            if ((int)_dp[_i] < _lo) _lo = _dp[_i]; \
            if ((int)_dp[_i] > _hi) _hi = _dp[_i]; \
        } \
        printf(" : sum=%lld : min=%d : max=%d", (long long)_sum, _lo, _hi); \
    } else { \
        printf(" : (unsupported dtype)"); \
    } \
    printf("\n"); \
} while(0)
