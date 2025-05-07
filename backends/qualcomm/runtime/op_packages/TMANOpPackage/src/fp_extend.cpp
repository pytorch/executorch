// for x86_64 simulation
#ifndef __hexagon__

#include <limits.h>
#include <stdint.h>

typedef uint16_t src_t;
typedef uint16_t src_rep_t;
#define SRC_REP_C UINT16_C
static const int srcSigBits = 10;
#define src_rep_t_clz __builtin_clz

typedef float dst_t;
typedef uint32_t dst_rep_t;
#define DST_REP_C UINT32_C
static const int dstSigBits = 23;

// End of specialization parameters.  Two helper routines for conversion to and
// from the representation of floating-point data as integer values follow.

static __inline src_rep_t srcToRep(src_t x) {
    const union { src_t f; src_rep_t i; } rep = {.f = x};
    return rep.i;
}

static __inline dst_t dstFromRep(dst_rep_t x) {
    const union { dst_t f; dst_rep_t i; } rep = {.i = x};
    return rep.f;
}
// End helper routines.  Conversion implementation follows.

static __inline dst_t __extendXfYf2__(src_t a) {
    // Various constants whose values follow from the type parameters.
    // Any reasonable optimizer will fold and propagate all of these.
    const int srcBits = sizeof(src_t)*CHAR_BIT;
    const int srcExpBits = srcBits - srcSigBits - 1;
    const int srcInfExp = (1 << srcExpBits) - 1;
    const int srcExpBias = srcInfExp >> 1;

    const src_rep_t srcMinNormal = SRC_REP_C(1) << srcSigBits;
    const src_rep_t srcInfinity = (src_rep_t)srcInfExp << srcSigBits;
    const src_rep_t srcSignMask = SRC_REP_C(1) << (srcSigBits + srcExpBits);
    const src_rep_t srcAbsMask = srcSignMask - 1;
    const src_rep_t srcQNaN = SRC_REP_C(1) << (srcSigBits - 1);
    const src_rep_t srcNaNCode = srcQNaN - 1;

    const int dstBits = sizeof(dst_t)*CHAR_BIT;
    const int dstExpBits = dstBits - dstSigBits - 1;
    const int dstInfExp = (1 << dstExpBits) - 1;
    const int dstExpBias = dstInfExp >> 1;

    const dst_rep_t dstMinNormal = DST_REP_C(1) << dstSigBits;

    // Break a into a sign and representation of the absolute value
    const src_rep_t aRep = srcToRep(a);
    const src_rep_t aAbs = aRep & srcAbsMask;
    const src_rep_t sign = aRep & srcSignMask;
    dst_rep_t absResult;

    // If sizeof(src_rep_t) < sizeof(int), the subtraction result is promoted
    // to (signed) int.  To avoid that, explicitly cast to src_rep_t.
    if ((src_rep_t)(aAbs - srcMinNormal) < srcInfinity - srcMinNormal) {
        // a is a normal number.
        // Extend to the destination type by shifting the significand and
        // exponent into the proper position and rebiasing the exponent.
        absResult = (dst_rep_t)aAbs << (dstSigBits - srcSigBits);
        absResult += (dst_rep_t)(dstExpBias - srcExpBias) << dstSigBits;
    }

    else if (aAbs >= srcInfinity) {
        // a is NaN or infinity.
        // Conjure the result by beginning with infinity, then setting the qNaN
        // bit (if needed) and right-aligning the rest of the trailing NaN
        // payload field.
        absResult = (dst_rep_t)dstInfExp << dstSigBits;
        absResult |= (dst_rep_t)(aAbs & srcQNaN) << (dstSigBits - srcSigBits);
        absResult |= (dst_rep_t)(aAbs & srcNaNCode) << (dstSigBits - srcSigBits);
    }

    else if (aAbs) {
        // a is denormal.
        // renormalize the significand and clear the leading bit, then insert
        // the correct adjusted exponent in the destination type.
        const int scale = src_rep_t_clz(aAbs) - src_rep_t_clz(srcMinNormal);
        absResult = (dst_rep_t)aAbs << (dstSigBits - srcSigBits + scale);
        absResult ^= dstMinNormal;
        const int resultExponent = dstExpBias - srcExpBias - scale + 1;
        absResult |= (dst_rep_t)resultExponent << dstSigBits;
    }

    else {
        // a is zero.
        absResult = 0;
    }

    // Apply the signbit to (dst_t)abs(a).
    const dst_rep_t result = absResult | (dst_rep_t)sign << (dstBits - srcBits);
    return dstFromRep(result);
}
// Use a forwarding definition and noinline to implement a poor man's alias,
// as there isn't a good cross-platform way of defining one.
__attribute__((noinline)) float __extendhfsf2(uint16_t a) {
    return __extendXfYf2__(a);
}

extern "C" float __gnu_h2f_ieee(uint16_t a) {
    return __extendhfsf2(a);
}

#endif // !__hexagon__
