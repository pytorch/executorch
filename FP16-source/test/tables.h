#include <cstdint>


namespace fp16 {
	/* FP32 conversion results for FP16 numbers in range [1.0h, 2.0h) */
	extern const uint32_t normalizedValues[1024];
	/* FP32 conversion results for FP16 numbers in range [0.0h, HALF_MIN) */
	extern const uint32_t denormalizedValues[1024];
	/* FP32 numbers such that FP16(fp32) < as_half(as_uint16(1.0h) | index) for fp32 < normalizedRanges[index] */
	extern const uint32_t normalizedRanges[1024];
	/* FP32 numbers such that FP16(fp32) < as_half(index) for fp32 < denormalizedRanges[index] */
	extern const uint32_t denormalizedRanges[1024];
}
