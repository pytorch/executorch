#include <gtest/gtest.h>

#include <cstdint>

#include <fp16.h>


extern "C" void fp16_alt_xmm_to_fp32_ymm_peachpy__avx2(const uint16_t* fp16, uint32_t* fp32);
const size_t vector_elements = 8;


TEST(FP16_ALT_XMM_TO_FP32_YMM, positive_normalized_values) {
	const uint32_t exponentBias = 15;
	for (int32_t e = -14; e <= 16; e++) {
		for (uint16_t h = 0; h < 0x0400; h += vector_elements) {
			uint16_t fp16[vector_elements];
			for (size_t i = 0; i < vector_elements; i++) {
				fp16[i] = h + ((e + exponentBias) << 10) + i;
			}
			uint32_t fp32[vector_elements];
			fp16_alt_xmm_to_fp32_ymm_peachpy__avx2(fp16, fp32);

			for (size_t i = 0; i < vector_elements; i++) {
				EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[i]), fp32[i]) <<
					std::hex << std::uppercase << std::setfill('0') <<
					"F16 = 0x" << std::setw(4) << fp16[i] << ", " <<
					"F32(F16) = 0x" << std::setw(8) << fp32[i] << ", " <<
					"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[i]) <<
					", lane " << i << "/" << vector_elements;
			}
		}
	}
}

TEST(FP16_ALT_XMM_TO_FP32_YMM, negative_normalized_values) {
	const uint32_t exponentBias = 15;
	for (int32_t e = -14; e <= 16; e++) {
		for (uint16_t h = 0; h < 0x0400; h += vector_elements) {
			uint16_t fp16[vector_elements];
			for (size_t i = 0; i < vector_elements; i++) {
				fp16[i] = 0x8000 + h + ((e + exponentBias) << 10) + i;
			}
			uint32_t fp32[vector_elements];
			fp16_alt_xmm_to_fp32_ymm_peachpy__avx2(fp16, fp32);

			for (size_t i = 0; i < vector_elements; i++) {
				EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[i]), fp32[i]) <<
					std::hex << std::uppercase << std::setfill('0') <<
					"F16 = 0x" << std::setw(4) << fp16[i] << ", " <<
					"F32(F16) = 0x" << std::setw(8) << fp32[i] << ", " <<
					"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[i]) <<
					", lane " << i << "/" << vector_elements;
			}
		}
	}
}

TEST(FP16_ALT_XMM_TO_FP32_YMM, positive_denormalized_values) {
	for (uint16_t h = 0; h < 0x0400; h += vector_elements) {
		uint16_t fp16[vector_elements];
		for (size_t i = 0; i < vector_elements; i++) {
			fp16[i] = h + i;
		}
		uint32_t fp32[vector_elements];
		fp16_alt_xmm_to_fp32_ymm_peachpy__avx2(fp16, fp32);

		for (size_t i = 0; i < vector_elements; i++) {
			EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[i]), fp32[i]) <<
				std::hex << std::uppercase << std::setfill('0') <<
				"F16 = 0x" << std::setw(4) << fp16[i] << ", " <<
				"F32(F16) = 0x" << std::setw(8) << fp32[i] << ", " <<
				"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[i]) <<
				", lane " << i << "/" << vector_elements;
		}
	}
}

TEST(FP16_ALT_XMM_TO_FP32_YMM, negative_denormalized_values) {
	for (uint16_t h = 0; h < 0x0400; h += vector_elements) {
		uint16_t fp16[vector_elements];
		for (size_t i = 0; i < vector_elements; i++) {
			fp16[i] = 0x8000 + h + i;
		}
		uint32_t fp32[vector_elements];
		fp16_alt_xmm_to_fp32_ymm_peachpy__avx2(fp16, fp32);

		for (size_t i = 0; i < vector_elements; i++) {
			EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[i]), fp32[i]) <<
				std::hex << std::uppercase << std::setfill('0') <<
				"F16 = 0x" << std::setw(4) << fp16[i] << ", " <<
				"F32(F16) = 0x" << std::setw(8) << fp32[i] << ", " <<
				"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[i]) <<
				", lane " << i << "/" << vector_elements;
		}
	}
}
