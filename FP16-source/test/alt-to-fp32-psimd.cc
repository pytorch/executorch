#include <gtest/gtest.h>

#include <cstdint>

#include <fp16.h>
#include <fp16/psimd.h>


TEST(FP16_ALT_TO_FP32_PSIMD, positive_normalized_values) {
	const uint32_t exponentBias = 15;
	for (int32_t e = -14; e <= 16; e++) {
		for (uint16_t h = 0; h < 0x0400; h += 4) {
			const psimd_u16 fp16 = {
				(uint16_t) (h + ((e + exponentBias) << 10) + 0),
				(uint16_t) (h + ((e + exponentBias) << 10) + 1),
				(uint16_t) (h + ((e + exponentBias) << 10) + 2),
				(uint16_t) (h + ((e + exponentBias) << 10) + 3),
			};
			const psimd_u32 fp32 = (psimd_u32) fp16_alt_to_fp32_psimd(fp16);

			EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[0]), fp32[0]) <<
				std::hex << std::uppercase << std::setfill('0') <<
				"F16 = 0x" << std::setw(4) << fp16[0] << ", " <<
				"F32(F16) = 0x" << std::setw(8) << fp32[0] << ", " <<
				"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[0]);
			EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[1]), fp32[1]) <<
				std::hex << std::uppercase << std::setfill('0') <<
				"F16 = 0x" << std::setw(4) << fp16[1] << ", " <<
				"F32(F16) = 0x" << std::setw(8) << fp32[1] << ", " <<
				"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[1]);
			EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[2]), fp32[2]) <<
				std::hex << std::uppercase << std::setfill('0') <<
				"F16 = 0x" << std::setw(4) << fp16[2] << ", " <<
				"F32(F16) = 0x" << std::setw(8) << fp32[2] << ", " <<
				"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[2]);
			EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[3]), fp32[3]) <<
				std::hex << std::uppercase << std::setfill('0') <<
				"F16 = 0x" << std::setw(4) << fp16[3] << ", " <<
				"F32(F16) = 0x" << std::setw(8) << fp32[3] << ", " <<
				"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[3]);
		}
	}
}

TEST(FP16_ALT_TO_FP32_PSIMD, negative_normalized_values) {
	const uint32_t exponentBias = 15;
	for (int32_t e = -14; e <= 16; e++) {
		for (uint16_t h = 0; h < 0x0400; h += 4) {
			const psimd_u16 fp16 = {
				(uint16_t) (h + ((e + exponentBias) << 10) + 0x8000),
				(uint16_t) (h + ((e + exponentBias) << 10) + 0x8001),
				(uint16_t) (h + ((e + exponentBias) << 10) + 0x8002),
				(uint16_t) (h + ((e + exponentBias) << 10) + 0x8003),
			};
			const psimd_u32 fp32 = (psimd_u32) fp16_alt_to_fp32_psimd(fp16);

			EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[0]), fp32[0]) <<
				std::hex << std::uppercase << std::setfill('0') <<
				"F16 = 0x" << std::setw(4) << fp16[0] << ", " <<
				"F32(F16) = 0x" << std::setw(8) << fp32[0] << ", " <<
				"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[0]);
			EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[1]), fp32[1]) <<
				std::hex << std::uppercase << std::setfill('0') <<
				"F16 = 0x" << std::setw(4) << fp16[1] << ", " <<
				"F32(F16) = 0x" << std::setw(8) << fp32[1] << ", " <<
				"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[1]);
			EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[2]), fp32[2]) <<
				std::hex << std::uppercase << std::setfill('0') <<
				"F16 = 0x" << std::setw(4) << fp16[2] << ", " <<
				"F32(F16) = 0x" << std::setw(8) << fp32[2] << ", " <<
				"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[2]);
			EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[3]), fp32[3]) <<
				std::hex << std::uppercase << std::setfill('0') <<
				"F16 = 0x" << std::setw(4) << fp16[3] << ", " <<
				"F32(F16) = 0x" << std::setw(8) << fp32[3] << ", " <<
				"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[3]);
		}
	}
}

TEST(FP16_ALT_TO_FP32_PSIMD, positive_denormalized_values) {
	for (uint16_t h = 0; h < 0x0400; h += 4) {
		const psimd_u16 fp16 = {
			(uint16_t) (h + 0),
			(uint16_t) (h + 1),
			(uint16_t) (h + 2),
			(uint16_t) (h + 3),
		};
		const psimd_u32 fp32 = (psimd_u32) fp16_alt_to_fp32_psimd(fp16);

		EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[0]), fp32[0]) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F16 = 0x" << std::setw(4) << fp16[0] << ", " <<
			"F32(F16) = 0x" << std::setw(8) << fp32[0] << ", " <<
			"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[0]);
		EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[1]), fp32[1]) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F16 = 0x" << std::setw(4) << fp16[1] << ", " <<
			"F32(F16) = 0x" << std::setw(8) << fp32[1] << ", " <<
			"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[1]);
		EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[2]), fp32[2]) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F16 = 0x" << std::setw(4) << fp16[2] << ", " <<
			"F32(F16) = 0x" << std::setw(8) << fp32[2] << ", " <<
			"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[2]);
		EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[3]), fp32[3]) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F16 = 0x" << std::setw(4) << fp16[3] << ", " <<
			"F32(F16) = 0x" << std::setw(8) << fp32[3] << ", " <<
			"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[3]);
	}
}

TEST(FP16_ALT_TO_FP32_PSIMD, negative_denormalized_values) {
	for (uint16_t h = 0; h < 0x0400; h += 4) {
		const psimd_u16 fp16 = {
			(uint16_t) (h + 0x8000),
			(uint16_t) (h + 0x8001),
			(uint16_t) (h + 0x8002),
			(uint16_t) (h + 0x8003),
		};
		const psimd_u32 fp32 = (psimd_u32) fp16_alt_to_fp32_psimd(fp16);

		EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[0]), fp32[0]) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F16 = 0x" << std::setw(4) << fp16[0] << ", " <<
			"F32(F16) = 0x" << std::setw(8) << fp32[0] << ", " <<
			"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[0]);
		EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[1]), fp32[1]) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F16 = 0x" << std::setw(4) << fp16[1] << ", " <<
			"F32(F16) = 0x" << std::setw(8) << fp32[1] << ", " <<
			"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[1]);
		EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[2]), fp32[2]) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F16 = 0x" << std::setw(4) << fp16[2] << ", " <<
			"F32(F16) = 0x" << std::setw(8) << fp32[2] << ", " <<
			"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[2]);
		EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[3]), fp32[3]) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F16 = 0x" << std::setw(4) << fp16[3] << ", " <<
			"F32(F16) = 0x" << std::setw(8) << fp32[3] << ", " <<
			"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[3]);
	}
}
