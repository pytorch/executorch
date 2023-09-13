#include <gtest/gtest.h>

#include <cstdint>

#include <fp16.h>
#include <fp16/psimd.h>


TEST(FP16_ALT_TO_FP32x2_PSIMD, positive_normalized_values) {
	const uint32_t exponentBias = 15;
	for (int32_t e = -14; e <= 16; e++) {
		for (uint16_t h = 0; h < 0x0400; h += 8) {
			const psimd_u16 fp16 = {
				(uint16_t) (h + ((e + exponentBias) << 10) + 0),
				(uint16_t) (h + ((e + exponentBias) << 10) + 1),
				(uint16_t) (h + ((e + exponentBias) << 10) + 2),
				(uint16_t) (h + ((e + exponentBias) << 10) + 3),
				(uint16_t) (h + ((e + exponentBias) << 10) + 4),
				(uint16_t) (h + ((e + exponentBias) << 10) + 5),
				(uint16_t) (h + ((e + exponentBias) << 10) + 6),
				(uint16_t) (h + ((e + exponentBias) << 10) + 7)
			};
			const psimd_u32x2 fp32 =
				psimd_cast_f32x2_u32x2(fp16_alt_to_fp32x2_psimd(fp16));

			EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[0]), fp32.lo[0]) <<
				std::hex << std::uppercase << std::setfill('0') <<
				"F16 = 0x" << std::setw(4) << fp16[0] << ", " <<
				"F32(F16) = 0x" << std::setw(8) << fp32.lo[0] << ", " <<
				"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[0]);
			EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[1]), fp32.lo[1]) <<
				std::hex << std::uppercase << std::setfill('0') <<
				"F16 = 0x" << std::setw(4) << fp16[1] << ", " <<
				"F32(F16) = 0x" << std::setw(8) << fp32.lo[1] << ", " <<
				"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[1]);
			EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[2]), fp32.lo[2]) <<
				std::hex << std::uppercase << std::setfill('0') <<
				"F16 = 0x" << std::setw(4) << fp16[2] << ", " <<
				"F32(F16) = 0x" << std::setw(8) << fp32.lo[2] << ", " <<
				"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[2]);
			EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[3]), fp32.lo[3]) <<
				std::hex << std::uppercase << std::setfill('0') <<
				"F16 = 0x" << std::setw(4) << fp16[3] << ", " <<
				"F32(F16) = 0x" << std::setw(8) << fp32.lo[3] << ", " <<
				"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[3]);
			EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[4]), fp32.hi[0]) <<
				std::hex << std::uppercase << std::setfill('0') <<
				"F16 = 0x" << std::setw(4) << fp16[4] << ", " <<
				"F32(F16) = 0x" << std::setw(8) << fp32.hi[0] << ", " <<
				"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[4]);
			EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[5]), fp32.hi[1]) <<
				std::hex << std::uppercase << std::setfill('0') <<
				"F16 = 0x" << std::setw(4) << fp16[5] << ", " <<
				"F32(F16) = 0x" << std::setw(8) << fp32.hi[1] << ", " <<
				"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[5]);
			EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[6]), fp32.hi[2]) <<
				std::hex << std::uppercase << std::setfill('0') <<
				"F16 = 0x" << std::setw(4) << fp16[6] << ", " <<
				"F32(F16) = 0x" << std::setw(8) << fp32.hi[2] << ", " <<
				"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[6]);
			EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[7]), fp32.hi[3]) <<
				std::hex << std::uppercase << std::setfill('0') <<
				"F16 = 0x" << std::setw(4) << fp16[7] << ", " <<
				"F32(F16) = 0x" << std::setw(8) << fp32.hi[3] << ", " <<
				"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[7]);
		}
	}
}

TEST(FP16_ALT_TO_FP32x2_PSIMD, negative_normalized_values) {
	const uint32_t exponentBias = 15;
	for (int32_t e = -14; e <= 16; e++) {
		for (uint16_t h = 0; h < 0x0400; h += 8) {
			const psimd_u16 fp16 = {
				(uint16_t) (h + ((e + exponentBias) << 10) + 0x8000),
				(uint16_t) (h + ((e + exponentBias) << 10) + 0x8001),
				(uint16_t) (h + ((e + exponentBias) << 10) + 0x8002),
				(uint16_t) (h + ((e + exponentBias) << 10) + 0x8003),
				(uint16_t) (h + ((e + exponentBias) << 10) + 0x8004),
				(uint16_t) (h + ((e + exponentBias) << 10) + 0x8005),
				(uint16_t) (h + ((e + exponentBias) << 10) + 0x8006),
				(uint16_t) (h + ((e + exponentBias) << 10) + 0x8007)
			};
			const psimd_u32x2 fp32 =
				psimd_cast_f32x2_u32x2(fp16_alt_to_fp32x2_psimd(fp16));

			EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[0]), fp32.lo[0]) <<
				std::hex << std::uppercase << std::setfill('0') <<
				"F16 = 0x" << std::setw(4) << fp16[0] << ", " <<
				"F32(F16) = 0x" << std::setw(8) << fp32.lo[0] << ", " <<
				"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[0]);
			EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[1]), fp32.lo[1]) <<
				std::hex << std::uppercase << std::setfill('0') <<
				"F16 = 0x" << std::setw(4) << fp16[1] << ", " <<
				"F32(F16) = 0x" << std::setw(8) << fp32.lo[1] << ", " <<
				"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[1]);
			EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[2]), fp32.lo[2]) <<
				std::hex << std::uppercase << std::setfill('0') <<
				"F16 = 0x" << std::setw(4) << fp16[2] << ", " <<
				"F32(F16) = 0x" << std::setw(8) << fp32.lo[2] << ", " <<
				"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[2]);
			EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[3]), fp32.lo[3]) <<
				std::hex << std::uppercase << std::setfill('0') <<
				"F16 = 0x" << std::setw(4) << fp16[3] << ", " <<
				"F32(F16) = 0x" << std::setw(8) << fp32.lo[3] << ", " <<
				"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[3]);
			EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[4]), fp32.hi[0]) <<
				std::hex << std::uppercase << std::setfill('0') <<
				"F16 = 0x" << std::setw(4) << fp16[4] << ", " <<
				"F32(F16) = 0x" << std::setw(8) << fp32.hi[0] << ", " <<
				"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[4]);
			EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[5]), fp32.hi[1]) <<
				std::hex << std::uppercase << std::setfill('0') <<
				"F16 = 0x" << std::setw(4) << fp16[5] << ", " <<
				"F32(F16) = 0x" << std::setw(8) << fp32.hi[1] << ", " <<
				"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[5]);
			EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[6]), fp32.hi[2]) <<
				std::hex << std::uppercase << std::setfill('0') <<
				"F16 = 0x" << std::setw(4) << fp16[6] << ", " <<
				"F32(F16) = 0x" << std::setw(8) << fp32.hi[2] << ", " <<
				"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[6]);
			EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[7]), fp32.hi[3]) <<
				std::hex << std::uppercase << std::setfill('0') <<
				"F16 = 0x" << std::setw(4) << fp16[7] << ", " <<
				"F32(F16) = 0x" << std::setw(8) << fp32.hi[3] << ", " <<
				"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[7]);
		}
	}
}

TEST(FP16_ALT_TO_FP32x2_PSIMD, positive_denormalized_values) {
	for (uint16_t h = 0; h < 0x0400; h += 8) {
		const psimd_u16 fp16 = {
			(uint16_t) (h + 0),
			(uint16_t) (h + 1),
			(uint16_t) (h + 2),
			(uint16_t) (h + 3),
			(uint16_t) (h + 4),
			(uint16_t) (h + 5),
			(uint16_t) (h + 6),
			(uint16_t) (h + 7)
		};
		const psimd_u32x2 fp32 =
			psimd_cast_f32x2_u32x2(fp16_alt_to_fp32x2_psimd(fp16));

		EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[0]), fp32.lo[0]) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F16 = 0x" << std::setw(4) << fp16[0] << ", " <<
			"F32(F16) = 0x" << std::setw(8) << fp32.lo[0] << ", " <<
			"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[0]);
		EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[1]), fp32.lo[1]) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F16 = 0x" << std::setw(4) << fp16[1] << ", " <<
			"F32(F16) = 0x" << std::setw(8) << fp32.lo[1] << ", " <<
			"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[1]);
		EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[2]), fp32.lo[2]) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F16 = 0x" << std::setw(4) << fp16[2] << ", " <<
			"F32(F16) = 0x" << std::setw(8) << fp32.lo[2] << ", " <<
			"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[2]);
		EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[3]), fp32.lo[3]) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F16 = 0x" << std::setw(4) << fp16[3] << ", " <<
			"F32(F16) = 0x" << std::setw(8) << fp32.lo[3] << ", " <<
			"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[3]);
		EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[4]), fp32.hi[0]) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F16 = 0x" << std::setw(4) << fp16[4] << ", " <<
			"F32(F16) = 0x" << std::setw(8) << fp32.hi[0] << ", " <<
			"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[4]);
		EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[5]), fp32.hi[1]) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F16 = 0x" << std::setw(4) << fp16[5] << ", " <<
			"F32(F16) = 0x" << std::setw(8) << fp32.hi[1] << ", " <<
			"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[5]);
		EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[6]), fp32.hi[2]) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F16 = 0x" << std::setw(4) << fp16[6] << ", " <<
			"F32(F16) = 0x" << std::setw(8) << fp32.hi[2] << ", " <<
			"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[6]);
		EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[7]), fp32.hi[3]) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F16 = 0x" << std::setw(4) << fp16[7] << ", " <<
			"F32(F16) = 0x" << std::setw(8) << fp32.hi[3] << ", " <<
			"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[7]);
	}
}

TEST(FP16_ALT_TO_FP32x2_PSIMD, negative_denormalized_values) {
	for (uint16_t h = 0; h < 0x0400; h += 8) {
		const psimd_u16 fp16 = {
			(uint16_t) (h + 0x8000),
			(uint16_t) (h + 0x8001),
			(uint16_t) (h + 0x8002),
			(uint16_t) (h + 0x8003),
			(uint16_t) (h + 0x8004),
			(uint16_t) (h + 0x8005),
			(uint16_t) (h + 0x8006),
			(uint16_t) (h + 0x8007)
		};
		const psimd_u32x2 fp32 =
			psimd_cast_f32x2_u32x2(fp16_alt_to_fp32x2_psimd(fp16));

		EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[0]), fp32.lo[0]) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F16 = 0x" << std::setw(4) << fp16[0] << ", " <<
			"F32(F16) = 0x" << std::setw(8) << fp32.lo[0] << ", " <<
			"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[0]);
		EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[1]), fp32.lo[1]) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F16 = 0x" << std::setw(4) << fp16[1] << ", " <<
			"F32(F16) = 0x" << std::setw(8) << fp32.lo[1] << ", " <<
			"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[1]);
		EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[2]), fp32.lo[2]) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F16 = 0x" << std::setw(4) << fp16[2] << ", " <<
			"F32(F16) = 0x" << std::setw(8) << fp32.lo[2] << ", " <<
			"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[2]);
		EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[3]), fp32.lo[3]) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F16 = 0x" << std::setw(4) << fp16[3] << ", " <<
			"F32(F16) = 0x" << std::setw(8) << fp32.lo[3] << ", " <<
			"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[3]);
		EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[4]), fp32.hi[0]) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F16 = 0x" << std::setw(4) << fp16[4] << ", " <<
			"F32(F16) = 0x" << std::setw(8) << fp32.hi[0] << ", " <<
			"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[4]);
		EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[5]), fp32.hi[1]) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F16 = 0x" << std::setw(4) << fp16[5] << ", " <<
			"F32(F16) = 0x" << std::setw(8) << fp32.hi[1] << ", " <<
			"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[5]);
		EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[6]), fp32.hi[2]) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F16 = 0x" << std::setw(4) << fp16[6] << ", " <<
			"F32(F16) = 0x" << std::setw(8) << fp32.hi[2] << ", " <<
			"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[6]);
		EXPECT_EQ(fp16_alt_to_fp32_bits(fp16[7]), fp32.hi[3]) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F16 = 0x" << std::setw(4) << fp16[7] << ", " <<
			"F32(F16) = 0x" << std::setw(8) << fp32.hi[3] << ", " <<
			"F32 = 0x" << std::setw(8) << fp16_alt_to_fp32_bits(fp16[7]);
	}
}
