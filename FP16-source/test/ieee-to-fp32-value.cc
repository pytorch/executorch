#include <gtest/gtest.h>

#include <cstdint>
#include <cmath>

#include <fp16.h>
#include <tables.h>


TEST(FP16_IEEE_TO_FP32_VALUE, normalized_powers_of_2) {
	const uint16_t min_po2_f16   = UINT16_C(0x0400);
	const uint16_t eighths_f16   = UINT16_C(0x3000);
	const uint16_t quarter_f16   = UINT16_C(0x3400);
	const uint16_t half_f16      = UINT16_C(0x3800);
	const uint16_t one_f16       = UINT16_C(0x3C00);
	const uint16_t two_f16       = UINT16_C(0x4000);
	const uint16_t four_f16      = UINT16_C(0x4400);
	const uint16_t eight_f16     = UINT16_C(0x4800);
	const uint16_t sixteen_f16   = UINT16_C(0x4C00);
	const uint16_t thirtytwo_f16 = UINT16_C(0x5000);
	const uint16_t sixtyfour_f16 = UINT16_C(0x5400);
	const uint16_t max_po2_f16   = UINT16_C(0x7800);

	const uint32_t min_po2_f32   = UINT32_C(0x38800000);
	const uint32_t eighths_f32   = UINT32_C(0x3E000000);
	const uint32_t quarter_f32   = UINT32_C(0x3E800000);
	const uint32_t half_f32      = UINT32_C(0x3F000000);
	const uint32_t one_f32       = UINT32_C(0x3F800000);
	const uint32_t two_f32       = UINT32_C(0x40000000);
	const uint32_t four_f32      = UINT32_C(0x40800000);
	const uint32_t eight_f32     = UINT32_C(0x41000000);
	const uint32_t sixteen_f32   = UINT32_C(0x41800000);
	const uint32_t thirtytwo_f32 = UINT32_C(0x42000000);
	const uint32_t sixtyfour_f32 = UINT32_C(0x42800000);
	const uint32_t max_po2_f32   = UINT32_C(0x47000000);

	const float min_po2_value = fp16_ieee_to_fp32_value(min_po2_f16);
	uint32_t min_po2_bits;
	memcpy(&min_po2_bits, &min_po2_value, sizeof(min_po2_bits));
	EXPECT_EQ(min_po2_f32, min_po2_bits) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F16 = 0x" << std::setw(4) << min_po2_f16 << ", " <<
		"F32(F16) = 0x" << std::setw(8) << min_po2_bits << ", " <<
		"F32 = 0x" << std::setw(8) << min_po2_f32;

	const float eighths_value = fp16_ieee_to_fp32_value(eighths_f16);
	uint32_t eighths_bits;
	memcpy(&eighths_bits, &eighths_value, sizeof(eighths_bits));
	EXPECT_EQ(eighths_f32, eighths_bits) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F16 = 0x" << std::setw(4) << eighths_f16 << ", " <<
		"F32(F16) = 0x" << std::setw(8) << eighths_bits << ", " <<
		"F32 = 0x" << std::setw(8) << eighths_f32;

	const float quarter_value = fp16_ieee_to_fp32_value(quarter_f16);
	uint32_t quarter_bits;
	memcpy(&quarter_bits, &quarter_value, sizeof(quarter_bits));
	EXPECT_EQ(quarter_f32, quarter_bits) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F16 = 0x" << std::setw(4) << quarter_f16 << ", " <<
		"F32(F16) = 0x" << std::setw(8) << quarter_bits << ", " <<
		"F32 = 0x" << std::setw(8) << quarter_f32;

	const float half_value = fp16_ieee_to_fp32_value(half_f16);
	uint32_t half_bits;
	memcpy(&half_bits, &half_value, sizeof(half_bits));
	EXPECT_EQ(half_f32, half_bits) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F16 = 0x" << std::setw(4) << half_f16 << ", " <<
		"F32(F16) = 0x" << std::setw(8) << half_bits << ", " <<
		"F32 = 0x" << std::setw(8) << half_f32;

	const float one_value = fp16_ieee_to_fp32_value(one_f16);
	uint32_t one_bits;
	memcpy(&one_bits, &one_value, sizeof(one_bits));
	EXPECT_EQ(one_f32, one_bits) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F16 = 0x" << std::setw(4) << one_f16 << ", " <<
		"F32(F16) = 0x" << std::setw(8) << one_bits << ", " <<
		"F32 = 0x" << std::setw(8) << one_f32;

	const float two_value = fp16_ieee_to_fp32_value(two_f16);
	uint32_t two_bits;
	memcpy(&two_bits, &two_value, sizeof(two_bits));
	EXPECT_EQ(two_f32, two_bits) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F16 = 0x" << std::setw(4) << two_f16 << ", " <<
		"F32(F16) = 0x" << std::setw(8) << two_bits << ", " <<
		"F32 = 0x" << std::setw(8) << two_f32;

	const float four_value = fp16_ieee_to_fp32_value(four_f16);
	uint32_t four_bits;
	memcpy(&four_bits, &four_value, sizeof(four_bits));
	EXPECT_EQ(four_f32, four_bits) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F16 = 0x" << std::setw(4) << four_f16 << ", " <<
		"F32(F16) = 0x" << std::setw(8) << four_bits << ", " <<
		"F32 = 0x" << std::setw(8) << four_f32;

	const float eight_value = fp16_ieee_to_fp32_value(eight_f16);
	uint32_t eight_bits;
	memcpy(&eight_bits, &eight_value, sizeof(eight_bits));
	EXPECT_EQ(eight_f32, eight_bits) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F16 = 0x" << std::setw(4) << eight_f16 << ", " <<
		"F32(F16) = 0x" << std::setw(8) << eight_bits << ", " <<
		"F32 = 0x" << std::setw(8) << eight_f32;

	const float sixteen_value = fp16_ieee_to_fp32_value(sixteen_f16);
	uint32_t sixteen_bits;
	memcpy(&sixteen_bits, &sixteen_value, sizeof(sixteen_bits));
	EXPECT_EQ(sixteen_f32, sixteen_bits) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F16 = 0x" << std::setw(4) << sixteen_f16 << ", " <<
		"F32(F16) = 0x" << std::setw(8) << sixteen_bits << ", " <<
		"F32 = 0x" << std::setw(8) << sixteen_f32;

	const float thirtytwo_value = fp16_ieee_to_fp32_value(thirtytwo_f16);
	uint32_t thirtytwo_bits;
	memcpy(&thirtytwo_bits, &thirtytwo_value, sizeof(thirtytwo_bits));
	EXPECT_EQ(thirtytwo_f32, thirtytwo_bits) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F16 = 0x" << std::setw(4) << thirtytwo_f16 << ", " <<
		"F32(F16) = 0x" << std::setw(8) << thirtytwo_bits << ", " <<
		"F32 = 0x" << std::setw(8) << thirtytwo_f32;

	const float sixtyfour_value = fp16_ieee_to_fp32_value(sixtyfour_f16);
	uint32_t sixtyfour_bits;
	memcpy(&sixtyfour_bits, &sixtyfour_value, sizeof(sixtyfour_bits));
	EXPECT_EQ(sixtyfour_f32, sixtyfour_bits) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F16 = 0x" << std::setw(4) << sixtyfour_f16 << ", " <<
		"F32(F16) = 0x" << std::setw(8) << sixtyfour_bits << ", " <<
		"F32 = 0x" << std::setw(8) << sixtyfour_f32;

	const float max_po2_value = fp16_ieee_to_fp32_value(max_po2_f16);
	uint32_t max_po2_bits;
	memcpy(&max_po2_bits, &max_po2_value, sizeof(max_po2_bits));
	EXPECT_EQ(max_po2_f32, max_po2_bits) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F16 = 0x" << std::setw(4) << max_po2_f16 << ", " <<
		"F32(F16) = 0x" << std::setw(8) << max_po2_bits << ", " <<
		"F32 = 0x" << std::setw(8) << max_po2_f32;
}

TEST(FP16_IEEE_TO_FP32_VALUE, denormalized_powers_of_2) {
	const uint16_t exp2_minus_15_f16 = UINT16_C(0x0200);
	const uint16_t exp2_minus_16_f16 = UINT16_C(0x0100);
	const uint16_t exp2_minus_17_f16 = UINT16_C(0x0080);
	const uint16_t exp2_minus_18_f16 = UINT16_C(0x0040);
	const uint16_t exp2_minus_19_f16 = UINT16_C(0x0020);
	const uint16_t exp2_minus_20_f16 = UINT16_C(0x0010);
	const uint16_t exp2_minus_21_f16 = UINT16_C(0x0008);
	const uint16_t exp2_minus_22_f16 = UINT16_C(0x0004);
	const uint16_t exp2_minus_23_f16 = UINT16_C(0x0002);
	const uint16_t exp2_minus_24_f16 = UINT16_C(0x0001);

	const uint32_t exp2_minus_15_f32 = UINT32_C(0x38000000);
	const uint32_t exp2_minus_16_f32 = UINT32_C(0x37800000);
	const uint32_t exp2_minus_17_f32 = UINT32_C(0x37000000);
	const uint32_t exp2_minus_18_f32 = UINT32_C(0x36800000);
	const uint32_t exp2_minus_19_f32 = UINT32_C(0x36000000);
	const uint32_t exp2_minus_20_f32 = UINT32_C(0x35800000);
	const uint32_t exp2_minus_21_f32 = UINT32_C(0x35000000);
	const uint32_t exp2_minus_22_f32 = UINT32_C(0x34800000);
	const uint32_t exp2_minus_23_f32 = UINT32_C(0x34000000);
	const uint32_t exp2_minus_24_f32 = UINT32_C(0x33800000);

	const float exp2_minus_15_value = fp16_ieee_to_fp32_value(exp2_minus_15_f16);
	uint32_t exp2_minus_15_bits;
	memcpy(&exp2_minus_15_bits, &exp2_minus_15_value, sizeof(exp2_minus_15_bits));
	EXPECT_EQ(exp2_minus_15_f32, exp2_minus_15_bits) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F16 = 0x" << std::setw(4) << exp2_minus_15_f16 << ", " <<
		"F32(F16) = 0x" << std::setw(8) << exp2_minus_15_bits << ", " <<
		"F32 = 0x" << std::setw(8) << exp2_minus_15_f32;

	const float exp2_minus_16_value = fp16_ieee_to_fp32_value(exp2_minus_16_f16);
	uint32_t exp2_minus_16_bits;
	memcpy(&exp2_minus_16_bits, &exp2_minus_16_value, sizeof(exp2_minus_16_bits));
	EXPECT_EQ(exp2_minus_16_f32, exp2_minus_16_bits) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F16 = 0x" << std::setw(4) << exp2_minus_16_f16 << ", " <<
		"F32(F16) = 0x" << std::setw(8) << exp2_minus_16_bits << ", " <<
		"F32 = 0x" << std::setw(8) << exp2_minus_16_f32;

	const float exp2_minus_17_value = fp16_ieee_to_fp32_value(exp2_minus_17_f16);
	uint32_t exp2_minus_17_bits;
	memcpy(&exp2_minus_17_bits, &exp2_minus_17_value, sizeof(exp2_minus_17_bits));
	EXPECT_EQ(exp2_minus_17_f32, exp2_minus_17_bits) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F16 = 0x" << std::setw(4) << exp2_minus_17_f16 << ", " <<
		"F32(F16) = 0x" << std::setw(8) << exp2_minus_17_bits << ", " <<
		"F32 = 0x" << std::setw(8) << exp2_minus_17_f32;

	const float exp2_minus_18_value = fp16_ieee_to_fp32_value(exp2_minus_18_f16);
	uint32_t exp2_minus_18_bits;
	memcpy(&exp2_minus_18_bits, &exp2_minus_18_value, sizeof(exp2_minus_18_bits));
	EXPECT_EQ(exp2_minus_18_f32, exp2_minus_18_bits) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F16 = 0x" << std::setw(4) << exp2_minus_18_f16 << ", " <<
		"F32(F16) = 0x" << std::setw(8) << exp2_minus_18_bits << ", " <<
		"F32 = 0x" << std::setw(8) << exp2_minus_18_f32;

	const float exp2_minus_19_value = fp16_ieee_to_fp32_value(exp2_minus_19_f16);
	uint32_t exp2_minus_19_bits;
	memcpy(&exp2_minus_19_bits, &exp2_minus_19_value, sizeof(exp2_minus_19_bits));
	EXPECT_EQ(exp2_minus_19_f32, exp2_minus_19_bits) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F16 = 0x" << std::setw(4) << exp2_minus_19_f16 << ", " <<
		"F32(F16) = 0x" << std::setw(8) << exp2_minus_19_bits << ", " <<
		"F32 = 0x" << std::setw(8) << exp2_minus_19_f32;

	const float exp2_minus_20_value = fp16_ieee_to_fp32_value(exp2_minus_20_f16);
	uint32_t exp2_minus_20_bits;
	memcpy(&exp2_minus_20_bits, &exp2_minus_20_value, sizeof(exp2_minus_20_bits));
	EXPECT_EQ(exp2_minus_20_f32, exp2_minus_20_bits) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F16 = 0x" << std::setw(4) << exp2_minus_20_f16 << ", " <<
		"F32(F16) = 0x" << std::setw(8) << exp2_minus_20_bits << ", " <<
		"F32 = 0x" << std::setw(8) << exp2_minus_20_f32;

	const float exp2_minus_21_value = fp16_ieee_to_fp32_value(exp2_minus_21_f16);
	uint32_t exp2_minus_21_bits;
	memcpy(&exp2_minus_21_bits, &exp2_minus_21_value, sizeof(exp2_minus_21_bits));
	EXPECT_EQ(exp2_minus_21_f32, exp2_minus_21_bits) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F16 = 0x" << std::setw(4) << exp2_minus_21_f16 << ", " <<
		"F32(F16) = 0x" << std::setw(8) << exp2_minus_21_bits << ", " <<
		"F32 = 0x" << std::setw(8) << exp2_minus_21_f32;

	const float exp2_minus_22_value = fp16_ieee_to_fp32_value(exp2_minus_22_f16);
	uint32_t exp2_minus_22_bits;
	memcpy(&exp2_minus_22_bits, &exp2_minus_22_value, sizeof(exp2_minus_22_bits));
	EXPECT_EQ(exp2_minus_22_f32, exp2_minus_22_bits) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F16 = 0x" << std::setw(4) << exp2_minus_22_f16 << ", " <<
		"F32(F16) = 0x" << std::setw(8) << exp2_minus_22_bits << ", " <<
		"F32 = 0x" << std::setw(8) << exp2_minus_22_f32;

	const float exp2_minus_23_value = fp16_ieee_to_fp32_value(exp2_minus_23_f16);
	uint32_t exp2_minus_23_bits;
	memcpy(&exp2_minus_23_bits, &exp2_minus_23_value, sizeof(exp2_minus_23_bits));
	EXPECT_EQ(exp2_minus_23_f32, exp2_minus_23_bits) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F16 = 0x" << std::setw(4) << exp2_minus_23_f16 << ", " <<
		"F32(F16) = 0x" << std::setw(8) << exp2_minus_23_bits << ", " <<
		"F32 = 0x" << std::setw(8) << exp2_minus_23_f32;

	const float exp2_minus_24_value = fp16_ieee_to_fp32_value(exp2_minus_24_f16);
	uint32_t exp2_minus_24_bits;
	memcpy(&exp2_minus_24_bits, &exp2_minus_24_value, sizeof(exp2_minus_24_bits));
	EXPECT_EQ(exp2_minus_24_f32, exp2_minus_24_bits) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F16 = 0x" << std::setw(4) << exp2_minus_24_f16 << ", " <<
		"F32(F16) = 0x" << std::setw(8) << exp2_minus_24_bits << ", " <<
		"F32 = 0x" << std::setw(8) << exp2_minus_24_f32;
}

TEST(FP16_IEEE_TO_FP32_VALUE, zero) {
	const uint16_t positive_zero_f16 = UINT16_C(0x0000);
	const uint16_t negative_zero_f16 = UINT16_C(0x8000);

	const uint32_t positive_zero_f32 = UINT32_C(0x00000000);
	const uint32_t negative_zero_f32 = UINT32_C(0x80000000);

	const float positive_zero_value = fp16_ieee_to_fp32_value(positive_zero_f16);
	uint32_t positive_zero_bits;
	memcpy(&positive_zero_bits, &positive_zero_value, sizeof(positive_zero_bits));
	EXPECT_EQ(positive_zero_f32, positive_zero_bits) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F16 = 0x" << std::setw(4) << positive_zero_f16 << ", " <<
		"F32(F16) = 0x" << std::setw(8) << positive_zero_bits << ", " <<
		"F32 = 0x" << std::setw(8) << positive_zero_f32;

	const float negative_zero_value = fp16_ieee_to_fp32_value(negative_zero_f16);
	uint32_t negative_zero_bits;
	memcpy(&negative_zero_bits, &negative_zero_value, sizeof(negative_zero_bits));
	EXPECT_EQ(negative_zero_f32, negative_zero_bits) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F16 = 0x" << std::setw(4) << negative_zero_f16 << ", " <<
		"F32(F16) = 0x" << std::setw(8) << negative_zero_bits << ", " <<
		"F32 = 0x" << std::setw(8) << negative_zero_f32;
}

TEST(FP16_IEEE_TO_FP32_VALUE, infinity) {
	const uint16_t positive_infinity_f16 = UINT16_C(0x7C00);
	const uint16_t negative_infinity_f16 = UINT16_C(0xFC00);

	const uint32_t positive_infinity_f32 = UINT32_C(0x7F800000);
	const uint32_t negative_infinity_f32 = UINT32_C(0xFF800000);

	const float positive_infinity_value = fp16_ieee_to_fp32_value(positive_infinity_f16);
	uint32_t positive_infinity_bits;
	memcpy(&positive_infinity_bits, &positive_infinity_value, sizeof(positive_infinity_bits));
	EXPECT_EQ(positive_infinity_f32, positive_infinity_bits) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F16 = 0x" << std::setw(4) << positive_infinity_f16 << ", " <<
		"F32(F16) = 0x" << std::setw(8) << positive_infinity_bits << ", " <<
		"F32 = 0x" << std::setw(8) << positive_infinity_f32;

	const float negative_infinity_value = fp16_ieee_to_fp32_value(negative_infinity_f16);
	uint32_t negative_infinity_bits;
	memcpy(&negative_infinity_bits, &negative_infinity_value, sizeof(negative_infinity_bits));
	EXPECT_EQ(negative_infinity_f32, negative_infinity_bits) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F16 = 0x" << std::setw(4) << negative_infinity_f16 << ", " <<
		"F32(F16) = 0x" << std::setw(8) << negative_infinity_bits << ", " <<
		"F32 = 0x" << std::setw(8) << negative_infinity_f32;
}

TEST(FP16_IEEE_TO_FP32_VALUE, positive_nan) {
	using std::signbit;
	using std::isnan;
	for (uint16_t m = UINT16_C(1); m < UINT16_C(0x0400); m++) {
		const uint16_t nan_f16 = UINT16_C(0x7C00) | m;
		const float nan_f32 = fp16_ieee_to_fp32_value(nan_f16);
		uint32_t nan_bits;
		memcpy(&nan_bits, &nan_f32, sizeof(nan_bits));

		/* Check if NaN */
		EXPECT_TRUE(isnan(nan_f32)) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F16 = 0x" << std::setw(4) << nan_f16 << ", " <<
			"F32(F16) = 0x" << std::setw(8) << nan_bits;

		/* Check sign */
		EXPECT_EQ(signbit(nan_f32), 0) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F16 = 0x" << std::setw(4) << nan_f16 << ", " <<
			"F32(F16) = 0x" << std::setw(8) << nan_bits;
	}
}

TEST(FP16_IEEE_TO_FP32_VALUE, negative_nan) {
	using std::signbit;
	using std::isnan;
	for (uint16_t m = UINT16_C(1); m < UINT16_C(0x0400); m++) {
		const uint16_t nan_f16 = UINT16_C(0xFC00) | m;
		const float nan_f32 = fp16_ieee_to_fp32_value(nan_f16);
		uint32_t nan_bits;
		memcpy(&nan_bits, &nan_f32, sizeof(nan_bits));

		/* Check if NaN */
		EXPECT_TRUE(isnan(nan_f32)) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F16 = 0x" << std::setw(4) << nan_f16 << ", " <<
			"F32(F16) = 0x" << std::setw(8) << nan_bits;

		/* Check sign */
		EXPECT_EQ(signbit(nan_f32), 1) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F16 = 0x" << std::setw(4) << nan_f16 << ", " <<
			"F32(F16) = 0x" << std::setw(8) << nan_bits;
	}
}

TEST(FP16_IEEE_TO_FP32_VALUE, positive_normalized_values) {
	const uint32_t exponentBias = 15;
	for (int32_t e = -14; e <= 15; e++) {
		for (uint16_t h = 0; h < 0x0400; h++) {
			const uint16_t fp16 = h + ((uint16_t) (e + exponentBias) << 10);
			const uint32_t fp32 = fp16::normalizedValues[h] + ((uint32_t) e << 23);
			const float value = fp16_ieee_to_fp32_value(fp16);
			uint32_t bits;
			memcpy(&bits, &value, sizeof(bits));
			EXPECT_EQ(fp32, bits) <<
				std::hex << std::uppercase << std::setfill('0') <<
				"F16 = 0x" << std::setw(4) << fp16 << ", " <<
				"F32(F16) = 0x" << std::setw(8) << bits << ", " <<
				"F32 = 0x" << std::setw(8) << fp32;
		}
	}
}

TEST(FP16_IEEE_TO_FP32_VALUE, negative_normalized_values) {
	const uint32_t exponentBias = 15;
	for (int32_t e = -14; e <= 15; e++) {
		for (uint16_t h = 0; h < 0x0400; h++) {
			const uint16_t fp16 = (h + ((uint16_t) (e + exponentBias) << 10)) ^ UINT16_C(0x8000);
			const uint32_t fp32 = (fp16::normalizedValues[h] + ((uint32_t) e << 23)) ^ UINT32_C(0x80000000);
			const float value = fp16_ieee_to_fp32_value(fp16);
			uint32_t bits;
			memcpy(&bits, &value, sizeof(bits));
			EXPECT_EQ(fp32, bits) <<
				std::hex << std::uppercase << std::setfill('0') <<
				"F16 = 0x" << std::setw(4) << fp16 << ", " <<
				"F32(F16) = 0x" << std::setw(8) << bits << ", " <<
				"F32 = 0x" << std::setw(8) << fp32;
		}
	}
}

TEST(FP16_IEEE_TO_FP32_VALUE, positive_denormalized_values) {
	for (uint16_t h = 0; h < 0x0400; h++) {
		const float value = fp16_ieee_to_fp32_value(h);
		uint32_t bits;
		memcpy(&bits, &value, sizeof(bits));
		EXPECT_EQ(fp16::denormalizedValues[h], bits) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F16 = 0x" << std::setw(4) << h << ", " <<
			"F32(F16) = 0x" << std::setw(8) << bits << ", " <<
			"F32 = 0x" << std::setw(8) << fp16::denormalizedValues[h];
	}
}

TEST(FP16_IEEE_TO_FP32_VALUE, negative_denormalized_values) {
	for (uint16_t h = 0; h < 0x0400; h++) {
		const uint16_t fp16 = h ^ UINT16_C(0x8000);
		const uint32_t fp32 = fp16::denormalizedValues[h] ^ UINT32_C(0x80000000);
		const float value = fp16_ieee_to_fp32_value(fp16);
		uint32_t bits;
		memcpy(&bits, &value, sizeof(bits));
		EXPECT_EQ(fp32, bits) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F16 = 0x" << std::setw(4) << fp16 << ", " <<
			"F32(F16) = 0x" << std::setw(8) << bits << ", " <<
			"F32 = 0x" << std::setw(8) << fp32;
	}
}
