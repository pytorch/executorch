#include <gtest/gtest.h>

#include <cstdint>

#include <fp16.h>
#include <tables.h>

#if (defined(__i386__) || defined(__x86_64__)) && defined(__F16C__)
	#include <x86intrin.h>
#endif


TEST(FP16_ALT_FROM_FP32_VALUE, normalized_powers_of_2) {
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
	const uint16_t max_po2_f16   = UINT16_C(0x7C00);

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
	const uint32_t max_po2_f32   = UINT32_C(0x47800000);

	float min_po2_value;
	memcpy(&min_po2_value, &min_po2_f32, sizeof(min_po2_value));
	EXPECT_EQ(min_po2_f16, fp16_alt_from_fp32_value(min_po2_value)) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F32 = 0x" << std::setw(8) << min_po2_f32 << ", " <<
		"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(min_po2_value) << ", " <<
		"F16 = 0x" << std::setw(4) << min_po2_f16;

	float eighths_value;
	memcpy(&eighths_value, &eighths_f32, sizeof(eighths_value));
	EXPECT_EQ(eighths_f16, fp16_alt_from_fp32_value(eighths_value)) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F32 = 0x" << std::setw(8) << eighths_f32 << ", " <<
		"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(eighths_value) << ", " <<
		"F16 = 0x" << std::setw(4) << eighths_f16;

	float quarter_value;
	memcpy(&quarter_value, &quarter_f32, sizeof(quarter_value));
	EXPECT_EQ(quarter_f16, fp16_alt_from_fp32_value(quarter_value)) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F32 = 0x" << std::setw(8) << quarter_f32 << ", " <<
		"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(quarter_value) << ", " <<
		"F16 = 0x" << std::setw(4) << quarter_f16;

	float half_value;
	memcpy(&half_value, &half_f32, sizeof(half_value));
	EXPECT_EQ(half_f16, fp16_alt_from_fp32_value(half_value)) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F32 = 0x" << std::setw(8) << half_f32 << ", " <<
		"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(half_value) << ", " <<
		"F16 = 0x" << std::setw(4) << half_f16;

	float one_value;
	memcpy(&one_value, &one_f32, sizeof(one_value));
	EXPECT_EQ(one_f16, fp16_alt_from_fp32_value(one_value)) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F32 = 0x" << std::setw(8) << one_f32 << ", " <<
		"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(one_value) << ", " <<
		"F16 = 0x" << std::setw(4) << one_f16;

	float two_value;
	memcpy(&two_value, &two_f32, sizeof(two_value));
	EXPECT_EQ(two_f16, fp16_alt_from_fp32_value(two_value)) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F32 = 0x" << std::setw(8) << two_f32 << ", " <<
		"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(two_value) << ", " <<
		"F16 = 0x" << std::setw(4) << two_f16;

	float four_value;
	memcpy(&four_value, &four_f32, sizeof(four_value));
	EXPECT_EQ(four_f16, fp16_alt_from_fp32_value(four_value)) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F32 = 0x" << std::setw(8) << four_f32 << ", " <<
		"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(four_value) << ", " <<
		"F16 = 0x" << std::setw(4) << four_f16;

	float eight_value;
	memcpy(&eight_value, &eight_f32, sizeof(eight_value));
	EXPECT_EQ(eight_f16, fp16_alt_from_fp32_value(eight_value)) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F32 = 0x" << std::setw(8) << eight_f32 << ", " <<
		"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(eight_value) << ", " <<
		"F16 = 0x" << std::setw(4) << eight_f16;

	float sixteen_value;
	memcpy(&sixteen_value, &sixteen_f32, sizeof(sixteen_value));
	EXPECT_EQ(sixteen_f16, fp16_alt_from_fp32_value(sixteen_value)) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F32 = 0x" << std::setw(8) << sixteen_f32 << ", " <<
		"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(sixteen_value) << ", " <<
		"F16 = 0x" << std::setw(4) << sixteen_f16;

	float thirtytwo_value;
	memcpy(&thirtytwo_value, &thirtytwo_f32, sizeof(thirtytwo_value));
	EXPECT_EQ(thirtytwo_f16, fp16_alt_from_fp32_value(thirtytwo_value)) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F32 = 0x" << std::setw(8) << thirtytwo_f32 << ", " <<
		"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(thirtytwo_value) << ", " <<
		"F16 = 0x" << std::setw(4) << thirtytwo_f16;

	float sixtyfour_value;
	memcpy(&sixtyfour_value, &sixtyfour_f32, sizeof(sixtyfour_value));
	EXPECT_EQ(sixtyfour_f16, fp16_alt_from_fp32_value(sixtyfour_value)) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F32 = 0x" << std::setw(8) << sixtyfour_f32 << ", " <<
		"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(sixtyfour_value) << ", " <<
		"F16 = 0x" << std::setw(4) << sixtyfour_f16;

	float max_po2_value;
	memcpy(&max_po2_value, &max_po2_f32, sizeof(max_po2_value));
	EXPECT_EQ(max_po2_f16, fp16_ieee_from_fp32_value(max_po2_value)) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F32 = 0x" << std::setw(8) << max_po2_f32 << ", " <<
		"F16(F32) = 0x" << std::setw(4) << fp16_ieee_from_fp32_value(max_po2_value) << ", " <<
		"F16 = 0x" << std::setw(4) << max_po2_f16;
}

TEST(FP16_ALT_FROM_FP32_VALUE, denormalized_powers_of_2) {
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
	const uint16_t exp2_minus_25_f16 = UINT16_C(0x0000);

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
	const uint32_t exp2_minus_25_f32 = UINT32_C(0x33000000);

	float exp2_minus_15_value;
	memcpy(&exp2_minus_15_value, &exp2_minus_15_f32, sizeof(exp2_minus_15_value));
	EXPECT_EQ(exp2_minus_15_f16, fp16_alt_from_fp32_value(exp2_minus_15_value)) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F32 = 0x" << std::setw(8) << exp2_minus_15_f32 << ", " <<
		"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(exp2_minus_15_value) << ", " <<
		"F16 = 0x" << std::setw(4) << exp2_minus_15_f16;

	float exp2_minus_16_value;
	memcpy(&exp2_minus_16_value, &exp2_minus_16_f32, sizeof(exp2_minus_16_value));
	EXPECT_EQ(exp2_minus_16_f16, fp16_alt_from_fp32_value(exp2_minus_16_value)) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F32 = 0x" << std::setw(8) << exp2_minus_16_f32 << ", " <<
		"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(exp2_minus_16_value) << ", " <<
		"F16 = 0x" << std::setw(4) << exp2_minus_16_f16;

	float exp2_minus_17_value;
	memcpy(&exp2_minus_17_value, &exp2_minus_17_f32, sizeof(exp2_minus_17_value));
	EXPECT_EQ(exp2_minus_17_f16, fp16_alt_from_fp32_value(exp2_minus_17_value)) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F32 = 0x" << std::setw(8) << exp2_minus_17_f32 << ", " <<
		"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(exp2_minus_17_value) << ", " <<
		"F16 = 0x" << std::setw(4) << exp2_minus_17_f16;

	float exp2_minus_18_value;
	memcpy(&exp2_minus_18_value, &exp2_minus_18_f32, sizeof(exp2_minus_18_value));
	EXPECT_EQ(exp2_minus_18_f16, fp16_alt_from_fp32_value(exp2_minus_18_value)) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F32 = 0x" << std::setw(8) << exp2_minus_18_f32 << ", " <<
		"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(exp2_minus_18_value) << ", " <<
		"F16 = 0x" << std::setw(4) << exp2_minus_18_f16;

	float exp2_minus_19_value;
	memcpy(&exp2_minus_19_value, &exp2_minus_19_f32, sizeof(exp2_minus_19_value));
	EXPECT_EQ(exp2_minus_19_f16, fp16_alt_from_fp32_value(exp2_minus_19_value)) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F32 = 0x" << std::setw(8) << exp2_minus_19_f32 << ", " <<
		"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(exp2_minus_19_value) << ", " <<
		"F16 = 0x" << std::setw(4) << exp2_minus_19_f16;

	float exp2_minus_20_value;
	memcpy(&exp2_minus_20_value, &exp2_minus_20_f32, sizeof(exp2_minus_20_value));
	EXPECT_EQ(exp2_minus_20_f16, fp16_alt_from_fp32_value(exp2_minus_20_value)) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F32 = 0x" << std::setw(8) << exp2_minus_20_f32 << ", " <<
		"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(exp2_minus_20_value) << ", " <<
		"F16 = 0x" << std::setw(4) << exp2_minus_20_f16;

	float exp2_minus_21_value;
	memcpy(&exp2_minus_21_value, &exp2_minus_21_f32, sizeof(exp2_minus_21_value));
	EXPECT_EQ(exp2_minus_21_f16, fp16_alt_from_fp32_value(exp2_minus_21_value)) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F32 = 0x" << std::setw(8) << exp2_minus_21_f32 << ", " <<
		"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(exp2_minus_21_value) << ", " <<
		"F16 = 0x" << std::setw(4) << exp2_minus_21_f16;

	float exp2_minus_22_value;
	memcpy(&exp2_minus_22_value, &exp2_minus_22_f32, sizeof(exp2_minus_22_value));
	EXPECT_EQ(exp2_minus_22_f16, fp16_alt_from_fp32_value(exp2_minus_22_value)) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F32 = 0x" << std::setw(8) << exp2_minus_22_f32 << ", " <<
		"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(exp2_minus_22_value) << ", " <<
		"F16 = 0x" << std::setw(4) << exp2_minus_22_f16;

	float exp2_minus_23_value;
	memcpy(&exp2_minus_23_value, &exp2_minus_23_f32, sizeof(exp2_minus_23_value));
	EXPECT_EQ(exp2_minus_23_f16, fp16_alt_from_fp32_value(exp2_minus_23_value)) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F32 = 0x" << std::setw(8) << exp2_minus_23_f32 << ", " <<
		"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(exp2_minus_23_value) << ", " <<
		"F16 = 0x" << std::setw(4) << exp2_minus_23_f16;

	float exp2_minus_24_value;
	memcpy(&exp2_minus_24_value, &exp2_minus_24_f32, sizeof(exp2_minus_24_value));
	EXPECT_EQ(exp2_minus_24_f16, fp16_alt_from_fp32_value(exp2_minus_24_value)) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F32 = 0x" << std::setw(8) << exp2_minus_24_f32 << ", " <<
		"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(exp2_minus_24_value) << ", " <<
		"F16 = 0x" << std::setw(4) << exp2_minus_24_f16;

	float exp2_minus_25_value;
	memcpy(&exp2_minus_25_value, &exp2_minus_25_f32, sizeof(exp2_minus_25_value));
	EXPECT_EQ(exp2_minus_25_f16, fp16_alt_from_fp32_value(exp2_minus_25_value)) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F32 = 0x" << std::setw(8) << exp2_minus_25_f32 << ", " <<
		"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(exp2_minus_25_value) << ", " <<
		"F16 = 0x" << std::setw(4) << exp2_minus_25_f16;
}

TEST(FP16_ALT_FROM_FP32_VALUE, zero) {
	const uint16_t positive_zero_f16 = UINT16_C(0x0000);
	const uint16_t negative_zero_f16 = UINT16_C(0x8000);

	const uint32_t positive_zero_f32 = UINT32_C(0x00000000);
	const uint32_t negative_zero_f32 = UINT32_C(0x80000000);

	float positive_zero_value;
	memcpy(&positive_zero_value, &positive_zero_f32, sizeof(positive_zero_value));
	EXPECT_EQ(positive_zero_f16, fp16_alt_from_fp32_value(positive_zero_value)) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F32 = 0x" << std::setw(8) << positive_zero_f32 << ", " <<
		"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(positive_zero_value) << ", " <<
		"F16 = 0x" << std::setw(4) << positive_zero_f16;

	float negative_zero_value;
	memcpy(&negative_zero_value, &negative_zero_f32, sizeof(negative_zero_value));
	EXPECT_EQ(negative_zero_f16, fp16_alt_from_fp32_value(negative_zero_value)) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F32 = 0x" << std::setw(8) << negative_zero_f32 << ", " <<
		"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(negative_zero_value) << ", " <<
		"F16 = 0x" << std::setw(4) << negative_zero_f16;
}

TEST(FP16_ALT_FROM_FP32_VALUE, infinity) {
	const uint16_t max_f16 = UINT16_C(0x7FFF);
	const uint16_t min_f16 = UINT16_C(0xFFFF);

	const uint32_t positive_infinity_f32 = UINT32_C(0x7F800000);
	const uint32_t negative_infinity_f32 = UINT32_C(0xFF800000);

	float positive_infinity_value;
	memcpy(&positive_infinity_value, &positive_infinity_f32, sizeof(positive_infinity_value));
	EXPECT_EQ(max_f16, fp16_alt_from_fp32_value(positive_infinity_value)) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F32 = 0x" << std::setw(8) << positive_infinity_f32 << ", " <<
		"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(positive_infinity_value) << ", " <<
		"F16 = 0x" << std::setw(4) << max_f16;

	float negative_infinity_value;
	memcpy(&negative_infinity_value, &negative_infinity_f32, sizeof(negative_infinity_value));
	EXPECT_EQ(min_f16, fp16_alt_from_fp32_value(negative_infinity_value)) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F32 = 0x" << std::setw(8) << negative_infinity_f32 << ", " <<
		"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(negative_infinity_value) << ", " <<
		"F16 = 0x" << std::setw(4) << min_f16;
}

TEST(FP16_ALT_FROM_FP32_VALUE, positive_nan) {
	for (uint32_t nan_f32 = UINT32_C(0x7FFFFFFF); nan_f32 > UINT32_C(0x7F800000); nan_f32--) {
		float nan_value;
		memcpy(&nan_value, &nan_f32, sizeof(nan_value));
		const uint16_t nan_f16 = fp16_alt_from_fp32_value(nan_value);

		/* Check sign */
		ASSERT_EQ(nan_f16 & UINT16_C(0x8000), 0) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F32 = 0x" << std::setw(8) << nan_f32 << ", " <<
			"F16(F32) = 0x" << std::setw(4) << nan_f16;

		/* Check exponent */
		ASSERT_EQ(nan_f16 & UINT16_C(0x7C00), UINT16_C(0x7C00)) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F32 = 0x" << std::setw(8) << nan_f32 << ", " <<
			"F16(F32) = 0x" << std::setw(4) << nan_f16;

		/* Check mantissa */
		ASSERT_NE(nan_f16 & UINT16_C(0x03FF), 0) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F32 = 0x" << std::setw(8) << nan_f32 << ", " <<
			"F16(F32) = 0x" << std::setw(4) << nan_f16;
	}
}

TEST(FP16_ALT_FROM_FP32_VALUE, negative_nan) {
	for (uint32_t nan_f32 = UINT32_C(0xFFFFFFFF); nan_f32 > UINT32_C(0xFF800000); nan_f32--) {
		float nan_value;
		memcpy(&nan_value, &nan_f32, sizeof(nan_value));
		const uint16_t nan_f16 = fp16_alt_from_fp32_value(nan_value);

		/* Check sign */
		ASSERT_EQ(nan_f16 & UINT16_C(0x8000), UINT16_C(0x8000)) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F32 = 0x" << std::setw(8) << nan_f32 << ", " <<
			"F16(F32) = 0x" << std::setw(4) << nan_f16;

		/* Check exponent */
		ASSERT_EQ(nan_f16 & UINT16_C(0x7C00), UINT16_C(0x7C00)) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F32 = 0x" << std::setw(8) << nan_f32 << ", " <<
			"F16(F32) = 0x" << std::setw(4) << nan_f16;

		/* Check mantissa */
		ASSERT_NE(nan_f16 & UINT16_C(0x03FF), 0) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F32 = 0x" << std::setw(8) << nan_f32 << ", " <<
			"F16(F32) = 0x" << std::setw(4) << nan_f16;
	}
}

TEST(FP16_ALT_FROM_FP32_VALUE, revertible) {
	/* Positive values */
	for (uint16_t f16 = UINT16_C(0x0000); f16 <= UINT16_C(0x7FFF); f16++) {
		const float value_f32 = fp16_alt_to_fp32_value(f16);
		uint32_t bits_f32;
		memcpy(&bits_f32, &value_f32, sizeof(bits_f32));

		ASSERT_EQ(f16, fp16_alt_from_fp32_value(value_f32)) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F16 = 0x" << std::setw(4) << f16 << ", " <<
			"F32(F16) = 0x" << std::setw(8) << bits_f32 << ", " <<
			"F16(F32(F16)) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(value_f32);
	}

	/* Negative values */
	for (uint16_t f16 = UINT16_C(0xFFFF); f16 >= UINT16_C(0x8000); f16--) {
		const float value_f32 = fp16_alt_to_fp32_value(f16);
		uint32_t bits_f32;
		memcpy(&bits_f32, &value_f32, sizeof(bits_f32));

		ASSERT_EQ(f16, fp16_alt_from_fp32_value(value_f32)) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F16 = 0x" << std::setw(4) << f16 << ", " <<
			"F32(F16) = 0x" << std::setw(8) << bits_f32 << ", " <<
			"F16(F32(F16)) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(value_f32);
	}
}

TEST(FP16_ALT_FROM_FP32_VALUE, underflow) {
	const uint32_t min_nonzero_f32 = UINT32_C(0x33000001);
	const uint16_t zero_f16 = UINT16_C(0x0000);
	const uint16_t min_f16 = UINT16_C(0x0001);
	for (uint32_t bits = UINT32_C(0x00000001); bits < min_nonzero_f32; bits++) {
		float value;
		memcpy(&value, &bits, sizeof(value));
		ASSERT_EQ(zero_f16, fp16_alt_from_fp32_value(value)) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F32 = 0x" << std::setw(8) << bits << ", " <<
			"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(value) << ", " <<
			"F16 = 0x" << std::setw(4) << zero_f16;
	}
	float min_nonzero_value;
	memcpy(&min_nonzero_value, &min_nonzero_f32, sizeof(min_nonzero_value));
	ASSERT_EQ(min_f16, fp16_alt_from_fp32_value(min_nonzero_value)) <<
		std::hex << std::uppercase << std::setfill('0') <<
		"F32 = 0x" << std::setw(8) << min_nonzero_f32 << ", " <<
		"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(min_nonzero_value) << ", " <<
		"F16 = 0x" << std::setw(4) << min_f16;
}

TEST(FP16_ALT_FROM_FP32_VALUE, saturation) {
	const uint32_t max_f16_f32 = UINT32_C(0x47FFE000);
	const uint16_t max_f16 = UINT16_C(0x7FFF);
	const uint32_t positive_infinity_f32 = UINT32_C(0x7F800000);
	for (uint32_t bits = positive_infinity_f32; bits > max_f16_f32; bits--) {
		float value;
		memcpy(&value, &bits, sizeof(value));
		ASSERT_EQ(max_f16, fp16_alt_from_fp32_value(value)) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"F32 = 0x" << std::setw(8) << bits << ", " <<
			"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(value) << ", " <<
			"F16 = 0x" << std::setw(4) << max_f16;
	}
}

TEST(FP16_ALT_FROM_FP32_VALUE, positive_denormalized_values) {
	const uint32_t min_nonzero_f32 = UINT32_C(0x33000001);

	uint32_t f32_begin = min_nonzero_f32;
	for (uint16_t f16 = 0; f16 < UINT16_C(0x0400); f16++) {
		const uint32_t f32_end = fp16::denormalizedRanges[f16];
		for (uint32_t f32 = f32_begin; f32 < f32_end; f32++) {
			float value;
			memcpy(&value, &f32, sizeof(value));
			ASSERT_EQ(f16, fp16_alt_from_fp32_value(value)) <<
				std::hex << std::uppercase << std::setfill('0') <<
				"F32 = 0x" << std::setw(8) << f32 << ", " <<
				"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(value) << ", " <<
				"F16 = 0x" << std::setw(4) << f16;
		}
		f32_begin = f32_end;
	}
}

TEST(FP16_ALT_FROM_FP32_VALUE, negative_denormalized_values) {
	const uint32_t min_nonzero_f32 = UINT32_C(0x33000001);

	uint32_t f32_begin = min_nonzero_f32 | UINT32_C(0x80000000);
	for (uint16_t f16 = UINT16_C(0x8000); f16 < UINT16_C(0x8400); f16++) {
		const uint32_t f32_end = fp16::denormalizedRanges[f16 & UINT16_C(0x7FFF)] | UINT32_C(0x80000000);
		for (uint32_t f32 = f32_begin; f32 < f32_end; f32++) {
			float value;
			memcpy(&value, &f32, sizeof(value));
			ASSERT_EQ(f16, fp16_alt_from_fp32_value(value)) <<
				std::hex << std::uppercase << std::setfill('0') <<
				"F32 = 0x" << std::setw(8) << f32 << ", " <<
				"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(value) << ", " <<
				"F16 = 0x" << std::setw(4) << f16;
		}
		f32_begin = f32_end;
	}
}

TEST(FP16_ALT_FROM_FP32_VALUE, positive_normalized_values) {
	/* Minimum number that rounds to 1.0h when converted to half-precision */
	const uint32_t min_one_f32 = UINT32_C(0x3F7FF000);
	const uint32_t e_bias = 15;

	for (int32_t e = -14; e <= 16; e++) {
		uint32_t f32_begin = min_one_f32 + (uint32_t(e) << 23);
		for (uint16_t f16 = uint16_t(e + e_bias) << 10; f16 < uint16_t(e + e_bias + 1) << 10; f16++) {
			const uint32_t f32_end = fp16::normalizedRanges[f16 & UINT16_C(0x3FF)] + (uint32_t(e) << 23);
			for (uint32_t f32 = f32_begin; f32 < f32_end; f32++) {
				float value;
				memcpy(&value, &f32, sizeof(value));
				ASSERT_EQ(f16, fp16_alt_from_fp32_value(value)) <<
					std::hex << std::uppercase << std::setfill('0') <<
					"F32 = 0x" << std::setw(8) << f32 << ", " <<
					"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(value) << ", " <<
					"F16 = 0x" << std::setw(4) << f16;
			}
			f32_begin = f32_end;
		}
	}
}

TEST(FP16_ALT_FROM_FP32_VALUE, negative_normalized_values) {
	/* Minimum number that rounds to 1.0h when converted to half-precision */
	const uint32_t min_one_f32 = UINT32_C(0x3F7FF000);
	const uint32_t e_bias = 15;

	for (int32_t e = -14; e <= 16; e++) {
		uint32_t f32_begin = (min_one_f32 | UINT32_C(0x80000000)) + (uint32_t(e) << 23);
		for (uint16_t f16 = (UINT16_C(0x8000) | (uint16_t(e + e_bias) << 10)); f16 < (UINT16_C(0x8000) | (uint16_t(e + e_bias + 1) << 10)); f16++) {
			const uint32_t f32_end = (fp16::normalizedRanges[f16 & UINT16_C(0x3FF)] | UINT32_C(0x80000000)) + (uint32_t(e) << 23);
			for (uint32_t f32 = f32_begin; f32 < f32_end; f32++) {
				float value;
				memcpy(&value, &f32, sizeof(value));
				ASSERT_EQ(f16, fp16_alt_from_fp32_value(value)) <<
					std::hex << std::uppercase << std::setfill('0') <<
					"F32 = 0x" << std::setw(8) << f32 << ", " <<
					"F16(F32) = 0x" << std::setw(4) << fp16_alt_from_fp32_value(value) << ", " <<
					"F16 = 0x" << std::setw(4) << f16;
			}
			f32_begin = f32_end;
		}
	}
}
