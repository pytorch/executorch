#include <gtest/gtest.h>

#include <cstdint>

#include <fp16.h>


TEST(FP32_TO_BITS, positive) {
	for (uint32_t bits = UINT32_C(0x00000000); bits <= UINT32_C(0x7FFFFFFF); bits++) {
		float value;
		memcpy(&value, &bits, sizeof(value));

		ASSERT_EQ(bits, fp32_to_bits(value)) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"BITS = 0x" << std::setw(8) << bits << ", " <<
			"BITCAST(VALUE) = 0x" << std::setw(8) << fp32_to_bits(value);
	}
}

TEST(FP32_TO_BITS, negative) {
	for (uint32_t bits = UINT32_C(0xFFFFFFFF); bits >= UINT32_C(0x80000000); bits--) {
		float value;
		memcpy(&value, &bits, sizeof(value));

		ASSERT_EQ(bits, fp32_to_bits(value)) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"BITS = 0x" << std::setw(8) << bits << ", " <<
			"BITCAST(VALUE) = 0x" << std::setw(8) << fp32_to_bits(value);
	}
}

TEST(FP32_FROM_BITS, positive) {
	for (uint32_t bits = UINT32_C(0x00000000); bits <= UINT32_C(0x7FFFFFFF); bits++) {
		const float value = fp32_from_bits(bits);
		uint32_t bitcast;
		memcpy(&bitcast, &value, sizeof(bitcast));

		ASSERT_EQ(bits, bitcast) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"BITS = 0x" << std::setw(8) << bits << ", " <<
			"VALUE = 0x" << std::setw(8) << bitcast;
	}
}

TEST(FP32_FROM_BITS, negative) {
	for (uint32_t bits = UINT32_C(0xFFFFFFFF); bits >= UINT32_C(0x80000000); bits--) {
		const float value = fp32_from_bits(bits);
		uint32_t bitcast;
		memcpy(&bitcast, &value, sizeof(bitcast));

		ASSERT_EQ(bits, bitcast) <<
			std::hex << std::uppercase << std::setfill('0') <<
			"BITS = 0x" << std::setw(8) << bits << ", " <<
			"VALUE = 0x" << std::setw(8) << bitcast;
	}
}
