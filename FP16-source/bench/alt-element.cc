#include <benchmark/benchmark.h>

#include <fp16.h>
#ifndef EMSCRIPTEN
	#include <fp16/psimd.h>
#endif

static inline uint16_t next_xorshift16(uint16_t x) {
	x ^= x >> 8;
	x ^= x << 9;
	x ^= x >> 5;
	return x;
}

static inline uint32_t next_xorshift32(uint32_t x) {
	x ^= x >> 13;
	x ^= x << 17;
	x ^= x >> 5;
	return x;
}
#ifndef EMSCRIPTEN
	PSIMD_INTRINSIC psimd_u16 next_xorshift16_psimd(psimd_u16 x) {
		x ^= x >> psimd_splat_u16(8);
		x ^= x << psimd_splat_u16(9);
		x ^= x >> psimd_splat_u16(5);
		return x;
	}
#endif


static void fp16_alt_to_fp32_bits(benchmark::State& state) {
	uint16_t fp16 = UINT16_C(0x7C00);
	while (state.KeepRunning()) {
		const uint32_t fp32 = fp16_alt_to_fp32_bits(fp16);

		fp16 = next_xorshift16(fp16);
		benchmark::DoNotOptimize(fp32);
	}
}
BENCHMARK(fp16_alt_to_fp32_bits);

static void fp16_alt_to_fp32_value(benchmark::State& state) {
	uint16_t fp16 = UINT16_C(0x7C00);
	while (state.KeepRunning()) {
		const float fp32 = fp16_alt_to_fp32_value(fp16);

		fp16 = next_xorshift16(fp16);
		benchmark::DoNotOptimize(fp32);
	}
}
BENCHMARK(fp16_alt_to_fp32_value);

#ifndef EMSCRIPTEN
	static void fp16_alt_to_fp32_psimd(benchmark::State& state) {
		psimd_u16 fp16 = (psimd_u16) { 0x7C00, 0x7C01, 0x7C02, 0x7C03 };
		while (state.KeepRunning()) {
			const psimd_f32 fp32 = fp16_alt_to_fp32_psimd(fp16);

			fp16 = next_xorshift16_psimd(fp16);
			benchmark::DoNotOptimize(fp32);
		}
	}
	BENCHMARK(fp16_alt_to_fp32_psimd);

	static void fp16_alt_to_fp32x2_psimd(benchmark::State& state) {
		psimd_u16 fp16 =
			(psimd_u16) { 0x7C00, 0x7C01, 0x7C02, 0x7C03, 0x7C04, 0x7C05, 0x7C06, 0x7C07 };
		while (state.KeepRunning()) {
			const psimd_f32x2 fp32 = fp16_alt_to_fp32x2_psimd(fp16);

			fp16 = next_xorshift16_psimd(fp16);
			benchmark::DoNotOptimize(fp32);
		}
	}
	BENCHMARK(fp16_alt_to_fp32x2_psimd);
#endif

static void fp16_alt_from_fp32_value(benchmark::State& state) {
	uint32_t fp32 = UINT32_C(0x7F800000);
	while (state.KeepRunning()) {
		const uint16_t fp16 = fp16_alt_from_fp32_value(fp32_from_bits(fp32));

		fp32 = next_xorshift32(fp32);
		benchmark::DoNotOptimize(fp16);
	}
}
BENCHMARK(fp16_alt_from_fp32_value);

BENCHMARK_MAIN();
