#include <benchmark/benchmark.h>

#include <fp16.h>
#ifndef EMSCRIPTEN
	#include <fp16/psimd.h>
#endif

#if (defined(__i386__) || defined(__x86_64__)) && defined(__F16C__)
	#include <immintrin.h>
#endif

#ifdef FP16_COMPARATIVE_BENCHMARKS
	#include <third-party/THHalf.h>
	#include <third-party/npy-halffloat.h>
	#include <third-party/eigen-half.h>
	#include <third-party/float16-compressor.h>
	#include <third-party/half.hpp>
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


/* Conversion from IEEE FP16 to IEEE FP32 */

static void fp16_ieee_to_fp32_bits(benchmark::State& state) {
	uint16_t fp16 = UINT16_C(0x7C00);
	while (state.KeepRunning()) {
		const uint32_t fp32 = fp16_ieee_to_fp32_bits(fp16);

		fp16 = next_xorshift16(fp16);
		benchmark::DoNotOptimize(fp32);
	}
}
BENCHMARK(fp16_ieee_to_fp32_bits);

static void fp16_ieee_to_fp32_value(benchmark::State& state) {
	uint16_t fp16 = UINT16_C(0x7C00);
	while (state.KeepRunning()) {
		const float fp32 = fp16_ieee_to_fp32_value(fp16);

		fp16 = next_xorshift16(fp16);
		benchmark::DoNotOptimize(fp32);
	}
}
BENCHMARK(fp16_ieee_to_fp32_value);

#ifndef EMSCRIPTEN
	static void fp16_ieee_to_fp32_psimd(benchmark::State& state) {
		psimd_u16 fp16 = (psimd_u16) { 0x7C00, 0x7C01, 0x7C02, 0x7C03 };
		while (state.KeepRunning()) {
			const psimd_f32 fp32 = fp16_ieee_to_fp32_psimd(fp16);

			fp16 = next_xorshift16_psimd(fp16);
			benchmark::DoNotOptimize(fp32);
		}
	}
	BENCHMARK(fp16_ieee_to_fp32_psimd);

	static void fp16_ieee_to_fp32x2_psimd(benchmark::State& state) {
		psimd_u16 fp16 =
			(psimd_u16) { 0x7C00, 0x7C01, 0x7C02, 0x7C03, 0x7C04, 0x7C05, 0x7C06, 0x7C07 };
		while (state.KeepRunning()) {
			const psimd_f32x2 fp32 = fp16_ieee_to_fp32x2_psimd(fp16);

			fp16 = next_xorshift16_psimd(fp16);
			benchmark::DoNotOptimize(fp32);
		}
	}
	BENCHMARK(fp16_ieee_to_fp32x2_psimd);
#endif

#ifdef FP16_COMPARATIVE_BENCHMARKS
	static void TH_halfbits2float(benchmark::State& state) {
		uint16_t fp16 = UINT16_C(0x7C00);
		while (state.KeepRunning()) {
			float fp32;
			TH_halfbits2float(&fp16, &fp32);

			fp16 = next_xorshift16(fp16);
			benchmark::DoNotOptimize(fp32);
		}
	}
	BENCHMARK(TH_halfbits2float);

	static void npy_halfbits_to_floatbits(benchmark::State& state) {
		uint16_t fp16 = UINT16_C(0x7C00);
		while (state.KeepRunning()) {
			const uint32_t fp32 = npy_halfbits_to_floatbits(fp16);

			fp16 = next_xorshift16(fp16);
			benchmark::DoNotOptimize(fp32);
		}
	}
	BENCHMARK(npy_halfbits_to_floatbits);

	static void Eigen_half_to_float(benchmark::State& state) {
		uint16_t fp16 = UINT16_C(0x7C00);
		while (state.KeepRunning()) {
			const float fp32 =
				Eigen::half_impl::half_to_float(
					Eigen::half_impl::raw_uint16_to_half(fp16));

			fp16 = next_xorshift16(fp16);
			benchmark::DoNotOptimize(fp32);
		}
	}
	BENCHMARK(Eigen_half_to_float);

	static void Float16Compressor_decompress(benchmark::State& state) {
		uint16_t fp16 = UINT16_C(0x7C00);
		while (state.KeepRunning()) {
			const float fp32 = Float16Compressor::decompress(fp16);

			fp16 = next_xorshift16(fp16);
			benchmark::DoNotOptimize(fp32);
		}
	}
	BENCHMARK(Float16Compressor_decompress);

	static void half_float_detail_half2float_table(benchmark::State& state) {
		uint16_t fp16 = UINT16_C(0x7C00);
		while (state.KeepRunning()) {
			const float fp32 =
				half_float::detail::half2float_impl(fp16,
					half_float::detail::true_type());

			fp16 = next_xorshift16(fp16);
			benchmark::DoNotOptimize(fp32);
		}
	}
	BENCHMARK(half_float_detail_half2float_table);

	static void half_float_detail_half2float_branch(benchmark::State& state) {
		uint16_t fp16 = UINT16_C(0x7C00);
		while (state.KeepRunning()) {
			const float fp32 =
				half_float::detail::half2float_impl(fp16,
					half_float::detail::false_type());

			fp16 = next_xorshift16(fp16);
			benchmark::DoNotOptimize(fp32);
		}
	}
	BENCHMARK(half_float_detail_half2float_branch);
#endif

/* Conversion from IEEE FP32 to IEEE FP16 */

static void fp16_ieee_from_fp32_value(benchmark::State& state) {
	uint32_t fp32 = UINT32_C(0x7F800000);
	while (state.KeepRunning()) {
		const uint16_t fp16 = fp16_ieee_from_fp32_value(fp32_from_bits(fp32));

		fp32 = next_xorshift32(fp32);
		benchmark::DoNotOptimize(fp16);
	}
}
BENCHMARK(fp16_ieee_from_fp32_value);

#if (defined(__i386__) || defined(__x86_64__)) && defined(__F16C__)
	static void fp16_ieee_from_fp32_hardware(benchmark::State& state) {
		uint32_t fp32 = UINT32_C(0x7F800000);
		while (state.KeepRunning()) {
			const uint16_t fp16 = static_cast<uint16_t>(
				_mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(fp32), _MM_FROUND_CUR_DIRECTION)));

			fp32 = next_xorshift32(fp32);
			benchmark::DoNotOptimize(fp16);
		}
	}
	BENCHMARK(fp16_ieee_from_fp32_hardware);
#endif

#ifdef FP16_COMPARATIVE_BENCHMARKS
	static void TH_float2halfbits(benchmark::State& state) {
		uint32_t fp32 = UINT32_C(0x7F800000);
		while (state.KeepRunning()) {
			uint16_t fp16;
			float fp32_value = fp32_from_bits(fp32);
			TH_float2halfbits(&fp32_value, &fp16);

			fp32 = next_xorshift32(fp32);
			benchmark::DoNotOptimize(fp16);
		}
	}
	BENCHMARK(TH_float2halfbits);

	static void npy_floatbits_to_halfbits(benchmark::State& state) {
		uint32_t fp32 = UINT32_C(0x7F800000);
		while (state.KeepRunning()) {
			const uint16_t fp16 = npy_floatbits_to_halfbits(fp32);

			fp32 = next_xorshift32(fp32);
			benchmark::DoNotOptimize(fp16);
		}
	}
	BENCHMARK(npy_floatbits_to_halfbits);

	static void Eigen_float_to_half_rtne(benchmark::State& state) {
		uint32_t fp32 = UINT32_C(0x7F800000);
		while (state.KeepRunning()) {
			const Eigen::half_impl::__half fp16 =
				Eigen::half_impl::float_to_half_rtne(
					fp32_from_bits(fp32));

			fp32 = next_xorshift32(fp32);
			benchmark::DoNotOptimize(fp16);
		}
	}
	BENCHMARK(Eigen_float_to_half_rtne);

	static void Float16Compressor_compress(benchmark::State& state) {
		uint32_t fp32 = UINT32_C(0x7F800000);
		while (state.KeepRunning()) {
			const uint16_t fp16 = Float16Compressor::compress(fp32_from_bits(fp32));

			fp32 = next_xorshift32(fp32);
			benchmark::DoNotOptimize(fp16);
		}
	}
	BENCHMARK(Float16Compressor_compress);

	static void half_float_detail_float2half_table(benchmark::State& state) {
		uint32_t fp32 = UINT32_C(0x7F800000);
		while (state.KeepRunning()) {
			const uint16_t fp16 =
				half_float::detail::float2half_impl<std::round_to_nearest>(
					fp32_from_bits(fp32),
						half_float::detail::true_type());

			fp32 = next_xorshift32(fp32);
			benchmark::DoNotOptimize(fp16);
		}
	}
	BENCHMARK(half_float_detail_float2half_table);

	static void half_float_detail_float2half_branch(benchmark::State& state) {
		uint32_t fp32 = UINT32_C(0x7F800000);
		while (state.KeepRunning()) {
			const uint16_t fp16 =
				half_float::detail::float2half_impl<std::round_to_nearest>(
					fp32_from_bits(fp32),
						half_float::detail::false_type());

			fp32 = next_xorshift32(fp32);
			benchmark::DoNotOptimize(fp16);
		}
	}
	BENCHMARK(half_float_detail_float2half_branch);
#endif

BENCHMARK_MAIN();
