#include <benchmark/benchmark.h>

#include <fp16.h>
#ifndef EMSCRIPTEN
	#include <fp16/psimd.h>
#endif

#include <vector>
#include <random>
#include <chrono>
#include <functional>
#include <algorithm>

#if (defined(__i386__) || defined(__x86_64__)) && defined(__F16C__)
	#include <immintrin.h>
#endif

#if defined(__ARM_NEON__) || defined(__aarch64__)
	#include <arm_neon.h>
#endif

#ifdef FP16_COMPARATIVE_BENCHMARKS
	#include <third-party/THHalf.h>
	#include <third-party/npy-halffloat.h>
	#include <third-party/eigen-half.h>
	#include <third-party/float16-compressor.h>
	#include <third-party/half.hpp>
#endif


static void fp16_ieee_from_fp32_value(benchmark::State& state) {
	const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
	auto rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::mt19937(seed));

	std::vector<float> fp32(state.range(0));
	std::vector<uint16_t> fp16(state.range(0));
	std::generate(fp32.begin(), fp32.end(), std::ref(rng));

	while (state.KeepRunning()) {
		float* input = fp32.data();
		benchmark::DoNotOptimize(input);

		uint16_t* output = fp16.data();
		const size_t n = state.range(0);
		for (size_t i = 0; i < n; i++) {
			output[i] = fp16_ieee_from_fp32_value(input[i]);
		}

		benchmark::DoNotOptimize(output);
	}
	state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
}
BENCHMARK(fp16_ieee_from_fp32_value)->RangeMultiplier(2)->Range(1<<10, 64<<20);

#if (defined(__i386__) || defined(__x86_64__)) && defined(__F16C__)
	static void hardware_mm_cvtps_ph(benchmark::State& state) {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::mt19937(seed));

		std::vector<float> fp32(state.range(0));
		std::vector<uint16_t> fp16(state.range(0));
		std::generate(fp32.begin(), fp32.end(), std::ref(rng));

		while (state.KeepRunning()) {
			float* input = fp32.data();
			benchmark::DoNotOptimize(input);

			uint16_t* output = fp16.data();
			const size_t n = state.range(0);
			for (size_t i = 0; i < n; i += 4) {
				_mm_storel_epi64(
					static_cast<__m128i*>(static_cast<void*>(&output[i])),
					_mm_cvtps_ph(_mm_loadu_ps(&input[i]), _MM_FROUND_CUR_DIRECTION));
			}

			benchmark::DoNotOptimize(output);
		}
		state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
	}
	BENCHMARK(hardware_mm_cvtps_ph)->RangeMultiplier(2)->Range(1<<10, 64<<20);

	static void hardware_mm256_cvtps_ph(benchmark::State& state) {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::mt19937(seed));

		std::vector<float> fp32(state.range(0));
		std::vector<uint16_t> fp16(state.range(0));
		std::generate(fp32.begin(), fp32.end(), std::ref(rng));

		while (state.KeepRunning()) {
			float* input = fp32.data();
			benchmark::DoNotOptimize(input);

			uint16_t* output = fp16.data();
			const size_t n = state.range(0);
			for (size_t i = 0; i < n; i += 8) {
				_mm_storeu_si128(
					static_cast<__m128i*>(static_cast<void*>(&output[i])),
					_mm256_cvtps_ph(_mm256_loadu_ps(&input[i]), _MM_FROUND_CUR_DIRECTION));
			}

			benchmark::DoNotOptimize(output);
		}
		state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
	}
	BENCHMARK(hardware_mm256_cvtps_ph)->RangeMultiplier(2)->Range(1<<10, 64<<20);
#endif

#if defined(__ARM_NEON_FP) && (__ARM_NEON_FP & 0x2) || defined(__aarch64__)
	static void hardware_vcvt_f16_f32(benchmark::State& state) {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::mt19937(seed));

		std::vector<float> fp32(state.range(0));
		std::vector<uint16_t> fp16(state.range(0));
		std::generate(fp32.begin(), fp32.end(), std::ref(rng));

		while (state.KeepRunning()) {
			float* input = fp32.data();
			benchmark::DoNotOptimize(input);

			uint16_t* output = fp16.data();
			const size_t n = state.range(0);
			#if defined(__aarch64__)
				const unsigned int fpcr = __builtin_aarch64_get_fpcr();
				/* Disable flush-to-zero (bit 24) and Alternative FP16 format (bit 26) */
				__builtin_aarch64_set_fpcr(fpcr & 0xF6FFFFFFu);
			#else
				unsigned int fpscr;
				__asm__ __volatile__ ("VMRS %[fpscr], fpscr" : [fpscr] "=r" (fpscr));
				/* Disable flush-to-zero (bit 24) and Alternative FP16 format (bit 26) */
				__asm__ __volatile__ ("VMSR fpscr, %[fpscr]" :
					: [fpscr] "r" (fpscr & 0xF6FFFFFFu));
			#endif
			for (size_t i = 0; i < n; i += 4) {
				vst1_u16(&output[i],
					(uint16x4_t) vcvt_f16_f32(
						vld1q_f32(&input[i])));
			}
			#if defined(__aarch64__)
				__builtin_aarch64_set_fpcr(fpcr);
			#else
				__asm__ __volatile__ ("VMSR fpscr, %[fpscr]" :: [fpscr] "r" (fpscr));
			#endif

			benchmark::DoNotOptimize(output);
		}
		state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
	}
	BENCHMARK(hardware_vcvt_f16_f32)->RangeMultiplier(2)->Range(1<<10, 64<<20);
#endif

#ifdef FP16_COMPARATIVE_BENCHMARKS
	static void TH_float2halfbits(benchmark::State& state) {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::mt19937(seed));

		std::vector<float> fp32(state.range(0));
		std::vector<uint16_t> fp16(state.range(0));
		std::generate(fp32.begin(), fp32.end(), std::ref(rng));

		while (state.KeepRunning()) {
			float* input = fp32.data();
			benchmark::DoNotOptimize(input);

			uint16_t* output = fp16.data();
			const size_t n = state.range(0);
			for (size_t i = 0; i < n; i++) {
				TH_float2halfbits(&input[i], &output[i]);
			}

			benchmark::DoNotOptimize(output);
		}
		state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
	}
	BENCHMARK(TH_float2halfbits)->RangeMultiplier(2)->Range(1<<10, 64<<20);

	static void npy_floatbits_to_halfbits(benchmark::State& state) {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::mt19937(seed));

		std::vector<float> fp32(state.range(0));
		std::vector<uint16_t> fp16(state.range(0));
		std::generate(fp32.begin(), fp32.end(), std::ref(rng));

		while (state.KeepRunning()) {
			float* input = fp32.data();
			benchmark::DoNotOptimize(input);

			uint16_t* output = fp16.data();
			const size_t n = state.range(0);
			for (size_t i = 0; i < n; i++) {
				output[i] = npy_floatbits_to_halfbits(fp32_to_bits(input[i]));
			}

			benchmark::DoNotOptimize(output);
		}
		state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
	}
	BENCHMARK(npy_floatbits_to_halfbits)->RangeMultiplier(2)->Range(1<<10, 64<<20);

	static void Eigen_float_to_half_rtne(benchmark::State& state) {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::mt19937(seed));

		std::vector<float> fp32(state.range(0));
		std::vector<uint16_t> fp16(state.range(0));
		std::generate(fp32.begin(), fp32.end(), std::ref(rng));

		while (state.KeepRunning()) {
			float* input = fp32.data();
			benchmark::DoNotOptimize(input);

			uint16_t* output = fp16.data();
			const size_t n = state.range(0);
			for (size_t i = 0; i < n; i++) {
				output[i] = Eigen::half_impl::float_to_half_rtne(input[i]).x;
			}

			benchmark::DoNotOptimize(output);
		}
		state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
	}
	BENCHMARK(Eigen_float_to_half_rtne)->RangeMultiplier(2)->Range(1<<10, 64<<20);

	static void Float16Compressor_compress(benchmark::State& state) {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::mt19937(seed));

		std::vector<float> fp32(state.range(0));
		std::vector<uint16_t> fp16(state.range(0));
		std::generate(fp32.begin(), fp32.end(), std::ref(rng));

		while (state.KeepRunning()) {
			float* input = fp32.data();
			benchmark::DoNotOptimize(input);

			uint16_t* output = fp16.data();
			const size_t n = state.range(0);
			for (size_t i = 0; i < n; i++) {
				output[i] = Float16Compressor::compress(input[i]);
			}

			benchmark::DoNotOptimize(output);
		}
		state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
	}
	BENCHMARK(Float16Compressor_compress)->RangeMultiplier(2)->Range(1<<10, 64<<20);

	static void half_float_detail_float2half_table(benchmark::State& state) {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::mt19937(seed));

		std::vector<float> fp32(state.range(0));
		std::vector<uint16_t> fp16(state.range(0));
		std::generate(fp32.begin(), fp32.end(), std::ref(rng));

		while (state.KeepRunning()) {
			float* input = fp32.data();
			benchmark::DoNotOptimize(input);

			uint16_t* output = fp16.data();
			const size_t n = state.range(0);
			for (size_t i = 0; i < n; i++) {
				output[i] =
					half_float::detail::float2half_impl<std::round_to_nearest>(
						input[i], half_float::detail::true_type());
			}

			benchmark::DoNotOptimize(output);
		}
		state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
	}
	BENCHMARK(half_float_detail_float2half_table)->RangeMultiplier(2)->Range(1<<10, 64<<20);

	static void half_float_detail_float2half_branch(benchmark::State& state) {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::mt19937(seed));

		std::vector<float> fp32(state.range(0));
		std::vector<uint16_t> fp16(state.range(0));
		std::generate(fp32.begin(), fp32.end(), std::ref(rng));

		while (state.KeepRunning()) {
			float* input = fp32.data();
			benchmark::DoNotOptimize(input);

			uint16_t* output = fp16.data();
			const size_t n = state.range(0);
			for (size_t i = 0; i < n; i++) {
				output[i] =
					half_float::detail::float2half_impl<std::round_to_nearest>(
						input[i], half_float::detail::false_type());
			}

			benchmark::DoNotOptimize(output);
		}
		state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
	}
	BENCHMARK(half_float_detail_float2half_branch)->RangeMultiplier(2)->Range(1<<10, 64<<20);
#endif

BENCHMARK_MAIN();
