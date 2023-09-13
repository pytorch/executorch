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


static void fp16_ieee_to_fp32_bits(benchmark::State& state) {
	const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
	auto rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::mt19937(seed));

	std::vector<uint16_t> fp16(state.range(0));
	std::vector<uint32_t> fp32(state.range(0));
	std::generate(fp16.begin(), fp16.end(),
		[&rng]{ return fp16_ieee_from_fp32_value(rng()); });

	while (state.KeepRunning()) {
		uint16_t* input = fp16.data();
		benchmark::DoNotOptimize(input);

		uint32_t* output = fp32.data();
		const size_t n = state.range(0);
		for (size_t i = 0; i < n; i++) {
			output[i] = fp16_ieee_to_fp32_bits(input[i]);
		}

		benchmark::DoNotOptimize(output);
	}
	state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
}
BENCHMARK(fp16_ieee_to_fp32_bits)->RangeMultiplier(2)->Range(1<<10, 64<<20);

static void fp16_ieee_to_fp32_value(benchmark::State& state) {
	const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
	auto rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::mt19937(seed));

	std::vector<uint16_t> fp16(state.range(0));
	std::vector<float> fp32(state.range(0));
	std::generate(fp16.begin(), fp16.end(),
		[&rng]{ return fp16_ieee_from_fp32_value(rng()); });

	while (state.KeepRunning()) {
		uint16_t* input = fp16.data();
		benchmark::DoNotOptimize(input);

		float* output = fp32.data();
		const size_t n = state.range(0);
		for (size_t i = 0; i < n; i++) {
			output[i] = fp16_ieee_to_fp32_value(input[i]);
		}

		benchmark::DoNotOptimize(output);
	}
	state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
}
BENCHMARK(fp16_ieee_to_fp32_value)->RangeMultiplier(2)->Range(1<<10, 64<<20);

#ifndef EMSCRIPTEN
	static void fp16_ieee_to_fp32_psimd(benchmark::State& state) {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::mt19937(seed));

		std::vector<uint16_t> fp16(state.range(0));
		std::vector<float> fp32(state.range(0));
		std::generate(fp16.begin(), fp16.end(),
			[&rng]{ return fp16_ieee_from_fp32_value(rng()); });

		while (state.KeepRunning()) {
			uint16_t* input = fp16.data();
			benchmark::DoNotOptimize(input);

			float* output = fp32.data();
			const size_t n = state.range(0);
			for (size_t i = 0; i < n - 4; i += 4) {
				psimd_store_f32(&output[i],
					fp16_ieee_to_fp32_psimd(
						psimd_load_u16(&input[i])));
			}
			const psimd_u16 last_vector = { input[n - 4], input[n - 3], input[n - 2], input[n - 1] };
			psimd_store_f32(&output[n - 4],
				fp16_ieee_to_fp32_psimd(last_vector));

			benchmark::DoNotOptimize(output);
		}
		state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
	}
	BENCHMARK(fp16_ieee_to_fp32_psimd)->RangeMultiplier(2)->Range(1<<10, 64<<20);

	static void fp16_ieee_to_fp32x2_psimd(benchmark::State& state) {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::mt19937(seed));

		std::vector<uint16_t> fp16(state.range(0));
		std::vector<float> fp32(state.range(0));
		std::generate(fp16.begin(), fp16.end(),
			[&rng]{ return fp16_ieee_from_fp32_value(rng()); });

		while (state.KeepRunning()) {
			uint16_t* input = fp16.data();
			benchmark::DoNotOptimize(input);

			float* output = fp32.data();
			const size_t n = state.range(0);
			for (size_t i = 0; i < n; i += 8) {
				const psimd_f32x2 data =
					fp16_ieee_to_fp32x2_psimd(
						psimd_load_u16(&input[i]));
				psimd_store_f32(&output[i], data.lo);
				psimd_store_f32(&output[i + 4], data.hi);
			}

			benchmark::DoNotOptimize(output);
		}
		state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
	}
	BENCHMARK(fp16_ieee_to_fp32x2_psimd)->RangeMultiplier(2)->Range(1<<10, 64<<20);
#endif

#if (defined(__i386__) || defined(__x86_64__)) && defined(__F16C__)
	static void hardware_mm_cvtph_ps(benchmark::State& state) {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::mt19937(seed));

		std::vector<uint16_t> fp16(state.range(0));
		std::vector<float> fp32(state.range(0));
		std::generate(fp16.begin(), fp16.end(),
			[&rng]{ return fp16_ieee_from_fp32_value(rng()); });

		while (state.KeepRunning()) {
			uint16_t* input = fp16.data();
			benchmark::DoNotOptimize(input);

			float* output = fp32.data();
			const size_t n = state.range(0);
			for (size_t i = 0; i < n; i += 4) {
				_mm_storeu_ps(&output[i],
					_mm_cvtph_ps(
						_mm_loadl_epi64(static_cast<const __m128i*>(static_cast<const void*>(&input[i])))));
			}

			benchmark::DoNotOptimize(output);
		}
		state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
	}
	BENCHMARK(hardware_mm_cvtph_ps)->RangeMultiplier(2)->Range(1<<10, 64<<20);

	static void hardware_mm256_cvtph_ps(benchmark::State& state) {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::mt19937(seed));

		std::vector<uint16_t> fp16(state.range(0));
		std::vector<float> fp32(state.range(0));
		std::generate(fp16.begin(), fp16.end(),
			[&rng]{ return fp16_ieee_from_fp32_value(rng()); });

		while (state.KeepRunning()) {
			uint16_t* input = fp16.data();
			benchmark::DoNotOptimize(input);

			float* output = fp32.data();
			const size_t n = state.range(0);
			for (size_t i = 0; i < n; i += 8) {
				_mm256_storeu_ps(&output[i],
					_mm256_cvtph_ps(
						_mm_loadu_si128(static_cast<const __m128i*>(static_cast<const void*>(&input[i])))));
			}

			benchmark::DoNotOptimize(output);
		}
		state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
	}
	BENCHMARK(hardware_mm256_cvtph_ps)->RangeMultiplier(2)->Range(1<<10, 64<<20);
#endif

#if defined(__ARM_NEON_FP) && (__ARM_NEON_FP & 0x2) || defined(__aarch64__)
	static void hardware_vcvt_f32_f16(benchmark::State& state) {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::mt19937(seed));

		std::vector<uint16_t> fp16(state.range(0));
		std::vector<float> fp32(state.range(0));
		std::generate(fp16.begin(), fp16.end(),
			[&rng]{ return fp16_ieee_from_fp32_value(rng()); });

		while (state.KeepRunning()) {
			uint16_t* input = fp16.data();
			benchmark::DoNotOptimize(input);

			float* output = fp32.data();
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
				vst1q_f32(&output[i],
					vcvt_f32_f16(
						(float16x4_t) vld1_u16(&input[i])));
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
	BENCHMARK(hardware_vcvt_f32_f16)->RangeMultiplier(2)->Range(1<<10, 64<<20);
#endif

#ifdef FP16_COMPARATIVE_BENCHMARKS
	static void TH_halfbits2float(benchmark::State& state) {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::mt19937(seed));

		std::vector<uint16_t> fp16(state.range(0));
		std::vector<float> fp32(state.range(0));
		std::generate(fp16.begin(), fp16.end(),
			[&rng]{ return fp16_ieee_from_fp32_value(rng()); });

		while (state.KeepRunning()) {
			uint16_t* input = fp16.data();
			benchmark::DoNotOptimize(input);

			float* output = fp32.data();
			const size_t n = state.range(0);
			for (size_t i = 0; i < n; i++) {
				TH_halfbits2float(&input[i], &output[i]);
			}

			benchmark::DoNotOptimize(output);
		}
		state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
	}
	BENCHMARK(TH_halfbits2float)->RangeMultiplier(2)->Range(1<<10, 64<<20);

	static void npy_halfbits_to_floatbits(benchmark::State& state) {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::mt19937(seed));

		std::vector<uint16_t> fp16(state.range(0));
		std::vector<uint32_t> fp32(state.range(0));
		std::generate(fp16.begin(), fp16.end(),
			[&rng]{ return fp16_ieee_from_fp32_value(rng()); });

		while (state.KeepRunning()) {
			uint16_t* input = fp16.data();
			benchmark::DoNotOptimize(input);

			uint32_t* output = fp32.data();
			const size_t n = state.range(0);
			for (size_t i = 0; i < n; i++) {
				output[i] = npy_halfbits_to_floatbits(input[i]);
			}

			benchmark::DoNotOptimize(output);
		}
		state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
	}
	BENCHMARK(npy_halfbits_to_floatbits)->RangeMultiplier(2)->Range(1<<10, 64<<20);

	static void Eigen_half_to_float(benchmark::State& state) {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::mt19937(seed));

		std::vector<uint16_t> fp16(state.range(0));
		std::vector<float> fp32(state.range(0));
		std::generate(fp16.begin(), fp16.end(),
			[&rng]{ return fp16_ieee_from_fp32_value(rng()); });

		while (state.KeepRunning()) {
			uint16_t* input = fp16.data();
			benchmark::DoNotOptimize(input);

			float* output = fp32.data();
			const size_t n = state.range(0);
			for (size_t i = 0; i < n; i++) {
				output[i] =
					Eigen::half_impl::half_to_float(
						Eigen::half_impl::raw_uint16_to_half(input[i]));
			}

			benchmark::DoNotOptimize(output);
		}
		state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
	}
	BENCHMARK(Eigen_half_to_float)->RangeMultiplier(2)->Range(1<<10, 64<<20);

	static void Float16Compressor_decompress(benchmark::State& state) {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::mt19937(seed));

		std::vector<uint16_t> fp16(state.range(0));
		std::vector<float> fp32(state.range(0));
		std::generate(fp16.begin(), fp16.end(),
			[&rng]{ return fp16_ieee_from_fp32_value(rng()); });

		while (state.KeepRunning()) {
			uint16_t* input = fp16.data();
			benchmark::DoNotOptimize(input);

			float* output = fp32.data();
			const size_t n = state.range(0);
			for (size_t i = 0; i < n; i++) {
				output[i] = Float16Compressor::decompress(input[i]);
			}

			benchmark::DoNotOptimize(output);
		}
		state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
	}
	BENCHMARK(Float16Compressor_decompress)->RangeMultiplier(2)->Range(1<<10, 64<<20);

	static void half_float_detail_half2float_table(benchmark::State& state) {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::mt19937(seed));

		std::vector<uint16_t> fp16(state.range(0));
		std::vector<float> fp32(state.range(0));
		std::generate(fp16.begin(), fp16.end(),
			[&rng]{ return fp16_ieee_from_fp32_value(rng()); });

		while (state.KeepRunning()) {
			uint16_t* input = fp16.data();
			benchmark::DoNotOptimize(input);

			float* output = fp32.data();
			const size_t n = state.range(0);
			for (size_t i = 0; i < n; i++) {
				output[i] = half_float::detail::half2float_impl(input[i],
					half_float::detail::true_type());
			}

			benchmark::DoNotOptimize(output);
		}
		state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
	}
	BENCHMARK(half_float_detail_half2float_table)->RangeMultiplier(2)->Range(1<<10, 64<<20);

	static void half_float_detail_half2float_branch(benchmark::State& state) {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::mt19937(seed));

		std::vector<uint16_t> fp16(state.range(0));
		std::vector<float> fp32(state.range(0));
		std::generate(fp16.begin(), fp16.end(),
			[&rng]{ return fp16_ieee_from_fp32_value(rng()); });

		while (state.KeepRunning()) {
			uint16_t* input = fp16.data();
			benchmark::DoNotOptimize(input);

			float* output = fp32.data();
			const size_t n = state.range(0);
			for (size_t i = 0; i < n; i++) {
				output[i] = half_float::detail::half2float_impl(input[i],
					half_float::detail::false_type());
			}

			benchmark::DoNotOptimize(output);
		}
		state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
	}
	BENCHMARK(half_float_detail_half2float_branch)->RangeMultiplier(2)->Range(1<<10, 64<<20);
#endif

BENCHMARK_MAIN();
