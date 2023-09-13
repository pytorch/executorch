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

#if defined(__ARM_NEON__) || defined(__aarch64__)
	#include <arm_neon.h>
#endif


static void fp16_alt_to_fp32_bits(benchmark::State& state) {
	const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
	auto rng = std::bind(std::uniform_int_distribution<uint16_t>(0, 0x7BFF), std::mt19937(seed));

	std::vector<uint16_t> fp16(state.range(0));
	std::vector<uint32_t> fp32(state.range(0));
	std::generate(fp16.begin(), fp16.end(),
		[&rng]{ return fp16_alt_from_fp32_value(rng()); });

	while (state.KeepRunning()) {
		uint16_t* input = fp16.data();
		benchmark::DoNotOptimize(input);

		uint32_t* output = fp32.data();
		const size_t n = state.range(0);
		for (size_t i = 0; i < n; i++) {
			output[i] = fp16_alt_to_fp32_bits(input[i]);
		}

		benchmark::DoNotOptimize(output);
	}
	state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
}
BENCHMARK(fp16_alt_to_fp32_bits)->RangeMultiplier(2)->Range(1<<10, 64<<20);

static void fp16_alt_to_fp32_value(benchmark::State& state) {
	const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
	auto rng = std::bind(std::uniform_int_distribution<uint16_t>(0, 0x7BFF), std::mt19937(seed));

	std::vector<uint16_t> fp16(state.range(0));
	std::vector<float> fp32(state.range(0));
	std::generate(fp16.begin(), fp16.end(),
		[&rng]{ return fp16_alt_from_fp32_value(rng()); });

	while (state.KeepRunning()) {
		uint16_t* input = fp16.data();
		benchmark::DoNotOptimize(input);

		float* output = fp32.data();
		const size_t n = state.range(0);
		for (size_t i = 0; i < n; i++) {
			output[i] = fp16_alt_to_fp32_value(input[i]);
		}

		benchmark::DoNotOptimize(output);
	}
	state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
}
BENCHMARK(fp16_alt_to_fp32_value)->RangeMultiplier(2)->Range(1<<10, 64<<20);

#ifndef EMSCRIPTEN
	static void fp16_alt_to_fp32_psimd(benchmark::State& state) {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_int_distribution<uint16_t>(0, 0x7BFF), std::mt19937(seed));

		std::vector<uint16_t> fp16(state.range(0));
		std::vector<float> fp32(state.range(0));
		std::generate(fp16.begin(), fp16.end(),
			[&rng]{ return fp16_alt_from_fp32_value(rng()); });

		while (state.KeepRunning()) {
			uint16_t* input = fp16.data();
			benchmark::DoNotOptimize(input);

			float* output = fp32.data();
			const size_t n = state.range(0);
			for (size_t i = 0; i < n - 4; i += 4) {
				psimd_store_f32(&output[i],
					fp16_alt_to_fp32_psimd(
						psimd_load_u16(&input[i])));
			}
			const psimd_u16 last_vector = { input[n - 4], input[n - 3], input[n - 2], input[n - 1] };
			psimd_store_f32(&output[n - 4],
				fp16_alt_to_fp32_psimd(last_vector));

			benchmark::DoNotOptimize(output);
		}
		state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
	}
	BENCHMARK(fp16_alt_to_fp32_psimd)->RangeMultiplier(2)->Range(1<<10, 64<<20);

	static void fp16_alt_to_fp32x2_psimd(benchmark::State& state) {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_int_distribution<uint16_t>(0, 0x7BFF), std::mt19937(seed));

		std::vector<uint16_t> fp16(state.range(0));
		std::vector<float> fp32(state.range(0));
		std::generate(fp16.begin(), fp16.end(),
			[&rng]{ return fp16_alt_from_fp32_value(rng()); });

		while (state.KeepRunning()) {
			uint16_t* input = fp16.data();
			benchmark::DoNotOptimize(input);

			float* output = fp32.data();
			const size_t n = state.range(0);
			for (size_t i = 0; i < n; i += 8) {
				const psimd_f32x2 data =
					fp16_alt_to_fp32x2_psimd(
						psimd_load_u16(&input[i]));
				psimd_store_f32(&output[i], data.lo);
				psimd_store_f32(&output[i + 4], data.hi);
			}

			benchmark::DoNotOptimize(output);
		}
		state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
	}
	BENCHMARK(fp16_alt_to_fp32x2_psimd)->RangeMultiplier(2)->Range(1<<10, 64<<20);
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
				/* Disable flush-to-zero (bit 24) and enable Alternative FP16 format (bit 26) */
				__builtin_aarch64_set_fpcr((fpcr & 0xFEFFFFFFu) | 0x08000000u);
			#else
				unsigned int fpscr;
				__asm__ __volatile__ ("VMRS %[fpscr], fpscr" : [fpscr] "=r" (fpscr));
				/* Disable flush-to-zero (bit 24) and enable Alternative FP16 format (bit 26) */
				__asm__ __volatile__ ("VMSR fpscr, %[fpscr]" :
					: [fpscr] "r" ((fpscr & 0xFEFFFFFFu) | 0x08000000u));
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

BENCHMARK_MAIN();
