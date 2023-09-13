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


static void fp16_alt_from_fp32_value(benchmark::State& state) {
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
			output[i] = fp16_alt_from_fp32_value(input[i]);
		}

		benchmark::DoNotOptimize(output);
	}
	state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
}
BENCHMARK(fp16_alt_from_fp32_value)->RangeMultiplier(2)->Range(1<<10, 64<<20);

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

BENCHMARK_MAIN();
