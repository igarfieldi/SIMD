#pragma once

#include <iostream>

#define SIMD_NONE 0
#define SIMD_SSE 10
#define SIMD_SSE2 20
#define SIMD_SSE3 30
#define SIMD_SSSE3 31
#define SIMD_SSE4_1 41
#define SIMD_SSE4_2 42
#define SIMD_AVX 50
#define SIMD_FMA3 51
#define SIMD_AVX2 60

#ifdef _MSC_VER
#include <intrin.h>
#if defined(__AVX2__)
	#define SIMD_SSE_VERSION SIMD_AVX2
#elif defined(__AVX__)
	#define SIMD_SSE_VERSION SIMD_AVX
#elif (_M_IX86_FP == 2) || defined(_M_X64) || defined(_M_AMD64)
	#define SIMD_SSE_VERSION SIMD_SSE2
#elif _M_IX86_FP == 1
	#define SIMD_SSE_VERSION SIMD_SSE
#else 
	#define SIMD_SSE_VERSION SIMD_NONE
#endif
#else // _MSVC_VER
#include <x86intrin.h>
#include <cpuid.h>
#if defined(__AVX2__)
	#define SIMD_SSE_VERSION SIMD_AVX2
#elif defined(__FMA__)
	#define SIMD_SSE_VERSION SIMD_FMA3
#elif defined(__AVX__)
	#define SIMD_SSE_VERSION SIMD_AVX
#elif defined(__SSE4_2__)
	#define SIMD_SSE_VERSION SIMD_SSE4_2
#elif defined(__SSE4_1__)
	#define SIMD_SSE_VERSION SIMD_SSE4_1
#elif defined(__SSSE3__)
	#define SIMD_SSE_VERSION SIMD_SSSE3
#elif defined(__SSE3__)
	#define SIMD_SSE_VERSION SIMD_SSE3
#elif defined(__SSE2__)
	#define SIMD_SSE_VERSION SIMD_SSE2
#elif defined(__SSE__)
	#define SIMD_SSE_VERSION SIMD_SSE
#else
	#define SIMD_SSE_VERSION SIMD_NONE
#endif
#endif // _MSVC_VER

#define SIMD_SUPPORTS(ver) SIMD_SSE_VERSION >= ver

namespace simd {

	constexpr bool supports(int version) {
		return SIMD_SUPPORTS(version);
	}

	constexpr int sse_compile_version() {
		return SIMD_SSE_VERSION;
	}

	int sse_runtime_version() {
	#ifdef _MSC_VER
		int cpuInfo[4];
		__cpuid(cpuInfo, 0);
		int id_count = cpuInfo[0];
		
		if(id_count >= 7) {
			__cpuid(cpuInfo, 7);
			if(cpuInfo[1] & (1 << 5))
				return SIMD_AVX2;
		}
		if(id_count >= 1) {
			__cpuid(cpuInfo, 1);
			if(cpuInfo[2] & (1 << 12))
				return SIMD_FMA3;
			if(cpuInfo[2] & (1 << 28))
				return SIMD_AVX;
			if(cpuInfo[2] & (1 << 20))
				return SIMD_SSE4_2;
			if(cpuInfo[2] & (1 << 19))
				return SIMD_SSE4_1;
			if(cpuInfo[2] & (1 << 9))
				return SIMD_SSSE3;
			if(cpuInfo[2] & (1 << 0))
				return SIMD_SSE3;
			if(cpuInfo[3] & (1 << 26))
				return SIMD_SSE2;
			if(cpuInfo[3] & (1 << 25))
				return SIMD_SSE;
		}
	#else // _MSC_VER
		unsigned int eax, ebx, ecx, edx;
		unsigned int id_count = __get_cpuid_max(0, nullptr);
		
		if(id_count >= 7) {
			__asm__ ("mov $0, %%ecx\n\t"
				"cpuid\n\t"
				: "=b"(ebx)
				: "0"(7)
				: "eax", "ecx", "edx");
			if(ebx & bit_AVX2)
				return SIMD_AVX2;
		}
		if(id_count >= 1) {
			__asm__ ("cpuid\n\t"
				: "=c"(ecx), "=d"(edx)
				: "0"(7)
				: "eax", "ebx");
			if(ecx & bit_FMA)
				return SIMD_FMA3;
			if(ecx & bit_AVX)
				return SIMD_AVX;
			if(ecx & bit_SSE4_2)
				return SIMD_SSE4_2;
			if(ecx & bit_SSE4_1)
				return SIMD_SSE4_1;
			if(ecx & bit_SSSE3)
				return SIMD_SSSE3;
			if(ecx & bit_SSE3)
				return SIMD_SSE3;
			if(edx & bit_SSE2)
				return SIMD_SSE2;
			if(edx & bit_SSE)
				return SIMD_SSE;
		}
	#endif // _MSC_VER
		return SIMD_NONE;
	}

	constexpr const char *version_name(int version) {
		switch(version) {
		case SIMD_SSE: return "SSE";
		case SIMD_SSE2: return "SSE2";
		case SIMD_SSE3: return "SSE3";
		case SIMD_SSSE3: return "SSSE3";
		case SIMD_SSE4_1: return "SSE4.1";
		case SIMD_SSE4_2: return "SSE4.2";
		case SIMD_AVX: return "AVX";
		case SIMD_AVX2: return "AVX2";
		case SIMD_FMA3: return "FMA3";
		default: return "none";
		}
	}

} // namespace simd