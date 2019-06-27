#pragma once

#include "versions.hpp"
#include <cstdint>
#include <type_traits>
#include <iostream>

namespace simd {

	template < class T, size_t W >
	struct native_vector;

#if SIMD_SUPPORTS(SIMD_SSE)
	template <>
	struct native_vector<float, 4> {
		static constexpr int required_version = SIMD_SSE;
		using native_type = __m128;
	};
#endif // SIMD_SUPPORTS(SIMD_SSE)

#if SIMD_SUPPORTS(SIMD_SSE2)
	template <>
	struct native_vector<double, 2> {
		static constexpr int required_version = SIMD_SSE2;
		using native_type = __m128d;
	};

	template <>
	struct native_vector<std::int32_t, 4> {
		static constexpr int required_version = SIMD_SSE2;
		using native_type = __m128i;
	};

	template <>
	struct native_vector<std::int64_t, 2> {
		static constexpr int required_version = SIMD_SSE2;
		using native_type = __m128i;
	};
#endif // SIMD_SUPPORTS(SIMD_SSE2)

#if SIMD_SUPPORTS(SIMD_AVX)
	template <>
	struct native_vector<float, 8> {
		static constexpr int required_version = SIMD_AVX;
		using native_type = __m256;
	};

	template <>
	struct native_vector<double, 4> {
		static constexpr int required_version = SIMD_AVX;
		using native_type = __m256d;
	};
#endif // SIMD_SUPPORTS(SIMD_AVX)

#if SIMD_SUPPORTS(SIMD_AVX2)

	template <>
	struct native_vector<std::int32_t, 8> {
		static constexpr int required_version = SIMD_AVX2;
		using native_type = __m256i;
	};

	template <>
	struct native_vector<std::int64_t, 4> {
		static constexpr int required_version = SIMD_AVX2;
		using native_type = __m256i;
	};
#endif // SIMD_SUPPORTS(SIMD_AVX2)

} // namespace simd