#pragma once

#include "float32x4.hpp"
#include "float32x8.hpp"
#include "float64x2.hpp"
#include "float64x4.hpp"
#include "int32x4.hpp"
#include "int32x8.hpp"

namespace simd {

#if SIMD_SUPPORTS(SIMD_SSE2)
	vector<float, 4>::vector(const vector<int, 4> &v) : vector_base(_mm_cvtepi32_ps(v.native())) {}
	vector<float, 4>::vector(const vector<double, 2> &v) : vector_base(_mm_cvtpd_ps(v.native())) {}
	vector<double, 2>::vector(const vector<int, 4> &v) : vector_base(_mm_cvtepi32_pd(v.native())) {}
	vector<double, 2>::vector(const vector<float, 4> &v) : vector_base(_mm_cvtps_pd(v.native())) {}
	vector<int, 4>::vector(const vector<float, 4> &v) : vector_base(_mm_cvtps_epi32(v.native())) {}
	vector<int, 4>::vector(const vector<double, 2> &v) : vector_base(_mm_cvtpd_epi32(v.native())) {}
#endif // SIMD_SUPPORTS(SIMD_SSE2)

#if SIMD_SUPPORTS(SIMD_AVX)
	vector<float, 4>::vector(const vector<double, 4> &v) : vector_base(_mm256_cvtpd_ps(v.native())) {}
	vector<double, 4>::vector(const vector<int, 4> &v) : vector_base(_mm256_cvtepi32_pd(v.native())) {}
	vector<double, 4>::vector(const vector<float, 4> &v) : vector_base(_mm256_cvtps_pd(v.native())) {}
	vector<int, 4>::vector(const vector<double, 4> &v) : vector_base(_mm256_cvtpd_epi32(v.native())) {}
#endif // SIMD_SUPPORTS(SIMD_AVX)

#if SIMD_SUPPORTS(SIMD_AVX2)
	vector<float, 8>::vector(const vector<int, 8> &v) : vector_base(_mm256_cvtepi32_ps(v.native())) {}
	vector<int, 8>::vector(const vector<float, 8> & v) : vector_base(_mm256_cvtps_epi32(v.native())) {}
#endif // SUPPORTS(SIMD_AVX2)

} // namespace simd