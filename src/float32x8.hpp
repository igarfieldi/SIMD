#pragma once

#include "base_types.hpp"
#include "simd.hpp"
#include "util.hpp"
#include <array>
#include <assert.h>
#include <exception>

namespace simd {

#if SIMD_SUPPORTS(SIMD_AVX)
	using float32x8 = vector<float, 8>;

	template <>
	class vector<float, 8> : public vector_base<float, 8> {
	public:
		static constexpr int required_version = native_vector<type, width>::required_version;

	public:
		SIMD_FORCEINLINE vector() : vector_base() {}
		explicit SIMD_FORCEINLINE vector(native_type v) : vector_base(v) {}
		explicit SIMD_FORCEINLINE vector(type f) : vector_base(_mm256_set1_ps(f)) {}
		explicit SIMD_FORCEINLINE vector(const type *vals) : vector_base(_mm256_loadu_ps(vals)) {}
		explicit SIMD_FORCEINLINE vector(const type *vals, aligned_load) : vector_base(_mm256_load_ps(vals)) {}
#if SIMD_SUPPORTS(SIMD_AVX2)
		explicit SIMD_FORCEINLINE vector(const vector<int, 8> &v);
#endif // SIMD_SUPPORTS(SIMD_AVX2)

		SIMD_FORCEINLINE vector<float, 8> &operator+=(const vector<float, 8> &v);
		SIMD_FORCEINLINE vector<float, 8> &operator-=(const vector<float, 8> &v);
		SIMD_FORCEINLINE vector<float, 8> &operator*=(const vector<float, 8> &v);
		SIMD_FORCEINLINE vector<float, 8> &operator/=(const vector<float, 8> &v);
		friend SIMD_FORCEINLINE vector<float, 8> operator+(vector<float, 8> v1, const vector<float, 8> &v2);
		friend SIMD_FORCEINLINE vector<float, 8> operator-(vector<float, 8> v1, const vector<float, 8> &v2);
		friend SIMD_FORCEINLINE vector<float, 8> operator*(vector<float, 8> v1, const vector<float, 8> &v2);
		friend SIMD_FORCEINLINE vector<float, 8> operator/(vector<float, 8> v1, const vector<float, 8> &v2);

		SIMD_FORCEINLINE vector<float, 8> &operator&=(const vector<float, 8> &v);
		SIMD_FORCEINLINE vector<float, 8> &operator|=(const vector<float, 8> &v);
		SIMD_FORCEINLINE vector<float, 8> &operator^=(const vector<float, 8> &v);
		friend SIMD_FORCEINLINE vector<float, 8> operator~(const vector<float, 8> &v);
		friend SIMD_FORCEINLINE vector<float, 8> operator&(vector<float, 8> v1, const vector<float, 8> &v2);
		friend SIMD_FORCEINLINE vector<float, 8> operator|(vector<float, 8> v1, const vector<float, 8> &v2);
		friend SIMD_FORCEINLINE vector<float, 8> operator^(vector<float, 8> v1, const vector<float, 8> &v2);

		friend SIMD_FORCEINLINE vector<float, 8> operator==(const vector<float, 8> &v1, const vector<float, 8> &v2);
		friend SIMD_FORCEINLINE vector<float, 8> operator!=(const vector<float, 8> &v1, const vector<float, 8> &v2);
		friend SIMD_FORCEINLINE vector<float, 8> operator>(const vector<float, 8> &v1, const vector<float, 8> &v2);
		friend SIMD_FORCEINLINE vector<float, 8> operator>=(const vector<float, 8> &v1, const vector<float, 8> &v2);
		friend SIMD_FORCEINLINE vector<float, 8> operator<(const vector<float, 8> &v1, const vector<float, 8> &v2);
		friend SIMD_FORCEINLINE vector<float, 8> operator<=(const vector<float, 8> &v1, const vector<float, 8> &v2);

		SIMD_FORCEINLINE vector<float, 8> &hadd(const vector<float, 8> &v);
		SIMD_FORCEINLINE vector<float, 8> &hsub(const vector<float, 8> &v);
		SIMD_FORCEINLINE type hadd() const;
		SIMD_FORCEINLINE type hsub() const;
		friend SIMD_FORCEINLINE vector<float, 8> hadd(vector<float, 8> v1, const vector<float, 8> &v2);
		friend SIMD_FORCEINLINE vector<float, 8> hsub(vector<float, 8> v1, const vector<float, 8> &v2);

		SIMD_FORCEINLINE vector<float, 8> &abs();
		friend SIMD_FORCEINLINE vector<float, 8> abs(vector<float, 8> v);
		friend SIMD_FORCEINLINE vector<float, 8> min(const vector<float, 8> &v1, const vector<float, 8> &v2);
		friend SIMD_FORCEINLINE vector<float, 8> max(const vector<float, 8> &v1, const vector<float, 8> &v2);
		SIMD_FORCEINLINE vector<float, 8> &ceil();
		SIMD_FORCEINLINE vector<float, 8> &floor();
		SIMD_FORCEINLINE vector<float, 8> &round(int mode);
		friend SIMD_FORCEINLINE vector<float, 8> ceil(vector<float, 8> v);
		friend SIMD_FORCEINLINE vector<float, 8> floor(vector<float, 8> v);
		friend SIMD_FORCEINLINE vector<float, 8> round(vector<float, 8> v, int mode);

		SIMD_FORCEINLINE vector<float, 8> sqrt() const;
		SIMD_FORCEINLINE vector<float, 8> rsqrt() const;

		friend SIMD_FORCEINLINE vector<float, 8> select(const vector<float, 8> &v, const vector<float, 8> &alt, const mask<float, 8> &condition);
		friend SIMD_FORCEINLINE vector<float, 8> select(const vector<float, 8> &v, const vector<float, 8> &alt, const vector<float, 8> &condition);

		friend std::ostream &operator<<(std::ostream &stream, const vector<float, 8> &v);
	};

	template <>
	class mask<float, 8> : public vector_base<float, 8> {
	public:
		SIMD_FORCEINLINE mask() : vector_base() {}
		explicit SIMD_FORCEINLINE mask(native_type v) : vector_base(v) {}
		explicit SIMD_FORCEINLINE mask(bool b) : vector_base(_mm256_castsi256_ps(_mm256_set1_epi32(-static_cast<int>(b)))) {}
		explicit SIMD_FORCEINLINE mask(bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7, bool b8) : vector_base(_mm256_castsi256_ps(_mm256_set_epi32(
			-static_cast<int>(b8), -static_cast<int>(b7), -static_cast<int>(b6), -static_cast<int>(b5),
			-static_cast<int>(b4), -static_cast<int>(b3), -static_cast<int>(b2), -static_cast<int>(b1)))) {}
		explicit SIMD_FORCEINLINE mask(const vector<float, 8> &v) : vector_base(v) {}

		friend SIMD_FORCEINLINE mask<float, 8> operator==(const mask<float, 8> &v1, const mask<float, 8> &v2);
		friend SIMD_FORCEINLINE mask<float, 8> operator!=(const mask<float, 8> &v1, const mask<float, 8> &v2);

		SIMD_FORCEINLINE mask<float, 8> &operator&=(const mask<float, 8> &v);
		SIMD_FORCEINLINE mask<float, 8> &operator|=(const mask<float, 8> &v);
		SIMD_FORCEINLINE mask<float, 8> &operator^=(const mask<float, 8> &v);

		friend SIMD_FORCEINLINE mask<float, 8> operator~(const mask<float, 8> &v);
		friend SIMD_FORCEINLINE mask<float, 8> operator&(mask<float, 8> v1, const mask<float, 8> &v2);
		friend SIMD_FORCEINLINE mask<float, 8> operator|(mask<float, 8> v1, const mask<float, 8> &v2);
		friend SIMD_FORCEINLINE mask<float, 8> operator^(mask<float, 8> v1, const mask<float, 8> &v2);
		friend SIMD_FORCEINLINE mask<float, 8> andnot(const mask<float, 8> &v1, const mask<float, 8> &v2);

		SIMD_FORCEINLINE int get_mask() const;
		SIMD_FORCEINLINE bool all() const;
		SIMD_FORCEINLINE bool any() const;
		SIMD_FORCEINLINE bool none() const;
	};

	vector<float, 8> &vector<float, 8>::operator+=(const vector<float, 8> &v) {
		m_vec = _mm256_add_ps(m_vec, v.m_vec);
		return *this;
	}

	vector<float, 8> &vector<float, 8>::operator-=(const vector<float, 8> &v) {
		m_vec = _mm256_sub_ps(m_vec, v.m_vec);
		return *this;
	}

	vector<float, 8> &vector<float, 8>::operator*=(const vector<float, 8> &v) {
		m_vec = _mm256_mul_ps(m_vec, v.m_vec);
		return *this;
	}

	vector<float, 8> &vector<float, 8>::operator/=(const vector<float, 8> &v) {
		m_vec = _mm256_div_ps(m_vec, v.m_vec);
		return *this;
	}

	vector<float, 8> &vector<float, 8>::hadd(const vector<float, 8> &v) {
		m_vec = _mm256_hadd_ps(m_vec, v.m_vec);
		return *this;
	}

	vector<float, 8> &vector<float, 8>::hsub(const vector<float, 8> &v) {
		m_vec = _mm256_hsub_ps(m_vec, v.m_vec);
		return *this;
	}

	float vector<float, 8>::hadd() const {
		// Results in A1+A2, A3+A4, ..., ..., A5+A6, A7+A8
		auto t1 = _mm256_hadd_ps(m_vec, m_vec);
		// Permute to get A5+A6, A7+A8, ....
		auto t2 = _mm256_permute2f128_ps(m_vec, m_vec, 0b000001);
		// Add first 4 elements
		auto t3 = _mm256_add_ps(t1, t2);
		// And last 4 elements
		auto t4 = _mm256_hadd_ps(t3, t3);

		// Grab the sums from the first entries
		return _mm256_cvtss_f32(t4);
	}

	float vector<float, 8>::hsub() const {
		throw std::runtime_error("Operation not implemented");
	}

	vector<float, 8> hadd(vector<float, 8> v1, const vector<float, 8> &v2) {
		return v1.hadd(v2);
	}

	vector<float, 8> hsub(vector<float, 8> v1, const vector<float, 8> &v2) {
		return v1.hsub(v2);
	}

	vector<float, 8> operator+(vector<float, 8> v1, const vector<float, 8> &v2) {
		return (v1 += v2);
	}

	vector<float, 8> operator-(vector<float, 8> v1, const vector<float, 8> &v2) {
		return (v1 -= v2);
	}

	vector<float, 8> operator*(vector<float, 8> v1, const vector<float, 8> &v2) {
		return (v1 *= v2);
	}

	vector<float, 8> operator/(vector<float, 8> v1, const vector<float, 8> &v2) {
		return (v1 /= v2);
	}

	vector<float, 8> &vector<float, 8>::operator&=(const vector<float, 8> &v) {
		m_vec = _mm256_and_ps(m_vec, v.m_vec);
		return *this;
	}

	vector<float, 8> &vector<float, 8>::operator|=(const vector<float, 8> &v) {
		m_vec = _mm256_or_ps(m_vec, v.m_vec);
		return *this;
	}

	vector<float, 8> &vector<float, 8>::operator^=(const vector<float, 8> &v) {
		m_vec = _mm256_xor_ps(m_vec, v.m_vec);
		return *this;
	}

	vector<float, 8> operator~(const vector<float, 8> &v) {
		return vector<float, 8>(_mm256_xor_ps(v.m_vec, _mm256_castsi256_ps(_mm256_set1_epi32(-1))));
	}

	vector<float, 8> operator&(vector<float, 8> v1, const vector<float, 8> &v2) {
		return v1 &= v2;
	}

	vector<float, 8> operator|(vector<float, 8> v1, const vector<float, 8> &v2) {
		return v1 |= v2;
	}

	vector<float, 8> operator^(vector<float, 8> v1, const vector<float, 8> &v2) {
		return v1 ^= v2;
	}

	vector<float, 8> operator==(const vector<float, 8> &v1, const vector<float, 8> &v2) {
		// TODO: what is the proper comparison flag?
		return vector<float, 8>(_mm256_cmp_ps(v1.m_vec, v2.m_vec, _CMP_EQ_OQ));
	}

	vector<float, 8> operator!=(const vector<float, 8> &v1, const vector<float, 8> &v2) {
		return vector<float, 8>(_mm256_cmp_ps(v1.m_vec, v2.m_vec, _CMP_NEQ_OQ));
	}

	vector<float, 8> operator>(const vector<float, 8> &v1, const vector<float, 8> &v2) {
		return vector<float, 8>(_mm256_cmp_ps(v1.m_vec, v2.m_vec, _CMP_GT_OQ));
	}

	vector<float, 8> operator>=(const vector<float, 8> &v1, const vector<float, 8> &v2) {
		return vector<float, 8>(_mm256_cmp_ps(v1.m_vec, v2.m_vec, _CMP_GE_OQ));
	}

	vector<float, 8> operator<(const vector<float, 8> &v1, const vector<float, 8> &v2) {
		return vector<float, 8>(_mm256_cmp_ps(v1.m_vec, v2.m_vec, _CMP_LT_OQ));
	}

	vector<float, 8> operator<=(const vector<float, 8> &v1, const vector<float, 8> &v2) {
		return vector<float, 8>(_mm256_cmp_ps(v1.m_vec, v2.m_vec, _CMP_LE_OQ));
	}

	vector<float, 8> &vector<float, 8>::abs() {
		// TODO: benchmark for most performant solution
		m_vec = _mm256_max_ps(_mm256_sub_ps(_mm256_setzero_ps(), m_vec), m_vec);
		return *this;
	}

	vector<float, 8> abs(vector<float, 8> v) {
		return v.abs();
	}

	vector<float, 8> min(const vector<float, 8> &v1, const vector<float, 8> &v2) {
		return vector<float, 8>(_mm256_min_ps(v1.m_vec, v2.m_vec));
	}

	vector<float, 8> max(const vector<float, 8> &v1, const vector<float, 8> &v2) {
		return vector<float, 8>(_mm256_max_ps(v1.m_vec, v2.m_vec));
	}

	vector<float, 8> vector<float, 8>::sqrt() const {
		return vector<float, 8>(_mm256_sqrt_ps(m_vec));
	}

	vector<float, 8> vector<float, 8>::rsqrt() const {
		return vector<float, 8>(_mm256_rsqrt_ps(m_vec));
	}

	vector<float, 8> &vector<float, 8>::ceil() {
		m_vec = _mm256_ceil_ps(m_vec);
		return *this;
	}

	vector<float, 8> &vector<float, 8>::floor() {
		m_vec = _mm256_floor_ps(m_vec);
		return *this;
	}

	vector<float, 8> &vector<float, 8>::round(int mode) {
		// TODO: requires constant expression?
		m_vec = _mm256_round_ps(m_vec, 0);
		return *this;
	}

	vector<float, 8> ceil(vector<float, 8> v) {
		return v.ceil();
	}

	vector<float, 8> floor(vector<float, 8> v) {
		return v.floor();
	}

	vector<float, 8> round(vector<float, 8> v, int mode) {
		return v.round(mode);
	}

	vector<float, 8> select(const vector<float, 8> &v, const vector<float, 8> &alt, const mask<float, 8> &condition) {
		return vector<float, 8>(_mm256_blendv_ps(alt.m_vec, v.m_vec, condition.native()));
	}

	vector<float, 8> select(const vector<float, 8> &v, const vector<float, 8> &alt, const vector<float, 8> &condition) {
		return vector<float, 8>(_mm256_blendv_ps(alt.m_vec, v.m_vec, condition.m_vec));
	}

	std::ostream &operator<<(std::ostream &stream, const vector<float, 8> &v) {
		stream << '(';
		for (size_t i = 0; i < v.width - 1; ++i) {
			stream << v.m_array[i] << ' ';
		}
		stream << v.m_array[v.width - 1] << ')';
		return stream;
	}

	mask<float, 8> operator==(const mask<float, 8> &v1, const mask<float, 8> &v2) {
		return mask<float, 8>(_mm256_cmp_ps(v1.m_vec, v2.m_vec, _CMP_EQ_OQ));
	}

	mask<float, 8> operator!=(const mask<float, 8> &v1, const mask<float, 8> &v2) {
		return mask<float, 8>(_mm256_cmp_ps(v1.m_vec, v2.m_vec, _CMP_NEQ_OQ));
	}

	mask<float, 8> &mask<float, 8>::operator&=(const mask<float, 8> &v) {
		m_vec = _mm256_and_ps(m_vec, v.m_vec);
		return *this;
	}

	mask<float, 8> &mask<float, 8>::operator|=(const mask<float, 8> &v) {
		m_vec = _mm256_or_ps(m_vec, v.m_vec);
		return *this;
	}

	mask<float, 8> &mask<float, 8>::operator^=(const mask<float, 8> &v) {
		m_vec = _mm256_xor_ps(m_vec, v.m_vec);
		return *this;
	}

	mask<float, 8> operator~(const mask<float, 8> &v) {
		return mask<float, 8>(_mm256_xor_ps(v.m_vec, _mm256_castsi256_ps(_mm256_set1_epi32(-1))));
	}

	mask<float, 8> operator&(mask<float, 8> v1, const mask<float, 8> &v2) {
		return v1 &= v2;
	}

	mask<float, 8> operator|(mask<float, 8> v1, const mask<float, 8> &v2) {
		return v1 |= v2;
	}

	mask<float, 8> operator^(mask<float, 8> v1, const mask<float, 8> &v2) {
		return v1 ^= v2;
	}

	mask<float, 8> andnot(const mask<float, 8> &v1, const mask<float, 8> &v2) {
		return mask<float, 8>(_mm256_andnot_ps(v1.m_vec, v2.m_vec));
	}

	int mask<float, 8>::get_mask() const {
		return _mm256_movemask_ps(m_vec);
	}

	bool mask<float, 8>::all() const {
		// TODO: what is faster here?
		return _mm256_movemask_ps(m_vec) & 0b11111111;
	}

	bool mask<float, 8>::any() const {
		return _mm256_movemask_ps(m_vec);
	}

	bool mask<float, 8>::none() const {
		return !_mm256_movemask_ps(m_vec);
	}

#endif // SIMD_SUPPORTS(SIMD_AVX)

} // namespace simd