#pragma once

#include "base_types.hpp"
#include "simd.hpp"
#include "util.hpp"
#include <array>
#include <assert.h>
#include <exception>

namespace simd {

#if SIMD_SUPPORTS(SIMD_SSE)
	using float32x4 = vector<float, 4>;

	template <>
	class vector<float, 4> : public vector_base<float, 4> {
	public:
		static constexpr int required_version = native_vector<type, width>::required_version;

		SIMD_FORCEINLINE vector() : vector_base() {}
		explicit SIMD_FORCEINLINE vector(native_type v) : vector_base(v) {}
		explicit SIMD_FORCEINLINE vector(type f) : vector_base(_mm_set1_ps(f)) {}
		explicit SIMD_FORCEINLINE vector(type f1, type f2, type f3, type f4) : vector_base(_mm_set_ps(f4, f3, f2, f1)) {}
		explicit SIMD_FORCEINLINE vector(const type *vals) : vector_base(_mm_loadu_ps(vals)) {}
		explicit SIMD_FORCEINLINE vector(const type *vals, aligned_load) : vector_base(_mm_load_ps(vals)) {}
		explicit SIMD_FORCEINLINE vector(const vector<int, 4> &v);
		explicit SIMD_FORCEINLINE vector(const vector<double, 2> &v);
#if SIMD_SUPPORTS(SIMD_AVX)
		explicit SIMD_FORCEINLINE vector(const vector<double, 4> &v);
#endif // SIMD_SUPPORTS(SIMD_AVX)


		SIMD_FORCEINLINE vector<float, 4> &operator+=(const vector<float, 4> &v);
		SIMD_FORCEINLINE vector<float, 4> &operator-=(const vector<float, 4> &v);
		SIMD_FORCEINLINE vector<float, 4> &operator*=(const vector<float, 4> &v);
		SIMD_FORCEINLINE vector<float, 4> &operator/=(const vector<float, 4> &v);
		friend SIMD_FORCEINLINE vector<float, 4> operator+(vector<float, 4> v1, const vector<float, 4> &v2);
		friend SIMD_FORCEINLINE vector<float, 4> operator-(vector<float, 4> v1, const vector<float, 4> &v2);
		friend SIMD_FORCEINLINE vector<float, 4> operator*(vector<float, 4> v1, const vector<float, 4> &v2);
		friend SIMD_FORCEINLINE vector<float, 4> operator/(vector<float, 4> v1, const vector<float, 4> &v2);

		SIMD_FORCEINLINE vector<float, 4> &operator&=(const vector<float, 4> &v);
		SIMD_FORCEINLINE vector<float, 4> &operator|=(const vector<float, 4> &v);
		SIMD_FORCEINLINE vector<float, 4> &operator^=(const vector<float, 4> &v);
		friend SIMD_FORCEINLINE vector<float, 4> operator~(const vector<float, 4> &v);
		friend SIMD_FORCEINLINE vector<float, 4> operator&(vector<float, 4> v1, const vector<float, 4> &v2);
		friend SIMD_FORCEINLINE vector<float, 4> operator|(vector<float, 4> v1, const vector<float, 4> &v2);
		friend SIMD_FORCEINLINE vector<float, 4> operator^(vector<float, 4> v1, const vector<float, 4> &v2);

		SIMD_FORCEINLINE int get_mask() const;

		friend SIMD_FORCEINLINE vector<float, 4> operator==(const vector<float, 4> &v1, const vector<float, 4> &v2);
		friend SIMD_FORCEINLINE vector<float, 4> operator!=(const vector<float, 4> &v1, const vector<float, 4> &v2);
		friend SIMD_FORCEINLINE vector<float, 4> operator>(const vector<float, 4> &v1, const vector<float, 4> &v2);
		friend SIMD_FORCEINLINE vector<float, 4> operator>=(const vector<float, 4> &v1, const vector<float, 4> &v2);
		friend SIMD_FORCEINLINE vector<float, 4> operator<(const vector<float, 4> &v1, const vector<float, 4> &v2);
		friend SIMD_FORCEINLINE vector<float, 4> operator<=(const vector<float, 4> &v1, const vector<float, 4> &v2);

		SIMD_FORCEINLINE vector<float, 4> &hadd(const vector<float, 4> &v);
		SIMD_FORCEINLINE vector<float, 4> &hsub(const vector<float, 4> &v);
		SIMD_FORCEINLINE type hadd() const;
		SIMD_FORCEINLINE type hsub() const;
		friend SIMD_FORCEINLINE vector<float, 4> hadd(vector<float, 4> v1, const vector<float, 4> &v2);
		friend SIMD_FORCEINLINE vector<float, 4> hsub(vector<float, 4> v1, const vector<float, 4> &v2);

		SIMD_FORCEINLINE vector<float, 4> &abs();
		friend SIMD_FORCEINLINE vector<float, 4> abs(vector<float, 4> v);
		friend SIMD_FORCEINLINE vector<float, 4> min(const vector<float, 4> &v1, const vector<float, 4> &v2);
		friend SIMD_FORCEINLINE vector<float, 4> max(const vector<float, 4> &v1, const vector<float, 4> &v2);
		SIMD_FORCEINLINE vector<float, 4> &ceil();
		SIMD_FORCEINLINE vector<float, 4> &floor();
		SIMD_FORCEINLINE vector<float, 4> &round(int mode);
		friend SIMD_FORCEINLINE vector<float, 4> ceil(vector<float, 4> v);
		friend SIMD_FORCEINLINE vector<float, 4> floor(vector<float, 4> v);
		friend SIMD_FORCEINLINE vector<float, 4> round(vector<float, 4> v, int mode);


		SIMD_FORCEINLINE vector<float, 4> sqrt() const;
		SIMD_FORCEINLINE vector<float, 4> rsqrt() const;

		friend SIMD_FORCEINLINE vector<float, 4> select(const vector<float, 4> &v, const vector<float, 4> &alt, const mask<float, 4> &condition);
		friend SIMD_FORCEINLINE vector<float, 4> select(const vector<float, 4> &v, const vector<float, 4> &alt, const vector<float, 4> &condition);

		friend std::ostream &operator<<(std::ostream &stream, const vector<float, 4> &v);
	};

	template <>
	class mask<float, 4> : public vector_base<float, 4> {
	public:
		SIMD_FORCEINLINE mask() : vector_base() {}
		explicit SIMD_FORCEINLINE mask(native_type v) : vector_base(v) {}
		explicit SIMD_FORCEINLINE mask(bool b) : vector_base(_mm_castsi128_ps(_mm_set1_epi32(-static_cast<int>(b)))) {}
		explicit SIMD_FORCEINLINE mask(bool b1, bool b2, bool b3, bool b4) : vector_base(_mm_castsi128_ps(_mm_set_epi32(
			-static_cast<int>(b4), -static_cast<int>(b3), -static_cast<int>(b2), -static_cast<int>(b1)))) {}
		explicit SIMD_FORCEINLINE mask(const vector<float, 4> &v) : vector_base(v) {}

		friend SIMD_FORCEINLINE mask<float, 4> operator==(const mask<float, 4> &v1, const mask<float, 4> &v2);
		friend SIMD_FORCEINLINE mask<float, 4> operator!=(const mask<float, 4> &v1, const mask<float, 4> &v2);

		SIMD_FORCEINLINE mask<float, 4> &operator&=(const mask<float, 4> &v);
		SIMD_FORCEINLINE mask<float, 4> &operator|=(const mask<float, 4> &v);
		SIMD_FORCEINLINE mask<float, 4> &operator^=(const mask<float, 4> &v);

		friend SIMD_FORCEINLINE mask<float, 4> operator~(const mask<float, 4> &v);
		friend SIMD_FORCEINLINE mask<float, 4> operator&(mask<float, 4> v1, const mask<float, 4> &v2);
		friend SIMD_FORCEINLINE mask<float, 4> operator|(mask<float, 4> v1, const mask<float, 4> &v2);
		friend SIMD_FORCEINLINE mask<float, 4> operator^(mask<float, 4> v1, const mask<float, 4> &v2);
		friend SIMD_FORCEINLINE mask<float, 4> andnot(const mask<float, 4> &v1, const mask<float, 4> &v2);

		SIMD_FORCEINLINE int get_mask() const;
		SIMD_FORCEINLINE bool all() const;
		SIMD_FORCEINLINE bool any() const;
		SIMD_FORCEINLINE bool none() const;
	};

	vector<float, 4> &vector<float, 4>::operator+=(const vector<float, 4> &v) {
		m_vec = _mm_add_ps(m_vec, v.m_vec);
		return *this;
	}

	vector<float, 4> &vector<float, 4>::operator-=(const vector<float, 4> &v) {
		m_vec = _mm_sub_ps(m_vec, v.m_vec);
		return *this;
	}

	vector<float, 4> &vector<float, 4>::operator*=(const vector<float, 4> &v) {
		m_vec = _mm_mul_ps(m_vec, v.m_vec);
		return *this;
	}

	vector<float, 4> &vector<float, 4>::operator/=(const vector<float, 4> &v) {
		m_vec = _mm_div_ps(m_vec, v.m_vec);
		return *this;
	}

	vector<float, 4> &vector<float, 4>::hadd(const vector<float, 4> &v) {
#if SIMD_SUPPORTS(SIMD_SSE3)
		m_vec = _mm_hadd_ps(m_vec, v.m_vec);
#else // SIMD_SUPPORTS(SIMD_SSE3)
		throw std::runtime_error("Operation not implemented");
#endif // SIMD_SUPPORTS(SIMD_SSE3)
		return *this;
	}

	vector<float, 4> &vector<float, 4>::hsub(const vector<float, 4> &v) {
#if SIMD_SUPPORTS(SIMD_SSE3)
		m_vec = _mm_hsub_ps(m_vec, v.m_vec);
#else // SIMD_SUPPORTS(SIMD_SSE3)
		throw std::runtime_error("Operation not implemented");
#endif // SIMD_SUPPORTS(SIMD_SSE3)
		return *this;
	}

	float vector<float, 4>::hadd() const {
#if SIMD_SUPPORTS(SIMD_SSE3)
		auto t1 = _mm_hadd_ps(m_vec, m_vec);
		return _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
#else // SIMD_SUPPORTS(SIMD_SSE3)
			// Compute A1+A3, A2+A4, ...
		auto t1 = _mm_add_ps(m_vec, _mm_movehl_ps(m_vec, m_vec));
		// Compute A1+A2 w. shuffle
		auto t2 = _mm_add_ps(t1, _mm_shuffle_ps(t1, t1, 0b00000001));
		return _mm_cvtss_f32(t2);
#endif // SIMD_SUPPORTS(SIMD_SSE3)
	}

	float vector<float, 4>::hsub() const {
		throw std::runtime_error("Operation not implemented");
	}

	vector<float, 4> hadd(vector<float, 4> v1, const vector<float, 4> &v2) {
		return v1.hadd(v2);
	}

	vector<float, 4> hsub(vector<float, 4> v1, const vector<float, 4> &v2) {
		return v1.hsub(v2);
	}

	vector<float, 4> operator+(vector<float, 4> v1, const vector<float, 4> &v2) {
		return (v1 += v2);
	}

	vector<float, 4> operator-(vector<float, 4> v1, const vector<float, 4> &v2) {
		return (v1 -= v2);
	}

	vector<float, 4> operator*(vector<float, 4> v1, const vector<float, 4> &v2) {
		return (v1 *= v2);
	}

	vector<float, 4> operator/(vector<float, 4> v1, const vector<float, 4> &v2) {
		return (v1 /= v2);
	}

	vector<float, 4> &vector<float, 4>::operator&=(const vector<float, 4> &v) {
		m_vec = _mm_and_ps(m_vec, v.m_vec);
		return *this;
	}

	vector<float, 4> &vector<float, 4>::operator|=(const vector<float, 4> &v) {
		m_vec = _mm_or_ps(m_vec, v.m_vec);
		return *this;
	}

	vector<float, 4> &vector<float, 4>::operator^=(const vector<float, 4> &v) {
		m_vec = _mm_xor_ps(m_vec, v.m_vec);
		return *this;
	}

	vector<float, 4> operator~(const vector<float, 4> &v) {
		return vector<float, 4>(_mm_xor_ps(v.m_vec, _mm_castsi128_ps(_mm_set1_epi32(-1))));
	}

	vector<float, 4> operator&(vector<float, 4> v1, const vector<float, 4> &v2) {
		return v1 &= v2;
	}

	vector<float, 4> operator|(vector<float, 4> v1, const vector<float, 4> &v2) {
		return v1 |= v2;
	}

	vector<float, 4> operator^(vector<float, 4> v1, const vector<float, 4> &v2) {
		return v1 ^= v2;
	}

	int vector<float, 4>::get_mask() const {
		return _mm_movemask_ps(m_vec);
	}

	vector<float, 4> operator==(const vector<float, 4> &v1, const vector<float, 4> &v2) {
		return vector<float, 4>(_mm_cmpeq_ps(v1.m_vec, v2.m_vec));
	}

	vector<float, 4> operator!=(const vector<float, 4> &v1, const vector<float, 4> &v2) {
		return vector<float, 4>(_mm_cmpneq_ps(v1.m_vec, v2.m_vec));
	}

	vector<float, 4> operator>(const vector<float, 4> &v1, const vector<float, 4> &v2) {
		return vector<float, 4>(_mm_cmpgt_ps(v1.m_vec, v2.m_vec));
	}

	vector<float, 4> operator>=(const vector<float, 4> &v1, const vector<float, 4> &v2) {
		return vector<float, 4>(_mm_cmpge_ps(v1.m_vec, v2.m_vec));
	}

	vector<float, 4> operator<(const vector<float, 4> &v1, const vector<float, 4> &v2) {
		return vector<float, 4>(_mm_cmplt_ps(v1.m_vec, v2.m_vec));
	}

	vector<float, 4> operator<=(const vector<float, 4> &v1, const vector<float, 4> &v2) {
		return vector<float, 4>(_mm_cmple_ps(v1.m_vec, v2.m_vec));
	}

	vector<float, 4> &vector<float, 4>::abs() {
		// TODO: benchmark for most performant solution
		m_vec = _mm_max_ps(_mm_sub_ps(_mm_setzero_ps(), m_vec), m_vec);
		return *this;
	}

	vector<float, 4> abs(vector<float, 4> v) {
		return v.abs();
	}

	vector<float, 4> min(const vector<float, 4> &v1, const vector<float, 4> &v2) {
		return vector<float, 4>(_mm_min_ps(v1.m_vec, v2.m_vec));
	}

	vector<float, 4> max(const vector<float, 4> &v1, const vector<float, 4> &v2) {
		return vector<float, 4>(_mm_max_ps(v1.m_vec, v2.m_vec));
	}

	vector<float, 4> vector<float, 4>::sqrt() const {
		return vector<float, 4>(_mm_sqrt_ps(m_vec));
	}

	vector<float, 4> vector<float, 4>::rsqrt() const {
		return vector<float, 4>(_mm_rsqrt_ps(m_vec));
	}

	vector<float, 4> &vector<float, 4>::ceil() {
#if SIMD_SUPPORTS(SIMD_SSE4_1)
		m_vec = _mm_ceil_ps(m_vec);
#else // SIMD_SUPPORTS(SIMD_SSE4_1)
		throw std::runtime_error("Operation not implemented");
#endif // SIMD_SUPPORTS(SIMD_SSE4_1)
		return *this;
	}

	vector<float, 4> &vector<float, 4>::floor() {
#if SIMD_SUPPORTS(SIMD_SSE4_1)
		m_vec = _mm_floor_ps(m_vec);
#else // SIMD_SUPPORTS(SIMD_SSE4_1)
		throw std::runtime_error("Operation not implemented");
#endif // SIMD_SUPPORTS(SIMD_SSE4_1)
		return *this;
	}

	vector<float, 4> &vector<float, 4>::round(int mode) {
#if SIMD_SUPPORTS(SIMD_SSE4_1)
		// TODO: requires constant expression?
		m_vec = _mm_round_ps(m_vec, 0);
#else // SIMD_SUPPORTS(SIMD_SSE4_1)
		throw std::runtime_error("Operation not implemented");
#endif // SIMD_SUPPORTS(SIMD_SSE4_1)
		return *this;
	}

	vector<float, 4> ceil(vector<float, 4> v) {
		return v.ceil();
	}

	vector<float, 4> floor(vector<float, 4> v) {
		return v.floor();
	}

	vector<float, 4> round(vector<float, 4> v, int mode) {
		return v.round(mode);
	}

	vector<float, 4> select(const vector<float, 4> &v, const vector<float, 4> &alt, const mask<float, 4> &condition) {
#if SIMD_SUPPORTS(SIMD_SSE4_1)
		return vector<float, 4>(_mm_blendv_ps(alt.m_vec, v.m_vec, condition.native()));
#else // SIMD_SUPPORTS(SIMD_SSE4_1)
		// TODO: only works when condition has either all or none set!
		return vector<float, 4>(_mm_or_ps(_mm_and_ps(v.m_vec, condition.native()), _mm_andnot_ps(alt.m_vec, condition.native())));
#endif // SIMD_SUPPORTS(SIMD_SSE4_1)
	}

	vector<float, 4> select(const vector<float, 4> &v, const vector<float, 4> &alt, const vector<float, 4> &condition) {
#if SIMD_SUPPORTS(SIMD_SSE4_1)
		return vector<float, 4>(_mm_blendv_ps(alt.m_vec, v.m_vec, condition.m_vec));
#else // SIMD_SUPPORTS(SIMD_SSE4_1)
		// TODO: only works when condition has either all or none set!
		return vector<float, 4>(_mm_or_ps(_mm_and_ps(v.m_vec, condition.m_vec), _mm_andnot_ps(alt.m_vec, condition.m_vec)));
#endif // SIMD_SUPPORTS(SIMD_SSE4_1)
	}

	std::ostream &operator<<(std::ostream &stream, const vector<float, 4> &v) {
		stream << '(';
		for (size_t i = 0; i < v.width - 1; ++i) {
			stream << v.m_array[i] << ' ';
		}
		stream << v.m_array[v.width - 1] << ')';
		return stream;
	}

	mask<float, 4> operator==(const mask<float, 4> &v1, const mask<float, 4> &v2) {
		return mask<float, 4>(_mm_cmpeq_ps(v1.m_vec, v2.m_vec));
	}

	mask<float, 4> operator!=(const mask<float, 4> &v1, const mask<float, 4> &v2) {
		return mask<float, 4>(_mm_cmpeq_ps(v1.m_vec, v2.m_vec));
	}

	mask<float, 4> &mask<float, 4>::operator&=(const mask<float, 4> &v) {
		m_vec = _mm_and_ps(m_vec, v.m_vec);
		return *this;
	}

	mask<float, 4> &mask<float, 4>::operator|=(const mask<float, 4> &v) {
		m_vec = _mm_or_ps(m_vec, v.m_vec);
		return *this;
	}

	mask<float, 4> &mask<float, 4>::operator^=(const mask<float, 4> &v) {
		m_vec = _mm_xor_ps(m_vec, v.m_vec);
		return *this;
	}

	mask<float, 4> operator~(const mask<float, 4> &v) {
		return mask<float, 4>(_mm_xor_ps(v.m_vec, _mm_castsi128_ps(_mm_set1_epi32(-1))));
	}

	mask<float, 4> operator&(mask<float, 4> v1, const mask<float, 4> &v2) {
		return v1 &= v2;
	}

	mask<float, 4> operator|(mask<float, 4> v1, const mask<float, 4> &v2) {
		return v1 |= v2;
	}

	mask<float, 4> operator^(mask<float, 4> v1, const mask<float, 4> &v2) {
		return v1 ^= v2;
	}

	mask<float, 4> andnot(const mask<float, 4> &v1, const mask<float, 4> &v2) {
		return mask<float, 4>(_mm_andnot_ps(v1.m_vec, v2.m_vec));
	}

	int mask<float, 4>::get_mask() const {
		return _mm_movemask_ps(m_vec);
	}

	bool mask<float, 4>::all() const {
		// TODO: what is faster here?
		return _mm_movemask_ps(m_vec) == 0b1111;
	}

	bool mask<float, 4>::any() const {
		return _mm_movemask_ps(m_vec);
	}

	bool mask<float, 4>::none() const {
		return !_mm_movemask_ps(m_vec);
	}

#endif // SIMD_SUPPORTS(SIMD_SSE)

} // namespace simd