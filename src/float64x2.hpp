#pragma once

#include <array>
#include <exception>
#include "base_types.hpp"
#include "simd.hpp"

namespace simd {

#if SIMD_SUPPORTS(SIMD_SSE2)
	using float64x2 = vector<double, 2>;

	template <>
	class vector<double, 2> : public vector_base<double, 2> {
	public:
		static constexpr int required_version = native_vector<type, width>::required_version;

	public:
		vector() : vector_base() {}
		explicit vector(native_type v) : vector_base(v) {}
		explicit vector(type f) : vector_base(_mm_set1_pd(f)) {}
		explicit vector(type f1, type f2) : vector_base(_mm_set_pd(f2, f1)) {}
		explicit vector(const std::array<type, width>& arr) : vector_base(_mm_set_pd(arr[1], arr[0])) {}
		explicit vector(const type *vals) : vector_base(_mm_loadu_pd(vals)) {}
		explicit vector(const type *vals, aligned_load) : vector_base(_mm_load_pd(vals)) {}
		explicit vector(const vector<int, 4> &v);
		explicit vector(const vector<float, 4> &v);

		vector<double, 2> &operator+=(const vector<double, 2> &v);
		vector<double, 2> &operator-=(const vector<double, 2> &v);
		vector<double, 2> &operator*=(const vector<double, 2> &v);
		vector<double, 2> &operator/=(const vector<double, 2> &v);
		friend vector<double, 2> operator+(vector<double, 2> v1, const vector<double, 2> &v2);
		friend vector<double, 2> operator-(vector<double, 2> v1, const vector<double, 2> &v2);
		friend vector<double, 2> operator*(vector<double, 2> v1, const vector<double, 2> &v2);
		friend vector<double, 2> operator/(vector<double, 2> v1, const vector<double, 2> &v2);

		vector<double, 2> &operator&=(const vector<double, 2> &v);
		vector<double, 2> &operator|=(const vector<double, 2> &v);
		vector<double, 2> &operator^=(const vector<double, 2> &v);
		friend vector<double, 2> operator~(const vector<double, 2> &v);
		friend vector<double, 2> operator&(vector<double, 2> v1, const vector<double, 2> &v2);
		friend vector<double, 2> operator|(vector<double, 2> v1, const vector<double, 2> &v2);
		friend vector<double, 2> operator^(vector<double, 2> v1, const vector<double, 2> &v2);

		friend vector<double, 2> operator==(const vector<double, 2> &v1, const vector<double, 2> &v2);
		friend vector<double, 2> operator!=(const vector<double, 2> &v1, const vector<double, 2> &v2);
		friend vector<double, 2> operator>(const vector<double, 2> &v1, const vector<double, 2> &v2);
		friend vector<double, 2> operator>=(const vector<double, 2> &v1, const vector<double, 2> &v2);
		friend vector<double, 2> operator<(const vector<double, 2> &v1, const vector<double, 2> &v2);
		friend vector<double, 2> operator<=(const vector<double, 2> &v1, const vector<double, 2> &v2);

		vector<double, 2> &hadd(const vector<double, 2> &v);
		vector<double, 2> &hsub(const vector<double, 2> &v);
		type hadd() const;
		type hsub() const;
		friend vector<double, 2> hadd(vector<double, 2> v1, const vector<double, 2> &v2);
		friend vector<double, 2> hsub(vector<double, 2> v1, const vector<double, 2> &v2);

		vector<double, 2> &abs();
		friend vector<double, 2> abs(vector<double, 2> v);
		friend vector<double, 2> min(const vector<double, 2> &v1, const vector<double, 2> &v2);
		friend vector<double, 2> max(const vector<double, 2> &v1, const vector<double, 2> &v2);
		vector<double, 2> &ceil();
		vector<double, 2> &floor();
		vector<double, 2> &round(int mode);
		friend vector<double, 2> ceil(vector<double, 2> v);
		friend vector<double, 2> floor(vector<double, 2> v);
		friend vector<double, 2> round(vector<double, 2> v, int mode);

		vector<double, 2> sqrt() const;
		vector<double, 2> rsqrt() const;

		friend vector<double, 2> select(const vector<double, 2> &v, const vector<double, 2> &alt, const mask<double, 2> &condition);

		friend std::ostream &operator<<(std::ostream &stream, const vector<double, 2> &v);
	};

	template <>
	class mask<double, 2> : public vector_base<double, 2> {
	public:
		SIMD_FORCEINLINE mask() : vector_base() {}
		explicit SIMD_FORCEINLINE mask(native_type v) : vector_base(v) {}
		explicit SIMD_FORCEINLINE mask(bool b) : vector_base(_mm_castsi128_pd(_mm_set1_epi64x(-static_cast<int>(b)))) {}
		explicit SIMD_FORCEINLINE mask(bool b1, bool b2) : vector_base(_mm_castsi128_pd(_mm_set_epi64x(
			-static_cast<int>(b2), -static_cast<int>(b1)))) {}
		explicit SIMD_FORCEINLINE mask(const std::array<bool, width>& arr) : mask(arr[1], arr[0]) {}
		explicit SIMD_FORCEINLINE mask(const vector<double, 2> & v) : vector_base(v) {}

		friend SIMD_FORCEINLINE mask<double, 2> operator==(const mask<double, 2> & v1, const mask<double, 2> & v2);
		friend SIMD_FORCEINLINE mask<double, 2> operator!=(const mask<double, 2> & v1, const mask<double, 2> & v2);

		SIMD_FORCEINLINE mask<double, 2> & operator&=(const mask<double, 2> & v);
		SIMD_FORCEINLINE mask<double, 2> & operator|=(const mask<double, 2> & v);
		SIMD_FORCEINLINE mask<double, 2> & operator^=(const mask<double, 2> & v);

		friend SIMD_FORCEINLINE mask<double, 2> operator~(const mask<double, 2> & v);
		friend SIMD_FORCEINLINE mask<double, 2> operator&(mask<double, 2> v1, const mask<double, 2> & v2);
		friend SIMD_FORCEINLINE mask<double, 2> operator|(mask<double, 2> v1, const mask<double, 2> & v2);
		friend SIMD_FORCEINLINE mask<double, 2> operator^(mask<double, 2> v1, const mask<double, 2> & v2);
		friend SIMD_FORCEINLINE mask<double, 2> andnot(const mask<double, 2> & v1, const mask<double, 2> & v2);

		SIMD_FORCEINLINE int get_mask() const;
		SIMD_FORCEINLINE bool all() const;
		SIMD_FORCEINLINE bool any() const;
		SIMD_FORCEINLINE bool none() const;
	};

	vector<double, 2> &vector<double, 2>::operator+=(const vector<double, 2> &v) {
		m_vec = _mm_add_pd(m_vec, v.m_vec);
		return *this;
	}

	vector<double, 2> &vector<double, 2>::operator-=(const vector<double, 2> &v) {
		m_vec = _mm_sub_pd(m_vec, v.m_vec);
		return *this;
	}

	vector<double, 2> &vector<double, 2>::operator*=(const vector<double, 2> &v) {
		m_vec = _mm_mul_pd(m_vec, v.m_vec);
		return *this;
	}

	vector<double, 2> &vector<double, 2>::operator/=(const vector<double, 2> &v) {
		m_vec = _mm_div_pd(m_vec, v.m_vec);
		return *this;
	}

	vector<double, 2> &vector<double, 2>::hadd(const vector<double, 2> &v) {
#if SIMD_SUPPORTS(SIMD_SSE3)
		m_vec = _mm_hadd_pd(m_vec, v.m_vec);
#else // SIMD_SUPPORTS(SIMD_SSE3)
		throw std::runtime_error("Operation not implemented");
#endif // SIMD_SUPPORTS(SIMD_SSE3)
		return *this;
	}

	vector<double, 2> &vector<double, 2>::hsub(const vector<double, 2> &v) {
#if SIMD_SUPPORTS(SIMD_SSE3)
		m_vec = _mm_hsub_pd(m_vec, v.m_vec);
#else // SIMD_SUPPORTS(SIMD_SSE3)
		throw std::runtime_error("Operation not implemented");
#endif // SIMD_SUPPORTS(SIMD_SSE3)
		return *this;
	}

	double vector<double, 2>::hadd() const {
#if SIMD_SUPPORTS(SIMD_SSE3)
		return _mm_cvtsd_f64(_mm_hadd_pd(m_vec, m_vec));
#else // SIMD_SUPPORTS(SIMD_SSE3)
		// Compute A1+A2 via shuffling
		auto t1 = _mm_add_pd(m_vec, _mm_shuffle_pd(m_vec, m_vec, 0b01));
		return _mm_cvtsd_f64(t1);
#endif // SIMD_SUPPORTS(SIMD_SSE3)
	}

	double vector<double, 2>::hsub() const {
		throw std::runtime_error("Operation not implemented");
	}

	vector<double, 2> hadd(vector<double, 2> v1, const vector<double, 2> &v2) {
		return v1.hadd(v2);
	}

	vector<double, 2> hsub(vector<double, 2> v1, const vector<double, 2> &v2) {
		return v1.hsub(v2);
	}

	vector<double, 2> operator+(vector<double, 2> v1, const vector<double, 2> &v2) {
		return (v1 += v2);
	}

	vector<double, 2> operator-(vector<double, 2> v1, const vector<double, 2> &v2) {
		return (v1 -= v2);
	}

	vector<double, 2> operator*(vector<double, 2> v1, const vector<double, 2> &v2) {
		return (v1 *= v2);
	}

	vector<double, 2> operator/(vector<double, 2> v1, const vector<double, 2> &v2) {
		return (v1 /= v2);
	}

	vector<double, 2> &vector<double, 2>::operator&=(const vector<double, 2> &v) {
		m_vec = _mm_and_pd(m_vec, v.m_vec);
		return *this;
	}

	vector<double, 2> &vector<double, 2>::operator|=(const vector<double, 2> &v) {
		m_vec = _mm_or_pd(m_vec, v.m_vec);
		return *this;
	}

	vector<double, 2> &vector<double, 2>::operator^=(const vector<double, 2> &v) {
		m_vec = _mm_xor_pd(m_vec, v.m_vec);
		return *this;
	}

	vector<double, 2> operator~(const vector<double, 2> &v) {
		return vector<double, 2>(_mm_xor_pd(v.m_vec, _mm_castsi128_pd(_mm_set1_epi32(-1))));
	}

	vector<double, 2> operator&(vector<double, 2> v1, const vector<double, 2> &v2) {
		return v1 &= v2;
	}

	vector<double, 2> operator|(vector<double, 2> v1, const vector<double, 2> &v2) {
		return v1 |= v2;
	}

	vector<double, 2> operator^(vector<double, 2> v1, const vector<double, 2> &v2) {
		return v1 ^= v2;
	}

	vector<double, 2> operator==(const vector<double, 2> &v1, const vector<double, 2> &v2) {
		return vector<double, 2>(_mm_cmpeq_pd(v1.m_vec, v2.m_vec));
	}

	vector<double, 2> operator!=(const vector<double, 2> &v1, const vector<double, 2> &v2) {
		return vector<double, 2>(_mm_cmpneq_pd(v1.m_vec, v2.m_vec));
	}

	vector<double, 2> operator>(const vector<double, 2> &v1, const vector<double, 2> &v2) {
		return vector<double, 2>(_mm_cmpgt_pd(v1.m_vec, v2.m_vec));
	}

	vector<double, 2> operator>=(const vector<double, 2> &v1, const vector<double, 2> &v2) {
		return vector<double, 2>(_mm_cmpge_pd(v1.m_vec, v2.m_vec));
	}

	vector<double, 2> operator<(const vector<double, 2> &v1, const vector<double, 2> &v2) {
		return vector<double, 2>(_mm_cmplt_pd(v1.m_vec, v2.m_vec));
	}

	vector<double, 2> operator<=(const vector<double, 2> &v1, const vector<double, 2> &v2) {
		return vector<double, 2>(_mm_cmple_pd(v1.m_vec, v2.m_vec));
	}

	vector<double, 2> &vector<double, 2>::abs() {
		// TODO: benchmark for most performant solution
		m_vec = _mm_max_pd(_mm_sub_pd(_mm_setzero_pd(), m_vec), m_vec);
		return *this;
	}

	vector<double, 2> abs(vector<double, 2> v) {
		return v.abs();
	}

	vector<double, 2> min(const vector<double, 2> &v1, const vector<double, 2> &v2) {
		return vector<double, 2>(_mm_min_pd(v1.m_vec, v2.m_vec));
	}

	vector<double, 2> max(const vector<double, 2> &v1, const vector<double, 2> &v2) {
		return vector<double, 2>(_mm_max_pd(v1.m_vec, v2.m_vec));
	}

	vector<double, 2> vector<double, 2>::sqrt() const {
		return vector<double, 2>(_mm_sqrt_pd(m_vec));
	}

	vector<double, 2> vector<double, 2>::rsqrt() const {
		return vector<double, 2>(_mm_div_pd(_mm_set1_pd(1.0), _mm_sqrt_pd(m_vec)));
	}

	vector<double, 2> &vector<double, 2>::ceil() {
#if SIMD_SUPPORTS(SIMD_SSE4_1)
		m_vec = _mm_ceil_pd(m_vec);
#else // SIMD_SUPPORTS(SIMD_SSE4_1)
		throw std::runtime_error("Operation not implemented");
#endif // SIMD_SUPPORTS(SIMD_SSE4_1)
		return *this;
	}

	vector<double, 2> &vector<double, 2>::floor() {
#if SIMD_SUPPORTS(SIMD_SSE4_1)
		m_vec = _mm_floor_pd(m_vec);
#else // SIMD_SUPPORTS(SIMD_SSE4_1)
		throw std::runtime_error("Operation not implemented");
#endif // SIMD_SUPPORTS(SIMD_SSE4_1)
		return *this;
	}

	vector<double, 2> &vector<double, 2>::round(int mode) {
#if SIMD_SUPPORTS(SIMD_SSE4_1)
		// TODO: requires constant expression?
		m_vec = _mm_round_pd(m_vec, 0);
#else // SIMD_SUPPORTS(SIMD_SSE4_1)
		throw std::runtime_error("Operation not implemented");
#endif // SIMD_SUPPORTS(SIMD_SSE4_1)
		return *this;
	}

	vector<double, 2> ceil(vector<double, 2> v) {
		return v.ceil();
	}

	vector<double, 2> floor(vector<double, 2> v) {
		return v.floor();
	}

	vector<double, 2> round(vector<double, 2> v, int mode) {
		return v.round(mode);
	}

	vector<double, 2> select(const vector<double, 2> &v, const vector<double, 2> &alt, const mask<double, 2> &condition) {
#if SIMD_SUPPORTS(SIMD_SSE4_1)
		return vector<double, 2>(_mm_blendv_pd(alt.m_vec, v.m_vec, condition.native()));
#else // SIMD_SUPPORTS(SIMD_SSE4_1)
		// TODO: only works when condition has either all or none set!
		return vector<double, 2>(_mm_or_pd(_mm_and_pd(v.m_vec, condition.m_vec), _mm_andnot_pd(alt.m_vec, condition.native())));
#endif // SIMD_SUPPORTS(SIMD_SSE4_1)
	}

	std::ostream &operator<<(std::ostream &stream, const vector<double, 2> &v) {
		stream << '(';
		for (size_t i = 0; i < v.width - 1; ++i) {
			stream << v.m_array[i] << ' ';
		}
		stream << v.m_array[v.width - 1] << ')';
		return stream;
	}

	mask<double, 2> operator==(const mask<double, 2> & v1, const mask<double, 2> & v2) {
		return mask<double, 2>(_mm_cmp_pd(v1.m_vec, v2.m_vec, _CMP_EQ_OQ));
	}

	mask<double, 2> operator!=(const mask<double, 2> & v1, const mask<double, 2> & v2) {
		return mask<double, 2>(_mm_cmp_pd(v1.m_vec, v2.m_vec, _CMP_NEQ_OQ));
	}

	mask<double, 2> & mask<double, 2>::operator&=(const mask<double, 2> & v) {
		m_vec = _mm_and_pd(m_vec, v.m_vec);
		return *this;
	}

	mask<double, 2> & mask<double, 2>::operator|=(const mask<double, 2> & v) {
		m_vec = _mm_or_pd(m_vec, v.m_vec);
		return *this;
	}

	mask<double, 2> & mask<double, 2>::operator^=(const mask<double, 2> & v) {
		m_vec = _mm_xor_pd(m_vec, v.m_vec);
		return *this;
	}

	mask<double, 2> operator~(const mask<double, 2> & v) {
		return mask<double, 2>(_mm_xor_pd(v.m_vec, _mm_castsi128_pd(_mm_set1_epi64x(-1))));
	}

	mask<double, 2> operator&(mask<double, 2> v1, const mask<double, 2> & v2) {
		return v1 &= v2;
	}

	mask<double, 2> operator|(mask<double, 2> v1, const mask<double, 2> & v2) {
		return v1 |= v2;
	}

	mask<double, 2> operator^(mask<double, 2> v1, const mask<double, 2> & v2) {
		return v1 ^= v2;
	}

	mask<double, 2> andnot(const mask<double, 2> & v1, const mask<double, 2> & v2) {
		return mask<double, 2>(_mm_andnot_pd(v1.m_vec, v2.m_vec));
	}

	int mask<double, 2>::get_mask() const {
		return _mm_movemask_pd(m_vec);
	}

	bool mask<double, 2>::all() const {
		// TODO: what is faster here?
		return _mm_movemask_pd(m_vec) == 0b1111;
	}

	bool mask<double, 2>::any() const {
		return _mm_movemask_pd(m_vec);
	}

	bool mask<double, 2>::none() const {
		return !_mm_movemask_pd(m_vec);
	}

#endif // SIMD_SUPPORTS(SIMD_SSE2)

} // namespace simd