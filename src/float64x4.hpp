#pragma once

#include <array>
#include <exception>
#include "base_types.hpp"
#include "simd.hpp"

namespace simd {

#if SIMD_SUPPORTS(SIMD_AVX)
	using float64x4 = vector<double, 4>;

	template <>
	class vector<double, 4> : public vector_base<double, 4> {
	public:
		static constexpr int required_version = native_vector<type, width>::required_version;

	public:
		vector() : vector_base() {}
		explicit vector(native_type v) : vector_base(v) {}
		explicit vector(type f) : vector_base(_mm256_set1_pd(f)) {}
		explicit vector(type f1, type f2, type f3, type f4) : vector_base(_mm256_set_pd(f4, f3, f2, f1)) {}
		explicit vector(const std::array<type, width>& arr) : vector_base(_mm256_set_pd(arr[3], arr[2], arr[1], arr[0])) {}
		explicit vector(const type *vals) : vector_base(_mm256_loadu_pd(vals)) {}
		explicit vector(const type *vals, aligned_load) : vector_base(_mm256_load_pd(vals)) {}
		explicit vector(const vector<int, 4> &v);
		explicit vector(const vector<float, 4> &v);

		vector<double, 4> &operator+=(const vector<double, 4> &v);
		vector<double, 4> &operator-=(const vector<double, 4> &v);
		vector<double, 4> &operator*=(const vector<double, 4> &v);
		vector<double, 4> &operator/=(const vector<double, 4> &v);
		friend vector<double, 4> operator+(vector<double, 4> v1, const vector<double, 4> &v2);
		friend vector<double, 4> operator-(vector<double, 4> v1, const vector<double, 4> &v2);
		friend vector<double, 4> operator*(vector<double, 4> v1, const vector<double, 4> &v2);
		friend vector<double, 4> operator/(vector<double, 4> v1, const vector<double, 4> &v2);

		vector<double, 4> &operator&=(const vector<double, 4> &v);
		vector<double, 4> &operator|=(const vector<double, 4> &v);
		vector<double, 4> &operator^=(const vector<double, 4> &v);
		friend vector<double, 4> operator~(const vector<double, 4> &v);
		friend vector<double, 4> operator&(vector<double, 4> v1, const vector<double, 4> &v2);
		friend vector<double, 4> operator|(vector<double, 4> v1, const vector<double, 4> &v2);
		friend vector<double, 4> operator^(vector<double, 4> v1, const vector<double, 4> &v2);

		friend vector<double, 4> operator==(const vector<double, 4> &v1, const vector<double, 4> &v2);
		friend vector<double, 4> operator!=(const vector<double, 4> &v1, const vector<double, 4> &v2);
		friend vector<double, 4> operator>(const vector<double, 4> &v1, const vector<double, 4> &v2);
		friend vector<double, 4> operator>=(const vector<double, 4> &v1, const vector<double, 4> &v2);
		friend vector<double, 4> operator<(const vector<double, 4> &v1, const vector<double, 4> &v2);
		friend vector<double, 4> operator<=(const vector<double, 4> &v1, const vector<double, 4> &v2);

		vector<double, 4> &hadd(const vector<double, 4> &v);
		vector<double, 4> &hsub(const vector<double, 4> &v);
		type hadd() const;
		type hsub() const;
		friend vector<double, 4> hadd(vector<double, 4> v1, const vector<double, 4> &v2);
		friend vector<double, 4> hsub(vector<double, 4> v1, const vector<double, 4> &v2);

		vector<double, 4> &abs();
		friend vector<double, 4> abs(vector<double, 4> v);
		friend vector<double, 4> min(const vector<double, 4> &v1, const vector<double, 4> &v2);
		friend vector<double, 4> max(const vector<double, 4> &v1, const vector<double, 4> &v2);
		vector<double, 4> &ceil();
		vector<double, 4> &floor();
		vector<double, 4> &round(int mode);
		friend vector<double, 4> ceil(vector<double, 4> v);
		friend vector<double, 4> floor(vector<double, 4> v);
		friend vector<double, 4> round(vector<double, 4> v, int mode);

		vector<double, 4> sqrt() const;
		vector<double, 4> rsqrt() const;

		friend vector<double, 4> select(const vector<double, 4> &v, const vector<double, 4> &alt, const mask<double, 4> &condition);

		friend std::ostream &operator<<(std::ostream &stream, const vector<double, 4> &v);
	};

	template <>
	class mask<double, 4> : public vector_base<double, 4> {
	public:
		SIMD_FORCEINLINE mask() : vector_base() {}
		explicit SIMD_FORCEINLINE mask(native_type v) : vector_base(v) {}
		explicit SIMD_FORCEINLINE mask(bool b) : vector_base(_mm256_castsi256_pd(_mm256_set1_epi64x(-static_cast<int>(b)))) {}
		explicit SIMD_FORCEINLINE mask(bool b1, bool b2, bool b3, bool b4) : vector_base(_mm256_castsi256_pd(_mm256_set_epi64x(
			-static_cast<int>(b4), -static_cast<int>(b3), -static_cast<int>(b2), -static_cast<int>(b1)))) {}
		explicit SIMD_FORCEINLINE mask(const std::array<bool, width>& arr) : mask(arr[3], arr[2], arr[1], arr[0]) {}
		explicit SIMD_FORCEINLINE mask(const vector<double, 4> & v) : vector_base(v) {}

		friend SIMD_FORCEINLINE mask<double, 4> operator==(const mask<double, 4> & v1, const mask<double, 4> & v2);
		friend SIMD_FORCEINLINE mask<double, 4> operator!=(const mask<double, 4> & v1, const mask<double, 4> & v2);

		SIMD_FORCEINLINE mask<double, 4> & operator&=(const mask<double, 4> & v);
		SIMD_FORCEINLINE mask<double, 4> & operator|=(const mask<double, 4> & v);
		SIMD_FORCEINLINE mask<double, 4> & operator^=(const mask<double, 4> & v);

		friend SIMD_FORCEINLINE mask<double, 4> operator~(const mask<double, 4> & v);
		friend SIMD_FORCEINLINE mask<double, 4> operator&(mask<double, 4> v1, const mask<double, 4> & v2);
		friend SIMD_FORCEINLINE mask<double, 4> operator|(mask<double, 4> v1, const mask<double, 4> & v2);
		friend SIMD_FORCEINLINE mask<double, 4> operator^(mask<double, 4> v1, const mask<double, 4> & v2);
		friend SIMD_FORCEINLINE mask<double, 4> andnot(const mask<double, 4> & v1, const mask<double, 4> & v2);

		SIMD_FORCEINLINE int get_mask() const;
		SIMD_FORCEINLINE bool all() const;
		SIMD_FORCEINLINE bool any() const;
		SIMD_FORCEINLINE bool none() const;
	};

	vector<double, 4> &vector<double, 4>::operator+=(const vector<double, 4> &v) {
		m_vec = _mm256_add_pd(m_vec, v.m_vec);
		return *this;
	}

	vector<double, 4> &vector<double, 4>::operator-=(const vector<double, 4> &v) {
		m_vec = _mm256_sub_pd(m_vec, v.m_vec);
		return *this;
	}

	vector<double, 4> &vector<double, 4>::operator*=(const vector<double, 4> &v) {
		m_vec = _mm256_mul_pd(m_vec, v.m_vec);
		return *this;
	}

	vector<double, 4> &vector<double, 4>::operator/=(const vector<double, 4> &v) {
		m_vec = _mm256_div_pd(m_vec, v.m_vec);
		return *this;
	}

	vector<double, 4> &vector<double, 4>::hadd(const vector<double, 4> &v) {
#if SIMD_SUPPORTS(SIMD_SSE3)
		m_vec = _mm256_hadd_pd(m_vec, v.m_vec);
#else // SIMD_SUPPORTS(SIMD_SSE3)
		throw std::runtime_error("Operation not implemented");
#endif // SIMD_SUPPORTS(SIMD_SSE3)
		return *this;
	}

	vector<double, 4> &vector<double, 4>::hsub(const vector<double, 4> &v) {
#if SIMD_SUPPORTS(SIMD_SSE3)
		m_vec = _mm256_hsub_pd(m_vec, v.m_vec);
#else // SIMD_SUPPORTS(SIMD_SSE3)
		throw std::runtime_error("Operation not implemented");
#endif // SIMD_SUPPORTS(SIMD_SSE3)
		return *this;
	}

	double vector<double, 4>::hadd() const {
		throw std::runtime_error("Operation not implemented");
		return 0.0;
		// Results in A1+A2, ..., A3+A4, ...
		/*auto t1 = _mm256_hadd_pd(m_vec, m_vec);
		// Permute to get A1+A2, A3+A4, ...
		auto t2 = _mm256_permute2f128_pd(m_vec, m_vec, 0b000001);
		// Add (A1+A2)+(A3+A4), ...
		auto t3 = _mm256_hadd_ps(t1, t1);
		
		// Grab the first entry
		return _mm256_cvtsd_f64(t3);*/
	}

	double vector<double, 4>::hsub() const {
		throw std::runtime_error("Operation not implemented");
	}

	vector<double, 4> hadd(vector<double, 4> v1, const vector<double, 4> &v2) {
		return v1.hadd(v2);
	}

	vector<double, 4> hsub(vector<double, 4> v1, const vector<double, 4> &v2) {
		return v1.hsub(v2);
	}

	vector<double, 4> operator+(vector<double, 4> v1, const vector<double, 4> &v2) {
		return (v1 += v2);
	}

	vector<double, 4> operator-(vector<double, 4> v1, const vector<double, 4> &v2) {
		return (v1 -= v2);
	}

	vector<double, 4> operator*(vector<double, 4> v1, const vector<double, 4> &v2) {
		return (v1 *= v2);
	}

	vector<double, 4> operator/(vector<double, 4> v1, const vector<double, 4> &v2) {
		return (v1 /= v2);
	}

	vector<double, 4> &vector<double, 4>::operator&=(const vector<double, 4> &v) {
		m_vec = _mm256_and_pd(m_vec, v.m_vec);
		return *this;
	}

	vector<double, 4> &vector<double, 4>::operator|=(const vector<double, 4> &v) {
		m_vec = _mm256_or_pd(m_vec, v.m_vec);
		return *this;
	}

	vector<double, 4> &vector<double, 4>::operator^=(const vector<double, 4> &v) {
		m_vec = _mm256_xor_pd(m_vec, v.m_vec);
		return *this;
	}

	vector<double, 4> operator~(const vector<double, 4> &v) {
		return vector<double, 4>(_mm256_xor_pd(v.m_vec, _mm256_castsi256_pd(_mm256_set1_epi32(-1))));
	}

	vector<double, 4> operator&(vector<double, 4> v1, const vector<double, 4> &v2) {
		return v1 &= v2;
	}

	vector<double, 4> operator|(vector<double, 4> v1, const vector<double, 4> &v2) {
		return v1 |= v2;
	}

	vector<double, 4> operator^(vector<double, 4> v1, const vector<double, 4> &v2) {
		return v1 ^= v2;
	}

	vector<double, 4> operator==(const vector<double, 4> &v1, const vector<double, 4> &v2) {
		return vector<double, 4>(_mm256_cmp_pd(v1.m_vec, v2.m_vec, _CMP_EQ_OQ));
	}

	vector<double, 4> operator!=(const vector<double, 4> &v1, const vector<double, 4> &v2) {
		return vector<double, 4>(_mm256_cmp_pd(v1.m_vec, v2.m_vec, _CMP_NEQ_OQ));
	}

	vector<double, 4> operator>(const vector<double, 4> &v1, const vector<double, 4> &v2) {
		return vector<double, 4>(_mm256_cmp_pd(v1.m_vec, v2.m_vec, _CMP_GT_OQ));
	}

	vector<double, 4> operator>=(const vector<double, 4> &v1, const vector<double, 4> &v2) {
		return vector<double, 4>(_mm256_cmp_pd(v1.m_vec, v2.m_vec, _CMP_GE_OQ));
	}

	vector<double, 4> operator<(const vector<double, 4> &v1, const vector<double, 4> &v2) {
		return vector<double, 4>(_mm256_cmp_pd(v1.m_vec, v2.m_vec, _CMP_LT_OQ));
	}

	vector<double, 4> operator<=(const vector<double, 4> &v1, const vector<double, 4> &v2) {
		return vector<double, 4>(_mm256_cmp_pd(v1.m_vec, v2.m_vec, _CMP_LE_OQ));
	}

	vector<double, 4> &vector<double, 4>::abs() {
		// TODO: benchmark for most performant solution
		m_vec = _mm256_max_pd(_mm256_sub_pd(_mm256_setzero_pd(), m_vec), m_vec);
		return *this;
	}

	vector<double, 4> abs(vector<double, 4> v) {
		return v.abs();
	}

	vector<double, 4> min(const vector<double, 4> &v1, const vector<double, 4> &v2) {
		return vector<double, 4>(_mm256_min_pd(v1.m_vec, v2.m_vec));
	}

	vector<double, 4> max(const vector<double, 4> &v1, const vector<double, 4> &v2) {
		return vector<double, 4>(_mm256_max_pd(v1.m_vec, v2.m_vec));
	}

	vector<double, 4> vector<double, 4>::sqrt() const {
		return vector<double, 4>(_mm256_sqrt_pd(m_vec));
	}

	vector<double, 4> vector<double, 4>::rsqrt() const {
		return vector<double, 4>(_mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(m_vec)));
	}

	vector<double, 4> &vector<double, 4>::ceil() {
		m_vec = _mm256_ceil_pd(m_vec);
		return *this;
	}

	vector<double, 4> &vector<double, 4>::floor() {
		m_vec = _mm256_floor_pd(m_vec);
		return *this;
	}

	vector<double, 4> &vector<double, 4>::round(int mode) {
		// TODO: requires constant expression?
		m_vec = _mm256_round_pd(m_vec, 0);
		return *this;
	}

	vector<double, 4> ceil(vector<double, 4> v) {
		return v.ceil();
	}

	vector<double, 4> floor(vector<double, 4> v) {
		return v.floor();
	}

	vector<double, 4> round(vector<double, 4> v, int mode) {
		return v.round(mode);
	}

	vector<double, 4> select(const vector<double, 4> &v, const vector<double, 4> &alt, const mask<double, 4> &condition) {
		return vector<double, 4>(_mm256_blendv_pd(alt.m_vec, v.m_vec, condition.native()));
	}

	std::ostream &operator<<(std::ostream &stream, const vector<double, 4> &v) {
		stream << '(';
		for (size_t i = 0; i < v.width - 1; ++i) {
			stream << v.m_array[i] << ' ';
		}
		stream << v.m_array[v.width - 1] << ')';
		return stream;
	}

	mask<double, 4> operator==(const mask<double, 4> & v1, const mask<double, 4> & v2) {
		return mask<double, 4>(_mm256_cmp_pd(v1.m_vec, v2.m_vec, _CMP_EQ_OQ));
	}

	mask<double, 4> operator!=(const mask<double, 4> & v1, const mask<double, 4> & v2) {
		return mask<double, 4>(_mm256_cmp_pd(v1.m_vec, v2.m_vec, _CMP_NEQ_OQ));
	}

	mask<double, 4> & mask<double, 4>::operator&=(const mask<double, 4> & v) {
		m_vec = _mm256_and_pd(m_vec, v.m_vec);
		return *this;
	}

	mask<double, 4> & mask<double, 4>::operator|=(const mask<double, 4> & v) {
		m_vec = _mm256_or_pd(m_vec, v.m_vec);
		return *this;
	}

	mask<double, 4> & mask<double, 4>::operator^=(const mask<double, 4> & v) {
		m_vec = _mm256_xor_pd(m_vec, v.m_vec);
		return *this;
	}

	mask<double, 4> operator~(const mask<double, 4> & v) {
		return mask<double, 4>(_mm256_xor_pd(v.m_vec, _mm256_castsi256_pd(_mm256_set1_epi64x(-1))));
	}

	mask<double, 4> operator&(mask<double, 4> v1, const mask<double, 4> & v2) {
		return v1 &= v2;
	}

	mask<double, 4> operator|(mask<double, 4> v1, const mask<double, 4> & v2) {
		return v1 |= v2;
	}

	mask<double, 4> operator^(mask<double, 4> v1, const mask<double, 4> & v2) {
		return v1 ^= v2;
	}

	mask<double, 4> andnot(const mask<double, 4> & v1, const mask<double, 4> & v2) {
		return mask<double, 4>(_mm256_andnot_pd(v1.m_vec, v2.m_vec));
	}

	int mask<double, 4>::get_mask() const {
		return _mm256_movemask_pd(m_vec);
	}

	bool mask<double, 4>::all() const {
		// TODO: what is faster here?
		return _mm256_movemask_pd(m_vec) == 0b1111;
	}

	bool mask<double, 4>::any() const {
		return _mm256_movemask_pd(m_vec);
	}

	bool mask<double, 4>::none() const {
		return !_mm256_movemask_pd(m_vec);
	}

#endif // SIMD_SUPPORTS(SIMD_AVX)

} // namespace simd