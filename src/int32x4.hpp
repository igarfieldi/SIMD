#pragma once

#include <array>
#include <cstdint>
#include <exception>
#include "base_types.hpp"
#include "simd.hpp"

namespace simd {

#if SIMD_SUPPORTS(SIMD_SSE2)
	using int32x4 = vector<std::int32_t, 4>;

	template <>
	class vector<std::int32_t, 4> : public vector_base<std::int32_t, 4> {
	public:
		static constexpr int required_version = native_vector<type, width>::required_version;

		vector() : vector_base() {}
		explicit vector(native_type v) : vector_base(v) {}
		explicit vector(type f) : vector_base(_mm_set1_epi32(f)) {}
		explicit vector(type f1, type f2, type f3, type f4) : vector_base(_mm_set_epi32(f4, f3, f2, f1)) {}
		explicit vector(const std::array<type, width>& arr) : vector_base(_mm_set_epi32(arr[3], arr[2], arr[1], arr[0])) {}
#if SIMD_SUPPORTS(SIMD_SSE3)
		explicit vector(const type *vals) : vector_base(_mm_lddqu_si128(reinterpret_cast<const native_type *>(vals))) {}
#else // SIMD_SUPPORTS(SIMD_SSE3)
		explicit vector(const type *vals) : vector_base(_mm_loadu_si128(reinterpret_cast<const native_type *>(vals))) {}
#endif // SIMD_SUPPORTS(SIMD_SSE3)
		explicit vector(const type *vals, aligned_load) : vector_base(_mm_load_si128(reinterpret_cast<const native_type *>(vals))) {}
		explicit vector(const vector<float, 4> &v);
		explicit vector(const vector<double, 2> &v);
#if SIMD_SUPPORTS(SIMD_AVX)
		explicit vector(const vector<double, 4> &v);
#endif // SIMD_SUPPORTS(SIMD_AVX)

		vector<std::int32_t, 4> &operator+=(const vector<std::int32_t, 4> &v);
		vector<std::int32_t, 4> &operator-=(const vector<std::int32_t, 4> &v);
		vector<std::int32_t, 4> &operator*=(const vector<std::int32_t, 4> &v);
		vector<std::int32_t, 4> &operator/=(const vector<std::int32_t, 4> &v);
		friend vector<std::int32_t, 4> operator+(vector<std::int32_t, 4> v1, const vector<std::int32_t, 4> &v2);
		friend vector<std::int32_t, 4> operator-(vector<std::int32_t, 4> v1, const vector<std::int32_t, 4> &v2);
		friend vector<std::int32_t, 4> operator*(vector<std::int32_t, 4> v1, const vector<std::int32_t, 4> &v2);
		friend vector<std::int32_t, 4> operator/(vector<std::int32_t, 4> v1, const vector<std::int32_t, 4> &v2);

		vector<std::int32_t, 4> &operator&=(const vector<std::int32_t, 4> &v);
		vector<std::int32_t, 4> &operator|=(const vector<std::int32_t, 4> &v);
		vector<std::int32_t, 4> &operator^=(const vector<std::int32_t, 4> &v);
		friend vector<std::int32_t, 4> operator~(const vector<std::int32_t, 4> &v);
		friend vector<std::int32_t, 4> operator&(vector<std::int32_t, 4> v1, const vector<std::int32_t, 4> &v2);
		friend vector<std::int32_t, 4> operator|(vector<std::int32_t, 4> v1, const vector<std::int32_t, 4> &v2);
		friend vector<std::int32_t, 4> operator^(vector<std::int32_t, 4> v1, const vector<std::int32_t, 4> &v2);

		friend vector<std::int32_t, 4> operator<<(const vector<std::int32_t, 4> &v, std::int32_t bits);
		friend vector<std::int32_t, 4> operator>>(const vector<std::int32_t, 4> &v, std::int32_t bits);

		friend vector<std::int32_t, 4> operator==(const vector<std::int32_t, 4> &v1, const vector<std::int32_t, 4> &v2);
		friend vector<std::int32_t, 4> operator!=(const vector<std::int32_t, 4> &v1, const vector<std::int32_t, 4> &v2);
		friend vector<std::int32_t, 4> operator>(const vector<std::int32_t, 4> &v1, const vector<std::int32_t, 4> &v2);
		friend vector<std::int32_t, 4> operator>=(const vector<std::int32_t, 4> &v1, const vector<std::int32_t, 4> &v2);
		friend vector<std::int32_t, 4> operator<(const vector<std::int32_t, 4> &v1, const vector<std::int32_t, 4> &v2);
		friend vector<std::int32_t, 4> operator<=(const vector<std::int32_t, 4> &v1, const vector<std::int32_t, 4> &v2);

		vector<std::int32_t, 4> &hadd(const vector<std::int32_t, 4> &v);
		vector<std::int32_t, 4> &hsub(const vector<std::int32_t, 4> &v);
		type hadd() const;
		type hsub() const;
		friend vector<std::int32_t, 4> hadd(vector<std::int32_t, 4> v1, const vector<std::int32_t, 4> &v2);
		friend vector<std::int32_t, 4> hsub(vector<std::int32_t, 4> v1, const vector<std::int32_t, 4> &v2);

		vector<std::int32_t, 4> &abs();
		friend vector<std::int32_t, 4> abs(vector<std::int32_t, 4> v);
		friend vector<std::int32_t, 4> min(const vector<std::int32_t, 4> &v1, const vector<std::int32_t, 4> &v2);
		friend vector<std::int32_t, 4> max(const vector<std::int32_t, 4> &v1, const vector<std::int32_t, 4> &v2);

		friend vector<std::int32_t, 4> select(const vector<std::int32_t, 4> &v, const vector<std::int32_t, 4> &alt, const mask<std::int32_t, 4> &condition);

		friend std::ostream &operator<<(std::ostream &stream, const vector<std::int32_t, 4> &v);
	};

	template <>
	class mask<std::int32_t, 4> : public vector_base<std::int32_t, 4> {
	public:
		SIMD_FORCEINLINE mask() : vector_base() {}
		explicit SIMD_FORCEINLINE mask(native_type v) : vector_base(v) {}
		explicit SIMD_FORCEINLINE mask(bool b) : vector_base(_mm_set1_epi32(-static_cast<int>(b))) {}
		explicit SIMD_FORCEINLINE mask(bool b1, bool b2, bool b3, bool b4) : vector_base(_mm_set_epi32(
			-static_cast<int>(b4), -static_cast<int>(b3), -static_cast<int>(b2), -static_cast<int>(b1))) {}
		explicit SIMD_FORCEINLINE mask(const std::array<bool, width>& arr) : mask(arr[3], arr[2], arr[1], arr[0]) {}
		explicit SIMD_FORCEINLINE mask(const vector<std::int32_t, 4> & v) : vector_base(v) {}

		friend SIMD_FORCEINLINE mask<std::int32_t, 4> operator==(const mask<std::int32_t, 4> & v1, const mask<std::int32_t, 4> & v2);
		friend SIMD_FORCEINLINE mask<std::int32_t, 4> operator!=(const mask<std::int32_t, 4> & v1, const mask<std::int32_t, 4> & v2);

		SIMD_FORCEINLINE mask<std::int32_t, 4> & operator&=(const mask<std::int32_t, 4> & v);
		SIMD_FORCEINLINE mask<std::int32_t, 4> & operator|=(const mask<std::int32_t, 4> & v);
		SIMD_FORCEINLINE mask<std::int32_t, 4> & operator^=(const mask<std::int32_t, 4> & v);

		friend SIMD_FORCEINLINE mask<std::int32_t, 4> operator~(const mask<std::int32_t, 4> & v);
		friend SIMD_FORCEINLINE mask<std::int32_t, 4> operator&(mask<std::int32_t, 4> v1, const mask<std::int32_t, 4> & v2);
		friend SIMD_FORCEINLINE mask<std::int32_t, 4> operator|(mask<std::int32_t, 4> v1, const mask<std::int32_t, 4> & v2);
		friend SIMD_FORCEINLINE mask<std::int32_t, 4> operator^(mask<std::int32_t, 4> v1, const mask<std::int32_t, 4> & v2);
		friend SIMD_FORCEINLINE mask<std::int32_t, 4> andnot(const mask<std::int32_t, 4> & v1, const mask<std::int32_t, 4> & v2);

		SIMD_FORCEINLINE int get_mask() const;
		SIMD_FORCEINLINE bool all() const;
		SIMD_FORCEINLINE bool any() const;
		SIMD_FORCEINLINE bool none() const;
	};

	vector<std::int32_t, 4> &vector<std::int32_t, 4>::operator+=(const vector<std::int32_t, 4> &v) {
		m_vec = _mm_add_epi32(m_vec, v.m_vec);
		return *this;
	}

	vector<std::int32_t, 4> &vector<std::int32_t, 4>::operator-=(const vector<std::int32_t, 4> &v) {
		m_vec = _mm_sub_epi32(m_vec, v.m_vec);
		return *this;
	}

	vector<std::int32_t, 4> &vector<std::int32_t, 4>::operator*=(const vector<std::int32_t, 4> &v) {
#if SIMD_SUPPORTS(SIMD_SSE4_1)
		m_vec = _mm_mul_epi32(m_vec, v.m_vec);
#else // SIMD_SUPPORTS(SIMD_SSE4_1)
		throw std::runtime_error("Operation not implemented");
#endif // SIMD_SUPPORTS(SIMD_SSE4_1)
		return *this;
	}

	vector<std::int32_t, 4> &vector<std::int32_t, 4>::operator/=(const vector<std::int32_t, 4> &v) {
		// TODO: there's gotta be a better way
		m_vec = _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(m_vec), _mm_cvtepi32_ps(v.m_vec)));
		return *this;
	}

	vector<std::int32_t, 4> &vector<std::int32_t, 4>::hadd(const vector<std::int32_t, 4> &v) {
#if SIMD_SUPPORTS(SIMD_SSE3)
		m_vec = _mm_hadd_epi32(m_vec, v.m_vec);
#else // SIMD_SUPPORTS(SIMD_SSE3)
		throw std::runtime_error("Operation not implemented");
#endif // SIMD_SUPPORTS(SIMD_SSE3)
		return *this;
	}

	vector<std::int32_t, 4> &vector<std::int32_t, 4>::hsub(const vector<std::int32_t, 4> &v) {
#if SIMD_SUPPORTS(SIMD_SSE3)
		m_vec = _mm_hsub_epi32(m_vec, v.m_vec);
#else // SIMD_SUPPORTS(SIMD_SSE3)
		throw std::runtime_error("Operation not implemented");
#endif // SIMD_SUPPORTS(SIMD_SSE3)
		return *this;
	}

	std::int32_t vector<std::int32_t, 4>::hadd() const {
#if SIMD_SUPPORTS(SIMD_SSE3)
		auto t1 = _mm_hadd_epi32(m_vec, m_vec);
		return _mm_cvtsi128_si32(_mm_hadd_epi32(t1, t1));
#else // SIMD_SUPPORTS(SIMD_SSE3)
		// Compute A1+A3, A2+A4, ...
		auto t1 = _mm_add_epi32(m_vec, _mm_shuffle_epi32(m_vec, 0b00001011));
		// Compute A1+A2 w. shuffle
		auto t2 = _mm_add_epi32(t1, _mm_shuffle_epi32(t1, 0b00000001));
		return _mm_cvtsi128_si32(t2);
#endif // SIMD_SUPPORTS(SIMD_SSE3)
	}

	std::int32_t vector<std::int32_t, 4>::hsub() const {
		throw std::runtime_error("Operation not implemented");
	}

	vector<std::int32_t, 4> hadd(vector<std::int32_t, 4> v1, const vector<std::int32_t, 4> &v2) {
		return v1.hadd(v2);
	}

	vector<std::int32_t, 4> hsub(vector<std::int32_t, 4> v1, const vector<std::int32_t, 4> &v2) {
		return v1.hsub(v2);
	}

	vector<std::int32_t, 4> operator+(vector<std::int32_t, 4> v1, const vector<std::int32_t, 4> &v2) {
		return (v1 += v2);
	}

	vector<std::int32_t, 4> operator-(vector<std::int32_t, 4> v1, const vector<std::int32_t, 4> &v2) {
		return (v1 -= v2);
	}

	vector<std::int32_t, 4> operator*(vector<std::int32_t, 4> v1, const vector<std::int32_t, 4> &v2) {
		return (v1 *= v2);
	}

	vector<std::int32_t, 4> operator/(vector<std::int32_t, 4> v1, const vector<std::int32_t, 4> &v2) {
		return (v1 /= v2);
	}

	vector<std::int32_t, 4> &vector<std::int32_t, 4>::operator&=(const vector<std::int32_t, 4> &v) {
		m_vec = _mm_and_si128(m_vec, v.m_vec);
		return *this;
	}

	vector<std::int32_t, 4> &vector<std::int32_t, 4>::operator|=(const vector<std::int32_t, 4> &v) {
		m_vec = _mm_or_si128(m_vec, v.m_vec);
		return *this;
	}

	vector<std::int32_t, 4> &vector<std::int32_t, 4>::operator^=(const vector<std::int32_t, 4> &v) {
		m_vec = _mm_xor_si128(m_vec, v.m_vec);
		return *this;
	}

	vector<std::int32_t, 4> operator~(const vector<std::int32_t, 4> &v) {
		return vector<std::int32_t, 4>(_mm_xor_si128(v.m_vec, _mm_set1_epi32(-1)));
	}

	vector<std::int32_t, 4> operator&(vector<std::int32_t, 4> v1, const vector<std::int32_t, 4> &v2) {
		return v1 &= v2;
	}

	vector<std::int32_t, 4> operator|(vector<std::int32_t, 4> v1, const vector<std::int32_t, 4> &v2) {
		return v1 |= v2;
	}

	vector<std::int32_t, 4> operator^(vector<std::int32_t, 4> v1, const vector<std::int32_t, 4> &v2) {
		return v1 ^= v2;
	}

	vector<std::int32_t, 4> operator<<(const vector<std::int32_t, 4> &v, std::int32_t bits) {
		return vector<std::int32_t, 4>(_mm_slli_epi32(v.m_vec, bits));
	}

	vector<std::int32_t, 4> operator>>(const vector<std::int32_t, 4> &v, std::int32_t bits) {
		return vector<std::int32_t, 4>(_mm_slli_epi32(v.m_vec, bits));
	}

	vector<std::int32_t, 4> operator==(const vector<std::int32_t, 4> &v1, const vector<std::int32_t, 4> &v2) {
		return vector<std::int32_t, 4>(_mm_cmpeq_epi32(v1.m_vec, v2.m_vec));
	}

	vector<std::int32_t, 4> operator!=(const vector<std::int32_t, 4> &v1, const vector<std::int32_t, 4> &v2) {
		// Invert because SSE doesn't have instruction for it
		return vector<std::int32_t, 4>(_mm_xor_si128(_mm_cmpeq_epi32(v1.m_vec, v2.m_vec), _mm_set1_epi32(-1)));
	}

	vector<std::int32_t, 4> operator>(const vector<std::int32_t, 4> &v1, const vector<std::int32_t, 4> &v2) {
		return vector<std::int32_t, 4>(_mm_cmpgt_epi32(v1.m_vec, v2.m_vec));
	}

	vector<std::int32_t, 4> operator>=(const vector<std::int32_t, 4> &v1, const vector<std::int32_t, 4> &v2) {
		// Invert because SSE doesn't have instruction for it
		return vector<std::int32_t, 4>(_mm_xor_si128(_mm_cmplt_epi32(v1.m_vec, v2.m_vec), _mm_set1_epi32(-1)));
	}

	vector<std::int32_t, 4> operator<(const vector<std::int32_t, 4> &v1, const vector<std::int32_t, 4> &v2) {
		return vector<std::int32_t, 4>(_mm_cmplt_epi32(v1.m_vec, v2.m_vec));
	}

	vector<std::int32_t, 4> operator<=(const vector<std::int32_t, 4> &v1, const vector<std::int32_t, 4> &v2) {
		// Invert because SSE doesn't have instruction for it
		return vector<std::int32_t, 4>(_mm_xor_si128(_mm_cmpgt_epi32(v1.m_vec, v2.m_vec), _mm_set1_epi32(-1)));
	}

	vector<std::int32_t, 4> &vector<std::int32_t, 4>::abs() {
#if SIMD_SUPPORTS(SIMD_SSE3)
		m_vec = _mm_abs_epi32(m_vec);
#else // SIMD_SUPPORTS(SIMD_SSE3)
		throw std::runtime_error("Operation not implemented");
#endif // SIMD_SUPPORTS(SIMD_SSE3)
		return *this;
	}

	vector<std::int32_t, 4> abs(vector<std::int32_t, 4> v) {
		return v.abs();
	}

	vector<std::int32_t, 4> min(const vector<std::int32_t, 4> &v1, const vector<std::int32_t, 4> &v2) {
#if SIMD_SUPPORTS(SIMD_SSE3)
		return vector<std::int32_t, 4>(_mm_min_epi32(v1.m_vec, v2.m_vec));
#else // SIMD_SUPPORTS(SIMD_SSE3)
		throw std::runtime_error("Operation not implemented");
#endif // SIMD_SUPPORTS(SIMD_SSE3)
	}

	vector<std::int32_t, 4> max(const vector<std::int32_t, 4> &v1, const vector<std::int32_t, 4> &v2) {
#if SIMD_SUPPORTS(SIMD_SSE3)
		return vector<std::int32_t, 4>(_mm_max_epi32(v1.m_vec, v2.m_vec));
#else // SIMD_SUPPORTS(SIMD_SSE3)
		throw std::runtime_error("Operation not implemented");
#endif // SIMD_SUPPORTS(SIMD_SSE3)
	}

	vector<std::int32_t, 4> select(const vector<std::int32_t, 4> &v, const vector<std::int32_t, 4> &alt, const mask<std::int32_t, 4> &condition) {
		// TODO: this only works when mask is either 0 or -1 (because it's byte-wise)
#if SIMD_SUPPORTS(SIMD_SSE4_1)
		return vector<std::int32_t, 4>(_mm_blendv_epi8(alt.m_vec, v.m_vec, condition.native()));
#else // SIMD_SUPPORTS(SIMD_SSE4_1)
		return vector<std::int32_t, 4>(_mm_or_si128(_mm_and_si128(v.m_vec, condition.native()), _mm_andnot_si128(alt.m_vec, condition.native())));
#endif // SIMD_SUPPORTS(SIMD_SSE4_1)
	}

	std::ostream &operator<<(std::ostream &stream, const vector<std::int32_t, 4> &v) {
		stream << '(';
		for (size_t i = 0; i < v.width - 1; ++i) {
			stream << v.m_array[i] << ' ';
		}
		stream << v.m_array[v.width - 1] << ')';
		return stream;
	}

	mask<std::int32_t, 4> operator==(const mask<std::int32_t, 4> & v1, const mask<std::int32_t, 4> & v2) {
		return mask<std::int32_t, 4>(_mm_cmpeq_epi32(v1.m_vec, v2.m_vec));
	}

	mask<std::int32_t, 4> operator!=(const mask<std::int32_t, 4> & v1, const mask<std::int32_t, 4> & v2) {
		// Invert because SSE doesn't have instruction for it
		return mask<std::int32_t, 4>(_mm_xor_si128(_mm_cmpeq_epi32(v1.m_vec, v2.m_vec), _mm_set1_epi32(-1)));
	}

	mask<std::int32_t, 4> & mask<std::int32_t, 4>::operator&=(const mask<std::int32_t, 4> & v) {
		m_vec = _mm_and_si128(m_vec, v.m_vec);
		return *this;
	}

	mask<std::int32_t, 4> & mask<std::int32_t, 4>::operator|=(const mask<std::int32_t, 4> & v) {
		m_vec = _mm_or_si128(m_vec, v.m_vec);
		return *this;
	}

	mask<std::int32_t, 4> & mask<std::int32_t, 4>::operator^=(const mask<std::int32_t, 4> & v) {
		m_vec = _mm_xor_si128(m_vec, v.m_vec);
		return *this;
	}

	mask<std::int32_t, 4> operator~(const mask<std::int32_t, 4> & v) {
		return mask<std::int32_t, 4>(_mm_xor_si128(v.m_vec, _mm_set1_epi32(-1)));
	}

	mask<std::int32_t, 4> operator&(mask<std::int32_t, 4> v1, const mask<std::int32_t, 4> & v2) {
		return v1 &= v2;
	}

	mask<std::int32_t, 4> operator|(mask<std::int32_t, 4> v1, const mask<std::int32_t, 4> & v2) {
		return v1 |= v2;
	}

	mask<std::int32_t, 4> operator^(mask<std::int32_t, 4> v1, const mask<std::int32_t, 4> & v2) {
		return v1 ^= v2;
	}

	mask<std::int32_t, 4> andnot(const mask<std::int32_t, 4> & v1, const mask<std::int32_t, 4> & v2) {
		return mask<std::int32_t, 4>(_mm_andnot_si128(v1.m_vec, v2.m_vec));
	}

	int mask<std::int32_t, 4>::get_mask() const {
		return _mm_movemask_epi8(m_vec);
	}

	bool mask<std::int32_t, 4>::all() const {
		// TODO: what is faster here?
		return _mm_movemask_epi8(m_vec) == 0b1111111111111111;
	}

	bool mask<std::int32_t, 4>::any() const {
		return _mm_movemask_epi8(m_vec);
	}

	bool mask<std::int32_t, 4>::none() const {
		return !_mm_movemask_epi8(m_vec);
	}

#endif // SIMD_SUPPORTS(SIMD_SSE2)

} // namespace simd