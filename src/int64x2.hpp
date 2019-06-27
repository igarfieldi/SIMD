#pragma once

#include <array>
#include <exception>
#include "base_types.hpp"
#include "simd.hpp"

namespace simd {

#if SIMD_SUPPORTS(SIMD_SSE2)
	using int64x2 = vector<std::int64_t, 2>;

	template <>
	class vector<std::int64_t, 2> : public vector_base<std::int64_t, 2> {
	public:
		static constexpr int required_version = native_vector<type, width>::required_version;

		vector() : vector_base() {}
		explicit vector(native_type v) : vector_base(v) {}
		explicit vector(type f) : vector_base(_mm_set1_epi64x(f)) {}
		explicit vector(type f1, type f2) : vector_base(_mm_set_epi64x(f2, f1)) {}
		explicit vector(const std::array<type, width>& arr) : vector_base(_mm_set_epi64x(arr[1], arr[0])) {}
#if SIMD_SUPPORTS(SIMD_SSE3)
		explicit vector(const type *vals) : vector_base(_mm_lddqu_si128(reinterpret_cast<const native_type *>(vals))) {}
#else // SIMD_SUPPORTS(SIMD_SSE3)
		explicit vector(const type *vals) : vector_base(_mm_loadu_si128(reinterpret_cast<const native_type *>(vals))) {}
#endif // SIMD_SUPPORTS(SIMD_SSE3)
		explicit vector(const type *vals, aligned_load) : vector_base(_mm_load_si128(reinterpret_cast<const native_type *>(vals))) {}
		//explicit vector(const vector<float, 4> &v);
		//explicit vector(const vector<double, 2> &v);
#if SIMD_SUPPORTS(SIMD_AVX)
		//explicit vector(const vector<double, 4> &v);
#endif // SIMD_SUPPORTS(SIMD_AVX)

		vector<std::int64_t, 2> &operator+=(const vector<std::int64_t, 2> &v);
		vector<std::int64_t, 2> &operator-=(const vector<std::int64_t, 2> &v);
		vector<std::int64_t, 2> &operator*=(const vector<std::int64_t, 2> &v);
		vector<std::int64_t, 2> &operator/=(const vector<std::int64_t, 2> &v);
		friend vector<std::int64_t, 2> operator+(vector<std::int64_t, 2> v1, const vector<std::int64_t, 2> &v2);
		friend vector<std::int64_t, 2> operator-(vector<std::int64_t, 2> v1, const vector<std::int64_t, 2> &v2);
		friend vector<std::int64_t, 2> operator*(vector<std::int64_t, 2> v1, const vector<std::int64_t, 2> &v2);
		friend vector<std::int64_t, 2> operator/(vector<std::int64_t, 2> v1, const vector<std::int64_t, 2> &v2);

		vector<std::int64_t, 2> &operator&=(const vector<std::int64_t, 2> &v);
		vector<std::int64_t, 2> &operator|=(const vector<std::int64_t, 2> &v);
		vector<std::int64_t, 2> &operator^=(const vector<std::int64_t, 2> &v);
		friend vector<std::int64_t, 2> operator~(const vector<std::int64_t, 2> &v);
		friend vector<std::int64_t, 2> operator&(vector<std::int64_t, 2> v1, const vector<std::int64_t, 2> &v2);
		friend vector<std::int64_t, 2> operator|(vector<std::int64_t, 2> v1, const vector<std::int64_t, 2> &v2);
		friend vector<std::int64_t, 2> operator^(vector<std::int64_t, 2> v1, const vector<std::int64_t, 2> &v2);

		friend vector<std::int64_t, 2> operator<<(const vector<std::int64_t, 2> &v, int bits);
		friend vector<std::int64_t, 2> operator>>(const vector<std::int64_t, 2> &v, int bits);

		friend vector<std::int64_t, 2> operator==(const vector<std::int64_t, 2> &v1, const vector<std::int64_t, 2> &v2);
		friend vector<std::int64_t, 2> operator!=(const vector<std::int64_t, 2> &v1, const vector<std::int64_t, 2> &v2);
		friend vector<std::int64_t, 2> operator>(const vector<std::int64_t, 2> &v1, const vector<std::int64_t, 2> &v2);
		friend vector<std::int64_t, 2> operator>=(const vector<std::int64_t, 2> &v1, const vector<std::int64_t, 2> &v2);
		friend vector<std::int64_t, 2> operator<(const vector<std::int64_t, 2> &v1, const vector<std::int64_t, 2> &v2);
		friend vector<std::int64_t, 2> operator<=(const vector<std::int64_t, 2> &v1, const vector<std::int64_t, 2> &v2);

		vector<std::int64_t, 2> &hadd(const vector<std::int64_t, 2> &v);
		vector<std::int64_t, 2> &hsub(const vector<std::int64_t, 2> &v);
		type hadd() const;
		type hsub() const;
		friend vector<std::int64_t, 2> hadd(vector<std::int64_t, 2> v1, const vector<std::int64_t, 2> &v2);
		friend vector<std::int64_t, 2> hsub(vector<std::int64_t, 2> v1, const vector<std::int64_t, 2> &v2);

		vector<std::int64_t, 2> &abs();
		friend vector<std::int64_t, 2> abs(vector<std::int64_t, 2> v);
		friend vector<std::int64_t, 2> min(const vector<std::int64_t, 2> &v1, const vector<std::int64_t, 2> &v2);
		friend vector<std::int64_t, 2> max(const vector<std::int64_t, 2> &v1, const vector<std::int64_t, 2> &v2);

		friend vector<std::int64_t, 2> select(const vector<std::int64_t, 2> &v, const vector<std::int64_t, 2> &alt, const mask<std::int64_t, 2> &condition);

		friend std::ostream &operator<<(std::ostream &stream, const vector<std::int64_t, 2> &v);
	};

	template <>
	class mask<std::int64_t, 2> : public vector_base<std::int64_t, 2> {
	public:
		SIMD_FORCEINLINE mask() : vector_base() {}
		explicit SIMD_FORCEINLINE mask(native_type v) : vector_base(v) {}
		explicit SIMD_FORCEINLINE mask(bool b) : vector_base(_mm_set1_epi64x(-static_cast<int>(b))) {}
		explicit SIMD_FORCEINLINE mask(bool b1, bool b2) : vector_base(_mm_set_epi64x(-static_cast<int>(b2), -static_cast<int>(b1))) {}
		explicit SIMD_FORCEINLINE mask(const std::array<bool, width>& arr) : mask(arr[1], arr[0]) {}
		explicit SIMD_FORCEINLINE mask(const vector<std::int64_t, 2> & v) : vector_base(v) {}

		friend SIMD_FORCEINLINE mask<std::int64_t, 2> operator==(const mask<std::int64_t, 2> & v1, const mask<std::int64_t, 2> & v2);
		friend SIMD_FORCEINLINE mask<std::int64_t, 2> operator!=(const mask<std::int64_t, 2> & v1, const mask<std::int64_t, 2> & v2);

		SIMD_FORCEINLINE mask<std::int64_t, 2> & operator&=(const mask<std::int64_t, 2> & v);
		SIMD_FORCEINLINE mask<std::int64_t, 2> & operator|=(const mask<std::int64_t, 2> & v);
		SIMD_FORCEINLINE mask<std::int64_t, 2> & operator^=(const mask<std::int64_t, 2> & v);

		friend SIMD_FORCEINLINE mask<std::int64_t, 2> operator~(const mask<std::int64_t, 2> & v);
		friend SIMD_FORCEINLINE mask<std::int64_t, 2> operator&(mask<std::int64_t, 2> v1, const mask<std::int64_t, 2> & v2);
		friend SIMD_FORCEINLINE mask<std::int64_t, 2> operator|(mask<std::int64_t, 2> v1, const mask<std::int64_t, 2> & v2);
		friend SIMD_FORCEINLINE mask<std::int64_t, 2> operator^(mask<std::int64_t, 2> v1, const mask<std::int64_t, 2> & v2);
		friend SIMD_FORCEINLINE mask<std::int64_t, 2> andnot(const mask<std::int64_t, 2> & v1, const mask<std::int64_t, 2> & v2);

		SIMD_FORCEINLINE int get_mask() const;
		SIMD_FORCEINLINE bool all() const;
		SIMD_FORCEINLINE bool any() const;
		SIMD_FORCEINLINE bool none() const;
	};

	vector<std::int64_t, 2> &vector<std::int64_t, 2>::operator+=(const vector<std::int64_t, 2> &v) {
		m_vec = _mm_add_epi64(m_vec, v.m_vec);
		return *this;
	}

	vector<std::int64_t, 2> &vector<std::int64_t, 2>::operator-=(const vector<std::int64_t, 2> &v) {
		m_vec = _mm_sub_epi64(m_vec, v.m_vec);
		return *this;
	}

	vector<std::int64_t, 2> &vector<std::int64_t, 2>::operator*=(const vector<std::int64_t, 2> &v) {
		// TODO: there's gotta be a better way
		// TODO: AVX 512
		throw std::runtime_error("Operation not implemented");
		//m_vec = _mm_cvtpd_epi64(_mm_mul_pd(_mm_cvtepi64_pd(m_vec), _mm_cvtepi64_pd(v.m_vec)));
		return *this;
	}

	vector<std::int64_t, 2> &vector<std::int64_t, 2>::operator/=(const vector<std::int64_t, 2> &v) {
		// TODO: there's gotta be a better way
		// TODO: AVX 512
		throw std::runtime_error("Operation not implemented");
		//m_vec = _mm_cvtps_epi64(_mm_div_ps(_mm_cvtepi32_ps(m_vec), _mm_cvtepi32_ps(v.m_vec)));
		return *this;
	}

	vector<std::int64_t, 2> &vector<std::int64_t, 2>::hadd(const vector<std::int64_t, 2> &v) {
		throw std::runtime_error("Operation not implemented");
		return *this;
	}

	vector<std::int64_t, 2> &vector<std::int64_t, 2>::hsub(const vector<std::int64_t, 2> &v) {
		throw std::runtime_error("Operation not implemented");
		return *this;
	}

	std::int64_t vector<std::int64_t, 2>::hadd() const {
		// Shuffle 32 bits due to lack of native intrinsic
		return _mm_cvtsi128_si64(_mm_add_epi64(m_vec, _mm_shuffle_epi32(m_vec, 0b00001011)));
	}

	std::int64_t vector<std::int64_t, 2>::hsub() const {
		// Shuffle 32 bits due to lack of native intrinsic
		return _mm_cvtsi128_si64(_mm_sub_epi64(m_vec, _mm_shuffle_epi32(m_vec, 0b00001011)));
	}

	vector<std::int64_t, 2> hadd(vector<std::int64_t, 2> v1, const vector<std::int64_t, 2> &v2) {
		return v1.hadd(v2);
	}

	vector<std::int64_t, 2> hsub(vector<std::int64_t, 2> v1, const vector<std::int64_t, 2> &v2) {
		return v1.hsub(v2);
	}

	vector<std::int64_t, 2> operator+(vector<std::int64_t, 2> v1, const vector<std::int64_t, 2> &v2) {
		return (v1 += v2);
	}

	vector<std::int64_t, 2> operator-(vector<std::int64_t, 2> v1, const vector<std::int64_t, 2> &v2) {
		return (v1 -= v2);
	}

	vector<std::int64_t, 2> operator*(vector<std::int64_t, 2> v1, const vector<std::int64_t, 2> &v2) {
		return (v1 *= v2);
	}

	vector<std::int64_t, 2> operator/(vector<std::int64_t, 2> v1, const vector<std::int64_t, 2> &v2) {
		return (v1 /= v2);
	}

	vector<std::int64_t, 2> &vector<std::int64_t, 2>::operator&=(const vector<std::int64_t, 2> &v) {
		m_vec = _mm_and_si128(m_vec, v.m_vec);
		return *this;
	}

	vector<std::int64_t, 2> &vector<std::int64_t, 2>::operator|=(const vector<std::int64_t, 2> &v) {
		m_vec = _mm_or_si128(m_vec, v.m_vec);
		return *this;
	}

	vector<std::int64_t, 2> &vector<std::int64_t, 2>::operator^=(const vector<std::int64_t, 2> &v) {
		m_vec = _mm_xor_si128(m_vec, v.m_vec);
		return *this;
	}

	vector<std::int64_t, 2> operator~(const vector<std::int64_t, 2> &v) {
		return vector<std::int64_t, 2>(_mm_xor_si128(v.m_vec, _mm_set1_epi64x(-1)));
	}

	vector<std::int64_t, 2> operator&(vector<std::int64_t, 2> v1, const vector<std::int64_t, 2> &v2) {
		return v1 &= v2;
	}

	vector<std::int64_t, 2> operator|(vector<std::int64_t, 2> v1, const vector<std::int64_t, 2> &v2) {
		return v1 |= v2;
	}

	vector<std::int64_t, 2> operator^(vector<std::int64_t, 2> v1, const vector<std::int64_t, 2> &v2) {
		return v1 ^= v2;
	}

	vector<std::int64_t, 2> operator<<(const vector<std::int64_t, 2> &v, int bits) {
		return vector<std::int64_t, 2>(_mm_slli_epi64(v.m_vec, bits));
	}

	vector<std::int64_t, 2> operator>>(const vector<std::int64_t, 2> &v, int bits) {
		return vector<std::int64_t, 2>(_mm_slli_epi64(v.m_vec, bits));
	}

	vector<std::int64_t, 2> operator==(const vector<std::int64_t, 2> &v1, const vector<std::int64_t, 2> &v2) {
#if SIMD_SUPPORTS(SIMD_SSE4_1)
		return vector<std::int64_t, 2>(_mm_cmpeq_epi64(v1.m_vec, v2.m_vec));
#else // SIMD_SUPPORTS(SIMD_SSE4_1)
		throw std::runtime_error("Operation not implemented");
#endif // SIMD_SUPPORTS(SIMD_SSE4_1)
	}

	vector<std::int64_t, 2> operator!=(const vector<std::int64_t, 2> &v1, const vector<std::int64_t, 2> &v2) {
#if SIMD_SUPPORTS(SIMD_SSE4_1)
		// Invert because SSE doesn't have instruction for it
		return vector<std::int64_t, 2>(_mm_xor_si128(_mm_cmpeq_epi64(v1.m_vec, v2.m_vec), _mm_set1_epi64x(-1)));
#else // SIMD_SUPPORTS(SIMD_SSE4_1)
		throw std::runtime_error("Operation not implemented");
#endif // SIMD_SUPPORTS(SIMD_SSE4_1)
	}

	vector<std::int64_t, 2> operator>(const vector<std::int64_t, 2> &v1, const vector<std::int64_t, 2> &v2) {
#if SIMD_SUPPORTS(SIMD_SSE4_1)
		return vector<std::int64_t, 2>(_mm_cmpgt_epi64(v1.m_vec, v2.m_vec));
#else // SIMD_SUPPORTS(SIMD_SSE4_1)
throw std::runtime_error("Operation not implemented");
#endif // SIMD_SUPPORTS(SIMD_SSE4_1)
	}

	vector<std::int64_t, 2> operator>=(const vector<std::int64_t, 2> &v1, const vector<std::int64_t, 2> &v2) {
#if SIMD_SUPPORTS(SIMD_SSE4_1)
		// Invert because SSE doesn't have instruction for it (also note the switched operands!)
		return vector<std::int64_t, 2>(_mm_xor_si128(_mm_cmpgt_epi64(v2.m_vec, v1.m_vec), _mm_set1_epi64x(-1)));
#else // SIMD_SUPPORTS(SIMD_SSE4_1)
		throw std::runtime_error("Operation not implemented");
#endif // SIMD_SUPPORTS(SIMD_SSE4_1)
	}

	vector<std::int64_t, 2> operator<(const vector<std::int64_t, 2> &v1, const vector<std::int64_t, 2> &v2) {
#if SIMD_SUPPORTS(SIMD_SSE4_1)
		// Note the switched operands!
		return vector<std::int64_t, 2>(_mm_cmpgt_epi64(v2.m_vec, v1.m_vec));
#else // SIMD_SUPPORTS(SIMD_SSE4_1)
		throw std::runtime_error("Operation not implemented");
#endif // SIMD_SUPPORTS(SIMD_SSE4_1)
	}

	vector<std::int64_t, 2> operator<=(const vector<std::int64_t, 2> &v1, const vector<std::int64_t, 2> &v2) {
#if SIMD_SUPPORTS(SIMD_SSE4_1)
		// Invert because SSE doesn't have instruction for it
		return vector<std::int64_t, 2>(_mm_xor_si128(_mm_cmpgt_epi64(v1.m_vec, v2.m_vec), _mm_set1_epi64x(-1)));
#else // SIMD_SUPPORTS(SIMD_SSE4_1)
throw std::runtime_error("Operation not implemented");
#endif // SIMD_SUPPORTS(SIMD_SSE4_1)
	}

	vector<std::int64_t, 2> &vector<std::int64_t, 2>::abs() {
		// TODO: benchmark for most performant solution
		// TODO: AVX 512
		throw std::runtime_error("Operation not implemented");
		return *this;
	}

	vector<std::int64_t, 2> abs(vector<std::int64_t, 2> v) {
		return v.abs();
	}

	vector<std::int64_t, 2> min(const vector<std::int64_t, 2> &v1, const vector<std::int64_t, 2> &v2) {
		// TODO: AVX 512
		throw std::runtime_error("Operation not implemented");
	}

	vector<std::int64_t, 2> max(const vector<std::int64_t, 2> &v1, const vector<std::int64_t, 2> &v2) {
		// TODO: AVX 512
		throw std::runtime_error("Operation not implemented");
	}

	vector<std::int64_t, 2> select(const vector<std::int64_t, 2> &v, const vector<std::int64_t, 2> &alt, const mask<std::int64_t, 2> &condition) {
		// TODO: this only works when mask is either 0 or -1 (because it's byte-wise)
#if SIMD_SUPPORTS(SIMD_SSE4_1)
		return vector<std::int64_t, 2>(_mm_blendv_epi8(alt.m_vec, v.m_vec, condition.native()));
#else // SIMD_SUPPORTS(SIMD_SSE4_1)
		return vector<std::int64_t, 2>(_mm_or_si128(_mm_and_si128(v.m_vec, condition.native()), _mm_andnot_si128(alt.m_vec, condition.native())));
#endif // SIMD_SUPPORTS(SIMD_SSE4_1)
	}

	std::ostream &operator<<(std::ostream &stream, const vector<std::int64_t, 2> &v) {
		stream << '(';
		for (size_t i = 0; i < v.width - 1; ++i) {
			stream << v.m_array[i] << ' ';
		}
		stream << v.m_array[v.width - 1] << ')';
		return stream;
	}

	mask<std::int64_t, 2> operator==(const mask<std::int64_t, 2> & v1, const mask<std::int64_t, 2> & v2) {
#if SIMD_SUPPORTS(SIMD_SSE4_1)
		return mask<std::int64_t, 2>(_mm_cmpeq_epi64(v1.m_vec, v2.m_vec));
#else // SIMD_SUPPORTS(SIMD_SSE4_1)
		throw std::runtime_error("Operation not implemented");
#endif // SIMD_SUPPORTS(SIMD_SSE4_1) 
	}

	mask<std::int64_t, 2> operator!=(const mask<std::int64_t, 2> & v1, const mask<std::int64_t, 2> & v2) {
#if SIMD_SUPPORTS(SIMD_SSE4_1)
		// Invert because SSE doesn't have instruction for it
		return mask<std::int64_t, 2>(_mm_xor_si128(_mm_cmpeq_epi64(v1.m_vec, v2.m_vec), _mm_set1_epi64x(-1)));
#else // SIMD_SUPPORTS(SIMD_SSE4_1)
		throw std::runtime_error("Operation not implemented");
#endif // SIMD_SUPPORTS(SIMD_SSE4_1) 
	}

	mask<std::int64_t, 2> & mask<std::int64_t, 2>::operator&=(const mask<std::int64_t, 2> & v) {
		m_vec = _mm_and_si128(m_vec, v.m_vec);
		return *this;
	}

	mask<std::int64_t, 2> & mask<std::int64_t, 2>::operator|=(const mask<std::int64_t, 2> & v) {
		m_vec = _mm_or_si128(m_vec, v.m_vec);
		return *this;
	}

	mask<std::int64_t, 2> & mask<std::int64_t, 2>::operator^=(const mask<std::int64_t, 2> & v) {
		m_vec = _mm_xor_si128(m_vec, v.m_vec);
		return *this;
	}

	mask<std::int64_t, 2> operator~(const mask<std::int64_t, 2> & v) {
		return mask<std::int64_t, 2>(_mm_xor_si128(v.m_vec, _mm_set1_epi64x(-1)));
	}

	mask<std::int64_t, 2> operator&(mask<std::int64_t, 2> v1, const mask<std::int64_t, 2> & v2) {
		return v1 &= v2;
	}

	mask<std::int64_t, 2> operator|(mask<std::int64_t, 2> v1, const mask<std::int64_t, 2> & v2) {
		return v1 |= v2;
	}

	mask<std::int64_t, 2> operator^(mask<std::int64_t, 2> v1, const mask<std::int64_t, 2> & v2) {
		return v1 ^= v2;
	}

	mask<std::int64_t, 2> andnot(const mask<std::int64_t, 2> & v1, const mask<std::int64_t, 2> & v2) {
		return mask<std::int64_t, 2>(_mm_andnot_si128(v1.m_vec, v2.m_vec));
	}

	int mask<std::int64_t, 2>::get_mask() const {
		return _mm_movemask_epi8(m_vec);
	}

	bool mask<std::int64_t, 2>::all() const {
		// TODO: what is faster here?
		return _mm_movemask_epi8(m_vec) == 0b1111111111111111;
	}

	bool mask<std::int64_t, 2>::any() const {
		return _mm_movemask_epi8(m_vec);
	}

	bool mask<std::int64_t, 2>::none() const {
		return !_mm_movemask_epi8(m_vec);
	}

#endif // SIMD_SUPPORTS(SIMD_SSE2)

} // namespace simd