#pragma once

#include <array>
#include <cstdint>
#include <exception>
#include "base_types.hpp"
#include "simd.hpp"

namespace simd {

#if SIMD_SUPPORTS(SIMD_AVX2)
	using int64x4 = vector<std::int64_t, 4>;

	template <>
	class vector<std::int64_t, 4> : public vector_base<std::int64_t, 4> {
	public:
		static constexpr int required_version = native_vector<type, width>::required_version;

		vector() : vector_base() {}
		explicit vector(native_type v) : vector_base(v) {}
		explicit vector(type f) : vector_base(_mm256_set1_epi64x(f)) {}
		explicit vector(type f1, type f2, type f3, type f4) : vector_base(_mm256_set_epi64x(f4, f3, f2, f1)) {}
		explicit vector(const std::array<type, width>& arr) : vector_base(_mm256_set_epi64x(arr[3], arr[2], arr[1], arr[0])) {}
		explicit vector(const type* vals) : vector_base(_mm256_lddqu_si256(reinterpret_cast<const native_type*>(vals))) {}
		explicit vector(const type* vals, aligned_load) : vector_base(_mm256_load_si256(reinterpret_cast<const native_type*>(vals))) {}

		vector<std::int64_t, 4> & operator+=(const vector<std::int64_t, 4> & v);
		vector<std::int64_t, 4> & operator-=(const vector<std::int64_t, 4> & v);
		vector<std::int64_t, 4> & operator*=(const vector<std::int64_t, 4> & v);
		vector<std::int64_t, 4> & operator/=(const vector<std::int64_t, 4> & v);
		friend vector<std::int64_t, 4> operator+(vector<std::int64_t, 4> v1, const vector<std::int64_t, 4> & v2);
		friend vector<std::int64_t, 4> operator-(vector<std::int64_t, 4> v1, const vector<std::int64_t, 4> & v2);
		friend vector<std::int64_t, 4> operator*(vector<std::int64_t, 4> v1, const vector<std::int64_t, 4> & v2);
		friend vector<std::int64_t, 4> operator/(vector<std::int64_t, 4> v1, const vector<std::int64_t, 4> & v2);

		vector<std::int64_t, 4> & operator&=(const vector<std::int64_t, 4> & v);
		vector<std::int64_t, 4> & operator|=(const vector<std::int64_t, 4> & v);
		vector<std::int64_t, 4> & operator^=(const vector<std::int64_t, 4> & v);
		friend vector<std::int64_t, 4> operator~(const vector<std::int64_t, 4> & v);
		friend vector<std::int64_t, 4> operator&(vector<std::int64_t, 4> v1, const vector<std::int64_t, 4> & v2);
		friend vector<std::int64_t, 4> operator|(vector<std::int64_t, 4> v1, const vector<std::int64_t, 4> & v2);
		friend vector<std::int64_t, 4> operator^(vector<std::int64_t, 4> v1, const vector<std::int64_t, 4> & v2);

		friend vector<std::int64_t, 4> operator<<(const vector<std::int64_t, 4> & v, int bits);
		friend vector<std::int64_t, 4> operator>>(const vector<std::int64_t, 4> & v, int bits);

		friend vector<std::int64_t, 4> operator==(const vector<std::int64_t, 4> & v1, const vector<std::int64_t, 4> & v2);
		friend vector<std::int64_t, 4> operator!=(const vector<std::int64_t, 4> & v1, const vector<std::int64_t, 4> & v2);
		friend vector<std::int64_t, 4> operator>(const vector<std::int64_t, 4> & v1, const vector<std::int64_t, 4> & v2);
		friend vector<std::int64_t, 4> operator>=(const vector<std::int64_t, 4> & v1, const vector<std::int64_t, 4> & v2);
		friend vector<std::int64_t, 4> operator<(const vector<std::int64_t, 4> & v1, const vector<std::int64_t, 4> & v2);
		friend vector<std::int64_t, 4> operator<=(const vector<std::int64_t, 4> & v1, const vector<std::int64_t, 4> & v2);

		vector<std::int64_t, 4> & hadd(const vector<std::int64_t, 4> & v);
		vector<std::int64_t, 4> & hsub(const vector<std::int64_t, 4> & v);
		type hadd() const;
		type hsub() const;
		friend vector<std::int64_t, 4> hadd(vector<std::int64_t, 4> v1, const vector<std::int64_t, 4> & v2);
		friend vector<std::int64_t, 4> hsub(vector<std::int64_t, 4> v1, const vector<std::int64_t, 4> & v2);

		vector<std::int64_t, 4> & abs();
		friend vector<std::int64_t, 4> abs(vector<std::int64_t, 4> v);
		friend vector<std::int64_t, 4> min(const vector<std::int64_t, 4> & v1, const vector<std::int64_t, 4> & v2);
		friend vector<std::int64_t, 4> max(const vector<std::int64_t, 4> & v1, const vector<std::int64_t, 4> & v2);

		friend vector<std::int64_t, 4> select(const vector<std::int64_t, 4> & v, const vector<std::int64_t, 4> & alt, const mask<std::int64_t, 4> & condition);

		friend std::ostream& operator<<(std::ostream& stream, const vector<std::int64_t, 4> & v);
	};

	template <>
	class mask<std::int64_t, 4> : public vector_base<std::int64_t, 4> {
	public:
		SIMD_FORCEINLINE mask() : vector_base() {}
		explicit SIMD_FORCEINLINE mask(native_type v) : vector_base(v) {}
		explicit SIMD_FORCEINLINE mask(bool b) : vector_base(_mm256_set1_epi64x(-static_cast<int>(b))) {}
		explicit SIMD_FORCEINLINE mask(bool b1, bool b2, bool b3, bool b4) : vector_base(_mm256_set_epi64x(
			-static_cast<int>(b4), -static_cast<int>(b3), -static_cast<int>(b2), -static_cast<int>(b1))) {}
		explicit SIMD_FORCEINLINE mask(const std::array<bool, width>& arr) : mask(arr[3], arr[2], arr[1], arr[0]) {}
		explicit SIMD_FORCEINLINE mask(const vector<std::int64_t, 4> & v) : vector_base(v) {}

		friend SIMD_FORCEINLINE mask<std::int64_t, 4> operator==(const mask<std::int64_t, 4> & v1, const mask<std::int64_t, 4> & v2);
		friend SIMD_FORCEINLINE mask<std::int64_t, 4> operator!=(const mask<std::int64_t, 4> & v1, const mask<std::int64_t, 4> & v2);

		SIMD_FORCEINLINE mask<std::int64_t, 4> & operator&=(const mask<std::int64_t, 4> & v);
		SIMD_FORCEINLINE mask<std::int64_t, 4> & operator|=(const mask<std::int64_t, 4> & v);
		SIMD_FORCEINLINE mask<std::int64_t, 4> & operator^=(const mask<std::int64_t, 4> & v);

		friend SIMD_FORCEINLINE mask<std::int64_t, 4> operator~(const mask<std::int64_t, 4> & v);
		friend SIMD_FORCEINLINE mask<std::int64_t, 4> operator&(mask<std::int64_t, 4> v1, const mask<std::int64_t, 4> & v2);
		friend SIMD_FORCEINLINE mask<std::int64_t, 4> operator|(mask<std::int64_t, 4> v1, const mask<std::int64_t, 4> & v2);
		friend SIMD_FORCEINLINE mask<std::int64_t, 4> operator^(mask<std::int64_t, 4> v1, const mask<std::int64_t, 4> & v2);
		friend SIMD_FORCEINLINE mask<std::int64_t, 4> andnot(const mask<std::int64_t, 4> & v1, const mask<std::int64_t, 4> & v2);

		SIMD_FORCEINLINE int get_mask() const;
		SIMD_FORCEINLINE bool all() const;
		SIMD_FORCEINLINE bool any() const;
		SIMD_FORCEINLINE bool none() const;
	};

	vector<std::int64_t, 4> & vector<std::int64_t, 4>::operator+=(const vector<std::int64_t, 4> & v) {
		m_vec = _mm256_add_epi64(m_vec, v.m_vec);
		return *this;
	}

	vector<std::int64_t, 4> & vector<std::int64_t, 4>::operator-=(const vector<std::int64_t, 4> & v) {
		m_vec = _mm256_sub_epi64(m_vec, v.m_vec);
		return *this;
	}

	vector<std::int64_t, 4> & vector<std::int64_t, 4>::operator*=(const vector<std::int64_t, 4> & v) {
		// TODO: AVX 512
		throw std::runtime_error("Operation not implemented");
		return *this;
	}

	vector<std::int64_t, 4> & vector<std::int64_t, 4>::operator/=(const vector<std::int64_t, 4> & v) {
		// TODO: AVX 512
		throw std::runtime_error("Operation not implemented");
		return *this;
	}

	vector<std::int64_t, 4> & vector<std::int64_t, 4>::hadd(const vector<std::int64_t, 4> & v) {
		throw std::runtime_error("Operation not implemented");
		return *this;
	}

	vector<std::int64_t, 4> & vector<std::int64_t, 4>::hsub(const vector<std::int64_t, 4> & v) {
		throw std::runtime_error("Operation not implemented");
		return *this;
	}

	std::int64_t vector<std::int64_t, 4>::hadd() const {
		// TODO: shuffle
		throw std::runtime_error("Operation not implemented");
	}

	std::int64_t vector<std::int64_t, 4>::hsub() const {
		// TODO: shuffle
		throw std::runtime_error("Operation not implemented");
	}

	vector<std::int64_t, 4> hadd(vector<std::int64_t, 4> v1, const vector<std::int64_t, 4> & v2) {
		return v1.hadd(v2);
	}

	vector<std::int64_t, 4> hsub(vector<std::int64_t, 4> v1, const vector<std::int64_t, 4> & v2) {
		return v1.hsub(v2);
	}

	vector<std::int64_t, 4> operator+(vector<std::int64_t, 4> v1, const vector<std::int64_t, 4> & v2) {
		return (v1 += v2);
	}

	vector<std::int64_t, 4> operator-(vector<std::int64_t, 4> v1, const vector<std::int64_t, 4> & v2) {
		return (v1 -= v2);
	}

	vector<std::int64_t, 4> operator*(vector<std::int64_t, 4> v1, const vector<std::int64_t, 4> & v2) {
		return (v1 *= v2);
	}

	vector<std::int64_t, 4> operator/(vector<std::int64_t, 4> v1, const vector<std::int64_t, 4> & v2) {
		return (v1 /= v2);
	}

	vector<std::int64_t, 4> & vector<std::int64_t, 4>::operator&=(const vector<std::int64_t, 4> & v) {
		m_vec = _mm256_and_si256(m_vec, v.m_vec);
		return *this;
	}

	vector<std::int64_t, 4> & vector<std::int64_t, 4>::operator|=(const vector<std::int64_t, 4> & v) {
		m_vec = _mm256_or_si256(m_vec, v.m_vec);
		return *this;
	}

	vector<std::int64_t, 4> & vector<std::int64_t, 4>::operator^=(const vector<std::int64_t, 4> & v) {
		m_vec = _mm256_xor_si256(m_vec, v.m_vec);
		return *this;
	}

	vector<std::int64_t, 4> operator~(const vector<std::int64_t, 4> & v) {
		return vector<std::int64_t, 4>(_mm256_xor_si256(v.m_vec, _mm256_set1_epi64x(-1)));
	}

	vector<std::int64_t, 4> operator&(vector<std::int64_t, 4> v1, const vector<std::int64_t, 4> & v2) {
		return v1 &= v2;
	}

	vector<std::int64_t, 4> operator|(vector<std::int64_t, 4> v1, const vector<std::int64_t, 4> & v2) {
		return v1 |= v2;
	}

	vector<std::int64_t, 4> operator^(vector<std::int64_t, 4> v1, const vector<std::int64_t, 4> & v2) {
		return v1 ^= v2;
	}

	vector<std::int64_t, 4> operator<<(const vector<std::int64_t, 4> & v, int bits) {
		return vector<std::int64_t, 4>(_mm256_slli_epi64(v.m_vec, bits));
	}

	vector<std::int64_t, 4> operator>>(const vector<std::int64_t, 4> & v, int bits) {
		return vector<std::int64_t, 4>(_mm256_slli_epi64(v.m_vec, bits));
	}

	vector<std::int64_t, 4> operator==(const vector<std::int64_t, 4> & v1, const vector<std::int64_t, 4> & v2) {
		return vector<std::int64_t, 4>(_mm256_cmpeq_epi64(v1.m_vec, v2.m_vec));
	}

	vector<std::int64_t, 4> operator!=(const vector<std::int64_t, 4> & v1, const vector<std::int64_t, 4> & v2) {
		// Invert because SSE doesn't have instruction for it
		return vector<std::int64_t, 4>(_mm256_xor_si256(_mm256_cmpeq_epi64(v1.m_vec, v2.m_vec), _mm256_set1_epi64x(-1)));
	}

	vector<std::int64_t, 4> operator>(const vector<std::int64_t, 4> & v1, const vector<std::int64_t, 4> & v2) {
		return vector<std::int64_t, 4>(_mm256_cmpgt_epi64(v1.m_vec, v2.m_vec));
	}

	vector<std::int64_t, 4> operator>=(const vector<std::int64_t, 4> & v1, const vector<std::int64_t, 4> & v2) {
		// Invert because SSE doesn't have instruction for it (also note the switched operands!)
		return vector<std::int64_t, 4>(_mm256_xor_si256(_mm256_cmpgt_epi64(v2.m_vec, v1.m_vec), _mm256_set1_epi64x(-1)));
	}

	vector<std::int64_t, 4> operator<(const vector<std::int64_t, 4> & v1, const vector<std::int64_t, 4> & v2) {
		// Note the switched operands!
		return vector<std::int64_t, 4>(_mm256_cmpgt_epi64(v2.m_vec, v1.m_vec));
	}

	vector<std::int64_t, 4> operator<=(const vector<std::int64_t, 4> & v1, const vector<std::int64_t, 4> & v2) {
		// Invert because SSE doesn't have instruction for it
		return vector<std::int64_t, 4>(_mm256_xor_si256(_mm256_cmpgt_epi64(v1.m_vec, v2.m_vec), _mm256_set1_epi64x(-1)));
	}

	vector<std::int64_t, 4> & vector<std::int64_t, 4>::abs() {
		// TODO: benchmark for most performant solution
		// TODO: AVX 512
		throw std::runtime_error("Operation not implemented");
		return *this;
	}

	vector<std::int64_t, 4> abs(vector<std::int64_t, 4> v) {
		return v.abs();
	}

	vector<std::int64_t, 4> min(const vector<std::int64_t, 4> & v1, const vector<std::int64_t, 4> & v2) {
		throw std::runtime_error("Operation not implemented");
		// TODO: AVX 512
	}

	vector<std::int64_t, 4> max(const vector<std::int64_t, 4> & v1, const vector<std::int64_t, 4> & v2) {
		throw std::runtime_error("Operation not implemented");
		// TODO: AVX 512
	}

	vector<std::int64_t, 4> select(const vector<std::int64_t, 4> & v, const vector<std::int64_t, 4> & alt, const mask<std::int64_t, 4> & condition) {
		// TODO: this only works when mask is either 0 or -1 (because it's byte-wise)
		return vector<std::int64_t, 4>(_mm256_blendv_epi8(alt.m_vec, v.m_vec, condition.native()));
	}

	std::ostream& operator<<(std::ostream& stream, const vector<std::int64_t, 4> & v) {
		stream << '(';
		for(size_t i = 0; i < v.width - 1; ++i) {
			stream << v.m_array[i] << ' ';
		}
		stream << v.m_array[v.width - 1] << ')';
		return stream;
	}

	mask<std::int64_t, 4> operator==(const mask<std::int64_t, 4> & v1, const mask<std::int64_t, 4> & v2) {
		return mask<std::int64_t, 4>(_mm256_cmpeq_epi64(v1.m_vec, v2.m_vec));
	}

	mask<std::int64_t, 4> operator!=(const mask<std::int64_t, 4> & v1, const mask<std::int64_t, 4> & v2) {
		// Invert because SSE doesn't have instruction for it
		return mask<std::int64_t, 4>(_mm256_xor_si256(_mm256_cmpeq_epi64(v1.m_vec, v2.m_vec), _mm256_set1_epi64x(-1)));
	}

	mask<std::int64_t, 4> & mask<std::int64_t, 4>::operator&=(const mask<std::int64_t, 4> & v) {
		m_vec = _mm256_and_si256(m_vec, v.m_vec);
		return *this;
	}

	mask<std::int64_t, 4> & mask<std::int64_t, 4>::operator|=(const mask<std::int64_t, 4> & v) {
		m_vec = _mm256_or_si256(m_vec, v.m_vec);
		return *this;
	}

	mask<std::int64_t, 4> & mask<std::int64_t, 4>::operator^=(const mask<std::int64_t, 4> & v) {
		m_vec = _mm256_xor_si256(m_vec, v.m_vec);
		return *this;
	}

	mask<std::int64_t, 4> operator~(const mask<std::int64_t, 4> & v) {
		return mask<std::int64_t, 4>(_mm256_xor_si256(v.m_vec, _mm256_set1_epi64x(-1)));
	}

	mask<std::int64_t, 4> operator&(mask<std::int64_t, 4> v1, const mask<std::int64_t, 4> & v2) {
		return v1 &= v2;
	}

	mask<std::int64_t, 4> operator|(mask<std::int64_t, 4> v1, const mask<std::int64_t, 4> & v2) {
		return v1 |= v2;
	}

	mask<std::int64_t, 4> operator^(mask<std::int64_t, 4> v1, const mask<std::int64_t, 4> & v2) {
		return v1 ^= v2;
	}

	mask<std::int64_t, 4> andnot(const mask<std::int64_t, 4> & v1, const mask<std::int64_t, 4> & v2) {
		return mask<std::int64_t, 4>(_mm256_andnot_si256(v1.m_vec, v2.m_vec));
	}

	int mask<std::int64_t, 4>::get_mask() const {
		return _mm256_movemask_epi8(m_vec);
	}

	bool mask<std::int64_t, 4>::all() const {
		// TODO: what is faster here?
		return _mm256_movemask_epi8(m_vec) == 0xFFFFFFFF;
	}

	bool mask<std::int64_t, 4>::any() const {
		return _mm256_movemask_epi8(m_vec);
	}

	bool mask<std::int64_t, 4>::none() const {
		return !_mm256_movemask_epi8(m_vec);
	}

#endif // SIMD_SUPPORTS(SIMD_AVX2)

} // namespace simd