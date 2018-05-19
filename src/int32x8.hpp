#pragma once

#include "base_types.hpp"
#include "simd.hpp"
#include <array>
#include <assert.h>
#include <cstdint>
#include <exception>

namespace simd {

#if SIMD_SUPPORTS(SIMD_AVX2)
	using int32x8 = vector<std::int32_t, 8>;

	template <>
	class vector<std::int32_t, 8> : public vector_base<std::int32_t, 8> {
	public:
		static constexpr int required_version = native_vector<type, width>::required_version;

	public:
		vector() : vector_base() {}
		explicit vector(native_type v) : vector_base(v) {}
		explicit vector(type f) : vector_base(_mm256_set1_epi32(f)) {}
		explicit vector(const type *vals) : vector_base(_mm256_lddqu_si256(reinterpret_cast<const native_type *>(vals))) {}
		explicit vector(const type *vals, aligned_load) : vector_base(_mm256_load_si256(reinterpret_cast<const native_type *>(vals))) {}
		explicit vector(const vector<float, 8> &v);

		vector<std::int32_t, 8> &operator+=(const vector<std::int32_t, 8> &v);
		vector<std::int32_t, 8> &operator-=(const vector<std::int32_t, 8> &v);
		vector<std::int32_t, 8> &operator*=(const vector<std::int32_t, 8> &v);
		vector<std::int32_t, 8> &operator/=(const vector<std::int32_t, 8> &v);
		friend vector<std::int32_t, 8> operator+(vector<std::int32_t, 8> v1, const vector<std::int32_t, 8> &v2);
		friend vector<std::int32_t, 8> operator-(vector<std::int32_t, 8> v1, const vector<std::int32_t, 8> &v2);
		friend vector<std::int32_t, 8> operator*(vector<std::int32_t, 8> v1, const vector<std::int32_t, 8> &v2);
		friend vector<std::int32_t, 8> operator/(vector<std::int32_t, 8> v1, const vector<std::int32_t, 8> &v2);

		vector<std::int32_t, 8> &operator&=(const vector<std::int32_t, 8> &v);
		vector<std::int32_t, 8> &operator|=(const vector<std::int32_t, 8> &v);
		vector<std::int32_t, 8> &operator^=(const vector<std::int32_t, 8> &v);
		friend vector<std::int32_t, 8> operator~(const vector<std::int32_t, 8> &v);
		friend vector<std::int32_t, 8> operator&(vector<std::int32_t, 8> v1, const vector<std::int32_t, 8> &v2);
		friend vector<std::int32_t, 8> operator|(vector<std::int32_t, 8> v1, const vector<std::int32_t, 8> &v2);
		friend vector<std::int32_t, 8> operator^(vector<std::int32_t, 8> v1, const vector<std::int32_t, 8> &v2);

		friend vector<std::int32_t, 8> operator<<(const vector<std::int32_t, 8> &v, std::int32_t bits);
		friend vector<std::int32_t, 8> operator>>(const vector<std::int32_t, 8> &v, std::int32_t bits);

		friend vector<std::int32_t, 8> operator==(const vector<std::int32_t, 8> &v1, const vector<std::int32_t, 8> &v2);
		friend vector<std::int32_t, 8> operator!=(const vector<std::int32_t, 8> &v1, const vector<std::int32_t, 8> &v2);
		friend vector<std::int32_t, 8> operator>(const vector<std::int32_t, 8> &v1, const vector<std::int32_t, 8> &v2);
		friend vector<std::int32_t, 8> operator>=(const vector<std::int32_t, 8> &v1, const vector<std::int32_t, 8> &v2);
		friend vector<std::int32_t, 8> operator<(const vector<std::int32_t, 8> &v1, const vector<std::int32_t, 8> &v2);
		friend vector<std::int32_t, 8> operator<=(const vector<std::int32_t, 8> &v1, const vector<std::int32_t, 8> &v2);

		vector<std::int32_t, 8> &hadd(const vector<std::int32_t, 8> &v);
		vector<std::int32_t, 8> &hsub(const vector<std::int32_t, 8> &v);
		type hadd() const;
		type hsub() const;
		friend vector<std::int32_t, 8> hadd(vector<std::int32_t, 8> v1, const vector<std::int32_t, 8> &v2);
		friend vector<std::int32_t, 8> hsub(vector<std::int32_t, 8> v1, const vector<std::int32_t, 8> &v2);

		vector<std::int32_t, 8> &abs();
		friend vector<std::int32_t, 8> abs(vector<std::int32_t, 8> v);
		friend vector<std::int32_t, 8> min(const vector<std::int32_t, 8> &v1, const vector<std::int32_t, 8> &v2);
		friend vector<std::int32_t, 8> max(const vector<std::int32_t, 8> &v1, const vector<std::int32_t, 8> &v2);

		friend vector<std::int32_t, 8> select(const vector<std::int32_t, 8> &v, const vector<std::int32_t, 8> &alt, const vector<std::int32_t, 8> &condition);

		friend std::ostream &operator<<(std::ostream &stream, const vector<std::int32_t, 8> &v);
	};

	vector<std::int32_t, 8> &vector<std::int32_t, 8>::operator+=(const vector<std::int32_t, 8> &v) {
		m_vec = _mm256_add_epi32(m_vec, v.m_vec);
		return *this;
	}

	vector<std::int32_t, 8> &vector<std::int32_t, 8>::operator-=(const vector<std::int32_t, 8> &v) {
		m_vec = _mm256_sub_epi32(m_vec, v.m_vec);
		return *this;
	}

	vector<std::int32_t, 8> &vector<std::int32_t, 8>::operator*=(const vector<std::int32_t, 8> &v) {
		m_vec = _mm256_mul_epi32(m_vec, v.m_vec);
		return *this;
	}

	vector<std::int32_t, 8> &vector<std::int32_t, 8>::operator/=(const vector<std::int32_t, 8> &v) {
		// TODO: there's gotta be a better way
		m_vec = _mm256_cvtps_epi32(_mm256_div_ps(_mm256_cvtepi32_ps(m_vec), _mm256_cvtepi32_ps(v.m_vec)));
		return *this;
	}

	vector<std::int32_t, 8> &vector<std::int32_t, 8>::hadd(const vector<std::int32_t, 8> &v) {
		m_vec = _mm256_hadd_epi32(m_vec, v.m_vec);
		return *this;
	}

	vector<std::int32_t, 8> &vector<std::int32_t, 8>::hsub(const vector<std::int32_t, 8> &v) {
		m_vec = _mm256_hsub_epi32(m_vec, v.m_vec);
		return *this;
	}

	std::int32_t vector<std::int32_t, 8>::hadd() const {
		// Results in A1+A2, A3+A4, ..., ..., A5+A6, A7+A8
		auto t1 = _mm256_hadd_epi32(m_vec, m_vec);
		// Permute to get A5+A6, A7+A8, ....
		auto t2 = _mm256_permute2x128_si256(t1, t1, 0b000001);
		// Compute (A1+A2)+(A5+A6), (A3+A4)+(A7+A8), ...
		auto t3 = _mm256_add_epi32(t1, t2);
		// Compute entire sum
		auto t4 = _mm256_hadd_epi32(t3, t3);

		// Grab the sums from the first entries
		// TODO: why does GCC not like _mm256_cvtsi256_si32?
		//return _mm256_cvtsi256_si32(t3) + _mm256_cvtsi256_si32(t4);
		return _mm256_extract_epi32(t4, 0);
	}

	std::int32_t vector<std::int32_t, 8>::hsub() const {
		throw std::runtime_error("Operation not implemented");
	}

	vector<std::int32_t, 8> hadd(vector<std::int32_t, 8> v1, const vector<std::int32_t, 8> &v2) {
		return v1.hadd(v2);
	}

	vector<std::int32_t, 8> hsub(vector<std::int32_t, 8> v1, const vector<std::int32_t, 8> &v2) {
		return v1.hsub(v2);
	}

	vector<std::int32_t, 8> operator+(vector<std::int32_t, 8> v1, const vector<std::int32_t, 8> &v2) {
		return (v1 += v2);
	}

	vector<std::int32_t, 8> operator-(vector<std::int32_t, 8> v1, const vector<std::int32_t, 8> &v2) {
		return (v1 -= v2);
	}

	vector<std::int32_t, 8> operator*(vector<std::int32_t, 8> v1, const vector<std::int32_t, 8> &v2) {
		return (v1 *= v2);
	}

	vector<std::int32_t, 8> operator/(vector<std::int32_t, 8> v1, const vector<std::int32_t, 8> &v2) {
		return (v1 /= v2);
	}

	vector<std::int32_t, 8> &vector<std::int32_t, 8>::operator&=(const vector<std::int32_t, 8> &v) {
		m_vec = _mm256_and_si256(m_vec, v.m_vec);
		return *this;
	}

	vector<std::int32_t, 8> &vector<std::int32_t, 8>::operator|=(const vector<std::int32_t, 8> &v) {
		m_vec = _mm256_or_si256(m_vec, v.m_vec);
		return *this;
	}

	vector<std::int32_t, 8> &vector<std::int32_t, 8>::operator^=(const vector<std::int32_t, 8> &v) {
		m_vec = _mm256_xor_si256(m_vec, v.m_vec);
		return *this;
	}

	vector<std::int32_t, 8> operator~(const vector<std::int32_t, 8> &v) {
		return vector<std::int32_t, 8>(_mm256_xor_si256(v.m_vec, _mm256_set1_epi32(-1)));
	}

	vector<std::int32_t, 8> operator&(vector<std::int32_t, 8> v1, const vector<std::int32_t, 8> &v2) {
		return v1 &= v2;
	}

	vector<std::int32_t, 8> operator|(vector<std::int32_t, 8> v1, const vector<std::int32_t, 8> &v2) {
		return v1 |= v2;
	}

	vector<std::int32_t, 8> operator^(vector<std::int32_t, 8> v1, const vector<std::int32_t, 8> &v2) {
		return v1 ^= v2;
	}

	vector<std::int32_t, 8> operator<<(const vector<std::int32_t, 8> &v, std::int32_t bits) {
		return vector<std::int32_t, 8>(_mm256_slli_epi32(v.m_vec, bits));
	}

	vector<std::int32_t, 8> operator>>(const vector<std::int32_t, 8> &v, std::int32_t bits) {
		return vector<std::int32_t, 8>(_mm256_slli_epi32(v.m_vec, bits));
	}

	vector<std::int32_t, 8> operator==(const vector<std::int32_t, 8> &v1, const vector<std::int32_t, 8> &v2) {
		return vector<std::int32_t, 8>(_mm256_cmpeq_epi32(v1.m_vec, v2.m_vec));
	}

	vector<std::int32_t, 8> operator!=(const vector<std::int32_t, 8> &v1, const vector<std::int32_t, 8> &v2) {
		// Invert because SSE doesn't have instruction for it
		return vector<std::int32_t, 8>(_mm256_xor_si256(_mm256_cmpeq_epi32(v1.m_vec, v2.m_vec), _mm256_set1_epi32(-1)));
	}

	vector<std::int32_t, 8> operator>(const vector<std::int32_t, 8> &v1, const vector<std::int32_t, 8> &v2) {
		return vector<std::int32_t, 8>(_mm256_cmpgt_epi32(v1.m_vec, v2.m_vec));
	}

	vector<std::int32_t, 8> operator>=(const vector<std::int32_t, 8> &v1, const vector<std::int32_t, 8> &v2) {
		// Emulate because SSE doesn't have instruction for it
		return vector<std::int32_t, 8>(_mm256_or_si256(_mm256_cmpeq_epi32(v1.m_vec, v2.m_vec), _mm256_cmpgt_epi32(v1.m_vec, v2.m_vec)));
	}

	vector<std::int32_t, 8> operator<(const vector<std::int32_t, 8> &v1, const vector<std::int32_t, 8> &v2) {
		// Emulate because SSE doesn't have instruction for it
		auto ge = _mm256_or_si256(_mm256_cmpeq_epi32(v1.m_vec, v2.m_vec), _mm256_cmpgt_epi32(v1.m_vec, v2.m_vec));
		return vector<std::int32_t, 8>(_mm256_xor_si256(ge, _mm256_set1_epi32(-1)));
	}

	vector<std::int32_t, 8> operator<=(const vector<std::int32_t, 8> &v1, const vector<std::int32_t, 8> &v2) {
		// Invert because SSE doesn't have instruction for it
		return vector<std::int32_t, 8>(_mm256_xor_si256(_mm256_cmpgt_epi32(v1.m_vec, v2.m_vec), _mm256_set1_epi32(-1)));
	}

	vector<std::int32_t, 8> &vector<std::int32_t, 8>::abs() {
		// TODO: benchmark for most performant solution
		m_vec = _mm256_abs_epi32(m_vec);
		return *this;
	}

	vector<std::int32_t, 8> abs(vector<std::int32_t, 8> v) {
		return v.abs();
	}

	vector<std::int32_t, 8> min(const vector<std::int32_t, 8> &v1, const vector<std::int32_t, 8> &v2) {
		return vector<std::int32_t, 8>(_mm256_min_epi32(v1.m_vec, v2.m_vec));
	}

	vector<std::int32_t, 8> max(const vector<std::int32_t, 8> &v1, const vector<std::int32_t, 8> &v2) {
		return vector<std::int32_t, 8>(_mm256_max_epi32(v1.m_vec, v2.m_vec));
	}

	vector<std::int32_t, 8> select(const vector<std::int32_t, 8> &v, const vector<std::int32_t, 8> &alt, const vector<std::int32_t, 8> &condition) {
		// This only works when mask is either 0 or -1 for each!
		return vector<std::int32_t, 8>(_mm256_blendv_epi8(alt.m_vec, v.m_vec, condition.m_vec));
	}

	std::ostream &operator<<(std::ostream &stream, const vector<std::int32_t, 8> &v) {
		stream << '(';
		for (size_t i = 0; i < v.width - 1; ++i) {
			stream << v.m_array[i] << ' ';
		}
		stream << v.m_array[v.width - 1] << ')';
		return stream;
	}
#endif // SIMD_SUPPORTS(SIMD_AVX2)

} // namespace simd