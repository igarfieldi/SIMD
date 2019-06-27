#pragma once

#include <array>
#include <cassert>
#include <ostream>
#include "base_types.hpp"
#include "util.hpp"

namespace simd {

	struct aligned_load {};
	struct unaligned_load {};

	template < class T, size_t W >
	class vector_base {
	public:
		using type = T;
		static constexpr size_t width = W;
		using native_type = typename native_vector<type, width>::native_type;

	protected:
		union {
			native_type m_vec;
			std::array<type, width> m_array;
		};

		SIMD_FORCEINLINE vector_base() : m_vec() {}
		SIMD_FORCEINLINE vector_base(native_type native) : m_vec(native) {}

	public:
		SIMD_FORCEINLINE native_type &native() {
			return m_vec;
		}

		SIMD_FORCEINLINE const native_type &native() const {
			return m_vec;
		}

		SIMD_FORCEINLINE std::array<type, width> &array() {
			return m_vec;
		}

		SIMD_FORCEINLINE const std::array<type, width> &array() const {
			return m_vec;
		}

		SIMD_FORCEINLINE type &operator[](size_t index) {
			assert(index < width);
			return m_array[index];
		}

		SIMD_FORCEINLINE const type &operator[](size_t index) const {
			assert(index < width);
			return m_array[index];
		}
	};

	template < class T, size_t W >
	class vector;

	
	template < class T, size_t W >
	class mask;


} // namespace simd