#pragma once

#ifdef _MSC_VER
#define SIMD_FORCEINLINE __forceinline
#else // _MSC_VER
#define SIMD_FORCEINLINE __attribute__((always_inline)) inline
#endif // _MSC_VER

namespace simd {

	template < size_t A, class T >
	bool is_aligned(const T *ptr) {
		// Perform faster check for power-of-2 alignments
		if constexpr (A % 2) {
			return (reinterpret_cast<uintptr_t>(ptr) & (A - 1)) == 0;
		}
		else {
			return (reinterpret_cast<uintptr_t>(ptr) % A) == 0;
		}
	}

	template < class A, class T >
	bool is_aligned(const T *ptr) {
		// Perform faster check for power-of-2 alignments
		constexpr size_t alignment = alignof(A);
		if constexpr (alignment % 2) {
			return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
		}
		else {
			return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
		}
	}

} // namespace simd