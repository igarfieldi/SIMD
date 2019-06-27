#include <iostream>
#include "simd.hpp"

namespace simd {

#define TEST_CHECK(operation, expected)																\
	try {																							\
		using vector_type = decltype(expected);														\
		using mask_type = mask<typename vector_type::type, vector_type::width>;						\
		const auto expct = expected;																\
		if(const auto res = (operation); mask_type(res == expct).all())								\
			std::cout << #operation << " : PASSED" << std::endl;									\
		else																						\
			std::cerr << #operation << " : FAILED (" << res << " != " << expct << ")" << std::endl;	\
	} catch(const std::runtime_error&) {															\
		std::cout << #operation << " : NOT IMPLEMENTED" << std::endl;								\
	}

template < class V >
inline V set_all_bits() {
	V v;
	std::memset(&v, -1, sizeof(V));
	return v;
}

template < class V, std::size_t M, class T, std::size_t N >
inline std::array<V, M> convert(const std::array<T, N>& arr) {
	static_assert(M <= N, "Can only convert from large to small");
	if constexpr(std::is_same_v<V, T>) {
		std::array<V, M> res;
		for(std::size_t i = 0u; i < M; ++i)
			res[i] = arr[i];
		return res;
	} else {
		std::array<V, M> res;
		for(std::size_t i = 0u; i < M; ++i)
			res[i] = static_cast<V>(arr[i]);
		return res;
	}
}

template < class T, std::size_t N >
std::array<T, N> operator+(const std::array<T, N>& l, const std::array<T, N>& r) {
	std::array<T, N> res;
	for(std::size_t i = 0u; i < N; ++i) res[i] = l[i] + r[i];
	return res;
}
template < class T, std::size_t N >
std::array<T, N> operator-(const std::array<T, N>& l, const std::array<T, N>& r) {
	std::array<T, N> res;
	for(std::size_t i = 0u; i < N; ++i) res[i] = l[i] - r[i];
	return res;
}
template < class T, std::size_t N >
std::array<T, N> operator*(const std::array<T, N>& l, const std::array<T, N>& r) {
	std::array<T, N> res;
	for(std::size_t i = 0u; i < N; ++i) res[i] = l[i] * r[i];
	return res;
}
template < class T, std::size_t N >
std::array<T, N> operator/(const std::array<T, N>& l, const std::array<T, N>& r) {
	std::array<T, N> res;
	for(std::size_t i = 0u; i < N; ++i) res[i] = l[i] / r[i];
	return res;
}

enum class Comparator {
	EQ, NEQ, LT, LE, GT, GE
};
template < class T, std::size_t N >
std::array<T, N> cmp(const std::array<T, N>& l, const std::array<T, N>& r, Comparator cmp) {
	std::array<T, N> res;
	switch(cmp) {
		case Comparator::EQ:
			for(std::size_t i = 0u; i < N; ++i) res[i] = l[i] == r[i] ? set_all_bits<T>() : T(0);
			break;
		case Comparator::NEQ:
			for(std::size_t i = 0u; i < N; ++i) res[i] = l[i] != r[i] ? set_all_bits<T>() : T(0);
			break;
		case Comparator::LT:
			for(std::size_t i = 0u; i < N; ++i) res[i] = l[i] < r[i] ? set_all_bits<T>() : T(0);
			break;
		case Comparator::LE:
			for(std::size_t i = 0u; i < N; ++i) res[i] = l[i] <= r[i] ? set_all_bits<T>() : T(0);
			break;
		case Comparator::GT:
			for(std::size_t i = 0u; i < N; ++i) res[i] = l[i] > r[i] ? set_all_bits<T>() : T(0);
			break;
		case Comparator::GE:
			for(std::size_t i = 0u; i < N; ++i) res[i] = l[i] >= r[i] ? set_all_bits<T>() : T(0);
			break;
	}
	return res;
}

template < class T, std::size_t N >
void test() {
	constexpr std::array<double, 8u> d1{ { 1., 5.25, -6.925, 7., 7., -10., 0., 3.2125 } };
	constexpr std::array<double, 8u> d2{ { -2.5, -3.25, 2.875, -2.225, -2.275, -2.25, 4.15, -1.475 } };
	//constexpr std::array<double, 8u> d3{ { -3.8, -3.4, -6.6, 5.2, 1., 0.6, 3.7, -6. } };

	using vector_type = vector<T, N>;

	vector_type a1{ convert<T, N>(d1) };
	vector_type a2{ convert<T, N>(d2) };

	TEST_CHECK(a1 + a2, (vector_type{ convert<T, N>(d1 + d2) }));
	TEST_CHECK(a1 - a2, (vector_type{ convert<T, N>(d1 - d2) }));
	TEST_CHECK(a1 * a2, (vector_type{ convert<T, N>(d1 * d2) }));
	TEST_CHECK(a1 / a2, (vector_type{ convert<T, N>(d1 / d2) }));
	TEST_CHECK(a1 == a2, (vector_type{ convert<T, N>(cmp(d1, d2, Comparator::EQ)) }));
	TEST_CHECK(a1 != a2, (vector_type{ convert<T, N>(cmp(d1, d2, Comparator::NEQ)) }));
	TEST_CHECK(a1 < a2, (vector_type{ convert<T, N>(cmp(d1, d2, Comparator::LT)) }));
	TEST_CHECK(a1 <= a2, (vector_type{ convert<T, N>(cmp(d1, d2, Comparator::LE)) }));
	TEST_CHECK(a1 > a2, (vector_type{ convert<T, N>(cmp(d1, d2, Comparator::GT)) }));
	TEST_CHECK(a1 >= a2, (vector_type{ convert<T, N>(cmp(d1, d2, Comparator::GE)) }));
}

} // namespace simd


int main(void) {
	using namespace simd;

	std::cout << "SSE compile version: " << version_name(sse_compile_version()) << std::endl;
	std::cout << "SSE runtime version: " << version_name(sse_runtime_version()) << std::endl;


	//if constexpr (supports(float32x4::required_version)) {
#if SIMD_SUPPORTS(SIMD_SSE)
	std::cout << std::endl << "--- float32x4 ---" << std::endl;
	test<float, 4u>();
#endif // SIMD_SUPPORTS(SIMD_SSE)

#if SIMD_SUPPORTS(SIMD_SSE2)
	std::cout << std::endl << "--- float64x2 ---" << std::endl;
	test<double, 2u>();
	std::cout << std::endl << "--- int32x4 ---" << std::endl;
	test<std::int32_t, 4u>();
	std::cout << std::endl << "--- int64x2 ---" << std::endl;
	test<std::int64_t, 2u>();
#endif // if SIMD_SUPPORTS(SIMD_SSE2)

#if SIMD_SUPPORTS(SIMD_AVX)
	std::cout << std::endl << "--- float32x8 ---" << std::endl;
	test<float, 8u>();
	std::cout << std::endl << "--- float64x4 ---" << std::endl;
	test<double, 4u>();
#endif // if SIMD_SUPPORTS(SIMD_AVX)

#if SIMD_SUPPORTS(SIMD_AVX2)
	std::cout << std::endl << "--- int32x8 ---" << std::endl;
	test<std::int32_t, 8u>();
	std::cout << std::endl << "--- int64x4 ---" << std::endl;
	test<std::int64_t, 4u>();
#endif // if SIMD_SUPPORTS(SIMD_AVX2)

	return EXIT_SUCCESS;
}