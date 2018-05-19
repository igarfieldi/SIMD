#include <iostream>
#include "conversion.hpp"

int main(void) {

	std::cout << "SSE compile version: " << simd::version_name(simd::sse_compile_version()) << std::endl;
	std::cout << "SSE runtime version: " << simd::version_name(simd::sse_runtime_version()) << std::endl;

	//if constexpr (simd::supports(simd::float32x4::required_version)) {
#if SIMD_SUPPORTS(SIMD_SSE)
	{
		simd::float32x4 a1(std::array<float, 4>{ 1.f, 2.f, -3.f, 4.f }.data());
		simd::float32x4 a2(std::array<float, 4>{ -5.f, -6.f, 7.f, 8.f }.data());

		std::cout << std::endl << "float32x4: " << a1 << " " << a2 << std::endl;
		std::cout << a1 + a2 << std::endl;
		std::cout << a1 - a2 << std::endl;
		std::cout << a1 * a2 << std::endl;
		std::cout << a1 / a2 << std::endl;
	}
#endif // if SIMD_SUPPORTS(SIMD_SSE)

#if SIMD_SUPPORTS(SIMD_SSE2)
	{
		simd::float64x2 a1(std::array<double, 2>{ 1., 2.}.data());
		simd::float64x2 a2(std::array<double, 2>{ -3., 4. }.data());

		std::cout << std::endl << "float64x2: " << a1 << " " << a2 << std::endl;
		std::cout << a1 + a2 << std::endl;
		std::cout << a1 - a2 << std::endl;
		std::cout << a1 * a2 << std::endl;
		std::cout << a1 / a2 << std::endl;
	}
	{
		simd::int32x4 a1(std::array<int, 4>{ 1, 2, -3, 4 }.data());
		simd::int32x4 a2(std::array<int, 4>{ -5, -6, 7, 8 }.data());

		std::cout << std::endl << "int32x4: " << a1 << " " << a2 << std::endl;
		std::cout << a1 + a2 << std::endl;
		std::cout << a1 - a2 << std::endl;
		std::cout << a1 * a2 << std::endl;
	}
#endif // if SIMD_SUPPORTS(SIMD_SSE2)

#if SIMD_SUPPORTS(SIMD_AVX)
	{
		simd::float32x8 a1(std::array<float, 8>{ 1.f, 2.f, -3.f, 4.f, 5.f, 6.f, 7.f, 8.f }.data());
		simd::float32x8 a2(std::array<float, 8>{ 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f }.data());

		std::cout << std::endl << "float32x8: " << a1 << " " << a2 << std::endl;
		std::cout << a1 + a2 << std::endl;
		std::cout << a1 - a2 << std::endl;
		std::cout << a1 * a2 << std::endl;
		std::cout << a1 / a2 << std::endl;
	}
	{
		simd::float64x4 a1(std::array<double, 4>{ 1., 2., -3., 4. }.data());
		simd::float64x4 a2(std::array<double, 4>{ -5., -6., 7., 8. }.data());

		std::cout << std::endl << "float64x4: " << a1 << " " << a2 << std::endl;
		std::cout << a1 + a2 << std::endl;
		std::cout << a1 - a2 << std::endl;
		std::cout << a1 * a2 << std::endl;
		std::cout << a1 / a2 << std::endl;
	}
#endif // if SIMD_SUPPORTS(SIMD_AVX)

#if SIMD_SUPPORTS(SIMD_AVX2)
	{
		simd::int32x8 a1(std::array<int, 8>{ 1, 2, -3, 4, 5, 6, 7, 8 }.data());
		simd::int32x8 a2(std::array<int, 8>{ 9, 10, 11, 12, 13, 14, 15, 16 }.data());

		std::cout << std::endl << "int32x8: " << a1 << " " << a2 << std::endl;
		std::cout << a1 + a2 << std::endl;
		std::cout << a1 - a2 << std::endl;
		std::cout << a1 * a2 << std::endl;
	}
#endif // if SIMD_SUPPORTS(SIMD_AVX2)

	return EXIT_SUCCESS;
}