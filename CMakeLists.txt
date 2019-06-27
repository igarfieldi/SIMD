cmake_minimum_required(VERSION 3.10)

project(simdwrapper)

set(SIMDWRAPPER_BUILD_TEST "No" CACHE STRING "Build test executable")
set_property(CACHE SIMDWRAPPER_BUILD_TEST PROPERTY
			 STRINGS "No" "SSE" "SSE2" "AVX" "AVX2")

add_library(simdwrapper INTERFACE)
target_sources(simdwrapper INTERFACE
	${CMAKE_CURRENT_SOURCE_DIR}/src/base_types.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/conversion.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/simd.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/vector.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/versions.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/util.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/float32x4.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/float32x8.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/float64x2.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/float64x4.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/int32x4.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/int32x8.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/int64x2.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/int64x4.hpp)
target_include_directories(simdwrapper INTERFACE ${PROJECT_SOURCE_DIR}/src/)

if(NOT SIMDWRAPPER_BUILD_TEST STREQUAL "No")
	add_executable(simdtest 
		${CMAKE_CURRENT_SOURCE_DIR}/test/main.cpp)
	target_include_directories(simdtest PRIVATE ${PROJECT_SOURCE_DIR}/src/)
	link_libraries(simdteset simdwrapper)
	set_target_properties(simdtest PROPERTIES CXX_STANDARD 17)
	if(MSVC)
		if(SIMDWRAPPER_BUILD_TEST STREQUAL "SSE")
			target_compile_options(simdtest PRIVATE "/arch:SSE")
		elseif(SIMDWRAPPER_BUILD_TEST STREQUAL "SSE2")
			target_compile_options(simdtest PRIVATE "/arch:SSE2")
		elseif(SIMDWRAPPER_BUILD_TEST STREQUAL "AVX")
			target_compile_options(simdtest PRIVATE "/arch:AVX")
		elseif(SIMDWRAPPER_BUILD_TEST STREQUAL "AVX2")
			target_compile_options(simdtest PRIVATE "/arch:AVX2")
		endif()
	else()
		if(SIMDWRAPPER_BUILD_TEST STREQUAL "SSE")
			target_compile_options(simdtest PRIVATE "-msse")
		elseif(SIMDWRAPPER_BUILD_TEST STREQUAL "SSE2")
			target_compile_options(simdtest PRIVATE "-msse2")
		elseif(SIMDWRAPPER_BUILD_TEST STREQUAL "AVX")
			target_compile_options(simdtest PRIVATE "-mavx")
		elseif(SIMDWRAPPER_BUILD_TEST STREQUAL "AVX2")
			target_compile_options(simdtest PRIVATE "-mavx2")
		endif()
	endif()
endif()

export(TARGETS simdwrapper NAMESPACE simd:: FILE SimdWrapperTargets.cmake)