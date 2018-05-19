cmake_minimum_required(VERSION 3.10)

project(simdwrapper)

add_executable(simdwrapper src/main.cpp 
		src/base_types.hpp src/conversion.hpp src/simd.hpp src/vector.hpp src/versions.hpp src/util.hpp
		src/float32x4.hpp src/float32x8.hpp src/float64x2.hpp src/float64x4.hpp
		src/int32x4.hpp src/int32x8.hpp src/int64x2.hpp src/int64x4.hpp)
target_include_directories(simdwrapper PRIVATE ${PROJECT_SOURCE_DIR}/src/)

set_target_properties(simdwrapper PROPERTIES CXX_STANDARD 17)

if(MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
endif(MSVC)