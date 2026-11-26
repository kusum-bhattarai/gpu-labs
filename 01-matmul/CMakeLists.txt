cmake_minimum_required(VERSION 3.20)
project(matmul LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# -O3 for speed, -g so Instruments / disassembly can map back to source lines
# (the Mac equivalent of nvcc -lineinfo). Release normally adds -DNDEBUG -O3.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-march=native HAS_MARCH_NATIVE)
if(HAS_MARCH_NATIVE)
  set(ARCH_FLAG -march=native)
else()
  set(ARCH_FLAG -mcpu=native)
endif()
message(STATUS "Arch flag: ${ARCH_FLAG}")
add_compile_options(-O3 -g ${ARCH_FLAG} -Wall -Wextra)

# ---- OpenMP via Homebrew libomp (Apple clang has no bundled OpenMP runtime) ----
if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  execute_process(COMMAND brew --prefix libomp
                  OUTPUT_VARIABLE LIBOMP_PREFIX
                  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(NOT LIBOMP_PREFIX)
    set(LIBOMP_PREFIX /opt/homebrew/opt/libomp)
  endif()
  set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
  set(OpenMP_CXX_LIB_NAMES omp)
  set(OpenMP_omp_LIBRARY ${LIBOMP_PREFIX}/lib/libomp.dylib)
endif()
find_package(OpenMP REQUIRED)

# ---- Your kernel ----
add_library(matmul_impl STATIC src/matmul.cpp)
target_include_directories(matmul_impl PUBLIC src)
target_link_libraries(matmul_impl PUBLIC OpenMP::OpenMP_CXX)

# ---- Shared harness code (naive reference, tolerance check, data gen) ----
add_library(harness STATIC bench/reference.cpp)
target_include_directories(harness PUBLIC bench src)
target_link_libraries(harness PUBLIC OpenMP::OpenMP_CXX)

add_executable(bench bench/bench.cpp)
target_link_libraries(bench PRIVATE matmul_impl harness)
target_compile_definitions(bench PRIVATE MATMUL_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/results)

add_executable(test_matmul tests/test_matmul.cpp)
target_link_libraries(test_matmul PRIVATE matmul_impl harness)

enable_testing()
add_test(NAME matmul_correctness COMMAND test_matmul)
