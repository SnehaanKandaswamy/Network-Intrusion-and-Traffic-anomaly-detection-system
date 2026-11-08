cmake_minimum_required(VERSION 3.18)
project(PCANetworkAnomalyDetection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(PCA_NATIVE "Optimise for the build machine's CPU (-march=native)" ON)

# ---------------------------------------------------------------- OpenMP
find_package(OpenMP REQUIRED COMPONENTS CXX)

# ---------------------------------------------------------------- Eigen
# Use a system Eigen if present; otherwise download the pinned 3.4.0 release
# (header-only, verified by SHA-256) into the build tree.
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
  set(EIGEN_VERSION 3.4.0)
  set(EIGEN_ARCHIVE "${CMAKE_BINARY_DIR}/_deps/eigen-${EIGEN_VERSION}.tar.gz")
  set(EIGEN_ROOT "${CMAKE_BINARY_DIR}/_deps/eigen-${EIGEN_VERSION}")
  if(NOT EXISTS "${EIGEN_ROOT}/Eigen/Dense")
    message(STATUS "Eigen3 not found on the system - downloading Eigen ${EIGEN_VERSION}")
    file(DOWNLOAD
      "https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.tar.gz"
      "${EIGEN_ARCHIVE}"
      EXPECTED_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
      SHOW_PROGRESS)
    file(ARCHIVE_EXTRACT INPUT "${EIGEN_ARCHIVE}" DESTINATION "${CMAKE_BINARY_DIR}/_deps")
  endif()
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  target_include_directories(Eigen3::Eigen SYSTEM INTERFACE "${EIGEN_ROOT}")
endif()

# ---------------------------------------------------------------- flags
add_library(pca_flags INTERFACE)
# Eigen must never start its own OpenMP threads: every parallel region in this
# project is one we wrote. This also prevents nested parallelism.
target_compile_definitions(pca_flags INTERFACE EIGEN_DONT_PARALLELIZE)
target_link_libraries(pca_flags INTERFACE Eigen3::Eigen)
if(MSVC)
  target_compile_options(pca_flags INTERFACE /W4 /permissive- /EHsc)
  target_compile_definitions(pca_flags INTERFACE _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
  target_compile_options(pca_flags INTERFACE -Wall -Wextra -Wpedantic)
  if(PCA_NATIVE)
    target_compile_options(pca_flags INTERFACE -march=native)
  endif()
endif()
if(WIN32)
  target_link_libraries(pca_flags INTERFACE psapi)
endif()

# OpenMP flags for the parallel code only.
add_library(pca_openmp_flags INTERFACE)
if(MSVC)
  # MSVC's default /openmp is OpenMP 2.0 (no simd, no max-reductions);
  # /openmp:llvm provides OpenMP 3.1+.
  target_compile_options(pca_openmp_flags INTERFACE /openmp:llvm)
else()
  target_link_libraries(pca_openmp_flags INTERFACE OpenMP::OpenMP_CXX)
endif()

# The sequential kernels are compiled WITHOUT the OpenMP runtime. On GCC/Clang
# -fopenmp-simd honours only `#pragma omp simd` (vectorisation hints, no
# threads), so both builds vectorise the inner loops identically.
add_library(pca_simd_flags INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(pca_simd_flags INTERFACE -fopenmp-simd)
endif()

# ---------------------------------------------------------------- libraries
add_library(pca_common STATIC
  src/common/matrix.cpp
  src/common/utils.cpp
  src/common/pca_common.cpp
  src/common/anomaly_detection.cpp)
target_include_directories(pca_common PUBLIC include)
target_link_libraries(pca_common PUBLIC pca_flags)

add_library(pca_sequential STATIC
  src/sequential/pca_sequential.cpp
  src/sequential/injection_sequential.cpp)
target_link_libraries(pca_sequential PUBLIC pca_common PRIVATE pca_simd_flags)

add_library(pca_parallel STATIC
  src/openmp/pca_openmp.cpp
  src/openmp/injection_openmp.cpp)
target_link_libraries(pca_parallel PUBLIC pca_common pca_sequential pca_openmp_flags)

# ---------------------------------------------------------------- executables
# One detection driver, compiled twice: once against the sequential kernels
# (no OpenMP at all) and once against the OpenMP kernels.
add_executable(pca_seq src/app/detect_main.cpp)
target_compile_definitions(pca_seq PRIVATE PCA_USE_OPENMP=0)
target_link_libraries(pca_seq PRIVATE pca_sequential)

add_executable(pca_omp src/app/detect_main.cpp)
target_compile_definitions(pca_omp PRIVATE PCA_USE_OPENMP=1)
target_link_libraries(pca_omp PRIVATE pca_parallel)

add_executable(injection_study src/app/injection_main.cpp)
target_link_libraries(injection_study PRIVATE pca_parallel)

add_executable(bench benchmarks/benchmark.cpp)
target_link_libraries(bench PRIVATE pca_parallel)

# ---------------------------------------------------------------- tests
enable_testing()
add_executable(pca_tests tests/test_main.cpp)
target_link_libraries(pca_tests PRIVATE pca_parallel)
add_test(NAME unit_tests COMMAND pca_tests)

find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import numpy"
                  RESULT_VARIABLE PCA_NUMPY_MISSING OUTPUT_QUIET ERROR_QUIET)
  if(PCA_NUMPY_MISSING EQUAL 0)
    add_test(NAME numpy_reference
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/reference/compare_with_numpy.py"
              --seq "$<TARGET_FILE:pca_seq>" --omp "$<TARGET_FILE:pca_omp>"
              --workdir "${CMAKE_BINARY_DIR}/numpy_reference")
  else()
    message(STATUS "NumPy not available for ${Python3_EXECUTABLE}: numpy_reference test disabled "
                   "(configure with -DPython3_EXECUTABLE=<venv python> to enable)")
  endif()
endif()
