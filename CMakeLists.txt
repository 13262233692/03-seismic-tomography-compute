cmake_minimum_required(VERSION 3.18)
project(seismic_tomography
    VERSION 1.1.0
    LANGUAGES CXX
    DESCRIPTION "Deep-sea seismic wave tomography imaging system"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(ENABLE_MPI "Enable MPI parallel computation" ON)
option(ENABLE_HDF5 "Enable HDF5 I/O support" ON)
option(ENABLE_PYBIND "Build pybind11 Python bindings" ON)
option(ENABLE_AVX2 "Enable AVX2 SIMD optimizations" ON)
option(ENABLE_TESTS "Build test executables" OFF)

if(ENABLE_AVX2 AND NOT MSVC)
    add_compile_options(-mavx2 -mfma)
endif()

if(MSVC)
    add_compile_options(/arch:AVX2 /W4 /O2)
else()
    add_compile_options(-Wall -Wextra -O3 -ffast-math)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-march=native)
    endif()
endif()

find_package(MPI REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

if(ENABLE_PYBIND)
    find_package(pybind11 QUIET)
    if(NOT pybind11_FOUND)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
            OUTPUT_VARIABLE pybind11_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        find_package(pybind11 REQUIRED)
    endif()
endif()

set(SEISMIC_SOURCES
    src/cpp/core/mpi_context.cpp
    src/cpp/core/grid.cpp
    src/cpp/core/config.cpp
    src/cpp/core/fault_tolerance.cpp
    src/cpp/solver/wave_equation.cpp
    src/cpp/solver/boundary.cpp
    src/cpp/sparse/csr_matrix.cpp
    src/cpp/sparse/sparse_ops.cpp
    src/cpp/io/hdf5_io.cpp
)

add_library(seismic_core STATIC ${SEISMIC_SOURCES})

target_include_directories(seismic_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
)

target_link_libraries(seismic_core PUBLIC
    MPI::MPI_CXX
    HDF5::HDF5_CXX
)

if(WIN32)
    target_compile_definitions(seismic_core PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

if(ENABLE_PYBIND)
    pybind11_add_module(seismic_engine src/cpp/pybind/bindings.cpp)

    target_link_libraries(seismic_engine PRIVATE seismic_core)

    target_compile_definitions(seismic_engine PRIVATE
        PYBIND11_DETAILED_ERROR_MESSAGES
    )

    install(TARGETS seismic_engine
        LIBRARY DESTINATION ${Python3_SITEARCH}/seismic_tomography
    )
endif()

if(ENABLE_TESTS)
    add_executable(test_wave_equation
        tests/test_wave_equation.cpp
    )
    target_link_libraries(test_wave_equation PRIVATE seismic_core)

    add_executable(test_sparse_matrix
        tests/test_sparse_matrix.cpp
    )
    target_link_libraries(test_sparse_matrix PRIVATE seismic_core)

    add_executable(test_boundary
        tests/test_boundary.cpp
    )
    target_link_libraries(test_boundary PRIVATE seismic_core)
endif()

install(TARGETS seismic_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY src/cpp/
    DESTINATION include/seismic_tomography
    FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY src/python/seismic_tomography/
    DESTINATION ${Python3_SITEARCH}/seismic_tomography
    FILES_MATCHING PATTERN "*.py"
)

install(FILES src/python/scripts/run_inversion.py
    DESTINATION bin
    RENAME seismic_tomography_run
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
)
