cmake_minimum_required(VERSION 3.5.1)

project(
    cuke-emlb
    VERSION 1.0.0
    LANGUAGES C CXX
)

# C++ standard settings.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Configurable features
set(CMAKE_BUILD_TYPE "RELEASE")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../modules)

# List of available options
option(EMLB_ENABLE_PYTHON "Build python bindings" ON)
option(EMLB_ENABLE_MODULES "Build dv modules" ON)

# Define libtorch direction
# If you do not want to set -DTORCH_DIR every time you build, you can directly below command.
# set(TORCH_DIR /home/kuga/Workspace/libtorch/share/cmake/Torch)
if(DEFINED TORCH_DIR)
    option(EMLB_ENABLE_TORCH "Torch" ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -Ofast -march=native") # enable avx
else()
    option(EMLB_ENABLE_TORCH "Torch" OFF)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native") # enable avx
endif()

# Print basic options info
message(STATUS "EMLB_ENABLE_PYTHON ${EMLB_ENABLE_PYTHON}")
message(STATUS "EMLB_ENABLE_MODULES ${EMLB_ENABLE_MODULES}")
message(STATUS "EMLB_ENABLE_SAMPLES ${EMLB_ENABLE_SAMPLES}")
message(STATUS "EMLB_ENABLE_LIBTORCH ${EMLB_ENABLE_TORCH}")

# Suppress warnings
if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
    set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

# Find OpenBlas supports
find_package(BLAS REQUIRED)
include_directories(${BLAS_INCLUDE_DIRS})

# Find OpenCV supports
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find dv-runtime supports
find_package(dv REQUIRED)

# Find eigen3 supports
find_package(Eigen3 REQUIRED)

# Find boost supports
find_package(Boost REQUIRED)

# Find libfmt supports
find_package(fmt REQUIRED)

# Find compression supports
INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(liblz4 IMPORTED_TARGET liblz4)
PKG_CHECK_MODULES(libzstd IMPORTED_TARGET libzstd)

# Find libtorch supports
if(EMLB_ENABLE_TORCH)
    set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")
    set(Torch_DIR ${TORCH_DIR})
    find_package(Torch REQUIRED)
endif()

# Inclue header files
add_subdirectory(include)

# Include external headers
add_subdirectory(external)

# Compile modules for python
if (EMLB_ENABLE_PYTHON)
    add_subdirectory(python)
endif()

# Compule moduls for dv
if (EMLB_ENABLE_MODULES)
    add_subdirectory(src)
endif()

# Compile samples
if (EMLB_ENABLE_SAMPLES)
    add_subdirectory(samples)
endif()
