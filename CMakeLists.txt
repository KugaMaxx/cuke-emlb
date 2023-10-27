cmake_minimum_required(VERSION 3.5.1)
project(emlb)

# Configurable features
set(CMAKE_BUILD_TYPE "RELEASE")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../modules)
if(DEFINED TORCH_DIR)
    message("libtorch directory" ${TORCH_DIR})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -Ofast -march=native") # enable avx
else()
    message("Undefined libtorch directory")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native") # enable avx
endif()
if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
    set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

# Find OpenBlas supports
find_package(BLAS REQUIRED)
include_directories(${BLAS_INCLUDE_DIRS})

# Find Pybind11 supports
find_package(pybind11 REQUIRED)

# # Find OpenCV supports
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find dv-runtime supports
find_package(dv REQUIRED)

# Find eigen3 supports
find_package(Eigen3 REQUIRED)

# Find boost supports
find_package(Boost REQUIRED)

# Find libtorch supports
if(DEFINED TORCH_DIR)
    set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")
    set(Torch_DIR ${TORCH_DIR})
    message(${Torch_DIR}) # set(Torch_DIR /home/kuga/Workspace/libtorch/share/cmake/Torch)
    find_package(Torch REQUIRED)
endif()

# Inclue header files.
add_subdirectory(include)
include_directories(${PROJECT_SOURCE_DIR}/include)

# Compile modules
add_subdirectory(src)

