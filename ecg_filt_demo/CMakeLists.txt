cmake_minimum_required(VERSION 3.20.0)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)
set(CMAKE_CXX_STANDARD 17)

add_compile_options(-Wall)

find_package(Torch REQUIRED)

message(STATUS "Libtorch version: ${Torch_VERSION}")
message(STATUS "Libtorch include path: ${TORCH_INCLUDE_DIRS}")

include_directories(${CMAKE_SOURCE_DIR})
include_directories(..)
include_directories(${TORCH_INCLUDE_DIRS})

include(GNUInstallDirs)

add_executable (ecg_filt_demo ecg_filt_demo.cpp)
target_link_libraries(ecg_filt_demo
    extension_training
    extension_tensor
    extension_module
    extension_data_loader
    extension_flat_tensor
    executorch
    executorch_core

    # XNNPACK backend
    # xnnpack_backend
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,xnnpack_backend>"

    # XNNPACK dependencies
    XNNPACK

    xnnpack-microkernels-prod

    pthreadpool
    extension_threadpool

    quantized_kernels
    quantized_ops_lib
    kernels_util_all_deps
    cpuinfo
    extension_named_data_map
)
