cmake_minimum_required(VERSION 3.20.0)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 17)

add_compile_options(-Wall)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(..)

include(GNUInstallDirs)

find_package(executorch CONFIG REQUIRED)

set(EXECUTORCH_LINK_LIBS 
  executorch
  optimized_native_cpu_ops_lib
  optimized_kernels
  extension_training
)

add_executable (test_load_model test_load_model.cpp)
target_link_libraries(test_load_model
  ${EXECUTORCH_LINK_LIBS}
)
add_test(TestLoadModel test_load_model)

add_executable (test_filtering test_filtering.cpp)
target_link_libraries(test_filtering
  ${EXECUTORCH_LINK_LIBS}
)
add_test(TestFiltering test_filtering)
