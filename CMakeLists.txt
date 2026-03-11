cmake_minimum_required(VERSION 3.20.0)

project(dnf_executorch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

add_subdirectory(ecg_filt_demo)

enable_testing()
add_subdirectory(test)
