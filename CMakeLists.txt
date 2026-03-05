cmake_minimum_required(VERSION 3.20.0)

project(dnf LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

find_package(Torch REQUIRED)

include_directories(${TORCH_INCLUDE_DIRS})

include(GNUInstallDirs)

add_subdirectory(ecg_filt_demo)

add_compile_options(-Wall -Wconversion -Wextra -pedantic)

add_library(dnf_torch STATIC dnf_torch.cpp)

target_link_libraries(dnf_torch ${TORCH_LIBRARIES})

set_target_properties(dnf_torch PROPERTIES
  POSITION_INDEPENDENT_CODE TRUE
  PUBLIC_HEADER dnf_torch.h)

install(TARGETS dnf_torch EXPORT dnf_torch-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(EXPORT dnf_torch-targets
  DESTINATION lib/cmake/dnf_torch
  FILE dnf_torch-config.cmake
)
