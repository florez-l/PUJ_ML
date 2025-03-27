## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

## -- Base configuration
cmake_minimum_required(VERSION 3.18)
project(PUJ_ML VERSION 1.0.0.1)

## -- Compilator version
set(CMAKE_CXX_STANDARD 23)

## -- Decide whether links are shared or static
if(WIN32 OR APPLE OR MINGW OR MSYS OR CYGWIN)
  set(PUJ_ML_BUILD_TYPE STATIC)
  set(Boost_USE_STATIC_RUNTIME ON)
else()
  set(PUJ_ML_BUILD_TYPE SHARED)
endif()

## -- Set a default build type if none was specified
set(default_build_type "Release")
if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
  set(default_build_type "Debug")
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(
    CMAKE_BUILD_TYPE "${default_build_type}" CACHE
    STRING "Choose the type of build." FORCE
    )
  # Set the possible values of build type for cmake-gui
  set_property(
    CACHE CMAKE_BUILD_TYPE
    PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo"
    )
endif()

## -- Output dirs
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

## -- CMake directives
include(GenerateExportHeader)

## -- Find Boost
find_package(Boost REQUIRED COMPONENTS filesystem program_options)

## -- Find Eigen
find_package(OpenMP QUIET)
set(PUJ_ML_OpenMP_LINK "")
if(OpenMP_CXX_FOUND)
  set(PUJ_ML_OpenMP_LINK "OpenMP::OpenMP_CXX")
endif()
find_package(Eigen3 REQUIRED)

## -- Subdirs
subdirs(examples lib)

## eof - $RCSfile$
