## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

file(GLOB_RECURSE _h ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB_RECURSE _hxx ${CMAKE_CURRENT_SOURCE_DIR}/*.hxx)
file(GLOB_RECURSE _cxx ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx)

include_directories(
  ${PROJECT_BINARY_DIR}/lib/cxx
  ${PROJECT_SOURCE_DIR}/lib/cxx
  )
add_library(PUJ_ML ${PUJ_ML_BUILD_TYPE} ${_cxx})
generate_export_header(
  PUJ_ML
  EXPORT_MACRO_NAME PUJ_ML_EXPORT
  EXPORT_FILE_NAME Export.h
  )
target_link_libraries(PUJ_ML Eigen3::Eigen)

## eof - $RCSfile$
