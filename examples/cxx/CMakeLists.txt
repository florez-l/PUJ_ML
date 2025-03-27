## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

include_directories(
  ${PROJECT_BINARY_DIR}/lib/cxx
  ${PROJECT_SOURCE_DIR}/lib/cxx
  )

set(
  _execs
  FeedForwardNeuralNetwork
  FeedForwardNeuralNetworkFit
  FitMNISTModel
  LinearRegressionClosedFit
  LogisticRegressionFit
  )

foreach(_e ${_execs})
  add_executable(PUJ_ML_example_${_e} ${_e}.cxx)
  target_link_libraries(PUJ_ML_example_${_e} PUJ_ML)
endforeach()

## eof - $RCSfile$
