// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <string>
#include <sstream>

#include <PUJ_ML/IO/ReadCSV.h>
#include <PUJ_ML/Model/Regression/Linear.h>

int main( int argc, char** argv )
{
  using TReal = long double;
  using TModel = PUJ_ML::Model::Regression::Linear< TReal >;
  using TMatrix = TModel::TMatrix;

  if( argc < 2 )
  {
    std::cerr
      << "Usage: " << argv[ 0 ]
      << " data.csv [L1=0] [L2=0] [delimiter=\",\"]"
      << std::endl;
    return( EXIT_FAILURE );
  } // end if
  std::string data_fname = argv[ 1 ];
  TReal L1 = 0, L2 = 0;
  unsigned char delimiter = ',';
  if( argc > 2 ) std::istringstream( argv[ 2 ] ) >> L1;
  if( argc > 3 ) std::istringstream( argv[ 3 ] ) >> L2;
  if( argc > 4 ) delimiter = argv[ 4 ][ 0 ];

  // Get train data
  TMatrix D;
  if( PUJ_ML::IO::ReadCSV( D, data_fname, delimiter ) )
  {
    auto X = D.block( 0, 0, D.rows( ), D.cols( ) - 1 );
    auto y = D.block( 0, D.cols( ) - 1, D.rows( ), 1 );

    // Prepare a model
    TModel m;
    std::cout << "Initial model: " << m << std::endl;
    std::cout << "  ---> Encoded model: " << m.encode64( ) << std::endl;

    // Fit model to train data
    m.fit( X, y, L1, L2 );

    // Show final model
    std::cout << "Fitted model: " << m << std::endl;
    std::cout << "  ---> Encoded model: " << m.encode64( ) << std::endl;
    std::cout << "Cost = " << m.cost( X, y ) << std::endl;

    return( EXIT_SUCCESS );
  }
  else
  {
    std::cerr
      << "Error: could not read data from \""
      << data_fname << "\""
      << std::endl;
    return( EXIT_FAILURE );
  } // end if
}

// eof - $RCSfile$
