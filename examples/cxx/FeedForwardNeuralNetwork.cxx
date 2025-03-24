// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <PUJ_ML/Model/NeuralNetwork/FeedForward.h>

int main( int argc, char** argv )
{
  using TReal = long double;
  using TModel = PUJ_ML::Model::NeuralNetwork::FeedForward< TReal >;

  if( argc < 2 )
  {
    std::cerr << "Usage: " << argv[ 0 ] << " model" << std::endl;
    return( EXIT_FAILURE );
  } // end if
  std::string model_fname = argv[ 1 ];

  // Prepare a model
  TModel m;
  m.load( model_fname );
  std::cout << "Read model: " << m << std::endl;
  std::cout << "  ---> Encoded model: " << m.encode64( ) << std::endl;

  /* TODO
     N = model.input_size( )
     M = 3
     X = numpy.reshape(
     numpy.matrix( [ random.random( ) for i in range( M * N ) ] ),
     shape = ( M, N )
     )
     print( '==================================================' )
     print( 'X      =\n', str( X ) )
     print( '==================================================' )
     print( 'y(X)   =\n', str( model( X, True ) ) )
     print( '==================================================' )
  */

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
