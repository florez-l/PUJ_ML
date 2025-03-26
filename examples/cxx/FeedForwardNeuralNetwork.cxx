// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <algorithm>
#include <iostream>
#include <random>
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

  // Evaluate on some random data
  TModel::TNatural N = m.input_size( );
  TModel::TNatural M = 7;
  TModel::TMatrix X( M, N );
  std::random_device rd;
  std::mt19937 rg( rd( ) );
  std::uniform_real_distribution< TReal > rdis( -100, 100 );
  std::generate(
    X.data( ), X.data( ) + X.size( ),
    [&]( ) -> TReal
    {
      return( rdis( rg ) );
    }
    );
  std::cout << "------------------------------------------" << std::endl;
  std::cout << "Input" << std::endl << X << std::endl;
  std::cout << "------------------------------------------" << std::endl;
  std::cout << "Output" << std::endl << m( X ) << std::endl;
  std::cout << "------------------------------------------" << std::endl;
  std::cout << "Threshold" << std::endl << m.threshold( X, true ) << std::endl;
  std::cout << "------------------------------------------" << std::endl;
  TModel::TColumn G = TModel::TColumn::Zero( m.size( ) );
  auto J = m.cost_gradient( G.data( ), X, m.threshold( X ), 0, 0 );
  std::cout << "Model size = " << G.size( ) << std::endl;
  std::cout << "Cost gradient norm = " << ( G.transpose( ) * G ) << std::endl;
  std::cout << "Cost = " << J << std::endl;
  std::cout << "------------------------------------------" << std::endl;

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
