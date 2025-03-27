// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <set>

#include <PUJ_ML/Model/NeuralNetwork/FeedForward.h>
#include <PUJ_ML/IO/ReadIDX.h>

int main( int argc, char** argv )
{
  using TReal = double;
  using TModel = PUJ_ML::Model::NeuralNetwork::FeedForward< TReal >;

  if( argc < 4 )
  {
    std::cerr << "Usage: " << argv[ 0 ] << " model X Y" << std::endl;
    return( EXIT_FAILURE );
  } // end if
  std::string model_fname = argv[ 1 ];
  std::string X_fname = argv[ 2 ];
  std::string Y_fname = argv[ 3 ];

  // Prepare a model
  TModel m;
  m.load( model_fname );
  /* TODO
     std::cout << "Read model: " << m << std::endl;
     std::cout << "  ---> Encoded model: " << m.encode64( ) << std::endl;
  */

  // Evaluate on some random data
  TModel::TMatrix X;
  PUJ_ML::IO::ReadIDX( X, X_fname );
  Eigen::Matrix< unsigned char, Eigen::Dynamic, Eigen::Dynamic > L;
  PUJ_ML::IO::ReadIDX( L, Y_fname );
  std::set< unsigned int > labels { L.data( ), L.data( ) + L.size( ) };
  TModel::TMatrix I = TModel::TMatrix::Identity( labels.size( ), labels.size( ) );
  TModel::TMatrix Y = I( L.col( 0 ), Eigen::all );

  std::cout << "------------------------------------------" << std::endl;
  TModel::TColumn G = TModel::TColumn::Zero( m.size( ) );
  auto J = m.cost_gradient( G.data( ), X, Y, 0, 0 );
  std::cout << "Model size = " << G.size( ) << std::endl;
  std::cout << "Cost gradient norm = " << ( G.transpose( ) * G ) << std::endl;
  std::cout << "Cost = " << J << std::endl;
  std::cout << "------------------------------------------" << std::endl;

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
