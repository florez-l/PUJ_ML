// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <set>
#include <string>

#include <PUJ_ML/Helpers.h>
#include <PUJ_ML/IO/ReadIDX.h>
#include <PUJ_ML/Model/NeuralNetwork/FeedForward.h>

int main( int argc, char** argv )
{
  using TModel = PUJ_ML::Model::NeuralNetwork::FeedForward< long double >;
  using TReal = TModel::TReal;
  using TNatural = TModel::TNatural;
  using TMatrix = TModel::TMatrix;

  // Parse command line arguments
  using TParser = PUJ_ML::Helpers::ParseFitArguments< TReal, TNatural >;
  TParser args( "FeedForward model fit", argc, argv );
  args.add_positional_option< std::string >( "model" );
  args.add_positional_option< std::string >( "Xtr" );
  args.add_positional_option< std::string >( "Ytr" );
  args.parse( );
  if( args.fail( ) )
  {
    args.show_error( std::cerr );
    return( EXIT_FAILURE );
  } // end if

  // Read data
  TMatrix Xtr;
  Eigen::Matrix< unsigned char, Eigen::Dynamic, Eigen::Dynamic > Ltr;
  if(
    PUJ_ML::IO::ReadIDX( Xtr, args.Strings[ "Xtr" ] )
    &&
    PUJ_ML::IO::ReadIDX( Ltr, args.Strings[ "Ytr" ] )
    )
  {
    std::set< unsigned char > labels { Ltr.data( ), Ltr.data( ) + Ltr.size( ) };
    TMatrix I = TModel::TMatrix::Identity( labels.size( ), labels.size( ) );
    TMatrix Ytr = I( Ltr.col( 0 ), Eigen::all );

    // Read model template
    TModel model;
    model.load( args.Strings[ "model" ] );
    /* TODO
       std::cout << "==============================================" << std::endl;
       std::cout << "Initial model: " <<  model << std::endl;
       std::cout << "  ---> Encoded model: " << model.encode64( ) << std::endl;
    */

    // Fit model
    PUJ_ML::Helpers::FitModel(
      model, args, Xtr, Ytr, TMatrix( ), TMatrix( )
      );

    // Show final models
    /* TODO
       std::cout << "==============================================" << std::endl;
       std::cout << "Fitted model: " <<  model << std::endl;
       std::cout << "  ---> Encoded model: " << model.encode64( ) << std::endl;
    */

    // Show final costs
    std::cout << "==============================================" << std::endl;
    std::cout << "Training cost = " << model.cost( Xtr, Ytr ) << std::endl;

    // Compute confussion matrices
    /* TODO
       TMatrix K_tr
       =
       PUJ_ML::Helpers::Confussion( y_tr, model.threshold( X_tr ) )
       .template cast< TReal >( );
       std::cout << "==============================================" << std::endl;
       std::cout << "Training confussion =" << std::endl << K_tr << std::endl;
    */

    // ROC curve
    /* TODO
       ROC_tr = PUJ_ML.Helpers.ROC( model, D_tr[ 0 ], D_tr[ 1 ] )
    */

    return( EXIT_SUCCESS );
  }
  else
  {
    std::cerr
      << "Error: could not read data from \""
      << args.Strings[ "train" ] << "\""
      << std::endl;
    return( EXIT_FAILURE );
  } // end if

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$




// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

/* TODO
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
*/

// eof - $RCSfile$
