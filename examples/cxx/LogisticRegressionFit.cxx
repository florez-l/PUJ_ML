// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <string>

#include <PUJ_ML/Helpers.h>
#include <PUJ_ML/IO/ReadCSV.h>
#include <PUJ_ML/Model/Regression/Logistic.h>

int main( int argc, char** argv )
{
  using TModel = PUJ_ML::Model::Regression::Logistic< long double >;
  using TReal = TModel::TReal;
  using TNatural = TModel::TNatural;
  using TMatrix = TModel::TMatrix;

  // Parse command line arguments
  using TParser = PUJ_ML::Helpers::ParseFitArguments< TReal, TNatural >;
  TParser args( "Logistic model fit", argc, argv );
  args.add_positional_option< std::string >( "train" );
  args.add_option< std::string >( "separator", ",", "CSV column separator" );
  args.parse( );
  if( args.fail( ) )
  {
    args.show_error( std::cerr );
    return( EXIT_FAILURE );
  } // end if

  // Read data
  TMatrix D_tr;
  if(
    PUJ_ML::IO::ReadCSV(
      D_tr, args.Strings[ "train" ], args.Strings[ "separator" ][ 0 ]
      )
    )
  {
    auto X_tr = D_tr.block( 0, 0, D_tr.rows( ), D_tr.cols( ) - 1 );
    auto y_tr = D_tr.col( D_tr.cols( ) - 1 );

    // Read model template
    TModel model( D_tr.cols( ) - 1 );
    std::cout << "==============================================" << std::endl;
    std::cout << "Initial model: " <<  model << std::endl;
    std::cout << "  ---> Encoded model: " << model.encode64( ) << std::endl;

    // Fit model
    PUJ_ML::Helpers::FitModel(
      model, args, X_tr, y_tr, TMatrix( ), TMatrix( )
      );

    // Show final models
    std::cout << "==============================================" << std::endl;
    std::cout << "Fitted model: " <<  model << std::endl;
    std::cout << "  ---> Encoded model: " << model.encode64( ) << std::endl;

    // Show final costs
    std::cout << "==============================================" << std::endl;
    std::cout << "Training cost = " << model.cost( X_tr, y_tr ) << std::endl;

    // Compute confussion matrices
    TMatrix K_tr
      =
      PUJ_ML::Helpers::Confussion( y_tr, model.threshold( X_tr ) )
      .template cast< TReal >( );
    std::cout << "==============================================" << std::endl;
    std::cout << "Training confussion =" << std::endl << K_tr << std::endl;

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
