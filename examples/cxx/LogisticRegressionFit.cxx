// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <string>

#include <PUJ_ML/Helpers/FitModel.h>
#include <PUJ_ML/Helpers/ParseFitArguments.h>
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
    // Read model template
    TModel model( D_tr.cols( ) - 1 );
    std::cout << "==============================================" << std::endl;
    std::cout << "Initial model: " <<  model << std::endl;
    std::cout << "  ---> Encoded model: " << model.encode64( ) << std::endl;

    // Fit model
    PUJ_ML::Helpers::FitModel( model, args, D_tr, TMatrix( ) );

    // Show final models
    std::cout << "==============================================" << std::endl;
    std::cout << "Fitted model: " <<  model << std::endl;
    std::cout << "  ---> Encoded model: " << model.encode64( ) << std::endl;

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

  // Show final costs
  /* TODO
     print( "==============================================" )
     print( "Training cost = " + str( model.cost( D_tr[ 0 ], D_tr[ 1 ] ) ) )
     if not D_te[ 0 ] is None:
     print( "Testing cost  = " + str( model.cost( D_te[ 0 ], D_te[ 1 ] ) ) )
     // end if
     */

  // Compute confussion matrices
  /* TODO
     K_tr = PUJ_ML.Helpers.Confussion( model, D_tr[ 0 ], D_tr[ 1 ] )
     print( "==============================================" )
     print( "Training confussion =\n", K_tr )
     if not D_te[ 0 ] is None:
     K_te = PUJ_ML.Helpers.Confussion( model, D_te[ 0 ], D_te[ 1 ] )
     print( "Testing confussion  =\n", K_te )
     // end if
     */

  // ROC curves
  /* TODO
     ROC_tr = PUJ_ML.Helpers.ROC( model, D_tr[ 0 ], D_tr[ 1 ] )
     ROC_te = None
     if not D_te[ 0 ] is None:
     ROC_te = PUJ_ML.Helpers.ROC( model, D_te[ 0 ], D_te[ 1 ] )
     // end if
     */

  /* TODO
     fig, ax = matplotlib.pyplot.subplots( )
     ax.plot( ROC_tr[ 0 ], ROC_tr[ 1 ], lw = 1 )
     if not ROC_te is None:
     ax.plot( ROC_te[ 0 ], ROC_te[ 1 ], lw = 1 )
     // end if
     ax.plot( [ 0, 1 ], [ 0, 1 ], lw = 0.5, linestyle = "--" )
     ax.set_aspect( 1 )
     matplotlib.pyplot.show( )
  */
  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
