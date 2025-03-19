## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import matplotlib.pyplot, sys
sys.path.append( '../lib' )
import PUJ_ML

if __name__ == '__main__':

  # Parse command line arguments
  args = PUJ_ML.Helpers.ParseFitArguments(
    sys.argv,
    mandatory = [ ( 'train', str ) ],
    optional = [
      ( '-t', '--test', str, '0' ),
      ( '-d', '--delimiter', str, ',' )
      ]
    )

  # Read data
  D_tr, D_te = PUJ_ML.IO.ReadCSV( args.train, args.test, args.delimiter )
  
  # Read model template
  model = PUJ_ML.Model.Regression.Logistic( D_tr[ 0 ].shape[ 1 ] )
  print( '==============================================' )
  print( 'Initial model: ' + str( model ) )
  print( '  ---> Encoded model: ' + model.encode64( ) )

  # Fit model
  PUJ_ML.Helpers.FitModel( model, args, D_tr, D_te )
  
  # Show final models
  print( '==============================================' )
  print( 'Fitted model: ' + str( model ) )
  print( '  ---> Encoded model: ' + model.encode64( ) )

  # Show final costs
  print( '==============================================' )
  print( 'Training cost = ' + str( model.cost( D_tr[ 0 ], D_tr[ 1 ] ) ) )
  if not D_te[ 0 ] is None:
    print( 'Testing cost  = ' + str( model.cost( D_te[ 0 ], D_te[ 1 ] ) ) )
  # end if

  # Compute confussion matrices
  K_tr = PUJ_ML.Helpers.Confussion( model, D_tr[ 0 ], D_tr[ 1 ] )
  print( '==============================================' )
  print( 'Training confussion =\n', K_tr )
  if not D_te[ 0 ] is None:
    K_te = PUJ_ML.Helpers.Confussion( model, D_te[ 0 ], D_te[ 1 ] )
    print( 'Testing confussion  =\n', K_te )
  # end if

  # ROC curves
  ROC_tr = PUJ_ML.Helpers.ROC( model, D_tr[ 0 ], D_tr[ 1 ] )
  ROC_te = None
  if not D_te[ 0 ] is None:
    ROC_te = PUJ_ML.Helpers.ROC( model, D_te[ 0 ], D_te[ 1 ] )
  # end if

  fig, ax = matplotlib.pyplot.subplots( )
  ax.plot( ROC_tr[ 0 ], ROC_tr[ 1 ], lw = 1 )
  if not ROC_te is None:
    ax.plot( ROC_te[ 0 ], ROC_te[ 1 ], lw = 1 )
  # end if
  ax.plot( [ 0, 1 ], [ 0, 1 ], lw = 0.5, linestyle = '--' )
  ax.set_aspect( 1 )
  matplotlib.pyplot.show( )

# end if

## eof - LogisticRegressionFit.py
