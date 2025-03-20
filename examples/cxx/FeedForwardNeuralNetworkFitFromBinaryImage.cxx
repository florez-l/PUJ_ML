## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import cv2, matplotlib.pyplot, numpy, sys
sys.path.append( '../lib' )
import PUJ_ML

if __name__ == '__main__':

  # Parse command line arguments
  args = PUJ_ML.Helpers.ParseFitArguments(
    sys.argv,
    mandatory = [ ( 'model', str ), ( 'image', str ) ],
    optional = [
      ( '-sampling', '--sampling', int, 0 ),
      ( '-out_image', '--out_image', str, 'out.png' )
      ]
    )

  # Read data
  D_tr, D_te = PUJ_ML.IO.ReadFromImage( args.image, args.sampling )

  # Read model template
  model = PUJ_ML.Model.NeuralNetwork.FeedForward( )
  model.load( args.model )
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

  # Save result image
  o_min = D_tr[ 0 ].min( axis = 0 )
  o_max = D_tr[ 0 ].max( axis = 0 )
  o_size = ( o_max - o_min ).astype( int ) + 1
  Y_range = numpy.linspace( o_min[ 0 ], o_max[ 0 ], num = o_size[ 0 ] ).astype( int )
  X_range = numpy.linspace( o_min[ 1 ], o_max[ 1 ], num = o_size[ 1 ] ).astype( int )

  grid = numpy.meshgrid( Y_range, X_range )
  N = grid[ 0 ].size
  X_image = numpy.concatenate(
    ( grid[ 0 ].reshape( ( N, 1 ) ), grid[ 1 ].reshape( ( N, 1 ) ) ),
    axis = 1
    ).astype( float )
  Y_image = model( X_image )
  m_Y_image = Y_image.min( )
  M_Y_image = Y_image.max( )
  print( 'Output range: ', m_Y_image, M_Y_image )
  Y_image -= m_Y_image
  Y_image /= M_Y_image - m_Y_image
  Y_image *= 255
  out_image = Y_image.reshape( grid[ 0 ].shape ).astype( int )

  cv2.imwrite( args.out_image, out_image )

# end if

## eof - FeedForwardNeuralNetworkFitFromBinaryImage.py
