## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import sys
sys.path.append( '../../lib/python3' )
import PUJ_ML

if __name__ == '__main__':

  # Parse command line arguments
  args = PUJ_ML.Helpers.ParseFitArguments(
      sys.argv,
      mandatory = [ ( 'dirname', str ), ( 'model', str ) ]
      )

  # Read model template
  model = PUJ_ML.Model.NeuralNetwork.FeedForward( )
  model.load( args.model )

  # Read data
  D_tr, D_te = PUJ_ML.IO.ReadMNIST( args.dirname )

  # Fit model
  PUJ_ML.Helpers.FitModel( model, args, D_tr, D_te )

# end if

## eof - FitMNISTModel.py
