## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import sys
sys.path.append( '../lib' )
import numpy
import PUJ_ML

if len( sys.argv ) < 2:
  print(
    'Usage: python '
    +
    sys.argv[ 0 ]
    +
    ' data.csv [L1=0] [L2=0]'
    )
  sys.exit( 1 )
# end if
data_fname = sys.argv[ 1 ]
L1, L2 = 0, 0
if len( sys.argv ) > 2: L1 = float( sys.argv[ 2 ] )
if len( sys.argv ) > 3: L2 = float( sys.argv[ 3 ] )

# Get train data
D = numpy.genfromtxt( data_fname, delimiter = ' ' )
X = numpy.asmatrix( D[ : , 0 : D.shape[ 1 ] - 1 ] )
y = numpy.asmatrix( D[ : , -1 ] ).T

# Prepare a model
m = PUJ_ML.Model.Regression.Linear( )
print( 'Initial model: ' + str( m ) )
print( '  ---> Encoded model: ' + m.encode64( ) )

# Fit model to train data
m.fit( X, y, L1, L2 )

# Show final model
print( 'Fitted model: ' + str( m ) )
print( '  ---> Encoded model: ' + m.encode64( ) )
print( 'Cost = ' + str( m.cost( X, y ) ) )

## eof - LinearRegressionClosedFit.py
