## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, random, sys
sys.path.append( '../../lib/python3' )
import PUJ_ML

# Prepare a model
model = PUJ_ML.Model.NeuralNetwork.FeedForward( )
model.load_and_decode( sys.argv[ 1 ] )

print( '==================================================' )
print( 'Read model:\n' + str( model ) )
print( '==================================================' )

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

## eof - FeedForwardNeuralNetwork.py
