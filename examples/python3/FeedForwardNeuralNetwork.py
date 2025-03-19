## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, random, sys
sys.path.append( '../lib' )
import PUJ_ML.Model.NeuralNetwork.FeedForward as Model

# Prepare a model
model = Model( )
model.load( sys.argv[ 1 ] )

print( '==================================================' )
print( 'Read model: ' + str( model ) )
print( '==================================================' )

N = model.input_size( )
M = 13
X = numpy.reshape(
    numpy.matrix( [ random.random( ) for i in range( M * N ) ] ),
    newshape = ( M, N )
    )
print( '==================================================' )
print( 'X      =\n', str( X ) )
print( '==================================================' )
print( 'y(X)   =\n', str( model( X ) ) )
print( '==================================================' )

## eof - FeedForwardNeuralNetwork.py
