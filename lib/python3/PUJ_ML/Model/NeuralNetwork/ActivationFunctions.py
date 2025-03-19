## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy

class ActivationFunctions:

  '''
  '''
  def get( name ):
    if name.lower( ) == 'identity':
      return ( 'Identity', ActivationFunctions.Identity )
    elif name.lower( ) == 'relu':
      return ( 'ReLU', ActivationFunctions.ReLU )
    elif name.lower( ) == 'sigmoid':
      return ( 'Sigmoid', ActivationFunctions.Sigmoid )
    elif name.lower( ) == 'tanh':
      return ( 'Tanh', ActivationFunctions.Tanh )
    elif name.lower( ) == 'softmax':
      return ( 'SoftMax', ActivationFunctions.SoftMax )
    else:
      return None
    # end if
  # end def

  '''
  '''
  def Identity( Z, d = False ):
    if d:
      return numpy.ones( Z.shape )
    else:
      return Z
    # end if
  # end def

  '''
  '''
  def ReLU( Z, d = False ):
    if d:
      return ( Z > 0 ).astype( float )
    else:
      return numpy.multiply( Z, ( Z > 0 ).astype( float ) )
    # end if
  # end def

  '''
  '''
  def Sigmoid( Z, d = False ):
    if d:
      s = Sigmoid( Z, False )
      return numpy.multiply( s, float( 1 ) - s )
    else:
      return float( 1 ) / ( float( 1 ) + numpy.exp( -Z ) )
    # end if
  # end def

  '''
  '''
  def Tanh( Z, d = False ):
    if d:
      T = Tanh( Z, False )
      return float( 1 ) - numpy.multiply( T, T )
    else:
      return numpy.tanh( Z )
    # end if
  # end def

  '''
  '''
  def SoftMax( Z, d = False ):
    if d:
      s = SoftMax( Z, False )
      return numpy.multiply( s, float( 1 ) - s )
    else:
      S = numpy.exp( Z )
      return numpy.divide( S, S.sum( axis = 1 ) )
    # end if
  # end def

# end class

## eof - ActivationFunctions.py
