## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy
from .Linear import Linear

"""
"""
class Logistic( Linear ):

  '''
  '''
  def __init__( self, n = 1 ):
    super( ).__init__( n )
  # end def

  '''
  '''
  def _evaluate( self, X, threshold ):
    z = super( )._evaluate( X, False )
    if threshold:
      return ( z >= 0.5 ).astype( float )
    else:
      return float( 1 ) / ( float( 1 ) + numpy.exp( -z ) )
    # end if
  # end def

  '''
  '''
  def fit( self, X, y, L1 = 0, L2 = 0 ):
    raise AssertionError(
      'There is no closed solution for a logistic regression.'
      )
  # end def

  '''
  '''
  def cost_gradient( self, X, y, L1, L2 ):
    z = self( X )

    zi = numpy.where( y == 0 )[ 0 ].tolist( )
    oi = numpy.where( y == 1 )[ 0 ].tolist( )

    J  = numpy.log( ( float( 1 ) + self.m_Epsilon ) - z[ zi , : ] ).sum( )
    J += numpy.log( z[ oi , : ] + self.m_Epsilon ).sum( )
    J /= float( X.shape[ 0 ] )

    G = numpy.zeros( self.m_P.shape )
    G[ 0 ] = ( z - y ).mean( )
    G[ 1 : ] = numpy.multiply( X, z - y ).mean( axis = 0 ).T

    return ( -J, G + self._regularization( L1, L2 ) )
  # end def

  '''
  '''
  def cost( self, X, y ):
    z = self( X )

    zi = numpy.where( y == 0 )[ 0 ].tolist( )
    oi = numpy.where( y == 1 )[ 0 ].tolist( )

    J  = numpy.log( float( 1 ) - z[ zi , : ] + self.m_Epsilon ).sum( )
    J += numpy.log( z[ oi , : ] + self.m_Epsilon ).sum( )
    J /= float( X.shape[ 0 ] )

    return -J
  # end def

# end class

## eof - Logistic.py
