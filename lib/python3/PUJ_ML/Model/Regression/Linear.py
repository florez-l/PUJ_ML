## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy
from ..Base import Base

"""
"""
class Linear( Base ):

  '''
  '''
  def __init__( self, n = 1 ):
    super( ).__init__( n + 1 )
  # end def

  '''
  '''
  def _evaluate( self, X, threshold ):
    return ( ( X @ self.m_P[ 1 : , 0 ] ) + self.m_P[ 0 , 0 ] ).T
  # end def

  '''
  '''
  def input_size( self ):
    return self.size( ) - 1
  # end def

  '''
  TODO: Use of L1 regularization is not yet solved
  '''
  def fit( self, X, y, L1 = 0, L2 = 0 ):

    n = 0
    if len( X.shape ) == 1:
      n = 1
    elif len( X.shape ) == 2:
      n = X.shape[ 1 ]
    # end if
    m = X.shape[ 0 ]

    if n == 0 or m != y.shape[ 0 ]:
      raise AssertionError( 'Incompatible sizes.' )
    # end if

    b = numpy.zeros( ( 1, n + 1 ) )
    b[ 0 , 0 ] = y.mean( )
    b[ 0 , 1 : ] = numpy.multiply( X, y ).mean( axis = 0 )

    A = numpy.zeros( ( n + 1, n + 1 ) )
    A[ 0 , 0 ] = 1 + L2
    A[ 1 : , 1 : ] \
      = \
      ( numpy.identity( n ) * L2 ) + ( ( X.T @ X ) / float( m ) )
    A[ 0 , 1 : ] = X.mean( axis = 0 )
    A[ 1 : , 0 ] = A[ 0 , 1 : ].T

    self.m_P = numpy.linalg.solve( A, b.T )
  # end def

  '''
  '''
  def cost_gradient( self, X, y, L1, L2 ):
    z = self( X ) - y

    G = numpy.zeros( self.m_P.shape )
    G[ 0 ] = z.mean( ) * float( 2 )
    G[ 1 : ] = numpy.multiply( X, z ).mean( axis = 0 ) * float( 2 )

    r = self._regularization( L1, L2 )

    return ( numpy.multiply( z, z ).mean( ), numpy.asmatrix( G ) + r )
  # end def

  '''
  '''
  def cost( self, X, y ):
    z = self( X ) - y
    return numpy.multiply( z, z ).mean( )
  # end def

# end class

## eof - Linear.py
