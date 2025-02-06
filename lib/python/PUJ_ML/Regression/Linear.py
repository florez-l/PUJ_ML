## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy

"""
"""
class Linear:

  m_W = None
  m_B = None

  '''
  '''
  def __init__( self ):
    pass
  # end def

  '''
  '''
  def fit( self, X, y ):
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
    b[ 0 , 0 ] = y.sum( axis = 0 )
    b[ 0 , 1 : ] = numpy.multiply( X, y ).sum( axis = 0 )

    A = numpy.zeros( ( n + 1, n + 1 ) )
    A[ 0 , 0 ] = float( m )
    A[ 1 : , 1 : ] = X.T @ X
    A[ 0 , 1 : ] = X.sum( axis = 0 )
    A[ 1 : , 0 ] = A[ 0 , 1 : ].T

    T = numpy.linalg.solve( A, b.T )
    self.m_B = T[ 0, 0 ]
    self.m_W = numpy.asmatrix( T[ 1 : , 0 ] ).T
  # end def

  '''
  '''
  def __str__( self ):
    s = ''
    if self.m_W is None:
      s = 'None'
    else:
      s = str( self.m_W.shape[ 0 ] + 1 ) + ' ' + str( self.m_B )
      for i in range( self.m_W.shape[ 0 ] ):
        s += ' ' + str( self.m_W[ i , 0 ] )
      # end for
    # end if
    return s
  # end def
  
# end class

## eof - $RCSfile$
