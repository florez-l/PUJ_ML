## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import base64, numpy, struct

"""
"""
class Base:

  '''
  '''
  m_P = None
  m_Epsilon = 0

  '''
  '''
  def __init__( self, n = 1 ):

    self.m_P = numpy.zeros( ( n, 1 ) )

    self.m_Epsilon = float( 1 )
    while self.m_Epsilon + 1 > 1:
      self.m_Epsilon /= 2
    # end while
    self.m_Epsilon *= 2

  # end def

  '''
  '''
  def __call__( self, X, threshold = False ):
    if not self.m_P is None:
      if isinstance( X, numpy.matrix ):
        return self._evaluate( X, threshold )
      elif isinstance( X, numpy.ndarray ):
        return self( numpy.asmatrix( X ), threshold )
      elif isinstance( X, list ):
        m = self.input_size( )
        s = ( len( X ) // m, m )
        n = s[ 0 ] * s[ 1 ]
        return self(
          numpy.reshape( numpy.matrix( X[ : n ] ), newshape = s ), threshold
          )
      else:
        return None
      # end if
    else:
      return None
    # end if
  # end def

  '''
  '''
  def __getitem__( self, i ):
    if not self.m_P is None:
      if i < self.m_P.shape[ 0 ]:
        return self.m_P[ i, 0 ]
      else:
        return float( 0 )
      # end if
    else:
      return float( 0 )
    # end if
  # end def

  '''
  '''
  def __setitem__( self, i, v ):
    if not self.m_P is None:
      if i < self.m_P.shape[ 0 ]:
        self.m_P[ i, 0 ] = v
      # end if
    # end if
  # end def

  '''
  '''
  def __iadd__( self, w ):
    self.m_P += w
    return self
  # end def

  '''
  '''
  def __isub__( self, w ):
    self.m_P -= w
    return self
  # end def

  '''
  '''
  def __str__( self ):
    if not self.m_P is None:
      N = self.m_P.size
      L = self.m_P.reshape( N ).tolist( )
      s = str( N )
      for l in L:
        s += ' ' + str( l )
      # end for
      return s
    else:
      return '0'
    # end if
  # end def

  '''
  '''
  def encode64( self ):
    N = 0
    P = None
    if not self.m_P is None:
      N = self.m_P.size
      L = self.m_P.reshape( N ).tolist( )
      P = bytearray( struct.pack( '%sd' % len( L ), *L ) )
    # end if
    B = N.to_bytes( 8 )
    if not P is None:
      B += P
    # end if
    return str( base64.b64encode( B ) )[ 2 : -1 ]
  # end def

  '''
  '''
  def decode64( self, m ):
    pass
    ## TODO
    #D = base64.b64decode( E )
    #N = int.from_bytes( D[ : 8 ] )

    #L = list( struct.unpack( '%sd' % N, D[ 8 : ] ) )
    #print( L )
    #print( '========================================' )
  # end def

  '''
  '''
  def input_size( self ):
    return self.size( )
  # end def

  '''
  '''
  def size( self ):
    return self.m_P.shape[ 0 ]
  # end def

  '''
  '''
  def init( self ):
    if not self.m_P is None:
      self.m_P *= float( 0 )
    # end if
  # end def

  '''
  '''
  def _regularization( self, L1, L2 ):
    r = self.m_P * ( float( 2 ) * L2 )
    pi = numpy.where( self.m_P > 0 )[ 0 ]
    ni = numpy.where( self.m_P < 0 )[ 0 ]
    if pi.size > 0: r[ pi ] += L1
    if ni.size > 0: r[ ni ] -= L1
    return numpy.asmatrix( r )
  # end def
# end class

## eof - Base.py
