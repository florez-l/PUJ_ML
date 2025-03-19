## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import math, numpy

"""
"""
class Base:

  '''
  '''
  m_Model = None
  m_Lambda1 = 0
  m_Lambda2 = 0
  m_Epsilon = None
  m_Debug = None

  '''
  '''
  def __init__( self, m ):
    self.m_Model = m

    self.m_Epsilon = float( 1 )
    while self.m_Epsilon + 1 > 1:
      self.m_Epsilon /= 2
    # end while
    self.m_Epsilon *= 2
  # end def

  '''
  '''
  def fit( self, D_tr, D_te = None, validation = 'normal', K = 5, batch_size = 0 ):
    if validation.lower( ) == 'normal':
      self._fit_normal( D_tr, D_te, batch_size )
    elif validation.lower( ) == 'loo':
      self._fit_loo( D_tr, D_te, batch_size )
    elif validation.lower( ) == 'kfold':
      self._fit_kfold( D_tr, D_te, K, batch_size )
    # end if
  # end def

  '''
  '''
  def _fit_normal( self, D_tr, D_te, batch_size ):

    # Get data
    X_tr, Y_tr = D_tr
    X_te, Y_te = D_te

    # Compute training batch indices
    indices = [ i for i in range( X_tr.shape[ 0 ] ) ]
    batches = self._batches( indices, batch_size )

    # Real fit
    self._fit( X_tr, Y_tr, X_te, Y_te, batches )
  # end def

  '''
  '''
  def _fit_loo( self, D_tr, D_te, batch_size ):
    M = X.shape[ 0 ]
    indices = [ i for i in range( M ) ]

    v = 0
    for m in range( M ):
      idx = indices[ : m ] + indices[ m + 1 : ]

      print( '*** Leave-one-out: ' + str( m ) + '/' + str( M - 1 ) )
      batches = self._batches( idx, batch_size )
      self._fit( D_tr[ 0 ], D_tr[ 1 ], None, None, batches )

      v += self.m_Model.cost( D_tr[ 0 ][ m , : ], D_tr[ 1 ][ m , : ] )
    # end for
    v /= float( M )

    message = '*** Leave-one-out final validation value = ' + str( v ) + ' ***'
    margin = ''.join( [ '*' for i in range( len( message ) ) ] )
    print( margin )
    print( message )
    print( margin )
  # end def

  '''
  '''
  def _fit_kfold( self, D_tr, D_te, K, batch_size ):
    M = X.shape[ 0 ]
    N = math.ceil( M / K )
    idx = []
    f = 0
    for k in range( K ):
      idx += [ [ f + i for i in range( N ) if ( f + i ) < M ] ]
      f += len( idx[ -1 ] )
    # end for

    v = 0
    for k in range( K ):
      idx_tr = sum( [ idx[ i ] for i in range( K ) if i != k ], [] )
      print( '*** Kfold (K=' + str( K ) + '): ' + str( k + 1 ) + '/' + str( K ) )
      batches = self._batches( idx_tr, batch_size )
      self._fit( D_tr[ 0 ], D_tr[ 1 ], None, None, batches )

      v += self.m_Model.cost( D_tr[ 0 ][ idx[ k ] , : ], D_tr[ 1 ][ idx[ k ] , : ] )
    # end for
    v /= float( M )

    message = \
            '*** Kfold (K=' + str( K ) \
            + \
            ') final validation value = ' \
            + \
            str( v ) \
            + \
            ' ***'
    margin = ''.join( [ '*' for i in range( len( message ) ) ] )
    print( margin )
    print( message )
    print( margin )
  # end def

  '''
  '''
  def _fit( self, X_tr, y_tr, X_te, y_te ):
    pass
  # end def

  '''
  '''
  def _batches( self, indices, batch_size ):
    batches = []
    n_batches = 1
    bs = batch_size
    if bs > 0 and bs < len( indices ):
      n_batches = len( indices ) // bs
    else:
      bs = len( indices )
    # end if

    for b in range( n_batches ):
      batches += [ indices[ b * bs : ( b + 1 ) * bs ] ]
    # end for

    if len( indices ) != ( n_batches * bs ):
      batches += [ indices[ n_batches * bs : ] ]
    # end if

    return batches
  # end def

# end class

## eof - Base.py
