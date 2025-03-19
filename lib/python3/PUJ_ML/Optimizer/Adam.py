## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import math, numpy
from .GradientDescent import GradientDescent

"""
"""
class Adam( GradientDescent ):
  '''
  '''
  m_Beta1 = 0.9
  m_Beta2 = 0.999

  '''
  '''
  def __init__( self, m ):
    super( ).__init__( m )
  # end def

  '''
  '''
  def _fit( self, X_tr, y_tr, X_te, y_te, batches ):

    e = self.m_Epsilon
    a = self.m_Alpha
    l1 = self.m_Lambda1
    l2 = self.m_Lambda2
    b1 = self.m_Beta1
    b2 = self.m_Beta2
    b1t = b1
    b2t = b2
    cb1 = float( 1 ) - b1
    cb2 = float( 1 ) - b2
    mt = numpy.zeros( ( self.m_Model.size( ), 1 ) )
    vt = numpy.zeros( ( self.m_Model.size( ), 1 ) )

    self.m_Model.init( )
    t = 0
    stop = False
    while not stop:
      t += 1

      for batch in batches:
        J_tr, G = self.m_Model.cost_gradient( X_tr[ batch , : ], y_tr[ batch , : ], self.m_Lambda1, self.m_Lambda2 )

        mt = ( mt * b1 ) + ( G * cb1 )
        vt = ( vt * b2 ) + ( numpy.multiply( G, G ) * cb2 )

        D = numpy.divide(
          mt * ( float( 1 ) / ( float( 1 ) - b1t ) ),
          numpy.sqrt( vt * ( float( 1 ) / ( float( 1 ) - b2t ) ) ) + e
          )

        self.m_Model -= D * a
      # end for

      if not math.isnan( J_tr ) and not math.isinf( J_tr ):
        J_te = None
        if not X_te is None:
          J_te = self.m_Model.cost( X_te, y_te )
        # end if

        if not self.m_Debug is None:
          stop = self.m_Debug( t, ( G.T @ G )[ 0 , 0 ] ** 0.5, J_tr, J_te )
        # end if

      else:
        stop = True
      # end if

      b1t *= b1
      b2t *= b2
    # end while
  # end def

## eof - Adam.py
