## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy
from PUJ_ML.Helpers import Confussion as Confussion

"""
"""
class ConfussionDebugger:

  '''
  '''
  def __init__( self, max_epochs, m, D_tr, D_te, acc_thr = 1, f1_thr = 1 ):
    self.m_MaxEpochs = max_epochs

    self.m_Model = m

    self.m_X_tr = D_tr[ 0 ]
    self.m_y_tr = D_tr[ 1 ]

    self.m_X_te = D_te[ 0 ]
    self.m_y_te = D_te[ 1 ]

    self.m_AccuracyThreshold = acc_thr
    self.m_F1Threshold = f1_thr
    
  # end def

  '''
  '''
  def __call__( self, t, nG, J_tr, J_te ):

    stop = not ( t < self.m_MaxEpochs )

    K_tr = Confussion( self.m_Model, self.m_X_tr, self.m_y_tr )
    acc_tr = K_tr[ 3 ]
    f1_tr = K_tr[ 4 ]

    stop = stop or not ( acc_tr < self.m_AccuracyThreshold )
    stop = stop or not ( f1_tr < self.m_F1Threshold )
    
    if \
      not J_te is None \
      and \
      not self.m_X_te is None \
      and not self.m_y_te is None\
      :
      K_te = Confussion( self.m_Model, self.m_X_te, self.m_y_te )
      acc_te = K_te[ 3 ]
      f1_te = K_te[ 4 ]

      stop = stop or not ( acc_te < self.m_AccuracyThreshold )
      stop = stop or not ( f1_te < self.m_F1Threshold )

      print( t, J_tr, J_te, acc_tr, acc_te, f1_tr, f1_te )
    else:
      print( t, J_tr, acc_tr, f1_tr )
    # end if
    return stop
  # end def
# end class

## eof - ConfussionDebugger.py
