## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

"""
"""
class Simple:

  '''
  '''
  def __init__( self, max_epochs, m, D_tr, D_te, acc_thr = 1, f1_thr = 1 ):
    self.m_MaxEpochs = max_epochs
    self.m_Model = m
  # end def

  '''
  '''
  def __call__( self, t, nG, J_tr, J_te ):
    stop = not ( t < self.m_MaxEpochs )
    if J_te is None:
      print( t, nG, J_tr )
    else:
      print( t, nG, J_tr, J_te )
    # end if
    return stop
  # end def
# end class

## eof - Simple.py
