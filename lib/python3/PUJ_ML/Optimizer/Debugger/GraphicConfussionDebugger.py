## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import matplotlib.pyplot, numpy, time
from PUJ_ML.Helpers import Confussion as Confussion

"""
"""
class GraphicConfussionDebugger:

  '''
  '''
  def __init__( self, max_epochs, m, D_tr, D_te, acc_thr = 1, f1_thr = 1 ):
    self.m_MaxEpochs = max_epochs
    self.m_Sleep = 1e-4
    self.m_MaxSize = 5000
    self.m_RenderOffset = 100

    self.m_Fig = None
    self.m_Ax = None
    self.m_LineJTr = None
    self.m_LineJTe = None
    self.m_LineAccTr = None
    self.m_LineAccTe = None
    self.m_LineF1Tr = None
    self.m_LineF1Te = None
    self.m_AxX = []
    self.m_AxJTr = []
    self.m_AxJTe = []
    self.m_AxAccTr = []
    self.m_AxAccTe = []
    self.m_AxF1Tr = []
    self.m_AxF1Te = []

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

    has_te = not J_te is None \
             and \
             not self.m_X_te is None \
             and not self.m_y_te is None

    # Initialize plots
    if len( self.m_AxX ) == 0:
      self.m_Fig, self.m_Ax = matplotlib.pyplot.subplots( )
      self.m_LineJTr, = self.m_Ax.plot( [], [] )
      self.m_LineAccTr, = self.m_Ax.plot( [], [] )
      self.m_LineF1Tr, = self.m_Ax.plot( [], [] )
      if has_te:
        self.m_LineJTe, = self.m_Ax.plot( [], [] )
        self.m_LineAccTe, = self.m_Ax.plot( [], [] )
        self.m_LineF1Te, = self.m_Ax.plot( [], [] )
      # end if
      self.m_Ax.set_xlim( 0, 1 )
      self.m_Ax.set_ylim( 0, 1 )
      self.m_Ax.set_xlabel( 'Epoch/iteration' )
      self.m_Ax.set_ylabel( 'Cost/Loss' )
      self.m_Ax.set_title( 'Cost/Loss evolution' )
      matplotlib.pyplot.ion( ) 
      matplotlib.pyplot.show( )
    # end if

    stop = not ( t < self.m_MaxEpochs )

    K_tr = Confussion( self.m_Model, self.m_X_tr, self.m_y_tr )
    acc_tr = K_tr[ 3 ]
    f1_tr = K_tr[ 4 ]

    self.m_AxX += [ t ]
    self.m_AxJTr += [ J_tr ]
    self.m_AxAccTr += [ acc_tr ]
    self.m_AxF1Tr += [ f1_tr ]

    stop = stop or not ( acc_tr < self.m_AccuracyThreshold )
    stop = stop or not ( f1_tr < self.m_F1Threshold )
    
    if has_te:
      K_te = Confussion( self.m_Model, self.m_X_te, self.m_y_te )
      acc_te = K_te[ 3 ]
      f1_te = K_te[ 4 ]

      self.m_AxJTe += [ J_te ]
      self.m_AxAccTe += [ acc_te ]
      self.m_AxF1Te += [ f1_te ]

      stop = stop or not ( acc_te < self.m_AccuracyThreshold )
      stop = stop or not ( f1_te < self.m_F1Threshold )
    else:
      pass
    # end if

    if len( self.m_AxX ) > self.m_MaxSize:
      self.m_AxX = self.m_AxX[ 1 : self.m_MaxSize ]
      self.m_AxJTr = self.m_AxJTr[ 1 : self.m_MaxSize ]
      self.m_AxAccTr = self.m_AxAccTr[ 1 : self.m_MaxSize ]
      self.m_AxF1Tr = self.m_AxF1Tr[ 1 : self.m_MaxSize ]
      if has_te:
        self.m_AxJTe = self.m_AxJTe[ 1 : self.m_MaxSize ]
        self.m_AxAccTe = self.m_AxAccTe[ 1 : self.m_MaxSize ]
        self.m_AxF1Te = self.m_AxF1Te[ 1 : self.m_MaxSize ]
      # end if
    # end if

    if t % self.m_RenderOffset == 0:
      self.m_LineJTr.set_data( self.m_AxX, self.m_AxJTr )
      self.m_LineAccTr.set_data( self.m_AxX, self.m_AxAccTr )
      self.m_LineF1Tr.set_data( self.m_AxX, self.m_AxF1Tr )
      if has_te:
        self.m_LineJTe.set_data( self.m_AxX, self.m_AxJTe )
        self.m_LineAccTe.set_data( self.m_AxX, self.m_AxAccTe )
        self.m_LineF1Te.set_data( self.m_AxX, self.m_AxF1Te )
      # end if

      self.m_Ax.set_xlim( self.m_AxX[ 0 ], self.m_AxX[ -1 ] )
      self.m_Fig.canvas.draw( )
      self.m_Fig.canvas.flush_events( )
      time.sleep( self.m_Sleep )
    # end if

    # Finish visualization
    if stop:
      matplotlib.pyplot.ioff( ) 
      matplotlib.pyplot.show( )
      matplotlib.pyplot.close( self.m_Fig )
    # end if

    return stop
  # end def
# end class

## eof - GraphicConfussionDebugger.py
