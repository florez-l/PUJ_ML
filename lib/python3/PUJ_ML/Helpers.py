## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import argparse, numpy, random, sys
from .Optimizer import *

'''
'''
def SplitDataForBinaryLabeling( A, train_coeff ):

  idx_z = numpy.where( A[ : , -1 ] == 0 )[ 0 ].tolist( )
  idx_o = numpy.where( A[ : , -1 ] == 1 )[ 0 ].tolist( )
  random.shuffle( idx_z )
  random.shuffle( idx_o )

  n = min( len( idx_z ), len( idx_o ) )
  n_tr = int( float( n ) * train_coeff )

  idx_tr = idx_z[ : n_tr ] + idx_o[ : n_tr ]
  random.shuffle( idx_tr )

  D_tr = A[ idx_tr , : ]

  D_te = None
  if n_tr < n:
    idx_te = idx_z[ n_tr : n ] + idx_o[ n_tr : n ]
    random.shuffle( idx_te )
    D_te = A[ idx_te , : ]
  # end if

  return ( D_tr, D_te )
# end def

'''
'''
def Confussion( m, X, y ):
  z = m( X, True )
  yp = numpy.concatenate( ( float( 1 ) - z, z ), axis = 1 )
  yo = numpy.concatenate( ( float( 1 ) - y, y ), axis = 1 )
  K = yo.T @ yp

  TP = float( K[ 0 , 0 ] )
  TN = float( K[ 1 , 1 ] )
  FN = float( K[ 0 , 1 ] )
  FP = float( K[ 1 , 0 ] )

  sensibility = 0
  specificity = 0
  accuracy = 0
  F1 = 0

  if TP + FN != 0: sensibility = TP / ( TP + FN )
  if TN + FP != 0: specificity = TN / ( TN + FP )
  if TP + FP != 0: accuracy = TP / ( TP + FP )
  if TP + ( ( FP + FN ) / 2 ) != 0: F1 = TP / ( TP + ( ( FP + FN ) / 2 ) )
  return ( K, sensibility, specificity, accuracy, F1 )
# end def

'''
'''
def ROC( m, X, y ):

  y_true = y.T.tolist( )[ 0 ]
  y_scores = m( X ).T.tolist( )[ 0 ]
  D = sorted( zip( y_scores, y_true ), reverse = True )

  n_pos = sum( y_true )
  n_neg = len( y_true ) - n_pos

  fpr = [ 0 ]
  tpr = [ 0 ]

  tp = 0
  fp = 0

  for i in range( len( D ) ):
    score, true_label = D[ i ]

    if i == 0 or score != D[ i - 1 ][ 0 ]:
      fpr.append( fp / n_neg )
      tpr.append( tp / n_pos )
    # end if

    if true_label == 1:
      tp += 1
    else:
      fp += 1
    # end if
  # end for

  fpr.append( 1 )
  tpr.append( 1 )

  return ( fpr, tpr )
# end def

'''
'''
def ParseFitArguments( args, mandatory = [], optional = [] ):
  parser = argparse.ArgumentParser(
    prog = args[ 0 ],
    description = 'Fit a model',
    epilog = 'Enjoy!'
    )

  for o in mandatory:
    parser.add_argument( o[ 0 ], type = o[ 1 ] )
  # end for

  for o in optional:
    parser.add_argument( o[ 0 ], o[ 1 ], type = o[ 2 ], default = o[ 3 ] )
  # end for

  parser.add_argument(
    '-o', '--optimizer', type = str, default = 'Adam',
    help = 'Adam|GradientDescent'
    )
  parser.add_argument(
    '-debugger', '--debugger', type = str,
    default = 'Simple', help = 'Simple|Graphic'
    )
  parser.add_argument(
    '-v', '--validation', type = str, default = 'normal',
    help = 'normal|LOO|KfoldK'
    )
  parser.add_argument( '-a', '--alpha', type = float, default = 1e-2 )
  parser.add_argument( '-b', '--batch_size', type = int, default = 0 )
  parser.add_argument( '-L1', '--L1', type = float, default = 0 )
  parser.add_argument( '-L2', '--L2', type = float, default = 0 )
  parser.add_argument( '-e', '--epochs', type = int, default = 1000 )
  try:
    return parser.parse_args( )
  except BaseException as error:
    sys.exit( 1 )
  # end try
# end def

'''
'''
def FitModel( model, args, D_tr, D_te ):

  # Prepare optimizer
  opt = None
  if args.optimizer.lower( ) == 'adam':
    opt = Adam( model )
  elif args.optimizer.lower( ) == 'gradientdescent':
    opt = GradientDescent( model )
  # end if
  if opt is None:
    print( 'Invalid optimizer "' + args.optimizer + '"' )
    sys.exit( 1 )
  # end if
  opt.m_Alpha = args.alpha
  opt.m_Lambda1 = args.L1
  opt.m_Lambda2 = args.L2

  #if args.debugger.lower( ) == 'simple':
  #opt.m_Debug = ConfussionDebugger( args.epochs, model, D_tr, D_te )
  #elif args.debugger.lower( ) == 'graphic':
  #opt.m_Debug = GraphicConfussionDebugger( args.epochs, model, D_tr, D_te )
  # end if
  opt.m_Debug = Debugger.Simple( args.epochs, model, D_tr, D_te )

  # Fit model to train data
  val = ''
  K = 1
  if args.validation.lower( ) == 'normal':
    val = 'normal'
  elif args.validation.lower( ) == 'loo':
    val = 'loo'
  elif args.validation.lower( )[ : 5 ] == 'kfold':
    val = 'kfold'
    K = int( args.validation.lower( )[ 5 : ] )
  else:
    print( 'Invalid validation strategy.' )
    sys.exit( 1 )
  # end if

  # Fit!
  opt.fit( D_tr, D_te, validation = val, K = K, batch_size = args.batch_size )

# end def

## eof - Helpers.py
