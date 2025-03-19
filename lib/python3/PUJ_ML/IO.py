## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import cv2, idx2numpy, numpy, os, random
from .Helpers import *

'''
'''
def ReadCSV( train, test, delimiter = ',' ):
  R_tr = None
  R_te = None
  try:
    train_coeff = abs( float( 1 ) - abs( float( test ) ) )
    if train_coeff > 1: train_coeff = 1
    if train_coeff < 1:
      R_tr, R_te = \
        SplitDataForBinaryLabeling( \
          numpy.genfromtxt( train, delimiter = delimiter ), \
          train_coeff
          )
    else:
      R_tr = numpy.genfromtxt( train, delimiter = delimiter )
    # end if
  except ValueError:
    R_tr = numpy.genfromtxt( train, delimiter = delimiter )
    R_te = numpy.genfromtxt( test, delimiter = delimiter )
  # end try

  D_tr = ( R_tr[ : , : -1 ], R_tr[ : , -1 : ] )
  D_te = ( None, None )
  if not R_te is None:
    D_te = ( R_te[ : , : -1 ], R_te[ : , -1 : ] )
  # end if

  return ( D_tr, D_te )
# end def

'''
'''
def ReadFromImage( fname, sampling ):

  # Read image and convert it to a data matrix
  image = cv2.imread( fname, cv2.IMREAD_GRAYSCALE ).astype( int )
  N = ( image.size, 1 )
  L = numpy.unique( image ).tolist( )
  mV = image.min( )
  MV = image.max( )

  d = ( float( len( L ) - 1 ) * ( image - mV ) / ( MV - mV ) ).astype( int )
  Y, X = numpy.indices( image.shape )
  R = numpy.concatenate( ( X.reshape( N ), Y.reshape( N ), d.reshape( N ) ), axis = 1 )
  indices = [ i for i in range( R.shape[ 0 ] ) ]
  random.shuffle( indices )
  R_tr = R[ indices , : ]
  R_te = None

  # Extract given number of samples
  samples = R_tr.shape[ 0 ]
  for v in L:
    l = int( float( len( L ) - 1 ) * ( v - mV ) / ( MV - mV ) )
    s = ( R_tr[ : , -1 ] == l ).astype( int ).sum( )
    if s < samples:
      samples = s
    # end if
  # end for
  if sampling > 0 and sampling < samples:
    samples = sampling
  # end if

  S_tr = None
  for v in L:
    l = int( float( len( L ) - 1 ) * ( v - mV ) / ( MV - mV ) )
    indices = numpy.where( R_tr[ : , -1 ] == l )[ 0 ].tolist( )
    random.shuffle( indices )
    if S_tr is None:
      S_tr = R_tr[ indices[ : samples ] , : ]
    else:
      S_tr \
        = \
        numpy.concatenate( \
          ( S_tr, R_tr[ indices[ : samples ] , : ] ), axis = 0 \
          )
    # end if
  # end for
  indices = [ i for i in range( S_tr.shape[ 0 ] ) ]
  random.shuffle( indices )

  return ( ( S_tr[ indices , : -1 ], S_tr[ indices , -1 ].reshape( ( S_tr.shape[ 0 ], 1 ) ) ), ( None, None ) )
# end def

'''
'''
def ReadMNIST( dn ):
  X_tr_name = 'train-images.idx3-ubyte'
  L_tr_name = 'train-labels.idx1-ubyte'
  X_te_name = 't10k-images.idx3-ubyte'
  L_te_name = 't10k-labels.idx1-ubyte'

  X_tr = idx2numpy.convert_from_file( os.path.join( dn, X_tr_name ) )
  L_tr = idx2numpy.convert_from_file( os.path.join( dn, L_tr_name ) )
  X_te = idx2numpy.convert_from_file( os.path.join( dn, X_te_name ) )
  L_te = idx2numpy.convert_from_file( os.path.join( dn, L_te_name ) )

  image_shape = X_tr.shape[ 1 : ]
  data_size = image_shape[ 0 ] * image_shape[ 1 ]
  X_tr = X_tr.reshape( ( X_tr.shape[ 0 ], data_size ) )
  X_te = X_te.reshape( ( X_te.shape[ 0 ], data_size ) )

  L = numpy.identity( len( numpy.unique( L_tr ).tolist( ) ) )
  Y_tr = L[ L_tr.tolist( ) , : ]
  Y_te = L[ L_te.tolist( ) , : ]

  return ( ( X_tr, Y_tr ), ( X_te, Y_te ) )
# end def

## eof - IO.py
