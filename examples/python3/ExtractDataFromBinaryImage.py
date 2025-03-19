## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import cv2, numpy, sys

image = cv2.imread( sys.argv[ 1 ], cv2.IMREAD_GRAYSCALE ).astype( int )
labels = numpy.unique( image )
if len( labels ) != 2:
  print( 'Input image is not binary.' )
  sys.exit( 1 )
# end if

N = ( image.size, 1 )
Y, X = numpy.indices( image.shape )
D = numpy.concatenate(
    ( X.reshape( N ),
      Y.reshape( N ),
      ( image == labels[ 1 ] ).astype( int ).reshape( N ) ),
    axis = 1
    )
numpy.savetxt( sys.argv[ 2 ], D, delimiter = ' ' )

## eof - ExtractDataFromBinaryImage.py
