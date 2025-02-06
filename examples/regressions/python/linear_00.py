## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

"""
Main function: read a CSV and perform a linear regression.
"""
if __name__ == '__main__':

  import numpy, sys
  import PUJ_ML.Regression.Linear

  if len( sys.argv ) <  3:
    print( 'Usage: python ' + sys.argv[ 0 ] + ' file.csv delimiter' )
    sys.exit( 1 )
  # end if
  fname = sys.argv[ 1 ]
  delim = sys.argv[ 2 ]
    
  D = numpy.genfromtxt( fname, delimiter = delim )
  X = numpy.asmatrix( D[ : , 0 : D.shape[ 1 ] - 1 ] )
  y = numpy.asmatrix( D[ : , -1 ] ).T

  model = PUJ_ML.Regression.Linear( )
  model.fit( X, y )
  print( model )
# end if

## eof - $RCSfile$
