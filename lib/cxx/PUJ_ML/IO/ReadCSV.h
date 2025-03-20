// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__IO__ReadCSV__h__
#define __PUJ_ML__IO__ReadCSV__h__

#include <PUJ_ML/Config.h>


#include <filesystem>
#include <fstream>
#include <sstream>
#include <deque>
#include <boost/algorithm/string.hpp>

namespace PUJ_ML
{
  namespace IO
  {
    /**
     */
    template< class _TMatrix >
    bool ReadCSV(
      _TMatrix& D, const std::string& fname,
      const unsigned char delimiter = ','
      )
    {
      std::stringstream seps;
      seps << delimiter;

      // Load buffer
      std::ifstream ifs( fname.c_str( ) );
      if( !ifs )
        return( false );
      ifs.seekg( 0, std::ios::end );
      std::size_t size = ifs.tellg( );
      ifs.seekg( 0, std::ios::beg );
      std::string buffer( size, 0 );
      ifs.read( &buffer[ 0 ], size );
      ifs.close( );
      std::istringstream input( buffer );

      // Read line by line
      std::deque< std::stringstream > lines;
      std::string line;
      unsigned long long n = 0;
      while( std::getline( input, line ) )
      {
        if( line != "" )
        {
          if( int( line[ 0 ] ) != 0 )
          {
            std::deque< std::string > tokens;
            boost::split( tokens, line, boost::is_any_of( seps.str( ) ) );
            unsigned int i = 0;
            lines.push_back( std::stringstream( ) );
            for( const std::string& t: tokens )
            {
              if( t != "" )
              {
                *lines.rbegin( ) << t << " ";
                i++;
              } // end if
            } // end for
            if( i > 0 )
              n = ( n < i )? i: n;
            else
              lines.pop_back( );
          } // end if
        } // end if
      } // end while

      // Pass to Eigen::Matrix
      unsigned long long m = lines.size( );
      D = _TMatrix::Zero( m, n );
      for( unsigned long long r = 0; r < m; ++r )
        for( unsigned long long c = 0; c < n; ++c )
          lines[ r ] >> D( r, c );

      return( true );
    }
  } // end namespace
} // end namespace

#endif // __PUJ_ML__IO__ReadCSV__h__

// eof - $RCSfile$
