// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__IO__ReadIDX__h__
#define __PUJ_ML__IO__ReadIDX__h__

#include <PUJ_ML/Config.h>


#include <bit>
#include <fstream>
#include <iostream>
#include <vector>
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
    bool ReadIDX( _TMatrix& D, const std::string& fname )
    {
      std::ifstream inputFile( fname.c_str( ), std::ios_base::binary );
      if( !inputFile )
        return( false );
      inputFile.seekg( 0, std::ios_base::end );
      auto length = inputFile.tellg( );
      inputFile.seekg( 0, std::ios_base::beg );
      std::vector< std::byte > buffer( length );
      inputFile.read( reinterpret_cast< char* >( buffer.data( ) ), length );
      inputFile.close( );

      unsigned short magic = *( reinterpret_cast< unsigned short* >( buffer.data( ) ) );
      unsigned char type = *( reinterpret_cast< unsigned char* >( buffer.data( ) + 2 ) );
      unsigned char dims = *( reinterpret_cast< unsigned char* >( buffer.data( ) + 3 ) );

      unsigned long long i = 4;
      unsigned int rows = std::byteswap( *( reinterpret_cast< unsigned int* >( buffer.data( ) + i ) ) );
      unsigned int cols = 1;
      for( unsigned char j = 1; j < dims; ++j )
      {
        i += sizeof( unsigned int );
        cols *= std::byteswap( *( reinterpret_cast< unsigned int* >( buffer.data( ) + i ) ) );
      } // end for
      i += sizeof( unsigned int );
      
      if( type == 0x08 )
        D = Eigen::Map< Eigen::Matrix< unsigned char, Eigen::Dynamic, Eigen::Dynamic > >( reinterpret_cast< unsigned char* >( buffer.data( ) + i ), rows, cols ).template cast< typename _TMatrix::Scalar >( );
      else if( type == 0x09 )
        D = Eigen::Map< Eigen::Matrix< char, Eigen::Dynamic, Eigen::Dynamic > >( reinterpret_cast< char* >( buffer.data( ) + i ), rows, cols ).template cast< typename _TMatrix::Scalar >( );
      else if( type == 0x0B )
        D = Eigen::Map< Eigen::Matrix< short, Eigen::Dynamic, Eigen::Dynamic > >( reinterpret_cast< short* >( buffer.data( ) + i ), rows, cols ).template cast< typename _TMatrix::Scalar >( );
      else if( type == 0x0C )
        D = Eigen::Map< Eigen::Matrix< int, Eigen::Dynamic, Eigen::Dynamic > >( reinterpret_cast< int* >( buffer.data( ) + i ), rows, cols ).template cast< typename _TMatrix::Scalar >( );
      else if( type == 0x0D )
        D = Eigen::Map< Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic > >( reinterpret_cast< float* >( buffer.data( ) + i ), rows, cols ).template cast< typename _TMatrix::Scalar >( );
      else if( type == 0x0E )
        D = Eigen::Map< Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > >( reinterpret_cast< double* >( buffer.data( ) + i ), rows, cols ).template cast< typename _TMatrix::Scalar >( );

      return( true );
    }
  } // end namespace
} // end namespace

#endif // __PUJ_ML__IO__ReadIDX__h__

// eof - $RCSfile$
