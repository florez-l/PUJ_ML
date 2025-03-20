// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
// http://www.adp-gmbh.ch/cpp/common/base64.html
// =========================================================================

#include <PUJ_ML/IO/Base64.h>

// -------------------------------------------------------------------------
bool PUJ_ML::IO::Base64::
is_base64( const unsigned char& c )
{
  return( ( std::isalnum( c ) || ( c == '+' ) || ( c == '/' ) ) );
}

// -------------------------------------------------------------------------
std::string PUJ_ML::IO::Base64::
real_encode( const unsigned char* bytes_to_encode, unsigned int in_len )
{
  std::string ret;
  int i = 0;
  int j = 0;
  unsigned char ca3[  3  ];
  unsigned char ca4[  4  ];

  while( in_len-- )
  {
    ca3[ i++ ] = *( bytes_to_encode++ );
    if( i == 3 )
    {
      ca4[ 0 ] = ( ca3[ 0 ] & 0xfc ) >> 2;
      ca4[ 1 ] = ( ( ca3[ 0 ] & 0x03 ) << 4 ) + ( ( ca3[ 1 ] & 0xf0 ) >> 4 );
      ca4[ 2 ] = ( ( ca3[ 1 ] & 0x0f ) << 2 ) + ( ( ca3[ 2 ] & 0xc0 ) >> 6 );
      ca4[ 3 ] = ca3[ 2 ] & 0x3f;

      for( i = 0; ( i <4 ) ; i++ )
        ret += base64_chars[ ca4[ i ] ];
      i = 0;
    } // end if
  } // end while

  if( i > 0 )
  {
    for( j = i; j < 3; j++ )
      ca3[ j ] = '\0';

    ca4[ 0 ] = ( ca3[ 0 ] & 0xfc ) >> 2;
    ca4[ 1 ] = ( ( ca3[ 0 ] & 0x03 ) << 4 ) + ( ( ca3[ 1 ] & 0xf0 ) >> 4 );
    ca4[ 2 ] = ( ( ca3[ 1 ] & 0x0f ) << 2 ) + ( ( ca3[ 2 ] & 0xc0 ) >> 6 );
    ca4[ 3 ] = ca3[ 2 ] & 0x3f;

    for( j = 0; ( j < i + 1 ); j++ )
      ret += base64_chars[ ca4[ j ] ];

    while( ( i++ < 3 ) )
      ret += '=';
  } // end if
  return( ret );
}

// -------------------------------------------------------------------------
std::string PUJ_ML::IO::Base64::
real_decode( const std::string& encoded_string )
{
  int in_len = encoded_string.size( );
  int i = 0;
  int j = 0;
  int in_ = 0;
  unsigned char ca4[ 4 ], ca3[ 3 ];
  std::string ret;

  while(
    in_len--
    &&
    ( encoded_string[ in_ ] != '=' )
    &&
    is_base64( encoded_string[ in_ ] )
    )
  {
    ca4[ i++ ] = encoded_string[ in_ ]; in_++;
    if( i == 4 )
    {
      for( i = 0; i < 4; i++ )
        ca4[ i ] = base64_chars.find( ca4[ i ] );

      ca3[ 0 ] = ( ca4[ 0 ] << 2 ) + ( ( ca4[ 1 ] & 0x30 ) >> 4 );
      ca3[ 1 ] = ( ( ca4[ 1 ] & 0xf ) << 4 ) + ( ( ca4[ 2 ] & 0x3c ) >> 2 );
      ca3[ 2 ] = ( ( ca4[ 2 ] & 0x3 ) << 6 ) + ca4[ 3 ];

      for( i = 0; i < 3; i++ )
        ret += ca3[ i ];
      i = 0;
    } // end if
  } // end while

  if( i )
  {
    for( j = i; j <4; j++ )
      ca4[ j ] = 0;

    for( j = 0; j <4; j++ )
      ca4[ j ] = base64_chars.find( ca4[ j ] );

    ca3[ 0 ] = ( ca4[ 0 ] << 2 ) + ( ( ca4[ 1 ] & 0x30 ) >> 4 );
    ca3[ 1 ] = ( ( ca4[ 1 ] & 0xf ) << 4 ) + ( ( ca4[ 2 ] & 0x3c ) >> 2 );
    ca3[ 2 ] = ( ( ca4[ 2 ] & 0x3 ) << 6 ) + ca4[ 3 ];

    for( j = 0; ( j < i - 1 ); j++ ) ret += ca3[ j ];
  } // end if
  return( ret );
}

// eof - $RCSfile$
