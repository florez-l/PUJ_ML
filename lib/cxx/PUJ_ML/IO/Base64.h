// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
// http://www.adp-gmbh.ch/cpp/common/base64.html
// =========================================================================
#ifndef __PUJ_ML__IO__Base64__h__
#define __PUJ_ML__IO__Base64__h__

#include <PUJ_ML/Export.h>
#include <string>

namespace PUJ_ML
{
  namespace IO
  {
    namespace Base64
    {
      /// TODO
      static const std::string base64_chars =
         "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
         "abcdefghijklmnopqrstuvwxyz"
         "0123456789+/";

      /**
       */
      bool PUJ_ML_EXPORT
      is_base64( const unsigned char& c );

      /**
       */
      std::string PUJ_ML_EXPORT
      real_encode( const unsigned char* B, unsigned int N );

      /**
       */
      std::string PUJ_ML_EXPORT
      real_decode( const std::string& E );

      /**
       */
      template< class _TV >
      std::string encode( const _TV& v )
      {
        return(
          real_encode(
            reinterpret_cast< const unsigned char* >( &v ), sizeof( _TV )
            )
          );
      }

      /**
       */
      template< class _TV >
      std::string encode( const _TV* p )
      {
        return(
          real_encode(
            reinterpret_cast< const unsigned char* >( p ), sizeof( _TV )
            )
          );
      }
    } // end namespace
  } // end namespace
} // end namespace

#endif // __PUJ_ML__IO__Base64__h__

// eof - $RCSfile$
