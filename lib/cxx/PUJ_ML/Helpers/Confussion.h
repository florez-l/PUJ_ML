// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Helpers__Confussion__h__
#define __PUJ_ML__Helpers__Confussion__h__

#include <set>

#include <PUJ_ML/Config.h>

namespace PUJ_ML
{
  namespace Helpers
  {
    /**
     */
    template< class _Ty, class _Tz >
    auto Confussion(
      const Eigen::EigenBase< _Ty >& by, const Eigen::EigenBase< _Tz >& bz
      )
    {
      using TNatural = unsigned long long;
      auto y = by.derived( ).template cast< TNatural >( ).col( 0 );
      auto z = bz.derived( ).template cast< TNatural >( ).col( 0 );

      std::set< TNatural > labels { y.begin( ), y.end( ) };
      labels.insert( z.begin( ), z.end( ) );
      TNatural L = labels.size( );
      Eigen::Matrix< TNatural, Eigen::Dynamic, Eigen::Dynamic > I( L, L );

      I.setIdentity( );
      return(
        (
          I( y.array( ), Eigen::all ).transpose( )
          *
          I( z.array( ), Eigen::all )
          ).eval( )
        );
    }
  } // end namespace
} // end namespace

#endif // __PUJ_ML__Helpers__Confussion__h__

// eof - $RCSfile$
