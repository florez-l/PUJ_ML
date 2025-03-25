// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__NeuralNetwork__Activations__h__
#define __PUJ_ML__Model__NeuralNetwork__Activations__h__

#include <PUJ_ML/Config.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <string>
#include <utility>

namespace PUJ_ML
{
  namespace Model
  {
    namespace NeuralNetwork
    {
      /**
       */
      template< class _TReal, class _TNatural >
      class Activations
      {
      public:
        using Self       = Activations;
        using TReal      = _TReal;
        using TNatural   = _TNatural;
        using TMatrix    = Eigen::Matrix< TReal, Eigen::Dynamic, Eigen::Dynamic >;
        using TMatrixMap = Eigen::Map< TMatrix >;

        using TFunction = std::function< void( TMatrixMap&, const TMatrixMap&, bool ) >;
        using TPair = std::pair< std::string, TFunction >;

      public:
        static TPair Get( const std::string& name )
        {
          if( Self::lower( name ) == "relu" )
          {
            return(
              std::make_pair(
                "ReLU",
                []( TMatrixMap& A, const TMatrixMap& Z, bool d ) -> void
                {
                  A
                  =
                  Z.unaryExpr( [&]( const TReal& z ) -> TReal
                    {
                      if( d )
                        return( ( z < TReal( 0 ) )? TReal( 0 ): TReal( 1 ) );
                      else
                        return( ( z < 0 )? TReal( 0 ): z );
                    }
                    );
                }
                )
              );
          }
          else if( Self::lower( name ) == "leakyrelu" )
          {
            return(
              std::make_pair(
                "LeakyReLU",
                []( TMatrixMap& A, const TMatrixMap& Z, bool d ) -> void
                {
                  A
                  =
                  Z.unaryExpr( [&]( const TReal& z ) -> TReal
                    {
                      if( d )
                        return( ( z < TReal( 0 ) )? TReal( 1e-2 ): TReal( 1 ) );
                      else
                        return( ( ( z < 0 )? TReal( 1e-2 ): TReal( 1 ) ) * z );
                    }
                    );
                }
                )
              );
          }
          else if( Self::lower( name ) == "tanh" )
          {
            return(
              std::make_pair(
                "Tanh",
                []( TMatrixMap& A, const TMatrixMap& Z, bool d ) -> void
                {
                  A
                  =
                  Z.unaryExpr( [&]( const TReal& z ) -> TReal
                    {
                      TReal a = std::tanh( z );
                      if( d )
                        return( TReal( 1 ) - ( a * a ) );
                      else
                        return( a );
                    }
                    );
                }
                )
              );
          }
          else if( Self::lower( name ) == "sigmoid" )
          {
            return(
              std::make_pair(
                "Sigmoid",
                []( TMatrixMap& A, const TMatrixMap& Z, bool d ) -> void
                {
                  static const TReal M  = std::numeric_limits< TReal >::max( );
                  static const TReal L  = std::log( M ) / TReal( 2 );
                  A
                  =
                  Z.unaryExpr( [&]( const TReal& z ) -> TReal
                    {
                      TReal a = TReal( 1 );
                      if     ( z < -L ) a = TReal( 0 );
                      else if( L < z  ) a = TReal( 1 );
                      else              a = TReal( 1 ) / ( TReal( 1 ) + std::exp( -z ) );
                      return( ( d )? ( a * ( TReal( 1 ) - a ) ): a );
                    }
                    );
                }
                )
              );
          }
          else if( Self::lower( name ) == "softmax" )
          {
            return( std::make_pair( "SoftMax", []( TMatrixMap& A, const TMatrixMap& Z, bool d ) -> void { } ) );
          }
          else // if( Self::lower( name ) == "linear" )
          {
            return(
              std::make_pair(
                "Linear",
                []( TMatrixMap& A, const TMatrixMap& Z, bool d ) -> void
                {
                  A
                  =
                  Z.unaryExpr( [&]( const TReal& z ) -> TReal
                    {
                      return( ( d )? TReal( 1 ): z );
                    }
                    );
                }
                )
              );
          } // end if
        }

      private:
        static inline std::string lower( const std::string& s )
          {
            std::string r = s;
            std::transform(
              r.begin( ), r.end( ), r.begin( ),
              []( const unsigned char& c )
              {
                return( std::tolower( c ) );
              }
              );
            return( r );
          }
      };
    } // end namespace
  } // end namespace
} // end namespace

// TODO: #include <PUJ_ML/Model/NeuralNetwork/Activations.hxx>

#endif // __PUJ_ML__Model__NeuralNetwork__Activations__h__

// eof - $RCSfile$
