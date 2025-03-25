// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <PUJ_ML/Model/NeuralNetwork/Activations.h>

#include <algorithm>
#include <cctype>
#include <cmath>

// -------------------------------------------------------------------------
template< class _TReal >
typename PUJ_ML::Model::NeuralNetwork::Activations< _TReal >::
TPair PUJ_ML::Model::NeuralNetwork::Activations< _TReal >::
Get( const std::string& name )
{
  auto _lwr = []( const std::string& s ) -> std::string
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
    };

  if( _lwr( name ) == "relu" )
  {
    return(
      std::make_pair(
        "ReLU",
        []( TMatrixMap& A, const TMatrixMap& Z, bool d ) -> void
        {
          A
            =
            Z.unaryExpr(
              [&]( const TReal& z ) -> TReal
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
  else if( _lwr( name ) == "leakyrelu" )
  {
    return(
      std::make_pair(
        "LeakyReLU",
        []( TMatrixMap& A, const TMatrixMap& Z, bool d ) -> void
        {
          A
            =
            Z.unaryExpr(
              [&]( const TReal& z ) -> TReal
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
  else if( _lwr( name ) == "tanh" )
  {
    return(
      std::make_pair(
        "Tanh",
        []( TMatrixMap& A, const TMatrixMap& Z, bool d ) -> void
        {
          A
            =
            Z.unaryExpr(
              [&]( const TReal& z ) -> TReal
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
  else if( _lwr( name ) == "sigmoid" )
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
            Z.unaryExpr(
              [&]( const TReal& z ) -> TReal
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
  else if( _lwr( name ) == "softmax" )
  {
    return(
      std::make_pair(
        "SoftMax",
        []( TMatrixMap& A, const TMatrixMap& Z, bool d ) -> void
        {
          TMatrix m = Z.rowwise( ).maxCoeff( );
          A = ( Z.colwise( ) - m.col( 0 ) ).array( ).exp( );
          m = A.rowwise( ).sum( );
          A.array( ).colwise( ) /= m.array( ).col( 0 );
          if( d )
            A
              =
              A.unaryExpr(
                []( const TReal& a ) -> TReal
                {
                  return( a * ( TReal( 1 ) - a ) );
                }
                );
        }
        )
      );
  }
  else // if( _lwr( name ) == "linear" )
  {
    return(
      std::make_pair(
        "Linear",
        []( TMatrixMap& A, const TMatrixMap& Z, bool d ) -> void
        {
          A
            =
            Z.unaryExpr(
              [&]( const TReal& z ) -> TReal
              {
                return( ( d )? TReal( 1 ): z );
              }
              );
        }
        )
      );
  } // end if
}

// -------------------------------------------------------------------------
namespace PUJ_ML
{
  namespace Model
  {
    namespace NeuralNetwork
    {
      template class PUJ_ML_EXPORT Activations< float >;
      template class PUJ_ML_EXPORT Activations< double >;
      template class PUJ_ML_EXPORT Activations< long double >;
    } // end namespace
  } // end namespace
} // end namespace

// eof - $RCSfile$
