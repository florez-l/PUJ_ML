// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__Regression__Logistic__hxx__
#define __PUJ_ML__Model__Regression__Logistic__hxx__

#include <Eigen/Dense>

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TX >
auto PUJ_ML::Model::Regression::Logistic< _TReal, _TNatural >::
operator()( const Eigen::EigenBase< _TX >& X ) const
{
  return( this->_eval( X, false ) );
}


// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TX >
auto PUJ_ML::Model::Regression::Logistic< _TReal, _TNatural >::
threshold( const Eigen::EigenBase< _TX >& X ) const
{
  return( this->_eval( X, true ) );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TX, class _Ty >
typename PUJ_ML::Model::Regression::Logistic< _TReal, _TNatural >::
TReal PUJ_ML::Model::Regression::Logistic< _TReal, _TNatural >::
cost(
  const Eigen::EigenBase< _TX >& X, const Eigen::EigenBase< _Ty >& y
  ) const
{
  return( this->_cost( this->operator()( X ), y ) );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TG, class _TX, class _Ty >
typename PUJ_ML::Model::Regression::Logistic< _TReal, _TNatural >::
TReal PUJ_ML::Model::Regression::Logistic< _TReal, _TNatural >::
cost_gradient(
  TReal* G,
  const Eigen::EigenBase< _TX >& bX,
  const Eigen::EigenBase< _Ty >& by,
  const TReal& L1, const TReal& L2
  ) const
{
  auto X = bX.derived( ).template cast< TReal >( );
  auto y = by.derived( ).template cast< TReal >( );

  TColumn z = this->operator()( X );
  *G = z.mean( );
  Eigen::Map< TMatrix >( G + 1, 1, X.cols( ) )
    =
    ( X.array( ).colwise( ) * z.array( ) ).colwise( ).mean( );

  return( this->_cost( z, y ) + this->_regularize( G, L1, L2 ) );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TX, class _Ty >
void PUJ_ML::Model::Regression::Logistic< _TReal, _TNatural >::
fit(
  const Eigen::EigenBase< _TX >& bX,
  const Eigen::EigenBase< _Ty >& by,
  const TReal& L1, const TReal& L2
  )
{
  /* TODO
     if( n == 0 || m != y.rows( ) )
     raise AssertionError( "Incompatible sizes." )
     # end if
  */
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TX >
auto PUJ_ML::Model::Regression::Logistic< _TReal, _TNatural >::
_eval( const Eigen::EigenBase< _TX >& X, const bool& threshold ) const
{
  static const TReal _0  = TReal( 0 );
  static const TReal _1  = TReal( 1 );
  static const TReal M  = std::numeric_limits< TReal >::max( );
  static const TReal L  = std::log( M ) / TReal( 2 );
  auto f = [&threshold]( TReal z ) -> TReal
    {
      TReal s;
      if     ( z >  L ) s = _1;
      else if( z < -L ) s = _0;
      else              s = _1 / ( _1 + std::exp( -z ) );
      return( ( threshold )? ( ( s < TReal( 0.5 ) )? _0: _1 ): s );
    };
  return( this->Superclass::operator()( X ).unaryExpr( f ) );
}

#endif // __PUJ_ML__Model__Regression__Logistic__hxx__


// eof - $RCSfile$
