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
  static const TReal _0  = TReal( 0 );
  static const TReal _1  = TReal( 1 );
  static const TReal _M  = std::numeric_limits< TReal >::max( );
  static const TReal _L  = std::log( _M ) / TReal( 2 );
  auto f = []( TReal z ) -> TReal
    {
      TReal s;
      if     ( z >  _L ) s = _1;
      else if( z < -_L ) s = _0;
      else               s = _1 / ( _1 + std::exp( -z ) );
      return( s );
    };
  return( this->Superclass::operator()( X ).unaryExpr( f ) );
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
  TColumn z = this->operator()( X );
  SVisitor v( z );
  y.derived( ).template cast< TReal >( ).visit( v );
  return( v.J / TReal( X.rows( ) ) );
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

#endif // __PUJ_ML__Model__Regression__Logistic__hxx__

// eof - $RCSfile$
