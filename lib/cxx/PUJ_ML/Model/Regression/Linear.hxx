// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__Regression__Linear__hxx__
#define __PUJ_ML__Model__Regression__Linear__hxx__

#include <Eigen/Dense>

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TX >
auto PUJ_ML::Model::Regression::Linear< _TReal, _TNatural >::
operator()( const Eigen::EigenBase< _TX >& X ) const
{
  return(
    (
      (
        X.derived( ).template cast< TReal >( )
        *
        Eigen::Map< TColumn >( this->m_P + 1, this->m_S - 1, 1 )
        ).array( ) + this->m_P[ 0 ] ).matrix( )
    );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TX, class _Ty >
typename PUJ_ML::Model::Regression::Linear< _TReal, _TNatural >::
TReal PUJ_ML::Model::Regression::Linear< _TReal, _TNatural >::
cost(
  const Eigen::EigenBase< _TX >& bX, const Eigen::EigenBase< _Ty >& by
  ) const
{
  auto X = bX.derived( ).template cast< TReal >( );
  auto y = by.derived( ).template cast< TReal >( ).col( 0 );

  /* TODO
     if( X.rows( ) != y.rows( ) )
     raise AssertionError( "Incompatible sizes." )
     # end if
  */
  return( ( this->operator()( X ) - y ).array( ).pow( 2 ).mean( ) );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TG, class _TX, class _Ty >
typename PUJ_ML::Model::Regression::Linear< _TReal, _TNatural >::
TReal PUJ_ML::Model::Regression::Linear< _TReal, _TNatural >::
cost_gradient(
  Eigen::EigenBase< _TG >& G,
  const Eigen::EigenBase< _TX >& bX,
  const Eigen::EigenBase< _Ty >& by,
  const TReal& L1, const TReal& L2
  ) const
{
  auto X = bX.derived( ).template cast< TReal >( );
  auto y = by.derived( ).template cast< TReal >( );

  TColumn z = this->operator()( X ) - y;
  G.derived( )( 0 , 0 ) = TReal( 2 ) * z.mean( );
  G.derived( ).block( 1, 0, X.cols( ), 1 )
    =
    ( X.array( ).colwise( ) * z.array( ) ).colwise( ).mean( ).transpose( );

  return( z.array( ).pow( 2 ).mean( ) );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TX, class _Ty >
void PUJ_ML::Model::Regression::Linear< _TReal, _TNatural >::
fit(
  const Eigen::EigenBase< _TX >& bX,
  const Eigen::EigenBase< _Ty >& by,
  const TReal& L1, const TReal& L2
  )
{
  auto X = bX.derived( ).template cast< TReal >( );
  auto y = by.derived( ).template cast< TReal >( ).col( 0 ).array( );

  TNatural n = X.cols( );
  TNatural m = X.rows( );
  /* TODO
     if( n == 0 || m != y.rows( ) )
     raise AssertionError( "Incompatible sizes." )
     # end if
  */

  // Fill vector
  TRow b = TRow::Zero( n + 1 );
  b( 0 ) = y.mean( );
  b.block( 0, 1, 1, n ) = ( X.array( ).colwise( ) * y ).colwise( ).mean( );

  // Fill matrix
  TMatrix A = TMatrix::Zero( n + 1, n + 1 );
  A( 0 , 0 ) = TReal( 1 ) + L2;
  A.block( 1, 1, n, n )
    =
    ( TMatrix::Identity( n, n ) * L2 )
    +
    ( ( X.transpose( ) * X ) / TReal( m ) );
  A.block( 0, 1, 1, n ) = X.colwise( ).mean( );
  A.block( 1, 0, n, 1 ) = A.block( 0, 1, 1, n ).transpose( );

  // Solve system
  this->_allocate( n + 1 );
  Eigen::Map< TColumn >( this->m_P, n + 1, 1 )
    =
    A.lu( ).solve( b.transpose( ) );
}

#endif // __PUJ_ML__Model__Regression__Linear__hxx__

// eof - $RCSfile$
