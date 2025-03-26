// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__Base__hxx__
#define __PUJ_ML__Model__Base__hxx__

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _Tw >
typename PUJ_ML::Model::Base< _TReal, _TNatural >::
Self& PUJ_ML::Model::Base< _TReal, _TNatural >::
operator+=( const Eigen::EigenBase< _Tw >& w )
{
  if( w.size( ) == this->m_S )
    Eigen::Map< TMatrix >( this->m_P, w.rows( ), w.cols( ) )
      +=
      w.derived( ).template cast< TReal >( );
  return( *this );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _Tw >
typename PUJ_ML::Model::Base< _TReal, _TNatural >::
Self& PUJ_ML::Model::Base< _TReal, _TNatural >::
operator-=( const Eigen::EigenBase< _Tw >& w )
{
  if( w.size( ) == this->m_S )
    Eigen::Map< TMatrix >( this->m_P, w.rows( ), w.cols( ) )
      -=
      w.derived( ).template cast< TReal >( );
  return( *this );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TA, class _TY >
typename PUJ_ML::Model::Base< _TReal, _TNatural >::
TReal PUJ_ML::Model::Base< _TReal, _TNatural >::
_cost(
  const Eigen::EigenBase< _TA >& bA, const Eigen::EigenBase< _TY >& bY
  ) const
{
  static const TReal E
    =
    std::pow(
      TReal( 10 ),
      std::log10( std::numeric_limits< TReal >::epsilon( ) ) * TReal( 0.5 )
      );
  static const TReal D = std::log( E );

  auto A = bA.derived( ).template cast< TReal >( );
  auto Y = bY.derived( ).template cast< TReal >( );

  /* TODO
     if( X.rows( ) != y.rows( ) )
     raise AssertionError( "Incompatible sizes." )
     # end if
  */

  if( this->cost_type( ) == "mse" )
  {
    return( ( A - Y ).array( ).pow( 2 ).mean( ) );
  }
  else if( this->cost_type( ) == "bce" )
  {
    return(
      Y.binaryExpr(
        A,
        [&]( const TReal& y, const TReal& a ) -> TReal
        {
          TReal v = ( y == TReal( 0 ) )? ( TReal( 1 ) - a ): a;
          return( -( ( E < v )? std::log( v ): D ) );
        }
        ).mean( )
      );
  }
  else if( this->cost_type( ) == "cce" )
  {
    return( TReal( 0 ) );
  }
  else
    return( TReal( 0 ) );
}

#endif // __PUJ_ML__Model__Base__hxx__

// eof - $RCSfile$

