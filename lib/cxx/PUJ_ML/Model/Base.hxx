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

#endif // __PUJ_ML__Model__Base__hxx__

// eof - $RCSfile$

