// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <PUJ_ML/Model/Base.h>

#include <cstring>
#include <PUJ_ML/IO/Base64.h>

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
PUJ_ML::Model::Base< _TReal, _TNatural >::
Base( const TNatural& n )
{
  this->_allocate( n );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
PUJ_ML::Model::Base< _TReal, _TNatural >::
~Base( )
{
  this->_allocate( 0 );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
typename PUJ_ML::Model::Base< _TReal, _TNatural >::
TNatural PUJ_ML::Model::Base< _TReal, _TNatural >::
size( ) const
{
  return( this->m_S );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
typename PUJ_ML::Model::Base< _TReal, _TNatural >::
TNatural PUJ_ML::Model::Base< _TReal, _TNatural >::
input_size( ) const
{
  return( this->m_S );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void PUJ_ML::Model::Base< _TReal, _TNatural >::
prepare_auxiliary_buffer( const TNatural& M ) const
{
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void PUJ_ML::Model::Base< _TReal, _TNatural >::
free_auxiliary_buffer( ) const
{
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
std::string PUJ_ML::Model::Base< _TReal, _TNatural >::
encode64( ) const
{
  std::string e
    =
    PUJ_ML::IO::Base64::encode( ( unsigned char )( sizeof( TReal ) ) )
    +
    PUJ_ML::IO::Base64::encode( ( unsigned char )( sizeof( TNatural ) ) );
  e += PUJ_ML::IO::Base64::encode( this->m_S );
  for( TReal* p = this->m_P; p != this->m_P + this->m_S; ++p )
    e += PUJ_ML::IO::Base64::encode( p );
  return( e );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void PUJ_ML::Model::Base< _TReal, _TNatural >::
_allocate( const TNatural& n )
{
  if( this->m_P != nullptr )
    std::free( this->m_P );
  this->m_P = nullptr;
  this->m_S = n;
  if( this->m_S > 0 )
    this->m_P
      =
      reinterpret_cast< TReal* >( std::calloc( this->m_S, sizeof( TReal ) ) );
  if( this->m_P != nullptr )
  {
    for( TReal* p = this->m_P; p != this->m_P + this->m_S; ++p )
      *p = TReal( 0 );
  }
  else
    this->m_S = 0;
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void PUJ_ML::Model::Base< _TReal, _TNatural >::
_to_stream( std::ostream& o ) const
{
  o << this->m_S;
  for( TReal* p = this->m_P; p != this->m_P + this->m_S; ++p )
    o << " " << *p;
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
typename PUJ_ML::Model::Base< _TReal, _TNatural >::
TReal PUJ_ML::Model::Base< _TReal, _TNatural >::
_regularize( TReal* G, const TReal& L1, const TReal& L2 ) const
{
  Eigen::Map< TMatrix > P( this->m_P, 1, this->m_S );

  Eigen::Map< TMatrix >( G, 1, this->m_S )
    +=
    P.unaryExpr(
      [&L1,&L2]( const TReal& p ) -> TReal
      {
        TReal l1 = TReal( 0 );
        if     ( p > TReal( 0 ) ) l1 =  L1;
        else if( p < TReal( 0 ) ) l1 = -L1;
        return( l1 + ( TReal( 2 ) * L2 * p ) );
      }
      );
  TReal J
    =
    P.unaryExpr(
      [&L1,&L2]( const TReal& p ) -> TReal
      {
        return( ( L1 * std::fabs( p ) ) + ( L2 * p * p ) );
      }
      ).sum( );
  return( J );
}

// -------------------------------------------------------------------------
namespace PUJ_ML { namespace Model { PUJ_ML_Model_Instance( Base ); } }

// eof - $RCSfile$

