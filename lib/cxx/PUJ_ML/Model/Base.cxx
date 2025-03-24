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
prepare_auxiliary_buffer( const TNatural& M )
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
namespace PUJ_ML
{
  namespace Model
  {
    template class PUJ_ML_EXPORT Base< float, unsigned int >;
    template class PUJ_ML_EXPORT Base< float, unsigned long >;
    template class PUJ_ML_EXPORT Base< float, unsigned long long >;

    template class PUJ_ML_EXPORT Base< double, unsigned int >;
    template class PUJ_ML_EXPORT Base< double, unsigned long >;
    template class PUJ_ML_EXPORT Base< double, unsigned long long >;

    template class PUJ_ML_EXPORT Base< long double, unsigned int >;
    template class PUJ_ML_EXPORT Base< long double, unsigned long >;
    template class PUJ_ML_EXPORT Base< long double, unsigned long long >;
  } // end namespace
} // end namespace

// eof - $RCSfile$

