// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <PUJ_ML/Model/Regression/Linear.h>

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
PUJ_ML::Model::Regression::Linear< _TReal, _TNatural >::
Linear( const TNatural& n )
  : Superclass( n + 1 )
{
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
PUJ_ML::Model::Regression::Linear< _TReal, _TNatural >::
~Linear( )
{
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
const std::string& PUJ_ML::Model::Regression::Linear< _TReal, _TNatural >::
cost_type( ) const
{
  static const std::string t = "mse";
  return( t );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
typename PUJ_ML::Model::Regression::Linear< _TReal, _TNatural >::
TNatural PUJ_ML::Model::Regression::Linear< _TReal, _TNatural >::
input_size( ) const
{
  return( this->m_S - 1 );
}

// -------------------------------------------------------------------------
namespace PUJ_ML { namespace Model { namespace Regression {
  PUJ_ML_Model_Instance( Linear );
} } }
  
// eof - $RCSfile$
