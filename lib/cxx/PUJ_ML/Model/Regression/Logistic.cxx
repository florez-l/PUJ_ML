// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <PUJ_ML/Model/Regression/Logistic.h>

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
PUJ_ML::Model::Regression::Logistic< _TReal, _TNatural >::
Logistic( const TNatural& n )
  : Superclass( n )
{
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
PUJ_ML::Model::Regression::Logistic< _TReal, _TNatural >::
~Logistic( )
{
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
const std::string& PUJ_ML::Model::Regression::Logistic< _TReal, _TNatural >::
cost_type( ) const
{
  static const std::string t = "bce";
  return( t );
}

// -------------------------------------------------------------------------
namespace PUJ_ML { namespace Model { namespace Regression {
  PUJ_ML_Model_Instance( Logistic );
} } }

// eof - $RCSfile$
