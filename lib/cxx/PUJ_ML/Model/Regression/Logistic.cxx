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
PUJ_ML::Model::Regression::Logistic< _TReal, _TNatural >::SVisitor::
SVisitor( const TColumn& Z )
{
  this->Z = &Z;
  this->E =
    std::pow(
      TReal( 10 ),
      std::log10( std::numeric_limits< TReal >::epsilon( ) ) * TReal( 0.5 )
      );
  this->D = std::log( this->E );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void PUJ_ML::Model::Regression::Logistic< _TReal, _TNatural >::SVisitor::
init( const TReal& y, const TIdx& i, const TIdx& j )
{
  this->J = TReal( 0 );
  this->operator()( y, i, j );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void PUJ_ML::Model::Regression::Logistic< _TReal, _TNatural >::SVisitor::
operator()( const TReal& y, const TIdx& i, const TIdx& j )
{
  TReal lv = this->Z->operator()( i );
  if( y == TReal( 0 ) )
    lv = TReal( 1 ) - lv;
  this->J -= ( this->E < lv )? std::log( lv ): this->D;
}

// -------------------------------------------------------------------------
namespace PUJ_ML
{
  namespace Model
  {
    namespace Regression
    {
      template class PUJ_ML_EXPORT Logistic< float, unsigned int >;
      template class PUJ_ML_EXPORT Logistic< float, unsigned long >;
      template class PUJ_ML_EXPORT Logistic< float, unsigned long long >;

      template class PUJ_ML_EXPORT Logistic< double, unsigned int >;
      template class PUJ_ML_EXPORT Logistic< double, unsigned long >;
      template class PUJ_ML_EXPORT Logistic< double, unsigned long long >;

      template class PUJ_ML_EXPORT Logistic< long double, unsigned int >;
      template class PUJ_ML_EXPORT Logistic< long double, unsigned long >;
      template class PUJ_ML_EXPORT Logistic< long double, unsigned long long >;
    } // end namespace
  } // end namespace
} // end namespace

// eof - $RCSfile$
