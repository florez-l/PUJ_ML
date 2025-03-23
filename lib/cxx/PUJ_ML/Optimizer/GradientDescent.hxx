// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Optimizer__GradientDescent__hxx__
#define __PUJ_ML__Optimizer__GradientDescent__hxx__

#include <cmath>

// -------------------------------------------------------------------------
template< class _TModel, class _TXTr, class _TYTr, class _TXTe, class _TYTe >
PUJ_ML::Optimizer::GradientDescent< _TModel, _TXTr, _TYTr, _TXTe, _TYTe >::
GradientDescent(
  const Eigen::EigenBase< _TXTr >& X_tr,
  const Eigen::EigenBase< _TYTr >& Y_tr,
  const Eigen::EigenBase< _TXTe >& X_te,
  const Eigen::EigenBase< _TYTe >& Y_te
  )
    : Superclass( X_tr, Y_tr, X_te, Y_te )
{
}

// -------------------------------------------------------------------------
template< class _TModel, class _TXTr, class _TYTr, class _TXTe, class _TYTe >
PUJ_ML::Optimizer::GradientDescent< _TModel, _TXTr, _TYTr, _TXTe, _TYTe >::
~GradientDescent( )
{
}

// -------------------------------------------------------------------------
template< class _TModel, class _TXTr, class _TYTr, class _TXTe, class _TYTe >
void PUJ_ML::Optimizer::GradientDescent< _TModel, _TXTr, _TYTr, _TXTe, _TYTe >::
_fit( TModel& model, const TBatches& batches )
{
  auto Xtr = this->m_Xtr->derived( ).template cast< TReal >( );
  auto Ytr = this->m_Ytr->derived( ).template cast< TReal >( );
  auto Xte = this->m_Xte->derived( ).template cast< TReal >( );
  auto Yte = this->m_Yte->derived( ).template cast< TReal >( );

  TNatural t = 0;
  TReal J_tr = 0, J_te = 0;
  bool stop = false;
  TColumn G( model.size( ) ), sG( model.size( ) );
  while( !stop )
  {
    t++;

    sG.fill( 0 );
    typename TBatches::const_iterator bIt = batches.begin( );
    while( bIt !=  batches.end( ) && !stop )
    {
      J_tr = model.cost_gradient( G, Xtr( *bIt, Eigen::all ), Ytr( *bIt, Eigen::all ), this->m_L1, this->m_L2 );
      if( !std::isnan( J_tr ) && !std::isinf( J_tr ) )
      {
        sG += G;
        model -= G * this->m_Alpha;
      }
      else
        stop = true;
      bIt++;
    } // end while
    if( !stop )
    {
      J_te = ( 0 < Xte.rows( ) )? model.cost( Xte, Yte ): 0;
      stop = this->m_Debug( t, std::sqrt( sG.array( ).pow( 2 ).sum( ) ), J_tr, J_te );
      stop |= ( t >= this->m_NumberOfMaximumIterations );
    }
    else
      stop = true;
  } // end while
  this->m_Debug( t, std::sqrt( sG.array( ).pow( 2 ).sum( ) ), J_tr, J_te );
}

#endif // __PUJ_ML__Optimizer__GradientDescent__hxx__

// eof - $RCSfile$
