// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Optimizer__Base__hxx__
#define __PUJ_ML__Optimizer__Base__hxx____

#include <algorithm>
#include <random>

// -------------------------------------------------------------------------
template< class _TModel, class _TXTr, class _TYTr, class _TXTe, class _TYTe >
PUJ_ML::Optimizer::Base< _TModel, _TXTr, _TYTr, _TXTe, _TYTe >::
Base(
  const Eigen::EigenBase< _TXTr >& X_tr,
  const Eigen::EigenBase< _TYTr >& Y_tr,
  const Eigen::EigenBase< _TXTe >& X_te,
  const Eigen::EigenBase< _TYTe >& Y_te
  )
{
  this->m_Xtr = &X_tr;
  this->m_Ytr = &Y_tr;
  this->m_Xte = &X_te;
  this->m_Yte = &Y_te;

  // TODO: check sizes
}

// -------------------------------------------------------------------------
template< class _TModel, class _TXTr, class _TYTr, class _TXTe, class _TYTe >
PUJ_ML::Optimizer::Base< _TModel, _TXTr, _TYTr, _TXTe, _TYTe >::
~Base( )
{
}

// -------------------------------------------------------------------------
template< class _TModel, class _TXTr, class _TYTr, class _TXTe, class _TYTe >
void PUJ_ML::Optimizer::Base< _TModel, _TXTr, _TYTr, _TXTe, _TYTe >::
setAlpha( const TReal& a )
{
  this->m_Alpha = a;
}

// -------------------------------------------------------------------------
template< class _TModel, class _TXTr, class _TYTr, class _TXTe, class _TYTe >
void PUJ_ML::Optimizer::Base< _TModel, _TXTr, _TYTr, _TXTe, _TYTe >::
setL1( const TReal& l )
{
  this->m_L1 = l;
}

// -------------------------------------------------------------------------
template< class _TModel, class _TXTr, class _TYTr, class _TXTe, class _TYTe >
void PUJ_ML::Optimizer::Base< _TModel, _TXTr, _TYTr, _TXTe, _TYTe >::
setL2( const TReal& l )
{
  this->m_L2 = l;
}

// -------------------------------------------------------------------------
template< class _TModel, class _TXTr, class _TYTr, class _TXTe, class _TYTe >
void PUJ_ML::Optimizer::Base< _TModel, _TXTr, _TYTr, _TXTe, _TYTe >::
setBatchSize( const TNatural& bs )
{
  this->m_BatchSize = bs;
}

// -------------------------------------------------------------------------
template< class _TModel, class _TXTr, class _TYTr, class _TXTe, class _TYTe >
void PUJ_ML::Optimizer::Base< _TModel, _TXTr, _TYTr, _TXTe, _TYTe >::
setValidationToNormal( )
{
  this->m_Validation = Self::Normal;
}

// -------------------------------------------------------------------------
template< class _TModel, class _TXTr, class _TYTr, class _TXTe, class _TYTe >
void PUJ_ML::Optimizer::Base< _TModel, _TXTr, _TYTr, _TXTe, _TYTe >::
setValidationToLeaveOneOut( )
{
  this->m_Validation = Self::LeaveOneOut;
}

// -------------------------------------------------------------------------
template< class _TModel, class _TXTr, class _TYTr, class _TXTe, class _TYTe >
void PUJ_ML::Optimizer::Base< _TModel, _TXTr, _TYTr, _TXTe, _TYTe >::
setValidationToKfold( const TNatural& K )
{
  this->m_Validation = Self::Kfold;
  this->m_K = K;
}

// -------------------------------------------------------------------------
template< class _TModel, class _TXTr, class _TYTr, class _TXTe, class _TYTe >
void PUJ_ML::Optimizer::Base< _TModel, _TXTr, _TYTr, _TXTe, _TYTe >::
setNumberOfMaximumIterations( const TNatural& i )
{
  this->m_NumberOfMaximumIterations = i;
}

// -------------------------------------------------------------------------
template< class _TModel, class _TXTr, class _TYTr, class _TXTe, class _TYTe >
void PUJ_ML::Optimizer::Base< _TModel, _TXTr, _TYTr, _TXTe, _TYTe >::
fit( TModel& model )
{
  if( this->m_Validation == Self::Normal )
    this->_fit_normal( model );
  else if( this->m_Validation == Self::LeaveOneOut )
    this->_fit_loo( model );
  else if( this->m_Validation == Self::Kfold )
    this->_fit_kfold( model, this->m_K );
}

// -------------------------------------------------------------------------
template< class _TModel, class _TXTr, class _TYTr, class _TXTe, class _TYTe >
void PUJ_ML::Optimizer::Base< _TModel, _TXTr, _TYTr, _TXTe, _TYTe >::
_fit_normal( TModel& model )
{
  TNatural M = this->m_Xtr->rows( );
  std::vector< Eigen::Index > idx( M );
  idx.shrink_to_fit( );
  std::iota( idx.begin( ), idx.end( ), 0 );
  std::random_device dev;
  std::mt19937 gen( dev( ) );
  std::shuffle( idx.begin( ), idx.end( ), gen );

  TNatural bs = this->m_BatchSize;
  if( bs == 0 || bs > M )
    bs = M;
  TBatches batches;
  for( TNatural b = 0; b < M; b += bs )
  {
    TNatural e = b + bs;
    if( !( e < M ) )
      e = M;
    batches.push_back( TIndices( idx.begin( ) + b, idx.begin( ) + e ) );
    batches.back( ).shrink_to_fit( );
  } // end for
  batches.shrink_to_fit( );

  this->_fit( model, batches );
}

// -------------------------------------------------------------------------
template< class _TModel, class _TXTr, class _TYTr, class _TXTe, class _TYTe >
void PUJ_ML::Optimizer::Base< _TModel, _TXTr, _TYTr, _TXTe, _TYTe >::
_fit_loo( TModel& model )
{
}

// -------------------------------------------------------------------------
template< class _TModel, class _TXTr, class _TYTr, class _TXTe, class _TYTe >
void PUJ_ML::Optimizer::Base< _TModel, _TXTr, _TYTr, _TXTe, _TYTe >::
_fit_kfold( TModel& model, const TNatural& K )
{
}

#endif // __PUJ_ML__Optimizer__Base__hxx__

// eof - $RCSfile$
