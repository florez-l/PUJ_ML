// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Optimizer__Adam__h__
#define __PUJ_ML__Optimizer__Adam__h__

#include <PUJ_ML/Optimizer/Base.h>


#include <cmath>



namespace PUJ_ML
{
  namespace Optimizer
  {
    /**
     */
    template< class _TModel, class _TXTr, class _TYTr, class _TXTe, class _TYTe >
    class Adam
      : public PUJ_ML::Optimizer::Base< _TModel, _TXTr, _TYTr, _TXTe, _TYTe >
    {
    public:
      using Self = Adam;
      using Superclass
      =
        PUJ_ML::Optimizer::Base< _TModel, _TXTr, _TYTr, _TXTe, _TYTe >;
      using TModel   = typename Superclass::TModel;
      using TReal    = typename Superclass::TReal;
      using TNatural = typename Superclass::TNatural;
      using TMatrix  = typename Superclass::TMatrix;
      using TColumn  = typename Superclass::TColumn;
      using TRow     = typename Superclass::TRow;
      using TIndices = typename Superclass::TIndices;
      using TBatches = typename Superclass::TBatches;

    public:
      Adam(
        const Eigen::EigenBase< _TXTr >& X_tr,
        const Eigen::EigenBase< _TYTr >& Y_tr,
        const Eigen::EigenBase< _TXTe >& X_te,
        const Eigen::EigenBase< _TYTe >& Y_te
        )
        : Superclass( X_tr, Y_tr, X_te, Y_te )
        {
        }
      virtual ~Adam( ) override
        {
        }

    protected:
      virtual void _fit( TModel& model, const TBatches& batches ) override
        {
          static const TReal _0 = TReal( 0 );
          static const TReal _1 = TReal( 1 );

          auto Xtr = this->m_Xtr->derived( ).template cast< TReal >( );
          auto Ytr = this->m_Ytr->derived( ).template cast< TReal >( );
          auto Xte = this->m_Xte->derived( ).template cast< TReal >( );
          auto Yte = this->m_Yte->derived( ).template cast< TReal >( );

          TReal e = this->m_Epsilon;
          TReal b1 = this->m_Beta1;
          TReal b2 = this->m_Beta2;
          TReal cb1 = _1 - b1;
          TReal cb2 = _1 - b2;
          TReal b1t = b1;
          TReal b2t = b2;
          TNatural t = 0;
		  TReal J_tr, J_te;
          bool stop = false;
          TColumn G( model.size( ) ), sG( model.size( ) );
          TColumn mt = TColumn::Zero( G.size( ) );
          TColumn vt = TColumn::Zero( G.size( ) );

          while( !stop )
          {
            t++;

			TReal i1 = _1 / ( _1 - b1t );
            TReal i2 = _1 / ( _1 - b2t );
            sG.fill( 0 );
			typename TBatches::const_iterator bIt = batches.begin( );
            while( bIt !=  batches.end( ) && !stop )
            {
              J_tr = model.cost_gradient( G, Xtr( *bIt, Eigen::all ), Ytr( *bIt, Eigen::all ), this->m_L1, this->m_L2 );
              if( !std::isnan( J_tr ) && !std::isinf( J_tr ) )
			  {
				sG += G;
                mt = ( mt * b1 ) + ( sG * cb1 );
                vt = ( vt * b2 ) + ( sG.array( ).pow( 2 ) * cb2 ).matrix( );
                model -= ( ( mt * i1 ).array( ) / ( ( ( vt * i2 ).array( ) ).sqrt( ) + e ) ).matrix( ) * this->m_Alpha;
			  }
			  else
				stop = true;
			  bIt++;
            } // end for
            if( !stop )
            {
              J_te = ( 0 < Xte.rows( ) )? model.cost( Xte, Yte ): 0;
              stop = this->m_Debug( t, std::sqrt( sG.array( ).pow( 2 ).sum( ) ), J_tr, J_te );
              stop |= ( t >= this->m_NumberOfMaximumIterations );
            }
            else
              stop = true;

            b1t *= b1;
            b2t *= b2;
          } // end while
		  this->m_Debug( t, std::sqrt( sG.array( ).pow( 2 ).sum( ) ), J_tr, J_te );
        }

    protected:
      TReal m_Beta1 { 0.9 };
      TReal m_Beta2 { 0.999 };
    };
  } // end namespace
} // end namespace

#endif // __PUJ_ML__Optimizer__Adam__h__

// eof - $RCSfile$
