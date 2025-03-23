// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Optimizer__GradientDescent__h__
#define __PUJ_ML__Optimizer__GradientDescent__h__

#include <PUJ_ML/Optimizer/Base.h>


#include <cmath>



namespace PUJ_ML
{
  namespace Optimizer
  {
    /**
     */
    template< class _TModel, class _TXTr, class _TYTr, class _TXTe, class _TYTe >
    class GradientDescent
      : public PUJ_ML::Optimizer::Base< _TModel, _TXTr, _TYTr, _TXTe, _TYTe >
    {
    public:
      using Self = GradientDescent;
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
      GradientDescent(
        const Eigen::EigenBase< _TXTr >& X_tr,
        const Eigen::EigenBase< _TYTr >& Y_tr,
        const Eigen::EigenBase< _TXTe >& X_te,
        const Eigen::EigenBase< _TYTe >& Y_te
        )
        : Superclass( X_tr, Y_tr, X_te, Y_te )
        {
        }
      virtual ~GradientDescent( ) override
        {
        }

    protected:
      virtual void _fit( TModel& model, const TBatches& batches ) override
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
            } // end for
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
    };
  } // end namespace
} // end namespace

#endif // __PUJ_ML__Optimizer__GradientDescent__h__

// eof - $RCSfile$
