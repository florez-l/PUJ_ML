// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Optimizer__Base__h__
#define __PUJ_ML__Optimizer__Base__h__

#include <PUJ_ML/Config.h>
#include <functional>
#include <vector>

namespace PUJ_ML
{
  namespace Optimizer
  {
    /**
     */
    template< class _TModel, class _TXTr, class _TYTr, class _TXTe, class _TYTe >
    class Base
    {
    public:
      using Self     = Base;
      using TModel   = _TModel;
      using TReal    = typename TModel::TReal;
      using TNatural = typename TModel::TNatural;
      using TMatrix  = typename TModel::TMatrix;
      using TColumn  = typename TModel::TColumn;
      using TRow     = typename TModel::TRow;

      using TIndices = std::vector< Eigen::Index >;
      using TBatches = std::vector< TIndices >;

      using TDebug
      =
        std::function< bool( const TNatural&, const TReal&, const TReal& ) >;

      enum EValidation
      {
        Normal = 0,
        LeaveOneOut,
        Kfold
      };

    public:
      Base(
        const Eigen::EigenBase< _TXTr >& X_tr,
        const Eigen::EigenBase< _TYTr >& Y_tr,
        const Eigen::EigenBase< _TXTe >& X_te,
        const Eigen::EigenBase< _TYTe >& Y_te
        );
      virtual ~Base( );

      void setAlpha( const TReal& a );
      void setL1( const TReal& l );
      void setL2( const TReal& l );
      void setBatchSize( const TNatural& bs );
      void setValidationToNormal( );
      void setValidationToLeaveOneOut( );
      void setValidationToKfold( const TNatural& K );
      void setNumberOfMaximumIterations( const TNatural& i );
      void setDebug( TDebug d );

      void fit( TModel& model );

    protected:
      void _fit_normal( TModel& model );
      void _fit_loo( TModel& model );
      void _fit_kfold( TModel& model, const TNatural& K );

      virtual void _fit( TModel& model, const TBatches& batches ) = 0;

    protected:
      const Eigen::EigenBase< _TXTr >* m_Xtr;
      const Eigen::EigenBase< _TYTr >* m_Ytr;
      const Eigen::EigenBase< _TXTe >* m_Xte;
      const Eigen::EigenBase< _TYTe >* m_Yte;

      TReal    m_Epsilon { std::numeric_limits< TReal >::epsilon( ) };
      TReal    m_Alpha { 1e-2 };
      TReal    m_L1 { 0 };
      TReal    m_L2 { 0 };
      TNatural m_BatchSize { 0 };
      TNatural m_NumberOfMaximumIterations { 1000 };

      TNatural m_K { 1 };
      Self::EValidation m_Validation { Self::Normal };

      TDebug m_Debug
        {
          [](
            const TNatural&, const TReal&, const TReal&
            ) -> bool { return( false ); }
        };
    };
  } // end namespace
} // end namespace

#include <PUJ_ML/Optimizer/Base.hxx>

#endif // __PUJ_ML__Optimizer__Base__h__

// eof - $RCSfile$
