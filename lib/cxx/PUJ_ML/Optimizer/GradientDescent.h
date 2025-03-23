// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Optimizer__GradientDescent__h__
#define __PUJ_ML__Optimizer__GradientDescent__h__

#include <PUJ_ML/Optimizer/Base.h>

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
        );
      virtual ~GradientDescent( ) override;

    protected:
      virtual void _fit( TModel& model, const TBatches& batches ) override;
    };
  } // end namespace
} // end namespace

#include <PUJ_ML/Optimizer/GradientDescent.hxx>

#endif // __PUJ_ML__Optimizer__GradientDescent__h__

// eof - $RCSfile$
