// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Helpers__FitModel__h__
#define __PUJ_ML__Helpers__FitModel__h__

#include <PUJ_ML/Config.h>

namespace PUJ_ML
{
  namespace Helpers
  {
    /**
     */
    template< class _TModel, class _TArgs, class _TTr, class _TTe >
    void FitModel(
      _TModel& model, const _TArgs& args,
      const Eigen::EigenBase< _TTr >& bD_tr,
      const Eigen::EigenBase< _TTe >& bD_te
      )
    {
#error aca voy
    }
  } // end namespace
} // end namespace

#endif // __PUJ_ML__Helpers__FitModel__h__

// eof - $RCSfile$
