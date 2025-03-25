// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__NeuralNetwork__Activations__h__
#define __PUJ_ML__Model__NeuralNetwork__Activations__h__

#include <PUJ_ML/Config.h>
#include <functional>
#include <string>
#include <utility>

namespace PUJ_ML
{
  namespace Model
  {
    namespace NeuralNetwork
    {
      /**
       */
      template< class _TReal >
      class Activations
      {
      public:
        using Self       = Activations;
        using TReal      = _TReal;
        using TMatrix    = Eigen::Matrix< TReal, Eigen::Dynamic, Eigen::Dynamic >;
        using TMatrixMap = Eigen::Map< TMatrix >;

        using TFunction = std::function< void( TMatrixMap&, const TMatrixMap&, bool ) >;
        using TPair = std::pair< std::string, TFunction >;

      public:
        static TPair Get( const std::string& name );
      };
    } // end namespace
  } // end namespace
} // end namespace

#endif // __PUJ_ML__Model__NeuralNetwork__Activations__h__

// eof - $RCSfile$
