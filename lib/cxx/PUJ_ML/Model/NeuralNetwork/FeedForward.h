// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__NeuralNetwork__FeedForward__h__
#define __PUJ_ML__Model__NeuralNetwork__FeedForward__h__

#include <PUJ_ML/Model/Base.h>
#include <PUJ_ML/Model/NeuralNetwork/Activations.h>
#include <vector>

namespace PUJ_ML
{
  namespace Model
  {
    namespace NeuralNetwork
    {
      /**
       */
      template< class _TReal, class _TNatural = unsigned long long >
      class FeedForward
        : public PUJ_ML::Model::Base< _TReal, _TNatural >
      {
      public:
        using Self       = FeedForward;
        using Superclass = PUJ_ML::Model::Base< _TReal, _TNatural >;

        using TReal    = typename Superclass::TReal;
        using TNatural = typename Superclass::TNatural;
        using TMatrix  = typename Superclass::TMatrix;
        using TColumn  = typename Superclass::TColumn;
        using TRow     = typename Superclass::TRow;

        using TMatrixMap = Eigen::Map< TMatrix >;
        using TColumnMap = Eigen::Map< TColumn >;
        using TRowMap    = Eigen::Map< TRow >;

        using TActivations
        =
          PUJ_ML::Model::NeuralNetwork::Activations< TReal >;
        using TActivationPair = typename TActivations::TPair;
        using TActivationFunction = typename TActivations::TFunction;

      public:
        FeedForward( );
        virtual ~FeedForward( ) override;

        virtual TNatural input_size( ) const override;

        virtual void prepare_auxiliary_buffer(
          const TNatural& M
          ) const override;
        virtual void free_auxiliary_buffer( ) const override;

        bool load( const std::string& fname );

        void set_input_layer(
          const TNatural& i, const TNatural& o, const std::string& a
          );
        void add_layer( const TNatural& o, const std::string& a );
        TNatural number_of_layers( ) const;

        virtual void init( );

        template< class _TX >
        auto operator()( const Eigen::EigenBase< _TX >& X ) const;

        template< class _TX >
        auto threshold( const Eigen::EigenBase< _TX >& X ) const;

        template< class _TX, class _Ty >
        TReal cost(
          const Eigen::EigenBase< _TX >& X, const Eigen::EigenBase< _Ty >& Y
          ) const;

        template< class _TX, class _Ty >
        TReal cost_gradient(
          TReal* bufferG,
          const Eigen::EigenBase< _TX >& bX,
          const Eigen::EigenBase< _Ty >& bY,
          const TReal& L1, const TReal& L2
          ) const;

      protected:
        void _prepare_buffers(
          TReal** bufferA, TReal** bufferZ,
          std::vector< TMatrixMap >* A, std::vector< TMatrixMap >* Z,
          const TNatural& M,
          bool keep_AZ = false
          ) const;

        void _free_buffers( TReal** bA, TReal** bZ ) const;
        void _eval(
          std::vector< TMatrixMap >& A, std::vector< TMatrixMap >& Z
          ) const;

        template< class _TA, class _TY >
        TReal _cost(
          const Eigen::EigenBase< _TA >& bA,
          const Eigen::EigenBase< _TY >& bY
          ) const;

      protected:
        std::vector< TNatural >    m_N;
        std::vector< TMatrixMap >  m_W;
        std::vector< TRowMap >     m_B;
        std::vector< TActivationPair > m_A;

        mutable TReal* m_BufferA { nullptr };
        mutable TReal* m_BufferZ { nullptr };

      private:
        static inline std::string lower( const std::string& s );
      };
    } // end namespace
  } // end namespace
} // end namespace

#include <PUJ_ML/Model/NeuralNetwork/FeedForward.hxx>

#endif // __PUJ_ML__Model__NeuralNetwork__FeedForward__h__

// eof - $RCSfile$
