// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Optimizer__Base__h__
#define __PUJ_ML__Optimizer__Base__h__

#include <PUJ_ML/Config.h>
#include <functional>



#include <algorithm>
#include <random>
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
        std::function<
          bool( const TNatural&, const TReal&, const TReal&, const TReal& )
          >;

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
        )
        {
          this->m_Xtr = &X_tr;
          this->m_Ytr = &Y_tr;
          this->m_Xte = &X_te;
          this->m_Yte = &Y_te;

          // TODO: check sizes
        }
      virtual ~Base( )
        {
        }

      void setAlpha( const TReal& a )
        {
          this->m_Alpha = a;
        }
      void setL1( const TReal& l )
        {
          this->m_L1 = l;
        }
      void setL2( const TReal& l )
        {
          this->m_L2 = l;
        }
      void setBatchSize( const TNatural& bs )
        {
          this->m_BatchSize = bs;
        }
      void setValidationToNormal( )
        {
          this->m_Validation = Self::Normal;
        }
      void setValidationToLeaveOneOut( )
        {
          this->m_Validation = Self::LeaveOneOut;
        }
      void setValidationToKfold( const TNatural& K )
        {
          this->m_Validation = Self::Kfold;
          this->m_K = K;
        }
      void setNumberOfMaximumIterations( const TNatural& i )
        {
          this->m_NumberOfMaximumIterations = i;
        }

      void fit( TModel& model )
        {
          if( this->m_Validation == Self::Normal )
            this->_fit_normal( model );
          else if( this->m_Validation == Self::LeaveOneOut )
            this->_fit_loo( model );
          else if( this->m_Validation == Self::Kfold )
            this->_fit_kfold( model, this->m_K );
        }

    protected:
      void _fit_normal( TModel& model )
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

      void _fit_loo( TModel& model )
        {
        }

      void _fit_kfold( TModel& model, const TNatural& K )
        {
        }

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
            const TNatural& t,
            const TReal& nG,
            const TReal& Jtr, const TReal& Jte
            ) -> bool
          {
            std::cout << t << " " << nG << " " << Jtr << " " << Jte << std::endl;
            return( false );
          }
        };
    };
  } // end namespace
} // end namespace
#endif // __PUJ_ML__Optimizer__Base__h__

// eof - $RCSfile$
