// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__Regression__Linear__h__
#define __PUJ_ML__Model__Regression__Linear__h__

#include <PUJ_ML/Model/Base.h>


#include <Eigen/Dense>

namespace PUJ_ML
{
  namespace Model
  {
    namespace Regression
    {
      /**
       */
      template< class _TReal, class _TNatural = unsigned long long >
      class Linear
        : public PUJ_ML::Model::Base< _TReal, _TNatural >
      {
      public:
        using Self       = Linear;
        using Superclass = PUJ_ML::Model::Base< _TReal, _TNatural >;

        using TReal    = typename Superclass::TReal;
        using TNatural = typename Superclass::TNatural;
        using TMatrix  = typename Superclass::TMatrix;
        using TColumn  = typename Superclass::TColumn;
        using TRow     = typename Superclass::TRow;

      public:
        Linear( const TNatural& n = 0 )
          : Superclass( n + 1 )
          {
          }
        virtual ~Linear( )
          {
          }

        template< class _TX >
        auto operator()( const Eigen::EigenBase< _TX >& X ) const
          {
            return( ( ( X.derived( ).template cast< TReal >( ) * Eigen::Map< TColumn >( this->m_P + 1, this->m_S - 1, 1 ) ).array( ) + this->m_P[ 0 ] ).matrix( ) );
          }

        template< class _TX, class _Ty >
        TReal cost(
          const Eigen::EigenBase< _TX >& bX,
          const Eigen::EigenBase< _Ty >& by
          ) const
          {
            auto X = bX.derived( ).template cast< TReal >( );
            auto y = by.derived( ).template cast< TReal >( ).col( 0 );

            /* TODO
               if( X.rows( ) != y.rows( ) )
               raise AssertionError( "Incompatible sizes." )
               # end if
            */
            return( ( this->operator()( X ) - y ).array( ).pow( 2 ).mean( ) );
          }

        /*
         * TODO: Use of L1 regularization is not yet solved
         */
        template< class _TX, class _Ty >
        void fit(
          const Eigen::EigenBase< _TX >& bX,
          const Eigen::EigenBase< _Ty >& by,
          const TReal& L1 = 0, const TReal& L2 = 0
          )
          {
            auto X = bX.derived( ).template cast< TReal >( );
            auto y = by.derived( ).template cast< TReal >( ).col( 0 ).array( );

            TNatural n = X.cols( );
            TNatural m = X.rows( );
            /* TODO
               if( n == 0 || m != y.rows( ) )
               raise AssertionError( "Incompatible sizes." )
               # end if
            */

            // Fill vector
            TRow b = TRow::Zero( n + 1 );
            b( 0 ) = y.mean( );
            b.block( 0, 1, 1, n ) = ( X.array( ).colwise( ) * y ).colwise( ).mean( );

            // Fill matrix
            TMatrix A = TMatrix::Zero( n + 1, n + 1 );
            A( 0 , 0 ) = TReal( 1 ) + L2;
            A.block( 1, 1, n, n ) = ( TMatrix::Identity( n, n ) * L2 ) + ( ( X.transpose( ) * X ) / TReal( m ) );
            A.block( 0, 1, 1, n ) = X.colwise( ).mean( );
            A.block( 1, 0, n, 1 ) = A.block( 0, 1, 1, n ).transpose( );

            // Solve system
            this->_allocate( n + 1 );
            Eigen::Map< TColumn >( this->m_P, n + 1, 1 ) = A.lu( ).solve( b.transpose( ) );
          }
      };
    } // end namespace
  } // end namespace
} // end namespace

#endif // __PUJ_ML__Model__Regression__Linear__h__

// eof - $RCSfile$
