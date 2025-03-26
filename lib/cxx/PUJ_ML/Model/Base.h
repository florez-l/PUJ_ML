// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__Base__h__
#define __PUJ_ML__Model__Base__h__

#include <PUJ_ML/Config.h>

// -------------------------------------------------------------------------
#define PUJ_ML_Model_Instance( model_name )                             \
  template class PUJ_ML_EXPORT model_name< float, unsigned int >;       \
  template class PUJ_ML_EXPORT model_name< float, unsigned long >;      \
  template class PUJ_ML_EXPORT model_name< float, unsigned long long >; \
  template class PUJ_ML_EXPORT model_name< double, unsigned int >;      \
  template class PUJ_ML_EXPORT model_name< double, unsigned long >;     \
  template class PUJ_ML_EXPORT model_name< double, unsigned long long >; \
  template class PUJ_ML_EXPORT model_name< long double, unsigned int >; \
  template class PUJ_ML_EXPORT model_name< long double, unsigned long >; \
  template class PUJ_ML_EXPORT model_name< long double, unsigned long long >


namespace PUJ_ML
{
  namespace Model
  {
    /**
     */
    template< class _TReal, class _TNatural = unsigned long long >
    class Base
    {
    public:
      using TReal    = _TReal;
      using TNatural = _TNatural;
      using Self     = Base;

      using TMatrix = Eigen::Matrix< TReal, Eigen::Dynamic, Eigen::Dynamic >;
      using TColumn = Eigen::Matrix< TReal, Eigen::Dynamic, 1 >;
      using TRow    = Eigen::Matrix< TReal, 1, Eigen::Dynamic >;

    public:
      Base( const TNatural& n = 0 );
      virtual ~Base( );

      virtual const std::string& cost_type( ) const = 0;

      TNatural size( ) const;
      virtual TNatural input_size( ) const;

      virtual void prepare_auxiliary_buffer( const TNatural& M ) const;
      virtual void free_auxiliary_buffer( ) const;

      template< class _Tw >
      Self& operator+=( const Eigen::EigenBase< _Tw >& w );

      template< class _Tw >
      Self& operator-=( const Eigen::EigenBase< _Tw >& w );

      std::string encode64( ) const;

    protected:
      virtual void _allocate( const TNatural& n );
      virtual void _to_stream( std::ostream& o ) const;
      virtual TReal _regularize(
        TReal* G, const TReal& L1, const TReal& L2
        ) const;

      template< class _TA, class _TY >
      TReal _cost(
        const Eigen::EigenBase< _TA >& bA,
        const Eigen::EigenBase< _TY >& bY
        ) const;

    protected:
      TReal*   m_P { nullptr };
      TNatural m_S { 0 };

    public:
      friend std::ostream& operator<<( std::ostream& o, const Self& m )
        {
          m._to_stream( o );
          return( o );
        }
    };
  } // end namespace
} // end namespace

#include <PUJ_ML/Model/Base.hxx>

#endif // __PUJ_ML__Model__Base__h__

// eof - $RCSfile$

