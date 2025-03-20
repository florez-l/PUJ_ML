// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__Base__h__
#define __PUJ_ML__Model__Base__h__

#include <PUJ_ML/Config.h>


#include <cstring>

#include <PUJ_ML/IO/Base64.h>

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
      Base( const TNatural& n = 0 )
        {
          this->_allocate( n );
        }
      ~Base( )
        {
          this->_allocate( 0 );
        }

      std::string encode64( ) const
        {
          std::string e = PUJ_ML::IO::Base64::encode( ( unsigned char )( sizeof( TReal ) ) ) + PUJ_ML::IO::Base64::encode( ( unsigned char )( sizeof( TNatural ) ) );
          e += PUJ_ML::IO::Base64::encode( this->m_S );
          for( TReal* p = this->m_P; p != this->m_P + this->m_S; ++p )
            e += PUJ_ML::IO::Base64::encode( p );
          return( e );
        }

    protected:
      virtual void _allocate( const TNatural& n )
        {
          if( this->m_P != nullptr )
            std::free( this->m_P );
          this->m_P = nullptr;
          this->m_S = n;
          if( this->m_S > 0 )
            this->m_P = reinterpret_cast< TReal* >( std::calloc( this->m_S, sizeof( TReal ) ) );
          if( this->m_P != nullptr )
          {
            for( TReal* p = this->m_P; p != this->m_P + this->m_S; ++p )
              *p = TReal( 0 );
          }
          else
          {
            this->m_P = nullptr;
            this->m_S = 0;
          } // end if
        }

      virtual void _to_stream( std::ostream& o ) const
        {
          o << this->m_S;
          for( TReal* p = this->m_P; p != this->m_P + this->m_S; ++p )
            o << " " << *p;
        }

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

#endif // __PUJ_ML__Model__Base__h__

// eof - $RCSfile$

