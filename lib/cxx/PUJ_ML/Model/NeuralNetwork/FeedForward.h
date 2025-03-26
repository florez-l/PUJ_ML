// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__NeuralNetwork__FeedForward__h__
#define __PUJ_ML__Model__NeuralNetwork__FeedForward__h__

#include <PUJ_ML/Model/Base.h>
#include <PUJ_ML/Model/NeuralNetwork/Activations.h>
#include <vector>






#include <algorithm>
#include <cctype>
#include <numeric>
#include <string>

#include <fstream>
#include <sstream>
#include <random>

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

        using TActivations = PUJ_ML::Model::NeuralNetwork::Activations< TReal >;
        using TActivationPair = typename TActivations::TPair;
        using TActivationFunction = typename TActivations::TFunction;

      public:
        FeedForward( )
          :Superclass( 0 )
          {
          }
        virtual ~FeedForward( ) override
          {
            this->free_auxiliary_buffer( );
          }

        virtual TNatural input_size( ) const override
          {
            return( this->m_N[ 0 ] );
          }

        virtual void prepare_auxiliary_buffer( const TNatural& M ) const override
          {
            this->free_auxiliary_buffer( );
            TNatural NA = std::accumulate( this->m_N.begin( ), this->m_N.end( ), 0 );
            TNatural NZ = NA - this->m_N[ 0 ];

            this->m_BufferA = reinterpret_cast< TReal* >( std::calloc( NA * M, sizeof( TReal ) ) );
            this->m_BufferZ = reinterpret_cast< TReal* >( std::calloc( NZ * M, sizeof( TReal ) ) );
            if( this->m_BufferA == nullptr || this->m_BufferZ == nullptr )
              this->free_auxiliary_buffer( );
          }

        virtual void free_auxiliary_buffer( ) const override
          {
            if( this->m_BufferA != nullptr )
              std::free( this->m_BufferA );
            if( this->m_BufferZ != nullptr )
              std::free( this->m_BufferZ );
            this->m_BufferA = nullptr;
            this->m_BufferZ = nullptr;
          }

        bool load( const std::string& fname )
          {
            // Load buffer
            std::ifstream ifs( fname.c_str( ) );
            if( !ifs )
              return( false );
            ifs.seekg( 0, std::ios::end );
            std::size_t size = ifs.tellg( );
            ifs.seekg( 0, std::ios::beg );
            std::string buffer( size, 0 );
            ifs.read( &buffer[ 0 ], size );
            ifs.close( );
            std::istringstream input( buffer );

            // Input layer
            std::string a;
            TNatural i, o;
            input >> i >> o >> a;
            this->set_input_layer( i, o, a );

            // Remaining layers
            input >> o;
            while( o != 0 )
            {
              input >> a;
              this->add_layer( o, a );
              input >> o;
            } // end while

            // Check remaining data
            input >> a;
            if( Self::lower( a ) == "random" )
              this->init( );


            return( true );
          }

        void set_input_layer( const TNatural& i, const TNatural& o, const std::string& a )
          {
            this->m_N.clear( );
            this->m_W.clear( );
            this->m_B.clear( );
            this->m_A.clear( );

            this->m_N.push_back( i );
            this->m_N.push_back( o );
            this->m_A.push_back( TActivations::Get( a ) );
          }
        void add_layer( const TNatural& o, const std::string& a )
          {
            this->m_N.push_back( o );
            this->m_A.push_back( TActivations::Get( a ) );
          }

        TNatural number_of_layers( ) const
          {
            return( this->m_N.size( ) - 1 );
          }

        virtual void init( )
          {
            // Reserve memory
            TNatural N = 0;
            TNatural L = this->number_of_layers( );
            for( TNatural l = 1; l <= L; ++l )
              N += ( this->m_N[ l - 1 ] + 1 ) * this->m_N[ l ];
            this->_allocate( N );

            // Fill with random numbers
            std::random_device rd;
            std::mt19937 rg( rd( ) );
            std::uniform_real_distribution< TReal > rdis(
              std::numeric_limits< TReal >::epsilon( ),
              TReal( 1 )
              );
            std::generate(
              this->m_P, this->m_P + this->m_S,
              [&]( ) -> TReal
              {
                return( ( TReal( 2 ) * rdis( rg ) ) - TReal( 1 ) );
              }
              );

            // Create maps
            this->m_W.clear( );
            this->m_B.clear( );

            TReal* m = this->m_P;
            for( TNatural l = 1; l <= L; ++l )
            {
              this->m_W.push_back( TMatrixMap( m, this->m_N[ l - 1 ], this->m_N[ l ] ) );
              m += this->m_W.back( ).size( );
              this->m_B.push_back( TRowMap( m, 1, this->m_N[ l ] ) );
              m += this->m_B.back( ).size( );
            } // end for
          }

        template< class _TX >
        auto operator()( const Eigen::EigenBase< _TX >& X ) const
          {
            TNatural nA = *( std::max_element( this->m_N.begin( ), this->m_N.end( ) ) );
            TNatural nZ = *( std::max_element( this->m_N.begin( ) + 1, this->m_N.end( ) ) );
            nA *= X.rows( );
            nZ *= X.rows( );
            TReal* buffer = reinterpret_cast< TReal* >( std::calloc( nA + nZ, sizeof( TReal ) ) );
            TReal* bA = buffer;
            TReal* bZ = bA + nA;

            TMatrixMap( bA, X.rows( ), X.cols( ) ) = X.derived( ).template cast< TReal >( );
            this->_eval( bA, bZ, X.rows( ), false );

            TMatrix A = TMatrixMap( bA, X.rows( ), this->m_N.back( ) );
            std::free( buffer );
            return( A );
          }

        template< class _TX >
        auto threshold( const Eigen::EigenBase< _TX >& X ) const
          {
            auto A = this->operator()( X );
            if( Self::lower( this->m_A.back( ).first ) == "sigmoid" )
            {
              return(
                A.unaryExpr(
                  []( const TReal& a ) -> TReal
                  {
                    return( ( a < TReal( 0.5 )? TReal( 0 ): TReal( 1 ) ) );
                  }
                  ).eval( )
                );
            }
            else if( Self::lower( this->m_A.back( ).first ) == "softmax" )
            {
              TMatrix T( A.rows( ), 1 );
              for( TNatural r = 0; r < A.rows( ); ++r )
                A.row( r ).maxCoeff( &( T( r, 0 ) ) );
              return( T );
            }
            else
              return( A );
          }

        template< class _TX, class _Ty >
        TReal cost(
          const Eigen::EigenBase< _TX >& bX,
          const Eigen::EigenBase< _Ty >& bY
          ) const
          {
            return( 0 );
          }

        template< class _TG, class _TX, class _Ty >
        TReal cost_gradient(
          Eigen::EigenBase< _TG >& G,
          const Eigen::EigenBase< _TX >& bX,
          const Eigen::EigenBase< _Ty >& bY,
          const TReal& L1, const TReal& L2
          ) const
          {
            auto X = bX.derived( ).template cast< TReal >( );
            auto Y = bY.derived( ).template cast< TReal >( );
            TNatural M = X.rows( );
            TNatural N = X.cols( );
            bool mem_owned = ( this->m_BufferA == nullptr || this->m_BufferZ == nullptr );
            if( mem_owned )
              this->prepare_auxiliary_buffer( M );

            /* TODO
               if( this->m_BufferA == nullptr || this->m_BufferZ == nullptr )
               throw error
            */

            // Forward propagation
            TMatrixMap( this->m_BufferA, M, N ) =  X;
            this->_eval( this->m_BufferA, this->m_BufferZ, M, true );

            // Some sizes
            TNatural NA = std::accumulate( this->m_N.begin( ), this->m_N.end( ), 0 );
            TNatural NZ = NA - this->m_N[ 0 ];

            // Backpropagate last layer
            TNatural L = this->number_of_layers( );
            TNatural oA = NA - ( this->m_N[ L ] * M );
            std::cout << "++++++++++++++++++++++++" << std::endl;
            std::cout << ( TMatrixMap( this->m_BufferA + oA, M, this->m_N[ L ] ) ) << std::endl;
            std::cout << "........................" << std::endl;
            std::cout << Y << std::endl;
            std::cout << "........................" << std::endl;
            std::cout << ( TMatrixMap( this->m_BufferA + oA, M, this->m_N[ L ] ) - Y ) << std::endl;
            std::cout << "++++++++++++++++++++++++" << std::endl;

            /* TODO
               m = float( 1 ) / float( X.shape[ 0 ] )
               DL = A[ L ] - Y
               i = self.m_B[ L - 2 ].size
               o = self.m_B[ L - 1 ].size
               k = self.size( ) - o
               G[ 0 , k : k + o ] = ( DL.sum( axis = 0 ) * m ).flatten( )
               k -= i * o
               G[ 0 , k : k + ( i * o ) ] = ( ( A[ L - 1 ].T @ DL ) * m ).flatten( )

               # Backpropagate remaining layers
               for l in range( L - 1, 0, -1 ):
               o = i
               i = self.m_W[ l - 1 ].shape[ 0 ]

               DL = numpy.multiply(
               ( DL @ self.m_W[ l ].T ),
               self.m_A[ l - 1 ][ 1 ]( Z[ l - 1 ], True )
               )
               k -= o
               G[ 0 , k : k + o ] = ( DL.sum( axis = 0 ) * m ).flatten( )
               k -= i * o
               G[ 0 , k : k + ( i * o ) ] = ( ( A[ l - 1 ].T @ DL ) * m ).flatten( )
               # end for

               # Cost (TODO: just MCE at the moment)
               zi = numpy.where( Y == 0 )[ 0 ].tolist( )
               oi = numpy.where( Y == 1 )[ 0 ].tolist( )

               J  = numpy.log( float( 1 ) - A[ -1 ][ zi , : ] + self.m_Epsilon ).sum( )
               J += numpy.log( A[ -1 ][ oi , : ] + self.m_Epsilon ).sum( )
               J /= float( X.shape[ 0 ] )
            */




            if( mem_owned )
              this->free_auxiliary_buffer( );

            return( 0 );
          }

      protected:
        void _eval( TReal* bA, TReal* bZ, const TNatural& M, bool keep_ZA ) const
        {
          TReal* A = bA;
          TReal* Z = bZ;
          TNatural L = this->number_of_layers( );
          for( TNatural l = 1; l <= L; ++l )
          {
            TMatrixMap( Z, M, this->m_N[ l ] ) = ( TMatrixMap( A, M, this->m_N[ l - 1 ] ) * this->m_W[ l - 1 ] ).rowwise( ) + this->m_B[ l - 1 ];
            A += ( keep_ZA )? ( this->m_N[ l - 1 ] * M ): 0;
            TMatrixMap mA( A, M, this->m_N[ l ] ), mZ( Z, M, this->m_N[ l ] );
            this->m_A[ l - 1 ].second( mA, mZ, false );
            Z += ( keep_ZA )? ( this->m_N[ l ] * M ): 0;
          } // end for
        }

      protected:
        std::vector< TNatural >    m_N;
        std::vector< TMatrixMap >  m_W;
        std::vector< TRowMap >     m_B;
        std::vector< TActivationPair > m_A;

        mutable TReal* m_BufferA { nullptr };
        mutable TReal* m_BufferZ { nullptr };

      private:
        static inline std::string lower( const std::string& s )
          {
            std::string r = s;
            std::transform(
              r.begin( ), r.end( ), r.begin( ),
              []( const unsigned char& c )
              {
                return( std::tolower( c ) );
              }
              );
            return( r );
          }
      };
    } // end namespace
  } // end namespace
} // end namespace

// TODO: #include <PUJ_ML/Model/NeuralNetwork/FeedForward.hxx>

#endif // __PUJ_ML__Model__NeuralNetwork__FeedForward__h__

// eof - $RCSfile$
