// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#include <PUJ_ML/Model/NeuralNetwork/FeedForward.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <numeric>
#include <sstream>
#include <random>

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
PUJ_ML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
FeedForward( )
  :Superclass( 0 )
{
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
PUJ_ML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
~FeedForward( )
{
  this->free_auxiliary_buffer( );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
const std::string&
PUJ_ML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
cost_type( ) const
{
  static const std::string none = "none";
  static const std::string mse = "mse";
  static const std::string bce = "bce";
  static const std::string cce = "cce";

  if( this->m_A.size( ) > 0 )
  {
    if( Self::lower( this->m_A.back( ).first ) == "sigmoid" )
      return( bce );
    else if( Self::lower( this->m_A.back( ).first ) == "softmax" )
      return( cce );
    else
      return( mse );
  }
  else
    return( none );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
typename PUJ_ML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
TNatural PUJ_ML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
input_size( ) const
{
  return( this->m_N[ 0 ] );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void PUJ_ML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
prepare_auxiliary_buffer( const TNatural& M ) const
{
  this->free_auxiliary_buffer( );
  this->_prepare_buffers(
    &( this->m_BufferA ), &( this->m_BufferZ ), nullptr, nullptr, M
    );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void PUJ_ML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
free_auxiliary_buffer( ) const
{
  this->_free_buffers( &this->m_BufferA, &this->m_BufferZ );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
bool PUJ_ML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
load( const std::string& fname )
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

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void  PUJ_ML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
set_input_layer( const TNatural& i, const TNatural& o, const std::string& a )
{
  this->m_N.clear( );
  this->m_W.clear( );
  this->m_B.clear( );
  this->m_A.clear( );

  this->m_N.push_back( i );
  this->m_N.push_back( o );
  this->m_A.push_back( TActivations::Get( a ) );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void PUJ_ML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
add_layer( const TNatural& o, const std::string& a )
{
  this->m_N.push_back( o );
  this->m_A.push_back( TActivations::Get( a ) );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
typename PUJ_ML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
TNatural PUJ_ML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
number_of_layers( ) const
{
  return( this->m_N.size( ) - 1 );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void PUJ_ML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
init( )
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

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void PUJ_ML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
_prepare_buffers(
  TReal** bufferA, TReal** bufferZ,
  std::vector< TMatrixMap >* A, std::vector< TMatrixMap >* Z,
  const TNatural& M,
  bool keep_AZ
  ) const
{
  TNatural NA = 0, NZ = 0;
  if( keep_AZ )
  {
    NA = std::accumulate( this->m_N.begin( ), this->m_N.end( ), 0 );
    NZ = NA - this->m_N[ 0 ];
  }
  else
  {
    NA = *( std::max_element( this->m_N.begin( ), this->m_N.end( ) ) );
    NZ = *( std::max_element( this->m_N.begin( ) + 1, this->m_N.end( ) ) );
  } // end if
  NA *= M;
  NZ *= M;
  if( *bufferA == nullptr )
    *bufferA
      =
      reinterpret_cast< TReal* >( std::calloc( NA, sizeof( TReal ) ) );
  if( *bufferZ == nullptr )
    *bufferZ
      =
      reinterpret_cast< TReal* >( std::calloc( NZ, sizeof( TReal ) ) );

  if( A != nullptr && Z != nullptr )
  {
    TReal* bA = *bufferA;
    TReal* bZ = *bufferZ;
    A->clear( );
    Z->clear( );
    A->push_back( TMatrixMap( bA, M, this->m_N[ 0 ] ) );
    for( TNatural l = 1; l <= this->number_of_layers( ); ++l )
    {
      bA += ( keep_AZ )? ( this->m_N[ l - 1 ] * M ): 0;
      A->push_back( TMatrixMap( bA, M, this->m_N[ l ] ) );
      Z->push_back( TMatrixMap( bZ, M, this->m_N[ l ] ) );
      bZ += ( keep_AZ )? ( this->m_N[ l ] * M ): 0;
    } // end for
  } // end for
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void PUJ_ML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
_free_buffers( TReal** bA, TReal** bZ ) const
{
  if( *bA != nullptr )
    std::free( *bA );
  if( *bZ != nullptr )
    std::free( *bZ );
  *bA = nullptr;
  *bZ = nullptr;
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void PUJ_ML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
_eval( std::vector< TMatrixMap >& A, std::vector< TMatrixMap >& Z ) const
{
  TNatural L = this->number_of_layers( );
  for( TNatural l = 1; l <= L; ++l )
  {
    Z[ l - 1 ]
      =
      ( A[ l - 1 ] * this->m_W[ l - 1 ] ).rowwise( ) + this->m_B[ l - 1 ];
    this->m_A[ l - 1 ].second( A[ l ], Z[ l - 1 ], false );
  } // end for
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
std::string PUJ_ML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
lower( const std::string& s )
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

// -------------------------------------------------------------------------
namespace PUJ_ML { namespace Model { namespace NeuralNetwork {
  PUJ_ML_Model_Instance( FeedForward );
} } }

// eof - $RCSfile$
