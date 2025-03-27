// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__NeuralNetwork__FeedForward__hxx__
#define __PUJ_ML__Model__NeuralNetwork__FeedForward__hxx__

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TX >
auto PUJ_ML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
operator()( const Eigen::EigenBase< _TX >& X ) const
{
  TReal *bA = nullptr, *bZ = nullptr;
  std::vector< TMatrixMap > A, Z;
  this->_prepare_buffers( &bA, &bZ, &A, &Z, X.rows( ), false );

  A[ 0 ] = X.derived( ).template cast< TReal >( );
  this->_eval( A, Z );
  TMatrix R = A.back( );
  this->_free_buffers( &bA, &bZ );
  return( R );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TX >
auto PUJ_ML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
threshold( const Eigen::EigenBase< _TX >& X, bool categorize ) const
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
    if( categorize )
    {
      TMatrix T( A.rows( ), 1 );
      for( TNatural r = 0; r < A.rows( ); ++r )
        A.row( r ).maxCoeff( &( T( r, 0 ) ) );
      return( T );
    }
    else
    {
      for( TNatural r = 0; r < A.rows( ); ++r )
      {
        Eigen::Index c;
        A.row( r ).maxCoeff( &c );
        A.row( r ).fill( TReal( 0 ) );
        A( r, c ) = TReal( 1 );
      } // end for
      return( A );
    } // end for
  }
  else
    return( A );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TX, class _Ty >
typename PUJ_ML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
TReal PUJ_ML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
cost(
  const Eigen::EigenBase< _TX >& X, const Eigen::EigenBase< _Ty >& Y
  ) const
{
  return( this->_cost( this->operator()( X ), Y ) );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TX, class _Ty >
typename PUJ_ML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
TReal PUJ_ML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
cost_gradient(
  TReal* bufferG,
  const Eigen::EigenBase< _TX >& bX,
  const Eigen::EigenBase< _Ty >& bY,
  const TReal& L1, const TReal& L2
  ) const
{
  auto X = bX.derived( ).template cast< TReal >( );
  auto Y = bY.derived( ).template cast< TReal >( );

  // Prepare auxiliary memory
  std::vector< TMatrixMap > A, Z;
  TNatural M = X.rows( );
  bool mem_owned
    =
    ( this->m_BufferA == nullptr || this->m_BufferZ == nullptr );
  this->_prepare_buffers(
    &( this->m_BufferA ), &( this->m_BufferZ ), &A, &Z, M, true
    );

  /* TODO
     if( this->m_BufferA == nullptr || this->m_BufferZ == nullptr )
     throw error
  */

  // Prepare gradient maps
  TNatural L = this->number_of_layers( );
  std::vector< TMatrixMap > G;
  TNatural offG = 0;
  for( TNatural l = 1; l <= L; ++l )
  {
    G.push_back(
      TMatrixMap( bufferG + offG, this->m_N[ l - 1 ], this->m_N[ l ] )
      );
    offG += G.back( ).size( );
    G.push_back( TMatrixMap( bufferG + offG, 1, this->m_N[ l ] ) );
    offG += G.back( ).size( );
  } // end for

  // Forward propagation
  A[ 0 ] = X;
  this->_eval( A, Z );

  // Compute unregularized cost
  TReal J = this->_cost( A[ L ], Y );

  // Backpropagate last layer
  A[ L ] -= Y;
  G[ ( L << 1 ) - 1 ] = A[ L ].colwise( ).mean( );
  G[ ( L << 1 ) - 2 ] = ( A[ L - 1 ].transpose( ) * A[ L ] ) / TReal( M );

  // Backpropagate remaining layers
  for( TNatural k = 0; k < L - 1; ++k )
  {
    TNatural l = L - k - 1;
    this->m_A[ l - 1 ].second( Z[ l - 1 ], Z[ l - 1 ], true );
    A[ l ].array( )
      =
      Z[ l - 1 ].array( )
      *
      ( A[ l + 1 ] * this->m_W[ l ].transpose( ) ).array( );

    G[ ( l << 1 ) - 1 ] = A[ l ].colwise( ).mean( );
    G[ ( l << 1 ) - 2 ] = ( A[ l - 1 ].transpose( ) * A[ l ] ) / TReal( M );
  } // end for

  if( mem_owned )
    this->free_auxiliary_buffer( );
  return( J + this->_regularize( bufferG, L1, L2 ) );
}

#endif // __PUJ_ML__Model__NeuralNetwork__FeedForward__hxx__

// eof - $RCSfile$
