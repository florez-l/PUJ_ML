// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <PUJ_ML/Helpers/ParseFitArguments.h>

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
PUJ_ML::Helpers::ParseFitArguments< _TReal, _TNatural >::
ParseFitArguments( const std::string& desc, int argc, char** argv )
{
  this->Description = new boost::program_options::options_description( desc );
  this->Description->add_options( )
    ( "help,h", "this message" )
    ( "alpha,a", boost::program_options::value< TReal >( &this->Alpha )->default_value( this->Alpha ), "Learning rate" )
    ( "L1", boost::program_options::value< TReal >( &this->L1 )->default_value( this->L1 ), "LASSO coefficient" )
    ( "L2", boost::program_options::value< TReal >( &this->L2 )->default_value( this->L2 ), "Ridge coefficient" )
    ( "epochs,e", boost::program_options::value< TNatural >( &this->Epochs )->default_value( this->Epochs ), "Maximum numer of epochs" )
    ( "batch_size,b", boost::program_options::value< TNatural >( &this->BatchSize )->default_value( this->BatchSize ), "Batch size" )
    ( "optimizer,o", boost::program_options::value< std::string >( &this->Optimizer )->default_value( this->Optimizer ), "Adam|GradientDescent" )
    ( "validation,v", boost::program_options::value< std::string >( &this->Validation )->default_value( this->Validation ), "normal|LOO|KFoldK" )
    ;
  this->Argc = argc;
  this->Argv = argv;
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
PUJ_ML::Helpers::ParseFitArguments< _TReal, _TNatural >::
~ParseFitArguments( )
{
  delete this->Description;
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void PUJ_ML::Helpers::ParseFitArguments< _TReal, _TNatural >::
parse( )
{
  try
  {
    boost::program_options::variables_map vm;
    boost::program_options::store(
      boost::program_options::command_line_parser( this->Argc, this->Argv )
      .options( *( this->Description ) )
      .positional( this->Positional )
      .run( ),
      vm
      );
    boost::program_options::notify( vm );

    this->Error << "";
    this->Success = 0;

    if( this->Argc == 1 || vm.count( "help" ) > 0 )
    {
      this->Error << *( this->Description );
      this->Success = 1;
      return;
    } // end if
  }
  catch( const std::exception& err )
  {
    this->Error << err.what( );
    this->Success = 3;
    return;
  } // end try
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void PUJ_ML::Helpers::ParseFitArguments< _TReal, _TNatural >::
show_error( std::ostream& o )
{
  o << this->Error.str( ) << std::endl;
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
bool PUJ_ML::Helpers::ParseFitArguments< _TReal, _TNatural >::
fail( )
{
  return( this->Success != 0 );
}

// -------------------------------------------------------------------------
namespace PUJ_ML
{
  namespace Helpers
  {
    template class PUJ_ML_EXPORT ParseFitArguments< float, unsigned int >;
    template class PUJ_ML_EXPORT ParseFitArguments< float, unsigned long >;
    template class PUJ_ML_EXPORT ParseFitArguments< float, unsigned long long >;

    template class PUJ_ML_EXPORT ParseFitArguments< double, unsigned int >;
    template class PUJ_ML_EXPORT ParseFitArguments< double, unsigned long >;
    template class PUJ_ML_EXPORT ParseFitArguments< double, unsigned long long >;

    template class PUJ_ML_EXPORT ParseFitArguments< long double, unsigned int >;
    template class PUJ_ML_EXPORT ParseFitArguments< long double, unsigned long >;
    template class PUJ_ML_EXPORT ParseFitArguments< long double, unsigned long long >;
  } // end namespace
} // end namespace

// eof - $RCSfile$
