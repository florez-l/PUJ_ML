// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Helpers__ParseFitArguments__h__
#define __PUJ_ML__Helpers__ParseFitArguments__h__

#include <map>
#include <string>
#include <boost/program_options.hpp>

#include <PUJ_ML/Config.h>

namespace PUJ_ML
{
  namespace Helpers
  {
    /**
     */
    template< class _TReal, class _TNatural >
    struct ParseFitArguments
    {
      using TReal = _TReal;
      using TNatural = _TNatural;

      ParseFitArguments( const std::string& desc, int argc, char** argv );
      virtual ~ParseFitArguments( );

      void parse( );
      void show_error( std::ostream& o );
      bool fail( );

      template< class _TV >
      void add_positional_option(
        const std::string& name, const std::string& help = ""
        );

      template< class _TV >
      void add_option(
        const std::string& name, const std::string& default_value,
        const std::string& help = ""
        );

      boost::program_options::options_description* Description;
      int Argc;
      char** Argv;

      int Success;
      std::stringstream Error;

      TReal Alpha { 1e-2 };
      TReal L1 { 0 };
      TReal L2 { 0 };
      TNatural BatchSize { 0 };
      TNatural Epochs { 1000 };

      std::string Optimizer  { "adam" };
      std::string Validation { "normal" };

      boost::program_options::positional_options_description Positional;
      std::map< std::string, std::string > Strings;
    };
  } // end namespace
} // end namespace

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TV >
void PUJ_ML::Helpers::ParseFitArguments< _TReal, _TNatural >::
add_positional_option( const std::string& name, const std::string& help )
{
  if( typeid( _TV ) == typeid( std::string ) )
  {
    auto i = this->Strings.insert( std::make_pair( name, "" ) ).first;
    this->Description->add_options( )
      (
        i->first.c_str( ),
        boost::program_options::value< std::string >( &( i->second ) ),
        help.c_str( )
        )
      ;
    this->Positional.add( i->first.c_str( ), 1 );
  } // end if
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TV >
void PUJ_ML::Helpers::ParseFitArguments< _TReal, _TNatural >::
add_option(
  const std::string& name, const std::string& default_value,
  const std::string& help
  )
{
  if( typeid( _TV ) == typeid( std::string ) )
  {
    auto i = this->Strings.insert( std::make_pair( name, "" ) ).first;
    this->Description->add_options( )
      (
        i->first.c_str( ),
        boost::program_options::value< std::string >( &( i->second ) )
        ->default_value( default_value ),
        help.c_str( )
        )
      ;
  } // end if
}

#endif // __PUJ_ML__Helpers__ParseFitArguments__h__

// eof - $RCSfile$
