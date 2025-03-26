// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Helpers__FitModel__h__
#define __PUJ_ML__Helpers__FitModel__h__

#include <algorithm>
#include <cctype>
#include <string>

#include <PUJ_ML/Config.h>
#include <PUJ_ML/Optimizer/Adam.h>
#include <PUJ_ML/Optimizer/GradientDescent.h>

namespace PUJ_ML
{
  namespace Helpers
  {
    /**
     */
    template< class _TModel, class _TArgs, class _TXTr, class _TYTr, class _TXTe, class _TYTe >
    void FitModel(
      _TModel& model, const _TArgs& args,
      const Eigen::EigenBase< _TXTr >& X_tr,
      const Eigen::EigenBase< _TYTr >& Y_tr,
      const Eigen::EigenBase< _TXTe >& X_te,
      const Eigen::EigenBase< _TYTe >& Y_te
      )
    {
      auto lwr = []( const std::string& s ) -> std::string
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
        };

      PUJ_ML::Optimizer::Base< _TModel, _TXTr, _TYTr, _TXTe, _TYTe >* opt
        =
        nullptr;
      if( lwr( args.Optimizer ) == "gradientdescent" )
      {
        opt
          =
          new PUJ_ML::Optimizer::GradientDescent
          < _TModel, _TXTr, _TYTr, _TXTe, _TYTe >
          ( X_tr, Y_tr, X_te, Y_te );
      }
      else if( lwr( args.Optimizer ) == "adam" )
      {
        opt
          =
          new PUJ_ML::Optimizer::Adam
          < _TModel, _TXTr, _TYTr, _TXTe, _TYTe >
          ( X_tr, Y_tr, X_te, Y_te );
      } // end if
      if( opt == nullptr )
      {
        // TODO: throw error
      } // end if

      // Assign common parameters
      opt->setAlpha( args.Alpha );
      opt->setL1( args.L1 );
      opt->setL2( args.L2 );
      opt->setBatchSize( args.BatchSize );
      opt->setNumberOfMaximumIterations( args.Epochs );
      if( lwr( args.Validation ) == "loo" )
        opt->setValidationToLeaveOneOut( );
      else if( lwr( args.Validation ) == "kfold" )
        opt->setValidationToKfold( args.K );
      else // if( lwr( args.Validation ) == "normal" )
        opt->setValidationToNormal( );

      // TODO: debugger
      opt->setDebug(
        []( const auto& t, const auto& Jtr, const auto& Jte )
        {
          std::cout << t << " " << Jtr << " " << Jte << std::endl;
          return( false );
        }
        );

      // Fit!
      opt->fit( model );

      // Free memory
      delete opt;
    }
  } // end namespace
} // end namespace

#endif // __PUJ_ML__Helpers__FitModel__h__

// eof - $RCSfile$
