#include <libplugin/plugin.h>
#include <psi4-dec.h>
#include <libparallel/parallel.h>
#include <liboptions/liboptions.h>
#include <libmints/mints.h>
#include <libpsio/psio.hpp>
#include <libciomr/libciomr.h>
#include "cks.h"

INIT_PLUGIN

namespace psi{ namespace scf {

extern "C" 
int read_options(std::string name, Options& options)
{
    if (name == "CKS"|| options.read_globals()) {
        /*- The constraint -*/
        options.add_double("VC", 0.0);
        /*- The amount of information printed to the output file -*/
        options.add_int("PRINT", 1);
    }
    return true;
}


extern "C" 
PsiReturnType cks(Options& options)
{
  tstart();
  boost::shared_ptr<PSIO> psio = PSIO::shared_object();

  std::string reference = options.get_str("REFERENCE");
  double energy;

//  if (reference == "RCKS") {
//  }else {
//      throw InputException("Unknown reference " + reference, "REFERENCE", __FILE__, __LINE__);
//      energy = 0.0;
//  }

  boost::shared_ptr<RCKS> scf = boost::shared_ptr<RCKS>(new RCKS(options, psio));

  // Set this early because the callback mechanism uses it.
  Process::environment.set_reference_wavefunction(scf);
  energy = scf->compute_energy();
  scf->Lowdin();
  scf->Lowdin2();
  Process::environment.reference_wavefunction().reset();

  Communicator::world->sync();

  // Set some environment variables
  Process::environment.globals["SCF TOTAL ENERGY"] = energy;
  Process::environment.globals["CURRENT ENERGY"] = energy;
  Process::environment.globals["CURRENT REFERENCE ENERGY"] = energy;

  // Shut down psi.


//    fprintf(outfile,"QDFT");
//    int print = options.get_int("PRINT");
//    boost::shared_ptr<Wavefunction> rcks(new RCKS(options));
//    Process::environment.set_reference_wavefunction(rcks);

//    rcks->compute_energy();
  tstop();
//    Process::environment.reference_wavefunction().reset();
    return Success;
}


}} // End namespaces
