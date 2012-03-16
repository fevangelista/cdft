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
        /*- Constraint on the charges -*/
        options.add("CHARGE", new ArrayType());
        /*- Select the way the charges are computed -*/
        options.add_str("CONSTRAINT_TYPE","LOWDIN", "LOWDIN");
        /*- The threshold for the gradient of the constraint -*/
        options.add_double("W_CONVERGENCE",1.0e-5);
        // Expert options
        /*- Apply a fixed Lagrange multiplier -*/
        options.add_bool("OPTIMIZE_VC", false);
        /*- Value of the Lagrange multiplier -*/
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
  scf->Lowdin2();
  Process::environment.reference_wavefunction().reset();

  Communicator::world->sync();

  // Set some environment variables
  Process::environment.globals["SCF TOTAL ENERGY"] = energy;
  Process::environment.globals["CURRENT ENERGY"] = energy;
  Process::environment.globals["CURRENT REFERENCE ENERGY"] = energy;

  // Shut down psi.

  tstop();
  return Success;
}


}} // End namespaces
