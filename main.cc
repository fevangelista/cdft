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
    if (name == "CKS" or options.read_globals()) {
        /*- Charge constraints -*/
        options.add("CHARGE", new ArrayType());
        /*- Spin constraints -*/
        options.add("SPIN", new ArrayType());
        /*- Excitation constraints -*/
        options.add("ALPHA_EXCITATION", new ArrayType());
        /*- Excitation constraints -*/
        options.add("BETA_EXCITATION", new ArrayType());
        /*- Excitation constraints on the HOMO orbital -*/
        options.add("HOMO_EXCITATION", new ArrayType());
        /*- Select the way the charges are computed -*/
        options.add_str("CONSTRAINT_TYPE","LOWDIN", "LOWDIN");
        /*- Select the way the charges are computed -*/
        options.add_str("W_ALGORITHM","NEWTON","NEWTON QUADRATIC");
        /*- The threshold for the gradient of the constraint -*/
        options.add_double("W_CONVERGENCE",1.0e-5);

        // Expert options
        /*- Apply a fixed Lagrange multiplier -*/
        options.add_bool("OPTIMIZE_VC", true);
        /*- Value of the Lagrange multiplier -*/
        options.add("VC", new ArrayType());
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

  // Run a ground state computation first
  if (reference == "RKS") {
      boost::shared_ptr<RCKS> scf = boost::shared_ptr<RCKS>(new RCKS(options, psio));
      Process::environment.set_reference_wavefunction(scf);
      energy = scf->compute_energy();
  }else if (reference == "UKS") {
      boost::shared_ptr<UCKS> scf = boost::shared_ptr<UCKS>(new UCKS(options, psio));
      Process::environment.set_reference_wavefunction(scf);
      energy = scf->compute_energy();
      // Additionally if excitation was specified, run an excited state computation
      if(options["ALPHA_EXCITATION"].size() != 0 or options["BETA_EXCITATION"].size() != 0 or options["HOMO_EXCITATION"].size() != 0){
        boost::shared_ptr<UCKS> scf_ex = boost::shared_ptr<UCKS>(new UCKS(options,psio,scf));
        energy = scf_ex->compute_energy();
      }
  }else {
      throw InputException("Unknown reference " + reference, "REFERENCE", __FILE__, __LINE__);
      energy = 0.0;
  }



  // Set this early because the callback mechanism uses it.
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
