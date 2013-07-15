#include <libplugin/plugin.h>
#include <psi4-dec.h>
#include <libparallel/parallel.h>
#include <liboptions/liboptions.h>
#include <libmints/mints.h>
#include <libpsio/psio.hpp>
#include <libciomr/libciomr.h>
#include <ucks.h>

#include <libmints/writer.h>
#include <libmints/writer_file_prefix.h>

INIT_PLUGIN

using namespace boost;

namespace psi{ namespace cdft {

extern "C"
int read_options(std::string name, Options& options)
{
    if (name == "CDFT" or options.read_globals()) {
        /*- Charge constraints -*/
        options.add("CHARGE", new ArrayType());
        /*- Spin constraints -*/
        options.add("SPIN", new ArrayType());
        /*- Number of excited states -*/
        options.add_int("NROOTS", 0);
        /*- Number of excited states per irrep, ROOTS_PER_IRREP has priority over NROOTS -*/
        options.add("ROOTS_PER_IRREP", new ArrayType());
        /*- Perform a correction of the triplet excitation energies -*/
        options.add_bool("TRIPLET_CORRECTION", true);
        /*- Perform a correction of the triplet excitation energies using the S+ formalism -*/
        options.add_bool("CDFT_SPIN_ADAPT_SP", true);
        /*- Perform a correction of the triplet excitation energies using a CI formalism -*/
        options.add_bool("CDFT_SPIN_ADAPT_CI", false);
        /*- Break the symmetry of the HOMO/LUMO pair (works only in C1 symmetry) -*/
        options.add("CDFT_BREAK_SYMMETRY", new ArrayType());
        /*- Select the way the charges are computed -*/
        options.add_str("CONSTRAINT_TYPE","LOWDIN", "LOWDIN");
        /*- Select the algorithm to optimize the constraints -*/
        options.add_str("W_ALGORITHM","NEWTON","NEWTON QUADRATIC");
        /*- Select the excited state method.  The valid options are:
        ``CP`` (constrained particle) which finds the optimal particle orbital
        while relaxing the other orbitals;
        ``CH`` (constrained hole) which finds the optimal hole orbital
        while relaxing the other orbitals;
        ``CHP`` (constrained hole/particle) which finds the optimal hole and
        particle orbitals while relaxing the other orbitals;
        ``CHP-F`` (frozen CHP) which is CHP without orbital relaxation.
        ``CHP-Fb`` (frozen beta CHP) which is CHP without beta orbital relaxation. Default is ``CHP``. -*/
        options.add_str("CDFT_EXC_METHOD","CHP","CP CH CHP CHP-F CHP-FB CIS");
        /*- An array of dimension equal to then number of irreps that allows to select a given hole/particle excitation -*/
        options.add("CDFT_EXC_SELECT", new ArrayType());
        /*- An array of dimension equal to then number of irreps that allows to select an excitation with given symmetry-*/
        options.add("CDFT_EXC_HOLE_SYMMETRY", new ArrayType());
        /*- Select the type of excited state to target -*/
        options.add_str("CDFT_EXC_TYPE","VALENCE","VALENCE CORE IP EA");
        /*- The threshold for the gradient of the constraint -*/
        options.add_double("W_CONVERGENCE",1.0e-5);
        /*- The Lagrange multiplier for the SUHF formalism -*/
        options.add_double("CDFT_SUHF_LAMBDA",0.0);
        /*- Charge constraints -*/
        options.add_double("LEVEL_SHIFT",0.0);


        // Expert options
        /*- Apply a fixed Lagrange multiplier -*/
        options.add_bool("OPTIMIZE_VC", true);
        /*- Value of the Lagrange multiplier -*/
        options.add("VC", new ArrayType());
        /*- The amount of information printed to the output file -*/
        options.add_int("PRINT", 1);
        /*- Compute an excited state by adding a penalty to the HOMO -*/
        options.add_bool("HOMO_PENALTY", false);
    }
    return true;
}

extern "C"
PsiReturnType cdft(Options& options)
{
    tstart();

    boost::shared_ptr<PSIO> psio = PSIO::shared_object();
    boost::shared_ptr<Wavefunction> ref_scf;
    std::string reference = options.get_str("REFERENCE");
    std::vector<double> energies;

    if (reference == "RKS") {
        throw InputException("Constrained RKS is not implemented ", "REFERENCE to UKS", __FILE__, __LINE__);
    }else if (reference == "UKS") {
        // Run a ground state computation first
        ref_scf = boost::shared_ptr<Wavefunction>(new scf::UCKS(options, psio));
        Process::environment.set_wavefunction(ref_scf);
        double gs_energy = ref_scf->compute_energy();
        // Print a molden file
        if ( options["MOLDEN_WRITE"].has_changed() ) {
           boost::shared_ptr<MoldenWriter> molden(new MoldenWriter(ref_scf));
           std::string filename = get_writer_file_prefix() + "." + to_string(0) + ".molden";
           molden->write(filename);
        }
        energies.push_back(gs_energy);

        if(options["ROOTS_PER_IRREP"].has_changed() and options["NROOTS"].has_changed()){
            throw InputException("NROOTS and ROOTS_PER_IRREP are simultaneously defined", "Please specify either NROOTS or ROOTS_PER_IRREP", __FILE__, __LINE__);
        }
        // Compute a number of excited states without specifying the symmetry
        if(options["NROOTS"].has_changed()){
            int nstates = options["NROOTS"].to_integer();
            for(int state = 1; state <= nstates; ++state){
                boost::shared_ptr<Wavefunction> new_scf = boost::shared_ptr<Wavefunction>(new scf::UCKS(options,psio,ref_scf,state));
                Process::environment.wavefunction().reset();
                Process::environment.set_wavefunction(new_scf);
                double new_energy = new_scf->compute_energy();
                // Print a molden file
                if ( options["MOLDEN_WRITE"].has_changed() ) {
                    boost::shared_ptr<MoldenWriter> molden(new MoldenWriter(new_scf));
                    std::string filename = get_writer_file_prefix() + "." + to_string(state) + ".molden";
                    molden->write(filename);
                }
                energies.push_back(new_energy);
                ref_scf = new_scf;
            }
        }
        // Compute a number of excited states of a given symmetry
        else if(options["ROOTS_PER_IRREP"].has_changed()){
            int maxnirrep = Process::environment.wavefunction()->nirrep();
            int nirrep = options["ROOTS_PER_IRREP"].size();
            if (nirrep != maxnirrep){
                throw InputException("The number of irreps specified in the option ROOTS_PER_IRREP does not match the number of irreps",
                                     "Please specify a correct number of irreps in ROOTS_PER_IRREP", __FILE__, __LINE__);
            }
            for (int h = 0; h < nirrep; ++h){
                int nstates = options["ROOTS_PER_IRREP"][h].to_integer();                
                if (nstates > 0){
                    fprintf(outfile,"\n\n  ==== Computing %d state%s of symmetry %d ====\n",nstates,nstates > 1 ? "s" : "",h);
                }
                int hole_num =  0;
                int part_num = -1;
                for (int state = 1; state <= nstates; ++state){
                    part_num += 1;
                    boost::shared_ptr<Wavefunction> new_scf = boost::shared_ptr<Wavefunction>(new scf::UCKS(options,psio,ref_scf,state,h,hole_num,part_num));
                    Process::environment.wavefunction().reset();
                    Process::environment.set_wavefunction(new_scf);
                    double new_energy = new_scf->compute_energy();
                    // Print a molden file
                    if ( options.get_bool("MOLDEN_WRITE") ) {
                       boost::shared_ptr<MoldenWriter> molden(new MoldenWriter(new_scf));
                       std::string filename = get_writer_file_prefix() + "." + to_string(h) + "." + to_string(state) + ".molden";
                       molden->write(filename);
                    }
                    energies.push_back(new_energy);
                    ref_scf = new_scf;
                    if (part_num > hole_num){
                       hole_num = part_num;
                       part_num = -1;
                    }
                }
            }
        }
    }else {
        throw InputException("Unknown reference " + reference, "REFERENCE", __FILE__, __LINE__);
    }

    // Set this early because the callback mechanism uses it.
    Process::environment.wavefunction().reset();

//    Communicator::world->sync();

//    // Set some environment variables
//    Process::environment.globals["SCF TOTAL ENERGY"] = energies.back();
//    Process::environment.globals["CURRENT ENERGY"] = energies.back();
//    Process::environment.globals["CURRENT REFERENCE ENERGY"] = energies[0];

    // Shut down psi.

    tstop();
    return Success;
}

}} // End namespaces
