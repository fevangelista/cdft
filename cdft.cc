#include <psi4-dec.h>
#include <physconst.h>

#include <libplugin/plugin.h>
#include <liboptions/liboptions.h>
#include <libmints/mints.h>
#include <libpsio/psio.hpp>
#include <libciomr/libciomr.h>
#include <libscf_solver/hf.h>

#include <libmints/wavefunction.h>
#include <libmints/writer.h>
#include <libmints/writer_file_prefix.h>

#include "ucks.h"
#include "ocdft.h"

INIT_PLUGIN

using namespace boost;

namespace psi{ namespace cdft {

void CDFT(Options& options);
void OCDFT(Options& options);
void NOCI(Options& options);

extern "C"
int read_options(std::string name, Options& options)
{
    if (name == "CDFT" or options.read_globals()) {
        /*- Select the constrained DFT method.  The valid options are:
        ``CDFT`` Constrained DFT;
        ``OCDFT`` Constrained DFT;   Default is ``OCDFTHP``. -*/
        options.add_str("METHOD","OCDFT", "OCDFT CDFT NOCI");

        // Options for Constrained DFT (CDFT)

        /*- Charge constraints -*/
        options.add("CHARGE", new ArrayType());
        /*- Spin constraints -*/
        options.add("SPIN", new ArrayType());
        /*- Select the way the charges are computed -*/
        options.add_str("CONSTRAINT_TYPE","LOWDIN", "LOWDIN");
        /*- Select the algorithm to optimize the constraints -*/
        options.add_str("W_ALGORITHM","NEWTON","NEWTON QUADRATIC");
        /*- The threshold for the gradient of the constraint -*/
        options.add_double("W_CONVERGENCE",1.0e-5);
        /*- The Lagrange multiplier for the SUHF formalism -*/
        options.add_double("CDFT_SUHF_LAMBDA",0.0);
        /*- Charge constraints -*/
        options.add_double("LEVEL_SHIFT",0.0);
        /*- Apply a fixed Lagrange multiplier -*/
        options.add_bool("OPTIMIZE_VC", true);
        /*- Value of the Lagrange multiplier -*/
        options.add("VC", new ArrayType());


        // Options for Orthogonality Constrained DFT (OCDFT)

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
        /*- Select the type of excited state to target -*/
        options.add_str("CDFT_PROJECT_OUT","H","H P HP");
        /*- Select the type of excited state to target -*/
        options.add_int("CDFT_NUM_PROJECT_OUT",1);


        // Expert options
        /*- The amount of information printed to the output file -*/
        options.add_int("PRINT", 1);
    }
    return true;
}

extern "C"
PsiReturnType cdft(Options& options)
{
    if (options.get_str("METHOD") == "CDFT"){
        outfile->Printf("\n  ==> Constrained DFT <==\n");
        CDFT(options);
    }else if (options.get_str("METHOD") == "OCDFT"){
        outfile->Printf("\n  ==> Orthogonality Constrained DFT <==\n");
        OCDFT(options);
    }else if (options.get_str("METHOD") == "NOCI"){
        outfile->Printf("\n  ==> NON-Orthogonality CI <==\n");
        NOCI(options);
     }

    // Set some environment variables
//    Process::environment.globals["SCF TOTAL ENERGY"] = energies.back();
//    Process::environment.globals["CURRENT ENERGY"] = energies.back();
//    Process::environment.globals["CURRENT REFERENCE ENERGY"] = energies[0];
    return Success;
}

void CDFT(Options& options)
{
    std::string reference = options.get_str("REFERENCE");

    if (reference == "RKS") {
        throw InputException("Constrained RKS is not implemented ", "REFERENCE to UKS", __FILE__, __LINE__);
    }else if (reference == "UKS") {
        // Run a ground state computation first
        boost::shared_ptr<PSIO> psio = PSIO::shared_object();
        boost::shared_ptr<Wavefunction> ref_scf(new scf::UCKS(options, psio));
        Process::environment.set_wavefunction(ref_scf);
        double gs_energy = ref_scf->compute_energy();

        // If requested, write a molden file
        if ( options["MOLDEN_WRITE"].has_changed() ) {
           boost::shared_ptr<MoldenWriter> molden(new MoldenWriter(ref_scf));
           std::string filename = get_writer_file_prefix() + "." + to_string(0) + ".molden";
           psi::scf::HF* hf = (psi::scf::HF*)ref_scf.get();
           SharedVector occA = hf->occupation_a();
           SharedVector occB = hf->occupation_b();
           molden->write(filename,ref_scf->Ca(),ref_scf->Cb(),ref_scf->epsilon_a(),ref_scf->epsilon_b(),occA,occB);
        }
    }
}


void NOCI(Options& options)
{
    std::string reference = options.get_str("REFERENCE");
}



void OCDFT(Options& options)
{
    boost::shared_ptr<PSIO> psio = PSIO::shared_object();
    boost::shared_ptr<Wavefunction> ref_scf;
    std::string reference = options.get_str("REFERENCE");
    std::vector<double> energies;
    // Store the irrep, multiplicity, total energy, excitation energy, oscillator strength
    std::vector<boost::tuple<int,int,double,double,double>> state_info;

    if (reference == "RKS") {
        throw InputException("Constrained RKS is not implemented ", "REFERENCE to UKS", __FILE__, __LINE__);
    }else if (reference == "UKS") {
        // Run a ground state computation first
        ref_scf = boost::shared_ptr<Wavefunction>(new scf::UOCDFT(options, psio));
        Process::environment.set_wavefunction(ref_scf);
        double gs_energy = ref_scf->compute_energy();

        energies.push_back(gs_energy);

        state_info.push_back(boost::make_tuple(0,1,gs_energy,0.0,0.0));

        // Print a molden file
        if ( options["MOLDEN_WRITE"].has_changed() ) {
           boost::shared_ptr<MoldenWriter> molden(new MoldenWriter(ref_scf));
           std::string filename = get_writer_file_prefix() + "." + to_string(0) + ".molden";
           psi::scf::HF* hf = (psi::scf::HF*)ref_scf.get();
           SharedVector occA = hf->occupation_a();
           SharedVector occB = hf->occupation_b();
           molden->write(filename,ref_scf->Ca(),ref_scf->Cb(),ref_scf->epsilon_a(),ref_scf->epsilon_b(),occA,occB);
        }

        if(options["ROOTS_PER_IRREP"].has_changed() and options["NROOTS"].has_changed()){
            throw InputException("NROOTS and ROOTS_PER_IRREP are simultaneously defined", "Please specify either NROOTS or ROOTS_PER_IRREP", __FILE__, __LINE__);
        }
        // Compute a number of excited states without specifying the symmetry
        if(options["NROOTS"].has_changed()){
            int nstates = options["NROOTS"].to_integer();
            for(int state = 1; state <= nstates; ++state){
                boost::shared_ptr<Wavefunction> new_scf = boost::shared_ptr<Wavefunction>(new scf::UOCDFT(options,psio,ref_scf,state));
                Process::environment.wavefunction().reset();
                Process::environment.set_wavefunction(new_scf);
                double new_energy = new_scf->compute_energy();
                energies.push_back(new_energy);

                scf::UOCDFT* uocdft_scf = dynamic_cast<scf::UOCDFT*>(new_scf.get());
                double singlet_exc_energy_s_plus = uocdft_scf->singlet_exc_energy_s_plus();
                double oscillator_strength_s_plus = uocdft_scf->oscillator_strength_s_plus();
                state_info.push_back(boost::make_tuple(state,1,new_energy,singlet_exc_energy_s_plus,oscillator_strength_s_plus));

                // Print a molden file
                if ( options["MOLDEN_WRITE"].has_changed() ) {
                    boost::shared_ptr<MoldenWriter> molden(new MoldenWriter(new_scf));
                    std::string filename = get_writer_file_prefix() + "." + to_string(state) + ".molden";
                    psi::scf::HF* hf = (psi::scf::HF*)new_scf.get();
                    SharedVector occA = hf->occupation_a();
                    SharedVector occB = hf->occupation_b();
                    molden->write(filename,new_scf->Ca(),new_scf->Cb(),new_scf->epsilon_a(),new_scf->epsilon_b(),occA,occB);
                }

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
                    outfile->Printf("\n\n  ==== Computing %d state%s of symmetry %d ====\n",nstates,nstates > 1 ? "s" : "",h);
                }
                int hole_num =  0;
                int part_num = -1;
                for (int state = 1; state <= nstates; ++state){
                    part_num += 1;
                    boost::shared_ptr<Wavefunction> new_scf = boost::shared_ptr<Wavefunction>(new scf::UOCDFT(options,psio,ref_scf,state,h));
                    Process::environment.wavefunction().reset();
                    Process::environment.set_wavefunction(new_scf);
                    double new_energy = new_scf->compute_energy();
                    energies.push_back(new_energy);

                    scf::UOCDFT* uocdft_scf = dynamic_cast<scf::UOCDFT*>(new_scf.get());
                    double singlet_exc_energy_s_plus = uocdft_scf->singlet_exc_energy_s_plus();
                    double oscillator_strength_s_plus = uocdft_scf->oscillator_strength_s_plus();
                    state_info.push_back(boost::make_tuple(0,1,new_energy,singlet_exc_energy_s_plus,oscillator_strength_s_plus));

                    // Print a molden file
                    if ( options.get_bool("MOLDEN_WRITE") ) {
                       boost::shared_ptr<MoldenWriter> molden(new MoldenWriter(new_scf));
                       std::string filename = get_writer_file_prefix() + "." + to_string(h) + "." + to_string(state) + ".molden";
                       psi::scf::HF* hf = (psi::scf::HF*)new_scf.get();
                       SharedVector occA = hf->occupation_a();
                       SharedVector occB = hf->occupation_b();
                       molden->write(filename,new_scf->Ca(),new_scf->Cb(),new_scf->epsilon_a(),new_scf->epsilon_b(),occA,occB);
                    }
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

    outfile->Printf("\n       ==> OCDFT Excited State Information <==\n");

    outfile->Printf("\n    ----------------------------------------------------");
    outfile->Printf("\n      State       Energy (Eh)    Omega (eV)   Osc. Str.");
    outfile->Printf("\n    ----------------------------------------------------");
    for (size_t n = 0; n < state_info.size(); ++n){
        double singlet_exc_en = state_info[n].get<3>();
        double osc_strength = state_info[n].get<4>();
        outfile->Printf("\n     @OCDFT-%-3d %13.7f %11.4f %11.4f",n,energies[n],(singlet_exc_en) * pc_hartree2ev,osc_strength);
    }
    outfile->Printf("\n    ----------------------------------------------------\n");

    // Set this early because the callback mechanism uses it.
    Process::environment.wavefunction().reset();
}

}} // End namespaces
