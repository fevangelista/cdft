#include <physconst.h>
#include <psifiles.h>
#include <libmints/mints.h>
#include <libfock/apps.h>
#include <libfock/v.h>
#include <libfock/jk.h>
#include <liboptions/liboptions.h>

#include "noci.h"

#define DEBUG_NOCI 0


using namespace psi;

namespace psi{ namespace scf{

NOCI::NOCI(Options &options, boost::shared_ptr<PSIO> psio): UHF(options, psio),do_noci(false),state_a_(0),state_b_(0)
{
    init();
}

NOCI::NOCI(Options &options, boost::shared_ptr<PSIO> psio, int state_a, std::pair<int,int> fmo, int state_b,
           std::vector<std::pair<int,int>>frozen_occ_a,std::vector<std::pair<int,int>>frozen_occ_b,
           std::vector<std::pair<int,int>>frozen_mos,
           std::vector<int>occ_frozen,std::vector<int>vir_frozen,
           SharedMatrix Ca_gs_, SharedMatrix Cb_gs_)
: UHF(options, psio),
  do_noci(true),do_alpha_states(true),
  state_a_(state_a),
  fmo_(fmo),
  state_b_(state_b),occ_(1),
  frozen_occ_a_(frozen_occ_a),
  frozen_occ_b_(frozen_occ_b),
  frozen_mos_(frozen_mos),
  occ_frozen_(occ_frozen),
  vir_frozen_(vir_frozen),
  Ca_gs(Ca_gs_),
  Cb_gs(Cb_gs_)
{
    //outfile->Printf("\n  ==> first excitation CI OFFFFGGG    <==\n\n");
    init();
    init_excitation();

}

void NOCI::init()
{
    //outfile->Printf("\n  ==> NON Orthogonality CI <==\n\n");
    zero_dim_ = Dimension(nirrep_);
    // Allocate matrices
    H_copy = factory_->create_shared_matrix("H_copy");
    TempMatrix = factory_->create_shared_matrix("Temp");
    TempMatrix2 = factory_->create_shared_matrix("Temp2");
    Dolda_ = factory_->create_shared_matrix("Dold alpha");
    Doldb_ = factory_->create_shared_matrix("Dold beta");

    Fpq_a = factory_->create_shared_matrix("Fpq_a");
    Fpq_b = factory_->create_shared_matrix("Fpq_b");


    Fpq = factory_->create_shared_matrix("Fpq");
    Dt_diff = factory_->create_shared_matrix("Density_diff");

    epsilon_a_ = SharedVector(factory_->create_vector());
    epsilon_b_ = epsilon_a_;;


    save_H_ = true;

}

NOCI::~NOCI()
{
}

void NOCI::init_excitation()
{
        Ua_ = SharedMatrix(new Matrix("C_aN",nsopi_,nmopi_));
        Ub_ = SharedMatrix(new Matrix("C_bN",nsopi_,nmopi_));

        Ca0  = SharedMatrix(new Matrix("Ca0",nsopi_,nmopi_));
        Cb0  = SharedMatrix(new Matrix("Cb0",nsopi_,nmopi_));

        U_fmo =    SharedMatrix(new Matrix("U_fmo",nsopi_,nmopi_));
        U_swap =    SharedMatrix(new Matrix("U_fmo",nsopi_,nmopi_));

        Ca_fmo =    SharedMatrix(new Matrix("Ca_fmo",nsopi_,nmopi_));
        Cb_fmo=    SharedMatrix(new Matrix("Cb_fmo",nsopi_,nmopi_));
        C_tmp =    SharedMatrix(new Matrix("C_fmo",nsopi_,nmopi_));

        rFpq_a = SharedMatrix(new Matrix("rFpq_a",nmopi_-(occ_frozen_+vir_frozen_),nmopi_-(occ_frozen_+vir_frozen_)));
        rFpq_b = SharedMatrix(new Matrix("rFpq_b",nmopi_-(occ_frozen_+vir_frozen_),nmopi_-(occ_frozen_+vir_frozen_)));

        rFpq = SharedMatrix(new Matrix("rFpq",nmopi_-(occ_frozen_+vir_frozen_),nmopi_-(occ_frozen_+vir_frozen_)));
        rUa_ = SharedMatrix(new Matrix("rUa_",nmopi_-(occ_frozen_+vir_frozen_),nmopi_-(occ_frozen_+vir_frozen_)));
        rUb_ = SharedMatrix(new Matrix("rUa_",nmopi_-(occ_frozen_+vir_frozen_),nmopi_-(occ_frozen_+vir_frozen_)));
        Repsilon_a_= SharedVector(new Vector("Dirac EigenValues",nmopi_-(occ_frozen_+vir_frozen_)));



        rCa0 = SharedMatrix(new Matrix("rCa0",nsopi_, nmopi_-(occ_frozen_+vir_frozen_)));
        rCb0 = SharedMatrix(new Matrix("rCb0",nsopi_, nmopi_-(occ_frozen_+vir_frozen_)));


        rCa_ = SharedMatrix(new Matrix("rCa0",nsopi_,nmopi_-(occ_frozen_+vir_frozen_)));
        rCb_ = SharedMatrix(new Matrix("rCb0",nsopi_,nmopi_-(occ_frozen_+vir_frozen_)));



} //end of the function


void NOCI::guess()
{
    if(do_noci){
        //        if (state_a_ == 1){
        //            iteration_ = 0;
        //            load_orbitals();
        //            form_D();
        //            E_ = compute_initial_E();
        //        }
        //        else {
        iteration_ = 0;
        load_orbitals();
        U_swap->identity();
        if(do_alpha_states){
            Ca_gs->print();
          //  for (auto &h_p : frozen_occ_a_){
              int irrep = std::get<0>(fmo_);
               int fmo   = std::get<1>(fmo_);
               int state = nalphapi_[irrep]-1+state_a_;
                outfile->Printf("irrep %d fmo %d state_a %d \n", irrep, fmo, state);
                U_swap->set(irrep,fmo,fmo,0.0);
                U_swap->set(irrep,state,state,0.0);
                U_swap->set(irrep,fmo,state,1.0);
                U_swap->set(irrep,state,fmo,1.0);
            //}
        Ca_fmo->gemm(false,false,1.0,Ca_gs,U_swap,0.0);
        Ca_fmo->print();
        Cb_fmo->copy(Cb_gs);
        }else{
            for (auto &h_p : frozen_occ_b_){
                int irrep = h_p.first;
                int fmo   = h_p.second;
                U_swap->set(irrep,fmo,fmo,0.0);
                U_swap->set(irrep,fmo+state_b_,fmo+state_b_,0.0);
                U_swap->set(irrep,fmo,fmo+state_b_,1.0);
                U_swap->set(irrep,fmo+state_b_,fmo,1.0);
            }
         Cb_fmo->gemm(false,false,1.0,Cb_gs,U_swap,0.0);
         Ca_fmo->copy(Ca_gs);
        }
        //print_occupation();
        Ca_->copy(Ca_gs);
        Cb_->copy(Cb_gs);
        form_D();
        E_ = compute_initial_E();
        // }
    }else{
        UHF::guess();
    }
}

void NOCI::save_density_and_energy()
{
    Dtold_->copy(Dt_);
    Dolda_->copy(Da_);
    Doldb_->copy(Db_);
    Eold_ = E_;

}

bool NOCI::test_convergency()
{
    double ediff = E_ - Eold_;
    Dt_diff->copy(Dtold_);
    Dt_diff->subtract(Dt_);
    den_diff=0.0;
    den_diff=Dt_diff->rms();
    if (fabs(ediff) < energy_threshold_ && den_diff < density_threshold_)
        return true;
    else
        return false;
}
void NOCI::form_G()
{
    // Push the C matrix on
    std::vector<SharedMatrix> & C = jk_->C_left();

    C.clear();
    C.push_back(Ca_subset("SO", "OCC"));
    C.push_back(Cb_subset("SO", "OCC"));

    // Addition to standard call, make sure that C_right is not set
    std::vector<SharedMatrix> & C_right = jk_->C_right();
    C_right.clear();

    // Run the JK object
    jk_->compute();

    // Pull the J and K matrices off
    const std::vector<SharedMatrix> & J = jk_->J();
    const std::vector<SharedMatrix> & K = jk_->K();
    J_->copy(J[0]);
    J_->add(J[1]);
    Ka_ = K[0];
    Kb_ = K[1];

    Ga_->copy(J_);
    Gb_->copy(J_);

    Ga_->subtract(Ka_);
    Gb_->subtract(Kb_);
}

void NOCI::form_F()
{
    // On the first iteration save H_
    if (save_H_){
        H_copy->copy(H_);
        save_H_ = false;
    }

    // Augement the one-electron potential (H_) with the CDFT terms
    H_->copy(H_copy);
    Fa_->copy(H_);
    Fa_->add(Ga_);

    H_->copy(H_copy);
    Fb_->copy(H_);
    Fb_->add(Gb_);

}

void NOCI::form_C()
{
  if(not do_noci){
        UHF::form_C();
     }
   else{
         form_C_noci();
  }
}


double NOCI::compute_energy()
{
    std::string reference = HF::options_.get_str("REFERENCE");

    bool converged = false;
    MOM_performed_ = false;
    diis_performed_ = false;
    // Neither of these are idempotent
    if (HF::options_.get_str("GUESS") == "SAD" || HF::options_.get_str("GUESS") == "READ")
        iteration_ = -1;
    else
        iteration_ = 0;

    if (print_)
        outfile->Printf( "  ==> Pre-Iterations <==\n\n");

    if (print_)
        print_preiterations();

    // Andy trick 2.0
    std::string old_scf_type = HF::options_.get_str("SCF_TYPE");
    if (HF::options_.get_bool("DF_SCF_GUESS") && !(old_scf_type == "DF" || old_scf_type == "CD")) {
        outfile->Printf( "  Starting with a DF guess...\n\n");
        if(!HF::options_["DF_BASIS_SCF"].has_changed()) {
            // TODO: Match Dunning basis sets
            HF::molecule_->set_basis_all_atoms("CC-PVDZ-JKFIT", "DF_BASIS_SCF");
        }
        scf_type_ = "DF";
        HF::options_.set_str("SCF","SCF_TYPE","DF"); // Scope is reset in proc.py. This is not pretty, but it works
    }

    if(attempt_number_ == 1){
        boost::shared_ptr<MintsHelper> mints (new MintsHelper(HF::options_, 0));
        mints->one_electron_integrals();

        integrals();

        timer_on("Form H");
        form_H(); //Core Hamiltonian
        timer_off("Form H");



        timer_on("Form S/X");
        form_Shalf(); //S and X Matrix
        timer_off("Form S/X");

        timer_on("Guess");
        guess(); // Guess
        timer_off("Guess");

    }else{
        // We're reading the orbitals from the previous set of iterations.
        form_D();
        E_ = compute_initial_E();
    }

    bool df = (HF::options_.get_str("SCF_TYPE") == "DF");

   // outfile->Printf( "  ==> NOCI::Iterations <==\n\n");
    outfile->Printf( "%s                        Total Energy        Delta E     RMS |[F,P]|\n\n", df ? "   " : "");



    // SCF iterations
    do {
        iteration_++;
        save_density_and_energy();
        // Call any preiteration callbacks
        call_preiteration_callbacks();
        E_ = 0.0;
        timer_on("Form G");
        form_G();
        timer_off("Form G");

        // Reset fractional SAD occupation
        if (iteration_ == 0 && HF::options_.get_str("GUESS") == "SAD")
            reset_SAD_occupation();

        timer_on("Form F");
        form_F();
        timer_off("Form F");

        if (print_>3) {
            Fa_->print("outfile");
            Fb_->print("outfile");
        }

        E_ += compute_E();

        timer_on("DIIS");
        bool add_to_diis_subspace = false;
        if (diis_enabled_ && iteration_ > 0 && iteration_ >= diis_start_ )
            add_to_diis_subspace = true;

        compute_orbital_gradient(add_to_diis_subspace);

        if (diis_enabled_ == true && iteration_ >= diis_start_ + min_diis_vectors_ - 1) {
            diis_performed_ = diis();
        } else {
            diis_performed_ = false;
        }
        timer_off("DIIS");

        if (print_>4 && diis_performed_) {
            outfile->Printf("  After DIIS:\n");
            Fa_->print("outfile");
            Fb_->print("outfile");
        }

        // If we're too well converged, or damping wasn't enabled, do DIIS
        damping_performed_ = (damping_enabled_ && iteration_ > 1 && Drms_ > damping_convergence_);

        std::string status = "";
        if(diis_performed_){
            if(status != "") status += "/";
            status += "DIIS";
        }
        if(MOM_performed_){
            if(status != "") status += "/";
            status += "MOM";
        }
        if(damping_performed_){
            if(status != "") status += "/";
            status += "DAMP";
        }
        if(frac_performed_){
            if(status != "") status += "/";
            status += "FRAC";
        }



        timer_on("Form C");
        form_C();
        timer_off("Form C");
        timer_on("Form D");
        form_D();
        timer_off("Form D");

        Process::environment.globals["SCF ITERATION ENERGY"] = E_;

        // After we've built the new D, damp the update if
        if(damping_performed_) damp_update();

        if (print_ > 3){
            Ca_->print("outfile");
            Cb_->print("outfile");
            Da_->print("outfile");
            Db_->print("outfile");
        }

        converged = test_convergency();

        df = (HF::options_.get_str("SCF_TYPE") == "DF");


        outfile->Printf( "   @%s%s iter %3d: %20.14f   %12.5e   %-11.5e %s\n", df ? "DF-" : "",
                         reference.c_str(), iteration_, E_, E_ - Eold_, Drms_, status.c_str());



        // If a an excited MOM is requested but not started, don't stop yet
        if (MOM_excited_ && !MOM_started_) converged = false;

        // If a fractional occupation is requested but not started, don't stop yet
        if (frac_enabled_ && !frac_performed_) converged = false;

        // If a DF Guess environment, reset the JK object, and keep running
        if (converged && HF::options_.get_bool("DF_SCF_GUESS") && !(old_scf_type == "DF" || old_scf_type == "CD")) {
            outfile->Printf( "\n  DF guess converged.\n\n"); // Be cool dude.
            converged = false;
            if(initialized_diis_manager_)
                diis_manager_->reset_subspace();
            scf_type_ = old_scf_type;
            HF::options_.set_str("SCF","SCF_TYPE",old_scf_type);
            old_scf_type = "DF";
            integrals();
        }

        // Call any postiteration callbacks
        call_postiteration_callbacks();

    } while (!converged && iteration_ < maxiter_ );




    outfile->Printf( "\n  ==> Post-Iterations <==\n\n");

    check_phases();
    compute_spin_contamination();
    frac_renormalize();

    if (converged || !fail_on_maxiter_) {
        // Need to recompute the Fock matrices, as they are modified during the SCF interation
        // and might need to be dumped to checkpoint later
        form_F();


        // Print the orbitals
        if(print_)
            print_orbitals();

        if (converged) {
            outfile->Printf( "  Energy converged.\n\n");
        }
        if (!converged) {
            outfile->Printf( "  Energy did not converge, but proceeding anyway.\n\n");
        }
        outfile->Printf( "  @%s%s Final Energy: %20.14f", df ? "DF-" : "", reference.c_str(), E_);
        if (perturb_h_) {
            outfile->Printf( " with %f perturbation", lambda_);
        }
        outfile->Printf( "\n\n");
        print_energies();


        // Properties
        if (print_) {
            boost::shared_ptr<OEProp> oe(new OEProp());
            oe->set_title("SCF");
            oe->add("DIPOLE");

            if (print_ >= 2) {
                oe->add("QUADRUPOLE");
                oe->add("MULLIKEN_CHARGES");
            }

            if (print_ >= 3) {
                oe->add("LOWDIN_CHARGES");
                oe->add("MAYER_INDICES");
                oe->add("WIBERG_LOWDIN_INDICES");
            }

            outfile->Printf( "  ==> Properties <==\n\n");
            oe->compute();


            Process::environment.globals["CURRENT DIPOLE X"] = Process::environment.globals["SCF DIPOLE X"];
            Process::environment.globals["CURRENT DIPOLE Y"] = Process::environment.globals["SCF DIPOLE Y"];
            Process::environment.globals["CURRENT DIPOLE Z"] = Process::environment.globals["SCF DIPOLE Z"];
        }

        save_information();
    } else {
        outfile->Printf( "  Failed to converged.\n");
        outfile->Printf( "    NOTE: MO Coefficients will not be saved to Checkpoint.\n");
        E_ = 0.0;
        if(HF::psio_->open_check(PSIF_CHKPT))
            HF::psio_->close(PSIF_CHKPT, 1);

        // Throw if we didn't converge?
        die_if_not_converged();
    }

    // Orbitals are always saved, in case an MO guess is requested later
    save_orbitals();
    if (HF::options_.get_str("SAPT") != "FALSE") //not a bool because it has types
        save_sapt_info();

    // Perform wavefunction stability analysis
    if(HF::options_.get_str("STABILITY_ANALYSIS") != "NONE")
        stability_analysis();

    // Clean memory off, handle diis closeout, etc
    finalize();


    //outfile->Printf("\nComputation Completed\n");

    return E_;
}


void NOCI::form_C_noci()
{
    //outfile->Printf("\n iteration_ %d \n", iteration_);
    if(iteration_ <= 1){
        Fpq_a->transform(Fa_,Ca_gs);
        Fpq_b->transform(Fb_,Cb_gs);
       // Fpq_a->print();
        for(int h=0; h <nirrep_; ++h){
           int oo=nalphapi_[h]-occ_frozen_[h];
           int aa=nalphapi_[h]+vir_frozen_[h];
           int ov;
           for (int i=0; i <oo; ++i){
               for (int j=0; j < oo; ++j){
                   double fpq =Fpq_a->get(h,i,j);
                   rFpq_a->set(h,i,j,fpq);
                  // rCa0->set(h,i,j,Cpq);
              }//j
              ov=oo;
              for (int a=aa;a <nmopi_[h];++a){
                   double fpq=Fpq_a->get(h,i,a);
                   rFpq_a->set(h,i,ov,fpq);
                   rFpq_a->set(h,ov,i,fpq);

                 //  rCa0->set(h,i,ov,Cpq);
                 //  rCa0->set(h,ov,i,Cpq);
                     ov+=1;
               }//a
           }//i

           int m=0;
            ov=oo;
           for (int a=aa;a <nmopi_[h];++a){
               int n=0;
               for (int b=aa;b <nmopi_[h];++b){
                   double fpq =Fpq_a->get(h,a,b);
                   double Cpq =Ca_gs->get(h,a,b);
                   rFpq_a->set(h,ov+m,ov+n,fpq);
                //   rCa0->set(h,ov+m,ov+n,Cpq);
                   n+=1;
               }//b
               m+=1;
           }//a
        } //irrep
 Fpq_a->diagonalize(Ua_,epsilon_a_);
 Fpq_b->diagonalize(Ub_,epsilon_b_);

 //epsilon_a_->print();
// epsilon_b_->print();

// rFpq_a->print();
// Beta component
        for(int h=0; h <nirrep_; ++h){
           int oo=nbetapi_[h]-occ_frozen_[h];
           int aa=nbetapi_[h]+vir_frozen_[h];
           int ov;
           for (int i=0; i <oo; ++i){
               for (int j=0; j < oo; ++j){
                   double fpq =Fpq_b->get(h,i,j);
                   rFpq_b->set(h,i,j,fpq);
              }//j
              ov=oo;
              for (int a=aa;a <nmopi_[h];++a){
                   double fpq=Fpq_b->get(h,i,a);
                   rFpq_b->set(h,i,ov,fpq);
                   rFpq_b->set(h,ov,i,fpq);
                     ov+=1;
               }//a
           }//i

           int m=0;
            ov=oo;
           for (int a=aa;a <nmopi_[h];++a){
               int n=0;
               for (int b=aa;b <nmopi_[h];++b){
                   double fpq =Fpq_b->get(h,a,b);
                   rFpq_b->set(h,ov+m,ov+n,fpq);
                   n+=1;
               }//b
               m+=1;
           }//a
        } //irrep
} //



    if(iteration_ <= 1){
        for(int h=0; h <nirrep_; ++h){
           int oo=nalphapi_[h]-occ_frozen_[h];
           int aa=nalphapi_[h]+vir_frozen_[h];
           int ov;
           for (int i=0; i <nsopi_[h]; ++i){
               for (int j=0; j < oo; ++j){
                   double Cpq =Ca_gs->get(h,i,j);
                   rCa0->set(h,i,j,Cpq);
              }//j
              ov=oo;
              for (int a=aa;a <nmopi_[h];++a){
                   double Cpq=Ca_gs->get(h,i,a);
                   rCa0->set(h,i,ov,Cpq);
                   ov+=1;
               }//a
           }//i
        } //irrep


        for(int h=0; h <nirrep_; ++h){
           int oo=nbetapi_[h]-occ_frozen_[h];
           int aa=nbetapi_[h]+vir_frozen_[h];
           int ov;
           for (int i=0; i <nsopi_[h]; ++i){
               for (int j=0; j < oo; ++j){
                   double Cpq =Cb_gs->get(h,i,j);
                   rCb0->set(h,i,j,Cpq);
              }//j
              ov=oo;
              for (int a=aa;a <nmopi_[h];++a){
                   double Cpq=Cb_gs->get(h,i,a);
                    rCb0->set(h,i,ov,Cpq);
                     ov+=1;
               }//a
           }//i
        } //irrep
    }





    if(iteration_ > 1){
        rFpq_a->transform(Fa_,rCa0);
        rFpq_b->transform(Fb_,rCb0);
    }

    rFpq->copy(rFpq_a);
    rFpq->add(rFpq_b);
    rFpq->scale(0.5);
    rFpq->diagonalize(rUa_,Repsilon_a_);

    rUb_->copy(rUa_);
    rCa_->gemm(false,false,1.0,rCa0,rUa_,0.0);
    rCb_->gemm(false,false,1.0,rCb0,rUb_,0.0);

    rCa0->copy(rCa_);
    rCb0->copy(rCb_);

  //  rCa_->print();

    Ca_->zero();
    for(int h=0; h <nirrep_; ++h){
        int oo=nalphapi_[h]-occ_frozen_[h];
        for (int i=0; i < nsopi_[h]; ++i){
            for(int j=0; j <(nalphapi_[h]-occ_frozen_[h]); ++j){
                Ca_->set(h,i,j,rCa_->get(h,i,j));
            }
            for(int jf=(nalphapi_[h]-occ_frozen_[h]); jf <nalphapi_[h];++jf){
                Ca_->set(h,i,jf,Ca_fmo->get(h,i,jf));
            }
            for(int af=nalphapi_[h]; af <(nalphapi_[h]+vir_frozen_[h]);++af){
                Ca_->set(h,i,af,Ca_fmo->get(h,i,af));
            }
            int m=oo;
            for (int a=(nalphapi_[h]+vir_frozen_[h]);a<nmopi_[h];++a){
                 Ca_->set(h,i,a,rCa_->get(h,i,m));
                 m=m+1;
            }
        }
    }

    for(int h=0; h <nirrep_; ++h){
        int oo=nbetapi_[h]-occ_frozen_[h];
        for (int i=0; i < nsopi_[h]; ++i){
            for(int j=0; j <nbetapi_[h]-occ_frozen_[h]; ++j){
                Cb_->set(h,i,j,rCb_->get(h,i,j));
            }
            for(int j=nbetapi_[h]-occ_frozen_[h]; j <nbetapi_[h];++j){
                Cb_->set(h,i,j,Cb_fmo->get(h,i,j));
            }
            for(int af=nbetapi_[h]; af <(nbetapi_[h]+vir_frozen_[h]);++af){
                Cb_->set(h,i,af,Cb_fmo->get(h,i,af));
            }
            int m=oo;
            for (int a=(nbetapi_[h]+vir_frozen_[h]);a<nmopi_[h];++a){
                 Cb_->set(h,i,a,rCb_->get(h,i,m));
                 m=m+1;
            }
        }
    }

//    Ca_fmo->print();
 //   Cb_fmo->print();
  //  Ca_->print();

}




void NOCI::form_D()
{
    for (int h = 0; h < nirrep_; ++h) {
        int nso = nsopi_[h];
        int nmo = nmopi_[h];
        int na = nalphapi_[h];
        //if(state_==1) na=nalphapi_[h]-occ_;
        int nb = nbetapi_[h];      
        if (nso == 0 || nmo == 0) continue;

        double** Ca = Ca_->pointer(h);
        double** Cb = Cb_->pointer(h);
        double** Da = Da_->pointer(h);
        double** Db = Db_->pointer(h);

        if (na == 0)
            ::memset(static_cast<void*>(Da[0]), '\0', sizeof(double)*nso*nso);
        if (nb == 0)
            ::memset(static_cast<void*>(Db[0]), '\0', sizeof(double)*nso*nso);

        C_DGEMM('N','T',nso,nso,na,1.0,Ca[0],nmo,Ca[0],nmo,0.0,Da[0],nso);
        C_DGEMM('N','T',nso,nso,nb,1.0,Cb[0],nmo,Cb[0],nmo,0.0,Db[0],nso);

    }

    Dt_->copy(Da_);
    Dt_->add(Db_);

    //outfile->Printf( "in NOCI::form_D:\n");
    //Ca_->print();
   // Cb_->print();

    if (debug_) {
        outfile->Printf( "in UHF::form_D:\n");
        Da_->print();
        Db_->print();
    }
}

void NOCI::save_information()
{
     //dets.push_back(SharedDeterminant(new Determinant(E_,Ca_,Cb_,nalphapi_,nbetapi_)));
     //dets[0]->print();
    Fpq_a->transform(S_,Ca_);
    Fpq_b->transform(S_,Cb_);
    outfile->Printf( "Printing C^t S C \n");
    Fpq_a->print();
    Fpq_b->print();
    double diag_a = 0.0;
    double ndiag_a = 0.0;
    double diag_b = 0.0;
    double ndiag_b = 0.0;
    for(int h = 0; h < nirrep_; ++h){
        for(int i = 0; i < nsopi_[h]; ++i){
            for(int j = 0; j < nsopi_[h]; ++j){
                double aaC = Fpq_a->get(h,i,j);
                double bbC = Fpq_b->get(h,i,j);
                if (i==j){
                    diag_a += aaC;
                    diag_b += bbC;
                    outfile->Printf("\n Diagonal Elements of C^SC_a: %f C^SC_b: %f \n", aaC,bbC);
                }
                else{
                    ndiag_a += aaC;
                    ndiag_b += bbC;
                }
        }}
    }
    outfile->Printf("\n Sum of the Off-Diagonal Elements of C^SC_a: %f C^SC_b: %f\n", ndiag_a,ndiag_b);
}

double NOCI::compute_E()
{

    // E_CDFT = 2.0 D*H + D*J - \alpha D*K + E_xc - Nc * Vc
    double one_electron_E = Da_->vector_dot(H_);
    one_electron_E += Db_->vector_dot(H_);

    double coulomb_E = Da_->vector_dot(J_);
    coulomb_E += Db_->vector_dot(J_);
    double exchange_E = 0.0;
    exchange_E -= Da_->vector_dot(Ka_);
    exchange_E -= Db_->vector_dot(Kb_);



    double Etotal = 0.0;
    Etotal += nuclearrep_;
    Etotal += one_electron_E;
    Etotal += 0.5 * coulomb_E;
    Etotal += 0.5 * exchange_E;


    if (debug_) {
        outfile->Printf( "   => Energetics <=\n\n");
        outfile->Printf( "    Nuclear Repulsion Energy = %24.14f\n", nuclearrep_);
        outfile->Printf( "    One-Electron Energy =      %24.14f\n", one_electron_E);
        outfile->Printf( "    Coulomb Energy =           %24.14f\n", 0.5 * coulomb_E);
        outfile->Printf( "    Hybrid Exchange Energy =   %24.14f\n", 0.5 * exchange_E);
           outfile->Printf( "=====>\n\n");
           //Fpq_a->transform(Fa_,Ca_);
          // Fpq_a -> print();
            outfile->Printf( "<<<<<=====\n\n");
    }
    return Etotal;
}
}} // Namespaces

