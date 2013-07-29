#include <ucks.h>
#include <physconst.h>
#include <libmints/view.h>
#include <libmints/mints.h>
#include <libfock/apps.h>
#include <libfock/v.h>
#include <libfock/jk.h>
#include <libdisp/dispersion.h>
#include <liboptions/liboptions.h>
#include <libciomr/libciomr.h>
#include <libqt/qt.h>
#include "boost/tuple/tuple_comparison.hpp"
#include <libiwl/iwl.hpp>
#include <psifiles.h>
//#include <libscf_solver/integralfunctors.h>
//#include <libscf_solver/omegafunctors.h>

#define DEBUG_THIS2(EXP) \
    fprintf(outfile,"\n  Starting " #EXP " ..."); fflush(outfile); \
    EXP \
    fprintf(outfile,"  done."); fflush(outfile); \


using namespace psi;

namespace psi{ namespace scf{

UCKS::UCKS(Options &options, boost::shared_ptr<PSIO> psio)
: UKS(options, psio),
  do_excitation(false),
  do_symmetry(false),
  optimize_Vc(false),
  gradW_threshold_(1.0e-9),
  nW_opt(0),
  ground_state_energy(0.0),
  ground_state_symmetry_(0),
  excited_state_symmetry_(0),
  state_(0)
{
    init();
    gs_Fa_ = Fa_;
    gs_Fb_ = Fb_;
}

UCKS::UCKS(Options &options, boost::shared_ptr<PSIO> psio, boost::shared_ptr<Wavefunction> ref_scf, int state)
: UKS(options, psio),
  do_excitation(true),
  do_symmetry(false),
  optimize_Vc(false),
  gradW_threshold_(1.0e-9),
  nW_opt(0),
  ground_state_energy(0.0),
  ground_state_symmetry_(0),
  excited_state_symmetry_(0),
  state_(state),
  hole_num_(0),
  part_num_(0)
{
    init();
    init_excitation(ref_scf);
    ground_state_energy = dets[0]->energy();
}

UCKS::UCKS(Options &options, boost::shared_ptr<PSIO> psio, boost::shared_ptr<Wavefunction> ref_scf, int state,int symmetry,int hole_num,int part_num)
: UKS(options, psio),
  do_excitation(true),
  do_symmetry(true),
  optimize_Vc(false),
  gradW_threshold_(1.0e-9),
  nW_opt(0),
  ground_state_energy(0.0),
  ground_state_symmetry_(0),
  excited_state_symmetry_(symmetry),
  state_(state),
  hole_num_(hole_num),
  part_num_(part_num)
{
    init();
    init_excitation(ref_scf);
    ground_state_energy = dets[0]->energy();
    ground_state_symmetry_ = dets[0]->symmetry();
}

void UCKS::init()
{
    fprintf(outfile,"\n  ==> Constrained DFT (UCKS) <==\n\n");


    optimize_Vc = false;
    if(KS::options_["CHARGE"].size() > 0 or KS::options_["SPIN"].size() > 0){
        KS::options_.get_bool("OPTIMIZE_VC");
    }
    gradW_threshold_ = KS::options_.get_double("W_CONVERGENCE");
    fprintf(outfile,"  gradW threshold = :%9.2e\n",gradW_threshold_);
    nfrag = basisset()->molecule()->nfragments();
    fprintf(outfile,"  Number of fragments: %d\n",nfrag);
    level_shift_ = KS::options_.get_double("LEVEL_SHIFT");
    fprintf(outfile,"  Level shift: %f\n",level_shift_);

    build_W_frag();

    // Check the option CHARGE, if it is defined use it to define the constrained charges, "-" skips the constraint
    for (int f = 0; f < int(KS::options_["CHARGE"].size()); ++f){
        if(KS::options_["CHARGE"][f].to_string() != "-"){
            double constrained_charge = KS::options_["CHARGE"][f].to_double();
            double Nc = frag_nuclear_charge[f] - constrained_charge;
            SharedConstraint constraint(new Constraint(W_frag[f],Nc,1.0,1.0,"charge(" + to_string(f) + ")"));
            constraints.push_back(constraint);
            fprintf(outfile,"  Fragment %d: constrained charge = %f .\n",f,constrained_charge);
        }else{
            fprintf(outfile,"  Fragment %d: no charge constraint specified .\n",f);
        }
    }
    // Check the option SPIN, if it is defined use it to define the constrained spins, "-" skips the constraint
    for (int f = 0; f < int(KS::options_["SPIN"].size()); ++f){
        if(KS::options_["SPIN"][f].to_string() != "-"){
            double constrained_spin = KS::options_["SPIN"][f].to_double();
            double Nc = constrained_spin;
            SharedConstraint constraint(new Constraint(W_frag[f],Nc,0.5,-0.5,"spin(" + to_string(f) + ")"));
            constraints.push_back(constraint);
            fprintf(outfile,"  Fragment %d: constrained spin   = %f .\n",f,constrained_spin);
        }else{
            fprintf(outfile,"  Fragment %d: no spin constraint specified .\n",f);
        }
    }

    saved_naholepi_ = Dimension(nirrep_,"Saved number of holes per irrep");
    saved_napartpi_ = Dimension(nirrep_,"Saved number of particles per irrep");
    zero_dim_ = Dimension(nirrep_);

    nconstraints = static_cast<int>(constraints.size());

    // Allocate vectors
    TempVector = factory_->create_shared_vector("SVD sigma");
    gradW = SharedVector(new Vector("gradW",nconstraints));
    gradW_old = SharedVector(new Vector("gradW_old",nconstraints));
    gradW_mo_resp = SharedVector(new Vector("gradW_mo_resp",nconstraints));
    Vc = SharedVector(new Vector("Vc",nconstraints));
    Vc_old = SharedVector(new Vector("Vc_old",nconstraints));

    // Allocate matrices
    H_copy = factory_->create_shared_matrix("H_copy");
    TempMatrix = factory_->create_shared_matrix("Temp");
    TempMatrix2 = factory_->create_shared_matrix("Temp2");
    Dolda_ = factory_->create_shared_matrix("Dold alpha");
    Doldb_ = factory_->create_shared_matrix("Dold beta");
    hessW = SharedMatrix(new Matrix("hessW",nconstraints,nconstraints));
    hessW_BFGS = SharedMatrix(new Matrix("hessW_BFGS",nconstraints,nconstraints));

    for (int f = 0; f < std::min(int(KS::options_["VC"].size()),nconstraints); ++f){
        if(KS::options_["VC"][f].to_string() != "-"){
            Vc->set(f,KS::options_["VC"][f].to_double());
        }else{
            Vc->set(f,0.0);
        }
        fprintf(outfile,"  The Lagrange multiplier for constraint %d will be initialized to Vc = %f .\n",f,KS::options_["VC"][f].to_double());
    }

    if(optimize_Vc){
        fprintf(outfile,"  The constraint(s) will be optimized.\n");
    }
    save_H_ = true;
}

void UCKS::init_excitation(boost::shared_ptr<Wavefunction> ref_scf)
{
    // Never recalculate the socc and docc arrays
    input_socc_ = true;
    input_docc_ = true;

    // Default: CHP algorithm
    do_holes = true;
    do_parts = true;
    do_opt_spectators = true;

    std::string exc_method = KS::options_.get_str("CDFT_EXC_METHOD");
    if(exc_method == "CHP-F"){
        do_opt_spectators = false;
    }else if(exc_method == "CH"){
        do_parts = false;
    }else if(exc_method == "CP"){
        do_holes = false;
    }

    std::string project_out = KS::options_.get_str("CDFT_PROJECT_OUT");
    do_project_out_holes = false;
    do_project_out_particles = false;
    if (project_out == "H"){
        do_project_out_holes = true;
    }else if (project_out == "P"){
        do_project_out_particles = true;
    }else if (project_out == "HP"){
        do_project_out_holes = true;
        do_project_out_particles = true;
    }

    // Save the reference state MOs and occupation numbers
    fprintf(outfile,"  Saving the reference orbitals for an excited state computation\n");
    UCKS* ucks_ptr = dynamic_cast<UCKS*>(ref_scf.get());

    gs_Fa_ = ucks_ptr->gs_Fa_;
    gs_Fb_ = ucks_ptr->gs_Fb_;
    Fa_->copy(gs_Fa_);
    Fb_->copy(gs_Fb_);
    naholepi_ = Dimension(nirrep_,"Number of holes per irrep");
    napartpi_ = Dimension(nirrep_,"Number of particles per irrep");
    gs_nalphapi_ = ucks_ptr->dets[0]->nalphapi();
    gs_navirpi_  = nmopi_ - gs_nalphapi_;
    gs_nbetapi_  = ucks_ptr->dets[0]->nbetapi();
    gs_nbvirpi_  = nmopi_ - gs_nbetapi_;

    // Grab the saved number of alpha holes/particles
    saved_naholepi_ = ucks_ptr->naholepi_;
    saved_napartpi_ = ucks_ptr->napartpi_;
    if (state_ == 1){
        dets.push_back(ucks_ptr->dets[0]);
        saved_Ch_ = SharedMatrix(new Matrix("Ch_",nsopi_,gs_nalphapi_));
        saved_Cp_ = SharedMatrix(new Matrix("Cp_",nsopi_,gs_navirpi_));
    }else{
        dets = ucks_ptr->dets;
        saved_Ch_ = ucks_ptr->Ch_;
        saved_Cp_ = ucks_ptr->Cp_;
    }

    PoFaPo_ = SharedMatrix(new Matrix("PoFPo",gs_nalphapi_,gs_nalphapi_));
    PvFaPv_ = SharedMatrix(new Matrix("PvFPo",gs_navirpi_,gs_navirpi_));
    Ua_o_ = SharedMatrix(new Matrix("Ua_o_",gs_nalphapi_,gs_nalphapi_));
    Ua_v_ = SharedMatrix(new Matrix("Ua_v_",gs_navirpi_,gs_navirpi_));
    lambda_a_o_ = SharedVector(new Vector("lambda_a_o_",gs_nalphapi_));
    lambda_a_v_ = SharedVector(new Vector("lambda_a_v_",gs_navirpi_));

    Ch_ = SharedMatrix(new Matrix("Ch_",nsopi_,gs_nalphapi_));
    Cp_ = SharedMatrix(new Matrix("Cp_",nsopi_,gs_navirpi_));

    QFQ_ = factory_->create_shared_matrix("QFQ");
    moFeffa_ = factory_->create_shared_matrix("MO alpha Feff");
    moFeffb_ = factory_->create_shared_matrix("MO beta Feff");


}

UCKS::~UCKS()
{
}

void UCKS::guess()
{
    if(do_excitation){
        iteration_ = 0;
        form_initial_C();
        //find_occupation();
//        Ca_ = dets[0]->Ca();
//        Cb_ = dets[0]->Cb();
        form_D();
        E_ = compute_initial_E();
    }else{
        UKS::guess();
    }
}

void UCKS::build_W_frag()
{
    // Compute the overlap matrix
    boost::shared_ptr<BasisSet> basisset_ = basisset();
    boost::shared_ptr<Molecule> mol = basisset_->molecule();
    boost::shared_ptr<OneBodyAOInt> overlap(integral_->ao_overlap());
    SharedMatrix S_ao(new Matrix("S_ao",basisset_->nbf(),basisset_->nbf()));
    overlap->compute(S_ao);

    // Form the S^(1/2) matrix
    S_ao->power(1.0/2.0);

    boost::shared_ptr<PetiteList> pet(new PetiteList(basisset_, integral_));
    SharedMatrix AO2SO_ = pet->aotoso();

    // Compute the W matrix for each fragment
    int min_a = 0;
    int max_a = 0;
    for (int f = 0; f < nfrag; ++f){
        std::vector<int> flist;
        std::vector<int> glist;
        flist.push_back(f);
        boost::shared_ptr<Molecule> frag = mol->extract_subsets(flist,glist);

        // Compute the nuclear charge on each fragment
        double frag_Z = 0.0;
        for (int a = 0; a < frag->natom(); ++a){
            frag_Z += frag->Z(a);
        }
        frag_nuclear_charge.push_back(frag_Z);

        // Form a copy of S_ao and zero the rows that are not on this fragment
        max_a = min_a + frag->natom();
        SharedMatrix S_f(S_ao->clone());
        for (int rho = 0; rho < basisset_->nbf(); rho++) {
            int shell = basisset_->function_to_shell(rho);
            int A = basisset_->shell_to_center(shell);
            if (A < min_a or max_a <= A){
                S_f->scale_row(0,rho,0.0);
            }
        }

        // Form W_f = (S_f)^T * S_f and transform it to the SO basis
        SharedMatrix W_f(new Matrix("W_f",basisset_->nbf(),basisset_->nbf()));
        SharedMatrix W_f_so(factory_->create_matrix("W_f_so"));
        W_f->gemm(true, false, 1.0, S_f, S_f, 0.0);
        W_f_so->apply_symmetry(W_f,AO2SO_);

        // Save W_f_so to the W_so vector
        W_frag.push_back(W_f_so);
        min_a = max_a;
    }
}

void UCKS::save_density_and_energy()
{
    Dtold_->copy(Dt_);
    Dolda_->copy(Da_);
    Doldb_->copy(Db_);
    Eold_ = E_;
}

void UCKS::form_G()
{
    timer_on("Form V");
    form_V();
    timer_off("Form V");

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
    const std::vector<SharedMatrix> & wK = jk_->wK();
    J_->copy(J[0]);
    J_->add(J[1]);
    if (functional_->is_x_hybrid()) {
        Ka_ = K[0];
        Kb_ = K[1];
    }
    if (functional_->is_x_lrc()) {
        wKa_ = wK[0];
        wKb_ = wK[1];
    }
    Ga_->copy(J_);
    Gb_->copy(J_);

    Ga_->add(Va_);
    Gb_->add(Vb_);

    double alpha = functional_->x_alpha();
    double beta = 1.0 - alpha;
    if (alpha != 0.0) {
        Ka_->scale(alpha);
        Kb_->scale(alpha);
        Ga_->subtract(Ka_);
        Gb_->subtract(Kb_);
        Ka_->scale(1.0/alpha);
        Kb_->scale(1.0/alpha);
    } else {
        Ka_->zero();
        Kb_->zero();
    }

    std::string functional_prefix = functional_->name().substr(0,2);
    if (functional_prefix == "sr"){
        wKa_->scale(-alpha);
        wKb_->scale(-alpha);
        Ga_->subtract(wKa_);
        Gb_->subtract(wKb_);
        wKa_->scale(-1.0/alpha);
        wKb_->scale(-1.0/alpha);
    } else{
        if (functional_->is_x_lrc()) {
            wKa_->scale(beta);
            wKb_->scale(beta);
            Ga_->subtract(wKa_);
            Gb_->subtract(wKb_);
            wKa_->scale(1.0/beta);
            wKb_->scale(1.0/beta);
        } else {
            wKa_->zero();
            wKb_->zero();
        }
    }

    if (debug_ > 2) {
        J_->print(outfile);
        Ka_->print(outfile);
        Kb_->print(outfile);
        wKa_->print(outfile);
        wKb_->print(outfile);
        Va_->print();
        Vb_->print();
    }
}

void UCKS::form_F()
{
    // On the first iteration save H_
    if (save_H_){
        H_copy->copy(H_);
        save_H_ = false;
    }

    // Augement the one-electron potential (H_) with the CDFT terms
    H_->copy(H_copy);
    for (int c = 0; c < nconstraints; ++c){
        TempMatrix->copy(constraints[c]->W_so());
        TempMatrix->scale(Vc->get(c) * constraints[c]->weight_alpha());
        H_->add(TempMatrix);
    }
    Fa_->copy(H_);
    Fa_->add(Ga_);

    H_->copy(H_copy);
    for (int c = 0; c < nconstraints; ++c){
        TempMatrix->copy(constraints[c]->W_so());
        TempMatrix->scale(Vc->get(c) * constraints[c]->weight_beta());
        H_->add(TempMatrix);
    }
    Fb_->copy(H_);
    Fb_->add(Gb_);

    gradient_of_W();

    // Form the effective Fock matrix
    if(do_excitation){
        // Form the projector onto the orbitals orthogonal to the holes and particles in the excited state mo representation
        TempMatrix->zero();
        if(do_holes){
            TempMatrix->gemm(false,true,1.0,Ch_,Ch_,1.0);
        }
        if(do_parts){
            TempMatrix->gemm(false,true,1.0,Cp_,Cp_,1.0);
        }
        TempMatrix->transform(S_);
        TempMatrix->transform(Ca_);
        TempMatrix2->identity();
        TempMatrix2->subtract(TempMatrix);
        // Form the Fock matrix in the excited state basis, project out the h/p
        QFQ_->transform(Fa_,Ca_);
        QFQ_->transform(TempMatrix2);
        moFeffa_->copy(QFQ_);
        // Form the Fock matrix in the excited state basis, project out the h/p
        TempMatrix->transform(Fb_,Cb_);
        moFeffb_->copy(TempMatrix);
////        QFQ_->print();
//        // Form the projector onto the ground state occuppied space in the excited state mo representation
//        TempMatrix->zero();
//        TempMatrix->gemm(false,true,1.0,Ch_,Ch_,0.0);
//        TempMatrix->gemm(false,true,1.0,Cp_,Cp_,1.0);
//        TempMatrix->transform(S_);
//        TempMatrix->transform(Ca_);
//        TempMatrix2->identity();
//        TempMatrix2->subtract(TempMatrix);
    }


    if (debug_) {
        Fa_->print(outfile);
        Fb_->print(outfile);
    }
}

void UCKS::form_C()
{
    if(not do_excitation){
        // Ground state: use the default form_C
        UKS::form_C();
        if(iteration_ == 4 and KS::options_["CDFT_BREAK_SYMMETRY"].has_changed()){
            // Mix the alpha and beta homo
            int np = KS::options_["CDFT_BREAK_SYMMETRY"][0].to_integer();
            int nq = KS::options_["CDFT_BREAK_SYMMETRY"][1].to_integer();
            double angle = KS::options_["CDFT_BREAK_SYMMETRY"][2].to_double();
            fprintf(outfile,"\n  Mixing the alpha orbitals %d and %d by %f.1 degrees\n\n",np,nq,angle);
            fflush(outfile);
            Ca_->rotate_columns(0,np-1,nq-1,pc_pi * angle / 180.0);
            Cb_->rotate_columns(0,np-1,nq-1,-pc_pi * angle / 180.0);
            // Reset the DIIS subspace
            diis_manager_->reset_subspace();
        }
    }else{
        // Excited state: use a special form_C
        form_C_ee();
    }
}

void UCKS::form_C_ee()
{
    // Compute the hole and the particle states
    compute_holes();
    compute_particles();

    // Find the hole/particle pair to follow
    find_ee_occupation(lambda_a_o_,lambda_a_v_);

    // Build the Ch and Cp matrices
    compute_hole_particle_mos();

    // Form and diagonalize the Fock matrix for the spectator orbitals
    if(do_opt_spectators){
        diagonalize_F_spectator_relaxed();
    }else{
        diagonalize_F_spectator_unrelaxed();
    }

    // Update the occupation and sort the MOs
    sort_ee_mos();

    // Beta always fully relaxed
    form_C_beta();
}

void UCKS::compute_holes()
{
    if(iteration_ > 1){
        // Form the projector Ca Ca^T S Ca_gs
        TempMatrix->zero();
        // Copy the occupied block of Ca
        copy_block(Ca_,1.0,TempMatrix,0.0,nsopi_,nalphapi_);
        // Copy Ch
        copy_block(Ch_,1.0,TempMatrix,0.0,nsopi_,naholepi_,zero_dim_,zero_dim_,zero_dim_,nalphapi_);

        TempMatrix2->gemm(false,true,1.0,TempMatrix,TempMatrix,0.0);
        TempMatrix->gemm(false,false,1.0,TempMatrix2,S_,0.0);
        TempMatrix2->gemm(false,false,1.0,TempMatrix,dets[0]->Ca(),0.0);

        TempMatrix->transform(Fa_,TempMatrix2);
    }else{
        // Transform Fa to the MO basis of the ground state
        TempMatrix->transform(Fa_,dets[0]->Ca());
    }

    // Grab the occ block of Fa
    copy_block(TempMatrix,1.0,PoFaPo_,0.0,gs_nalphapi_,gs_nalphapi_);

    // Form the projector 1 - Ph = 1 - (Ch^T S Ca_gs)^T Ch^T S Ca_gs
    TempMatrix->zero();
    TempMatrix->gemm(false,true,1.0,saved_Ch_,saved_Ch_,0.0);
    TempMatrix->transform(S_);
    TempMatrix->transform(dets[0]->Ca());
    SharedMatrix Ph = SharedMatrix(new Matrix("Ph",gs_nalphapi_,gs_nalphapi_));
    Ph->identity();
    copy_block(TempMatrix,-1.0,Ph,1.0,gs_nalphapi_,gs_nalphapi_);

    if(do_project_out_holes){
        // Project out the previous holes
        PoFaPo_->transform(Ph);
        fprintf(outfile,"  Projecting out the previous holes\n");
    }

    // Diagonalize the occ block
    PoFaPo_->diagonalize(Ua_o_,lambda_a_o_);
}

void UCKS::compute_particles()
{
    if(iteration_ > 1){
        // Form the projector Ca Ca^T S Ca_gs
        TempMatrix->zero();
        // Copy Cp
        copy_block(Cp_,1.0,TempMatrix,0.0,nsopi_,napartpi_);
        // Copy the virtual block of Ca
        copy_block(Ca_,1.0,TempMatrix,0.0,nsopi_,nmopi_ - nalphapi_,zero_dim_,nalphapi_,zero_dim_,napartpi_);

        TempMatrix2->gemm(false,true,1.0,TempMatrix,TempMatrix,0.0);
        TempMatrix->gemm(false,false,1.0,TempMatrix2,S_,0.0);
        TempMatrix2->gemm(false,false,1.0,TempMatrix,dets[0]->Ca(),0.0);

        TempMatrix->transform(Fa_,TempMatrix2);
    }else{
        // Transform Fa to the MO basis of the ground state
        TempMatrix->transform(Fa_,dets[0]->Ca());
    }

    // Grab the vir block of Fa
    copy_block(TempMatrix,1.0,PvFaPv_,0.0,gs_navirpi_,gs_navirpi_,gs_nalphapi_,gs_nalphapi_);

    // Form the projector Pp = 1 - (Cp^T S Ca_gs)^T Cp^T S Ca_gs
    TempMatrix->zero();
    TempMatrix->gemm(false,true,1.0,saved_Cp_,saved_Cp_,0.0);
    TempMatrix->transform(S_);
    TempMatrix->transform(dets[0]->Ca());
    SharedMatrix Pp = SharedMatrix(new Matrix("Pp",gs_navirpi_,gs_navirpi_));
    Pp->identity();
    copy_block(TempMatrix,-1.0,Pp,1.0,gs_navirpi_,gs_navirpi_,gs_nalphapi_,gs_nalphapi_);

    if (do_project_out_particles){
        // Project out the previous particles
        PvFaPv_->transform(Pp);
        fprintf(outfile,"  Projecting out the previous particles\n");
    }

    // Diagonalize the vir block
    PvFaPv_->diagonalize(Ua_v_,lambda_a_v_);
}

void UCKS::find_ee_occupation(SharedVector lambda_o,SharedVector lambda_v)
{
    // Find the hole/particle pair to follow
    boost::tuple<double,int,int> hole;
    boost::tuple<double,int,int> particle;
    std::vector<boost::tuple<double,int,int,double,int,int,double> > sorted_hp_pairs;

    // If we are doing core excitation just take the negative of the hole energy
    bool do_core_excitation = false;
    if(KS::options_.get_str("CDFT_EXC_TYPE") == "CORE"){
        do_core_excitation = true;
    }

    // Compute the symmetry adapted hole/particle pairs
    for (int occ_h = 0; occ_h < nirrep_; ++occ_h){
        int nocc = gs_nalphapi_[occ_h];
        for (int i = 0; i < nocc; ++i){
            double e_h = lambda_o->get(occ_h,i);
            for (int vir_h = 0; vir_h < nirrep_; ++vir_h){
                int nvir = gs_navirpi_[vir_h];
                for (int a = 0; a < nvir; ++a){
                    double e_p = lambda_v->get(vir_h,a);
                    double e_hp = do_core_excitation ? (e_p + e_h) : (e_p - e_h);
                    int symm = occ_h ^ vir_h ^ ground_state_symmetry_;
                    if(not do_symmetry or (symm == excited_state_symmetry_)){ // Test for symmetry
                        sorted_hp_pairs.push_back(boost::make_tuple(e_hp,occ_h,i,e_h,vir_h,a,e_p));  // N.B. shifted wrt to full indexing
                    }
                }
            }
        }
    }

    // Sort the hole/particle pairs according to the energy
    std::sort(sorted_hp_pairs.begin(),sorted_hp_pairs.end());

    CharacterTable ct = KS::molecule_->point_group()->char_table();
    if(iteration_ == 0){
        fprintf(outfile, "\n  Ground state symmetry: %s\n",ct.gamma(ground_state_symmetry_).symbol());
        fprintf(outfile, "  Excited state symmetry: %s\n",ct.gamma(excited_state_symmetry_).symbol());
        fprintf(outfile, "\n  Lowest energy excitations:\n");
        fprintf(outfile, "  --------------------------------------\n");
        fprintf(outfile, "    N   Occupied     Virtual     E(eV)  \n");
        fprintf(outfile, "  --------------------------------------\n");
        int maxstates = std::min(10,static_cast<int>(sorted_hp_pairs.size()));
        for (int n = 0; n < maxstates; ++n){
            double energy_hp = sorted_hp_pairs[n].get<6>() - sorted_hp_pairs[n].get<3>();
            fprintf(outfile,"   %2d:  %4d%-3s  -> %4d%-3s   %9.3f\n",n + 1,
                    sorted_hp_pairs[n].get<2>() + 1,
                    ct.gamma(sorted_hp_pairs[n].get<1>()).symbol(),
                    gs_nalphapi_[sorted_hp_pairs[n].get<4>()] + sorted_hp_pairs[n].get<5>() + 1,
                    ct.gamma(sorted_hp_pairs[n].get<4>()).symbol(),
                    energy_hp * pc_hartree2ev);
        }
        fprintf(outfile, "  --------------------------------------\n");

        int select_pair = 0;
        // Select the excitation pair using the energetic ordering
        if(KS::options_["CDFT_EXC_SELECT"].has_changed()){
            int input_select = KS::options_["CDFT_EXC_SELECT"][excited_state_symmetry_].to_integer();
            if (input_select > 0){
                select_pair = input_select - 1;
                fprintf(outfile, "\n  Following excitation #%d: ",input_select);
            }
        }
        // Select the excitation pair using the symmetry of the hole
        if(KS::options_["CDFT_EXC_HOLE_SYMMETRY"].has_changed()){
            int input_select = KS::options_["CDFT_EXC_HOLE_SYMMETRY"][excited_state_symmetry_].to_integer();
            if (input_select > 0){
                int maxstates = static_cast<int>(sorted_hp_pairs.size());
                for (int n = 0; n < maxstates; ++n){
                    if(sorted_hp_pairs[n].get<1>() == input_select - 1){
                        select_pair = n;
                        break;
                    }
                }
                fprintf(outfile, "\n  Following excitation #%d:\n",select_pair + 1);
            }
        }
        aholes.clear();
        aparts.clear();

        int ahole_h = sorted_hp_pairs[select_pair].get<1>();
        int ahole_mo = sorted_hp_pairs[select_pair].get<2>();
        double ahole_energy = sorted_hp_pairs[select_pair].get<3>();
        boost::tuple<int,int,double> ahole = boost::make_tuple(ahole_h,ahole_mo,ahole_energy);
        aholes.push_back(ahole);

        int apart_h = sorted_hp_pairs[select_pair].get<4>();
        int apart_mo = sorted_hp_pairs[select_pair].get<5>();
        double apart_energy = sorted_hp_pairs[select_pair].get<6>();
        boost::tuple<int,int,double> apart = boost::make_tuple(apart_h,apart_mo,apart_energy);
        aparts.push_back(apart);
    }else{
        if(not (KS::options_["CDFT_EXC_SELECT"].has_changed() or
                KS::options_["CDFT_EXC_HOLE_SYMMETRY"].has_changed())){
            aholes.clear();
            aparts.clear();

            int ahole_h = sorted_hp_pairs[0].get<1>();
            int ahole_mo = sorted_hp_pairs[0].get<2>();
            double ahole_energy = sorted_hp_pairs[0].get<3>();
            boost::tuple<int,int,double> ahole = boost::make_tuple(ahole_h,ahole_mo,ahole_energy);
            aholes.push_back(ahole);

            int apart_h = sorted_hp_pairs[0].get<4>();
            int apart_mo = sorted_hp_pairs[0].get<5>();
            double apart_energy = sorted_hp_pairs[0].get<6>();
            boost::tuple<int,int,double> apart = boost::make_tuple(apart_h,apart_mo,apart_energy);
            aparts.push_back(apart);
        }
    }
    fflush(outfile);

    for (int h = 0; h < nirrep_; ++h){
        naholepi_[h] = 0;
        napartpi_[h] = 0;
    }

    // Compute the number of hole and/or particle orbitals to compute
    fprintf(outfile,"\n  HOLES:     ");
    size_t naholes = aholes.size();
    for (size_t n = 0; n < naholes; ++n){
        naholepi_[aholes[n].get<0>()] += 1;
        fprintf(outfile,"%4d%-3s (%+.6f)",
                aholes[n].get<1>() + 1,
                ct.gamma(aholes[n].get<0>()).symbol(),
                aholes[n].get<2>());
    }
    fprintf(outfile,"\n  PARTICLES: ");
    size_t naparts = aparts.size();
    for (size_t n = 0; n < naparts; ++n){
        napartpi_[aparts[n].get<0>()] += 1;
        fprintf(outfile,"%4d%-3s (%+.6f)",
                gs_nalphapi_[aparts[n].get<0>()] + aparts[n].get<1>() + 1,
                ct.gamma(aparts[n].get<0>()).symbol(),
                aparts[n].get<2>());
    }
    fprintf(outfile,"\n\n");
}

void UCKS::compute_hole_particle_mos()
{
    SharedMatrix Ca0 = dets[0]->Ca();

    Ch_->zero();
    Cp_->zero();
    // Compute the hole orbitals
    size_t naholes = aholes.size();
    std::vector<int> hoffset(nirrep_,0);
    for (size_t n = 0; n < naholes; ++n){
        int ahole_h = aholes[n].get<0>();
        int ahole_mo = aholes[n].get<1>();
        int nhole = hoffset[ahole_h];
        int maxi = gs_nalphapi_[ahole_h];
        for (int p = 0; p < nsopi_[ahole_h]; ++p){
            double c_h = 0.0;
            for (int i = 0; i < maxi; ++i){
                c_h += Ca0->get(ahole_h,p,i) * Ua_o_->get(ahole_h,i,ahole_mo);
            }
            Ch_->set(ahole_h,p,nhole,c_h);
        }
        hoffset[ahole_h] += 1;
    }

    // Compute the particle orbital
    size_t naparts = aparts.size();
    std::vector<int> poffset(nirrep_,0);
    for (size_t n = 0; n < naparts; ++n){
        int apart_h = aparts[n].get<0>();
        int apart_mo = aparts[n].get<1>();
        int npart = poffset[apart_h];
        int maxa = gs_navirpi_[apart_h];
        for (int p = 0; p < nsopi_[apart_h]; ++p){
            double c_p = 0.0;
            for (int a = 0; a < maxa; ++a){
                c_p += Ca0->get(apart_h,p,gs_nalphapi_[apart_h] + a) * Ua_v_->get(apart_h,a,apart_mo) ;
            }
            Cp_->set(apart_h,p,npart,c_p);
        }
        poffset[apart_h] += 1;
    }
}

void UCKS::diagonalize_F_spectator_relaxed()
{
    // Form the projector onto the orbitals orthogonal to the holes and particles in the excited state mo representation
    TempMatrix->zero();
    // Project the hole, the particles, or both depending on the method
    if(do_holes){
        TempMatrix->gemm(false,true,1.0,Ch_,Ch_,1.0);
        if(do_project_out_holes){
            TempMatrix->gemm(false,true,1.0,saved_Ch_,saved_Ch_,1.0);
        }
    }
    if(do_parts){
        TempMatrix->gemm(false,true,1.0,Cp_,Cp_,1.0);
        if(do_project_out_particles){
            TempMatrix->gemm(false,true,1.0,saved_Cp_,saved_Cp_,1.0);
        }
    }
    TempMatrix->transform(S_);
    TempMatrix->transform(dets[0]->Ca());
    TempMatrix2->identity();
    TempMatrix2->subtract(TempMatrix);

    // Form the Fock matrix in the excited state basis, project out the h/p
    TempMatrix->transform(Fa_,dets[0]->Ca());
    TempMatrix->transform(TempMatrix2);

    // Diagonalize the Fock matrix and transform the MO coefficients
    TempMatrix->diagonalize(TempMatrix2,epsilon_a_);
    TempMatrix->zero();
    TempMatrix->gemm(false,false,1.0,dets[0]->Ca(),TempMatrix2,0.0);
    Ca_->copy(TempMatrix);
}

void UCKS::sort_ee_mos()
{
    // Set the occupation
    nalphapi_ = gs_nalphapi_ + napartpi_ - naholepi_;
    nbetapi_  = gs_nbetapi_;

    int old_socc[8];
    int old_docc[8];
    for(int h = 0; h < nirrep_; ++h){
        old_socc[h] = soccpi_[h];
        old_docc[h] = doccpi_[h];
    }

    for (int h = 0; h < nirrep_; ++h) {
        soccpi_[h] = std::abs(nalphapi_[h] - nbetapi_[h]);
        doccpi_[h] = std::min(nalphapi_[h] , nbetapi_[h]);
    }

    bool occ_changed = false;
    for(int h = 0; h < nirrep_; ++h){
        if( old_socc[h] != soccpi_[h] || old_docc[h] != doccpi_[h]){
            occ_changed = true;
            break;
        }
    }

    // At this point the orbitals are sorted according to the energy but we
    // want to make sure that the hole and particle MO appear where they should, that is
    // |(particles) (occupied spectators) | (virtual spectators) (hole)>
    TempMatrix->zero();
    TempVector->zero();
    for (int h = 0; h < nirrep_; ++h){
        int nso = nsopi_[h];
        int nmo = nmopi_[h];
        if (nso == 0 or nmo == 0)
            continue;
        double** T_h = TempMatrix->pointer(h);
        double** C_h = Ca_->pointer(h);
        double** Cp_h = Cp_->pointer(h);
        double** Ch_h = Ch_->pointer(h);
        double** saved_Cp_h = saved_Cp_->pointer(h);
        double** saved_Ch_h = saved_Ch_->pointer(h);

        int m = 0;
        // First place the particles
        if(do_project_out_holes){
            for (int p = 0; p < saved_naholepi_[h]; ++p){
                for (int q = 0; q < nso; ++q){
                    T_h[q][m] = saved_Ch_h[q][p];
                }
                m += 1;
            }
        }
        if(do_parts){
            for (int p = 0; p < napartpi_[h]; ++p){
                for (int q = 0; q < nso; ++q){
                    T_h[q][m] = Cp_h[q][p];
                }
                m += 1;
            }
        }
        // Then the spectators
        for (int p = 0; p < nmo; ++p){
            // Is this MO a hole or a particle?
            if(std::fabs(epsilon_a_->get(h,p)) > 1.0e-6){
                TempVector->set(h,m,epsilon_a_->get(h,p));
                for (int q = 0; q < nso; ++q){
                    T_h[q][m] = C_h[q][p];
                }
                m += 1;
            }
        }
        // Then the holes
        if(do_holes){
            for (int p = 0; p < naholepi_[h]; ++p){
                for (int q = 0; q < nso; ++q){
                    T_h[q][m] = Ch_h[q][p];
                }
                m += 1;
            }
        }
    }
    Ca_->copy(TempMatrix);
    epsilon_a_->copy(TempVector.get());
}

void UCKS::diagonalize_F_spectator_unrelaxed()
{
//    // Frozen spectator orbital algorithm
//    // Transform the ground state orbitals to the representation which diagonalizes the
//    // the PoFaPo and PvFaPv blocks
//    // Form the transformation matrix that diagonalizes the PoFaPo and PvFaPv blocks
//    // |----|----|
//    // | Uo | 0  |
//    // |----|----|
//    // | 0  | Uv |
//    // |----|----|
//    TempMatrix->zero();
//    for (int h = 0; h < nirrep_; ++h){
//        int nocc = dets[0]->nalphapi()[h];
//        int nvir = nmopi_[h] - nocc;
//        if (nocc != 0){
//            double** Temp_h = TempMatrix->pointer(h);
//            double** Uo_h = Ua_o_->pointer(h);
//            for (int i = 0; i < nocc; ++i){
//                epsilon_a_->set(h,i,lambda_a_o_->get(h,i));
//                for (int j = 0; j < nocc; ++j){
//                    Temp_h[i][j] = Uo_h[i][j];
//                }
//            }
//        }
//        if (nvir != 0){
//            double** Temp_h = TempMatrix->pointer(h);
//            double** Uv_h = Ua_v_->pointer(h);
//            for (int i = 0; i < nvir; ++i){
//                epsilon_a_->set(h,i + nocc,lambda_a_v_->get(h,i));
//                for (int j = 0; j < nvir; ++j){
//                    Temp_h[i + nocc][j + nocc] = Uv_h[i][j];
//                }
//            }
//        }
//    }
//    // Get the excited state orbitals: Ca(ex) = Ca(gs) * (Uo | Uv)
//    Ca_->gemm(false,false,1.0,dets[0]->Ca(),TempMatrix,0.0);

//    // Form the projector onto the orbitals orthogonal to the holes and particles in the excited state mo representation
//    TempMatrix->zero();
//    TempMatrix->gemm(false,true,1.0,Ch_,Ch_,0.0);
//    TempMatrix->gemm(false,true,1.0,Cp_,Cp_,1.0);
//    TempMatrix->transform(S_);
//    TempMatrix->transform(Ca_);
//    TempMatrix2->identity();
//    TempMatrix2->subtract(TempMatrix);

//    // Form the Fock matrix in the excited state basis, project out the h/p
//    TempMatrix->transform(Fa_,Ca_);
//    TempMatrix->transform(TempMatrix2);
//    // If we want the relaxed orbitals diagonalize the Fock matrix and transform the MO coefficients
//    if(KS::options_.get_str("CDFT_EXC_METHOD") == "CHP" or KS::options_.get_str("CDFT_EXC_METHOD") == "CHP-FB"){
//        TempMatrix->diagonalize(TempMatrix2,epsilon_a_);
//        TempMatrix->zero();
//        TempMatrix->gemm(false,false,1.0,Ca_,TempMatrix2,0.0);
//        Ca_->copy(TempMatrix);
//    }else{
//        // The orbitals don't change, but make sure that epsilon_a_ has the correct eigenvalues (some which are zero)
//        for (int h = 0; h < nirrep_; ++h){
//            for (int p = 0; p < nmopi_[h]; ++p){
//                epsilon_a_->set(h,p,TempMatrix->get(h,p,p));
//            }
//        }
//    }

//    std::vector<boost::tuple<double,int,int> > sorted_spectators;
//    for (int h = 0; h < nirrep_; ++h){
//        for (int p = 0; p < nmopi_[h]; ++p){
//            sorted_spectators.push_back(boost::make_tuple(epsilon_a_->get(h,p),h,p));
//        }
//    }
//    std::sort(sorted_spectators.begin(),sorted_spectators.end());

//    // Find the alpha occupation
//    int assigned = 0;
//    for (int h = 0; h < nirrep_; ++h){
//        nalphapi_[h] = apartpi[h];
//        assigned += apartpi[h];
//    }
//    for (int p = 0; p < nmo_; ++p){
//        if (assigned < nalpha_){
//            if(std::fabs(sorted_spectators[p].get<0>()) > 1.0e-6){  // !!! Check this out NB WARNING
//                int h = sorted_spectators[p].get<1>();
//                nalphapi_[h] += 1;
//                assigned += 1;
//            }
//        }
//    }
//    nbetapi_ = dets[0]->nbetapi();
//    int old_socc[8];
//    int old_docc[8];
//    for(int h = 0; h < nirrep_; ++h){
//        old_socc[h] = soccpi_[h];
//        old_docc[h] = doccpi_[h];
//    }

//    for (int h = 0; h < nirrep_; ++h) {
//        soccpi_[h] = std::abs(nalphapi_[h] - nbetapi_[h]);
//        doccpi_[h] = std::min(nalphapi_[h] , nbetapi_[h]);
//    }

//    bool occ_changed = false;
//    for(int h = 0; h < nirrep_; ++h){
//        if( old_socc[h] != soccpi_[h] || old_docc[h] != doccpi_[h]){
//            occ_changed = true;
//            break;
//        }
//    }

//    // At this point the orbitals are sorted according to the energy but we
//    // want to make sure that the hole and particle MO appear where they should, that is
//    // |(particles) (occupied spectators) | (virtual spectators) (hole)>
//    TempMatrix->zero();
//    TempVector->zero();
//    for (int h = 0; h < nirrep_; ++h){
//        int nso = nsopi_[h];
//        int nmo = nmopi_[h];
//        double** T_h = TempMatrix->pointer(h);
//        double** C_h = Ca_->pointer(h);
//        double** Cp_h = Cp_->pointer(h);
//        double** Ch_h = Ch_->pointer(h);
//        // First place the particles
//        int m = 0;
//        for (int p = 0; p < apartpi[h]; ++p){
//            for (int q = 0; q < nso; ++q){
//                T_h[q][m] = Cp_h[q][p];
//            }
//            m += 1;
//        }
//        // Then the spectators
//        for (int p = 0; p < nmo; ++p){
//            // Is this MO a hole or a particle?
//            if(std::fabs(epsilon_a_->get(h,p)) > 1.0e-6){
//                TempVector->set(h,m,epsilon_a_->get(h,p));
//                for (int q = 0; q < nso; ++q){
//                    T_h[q][m] = C_h[q][p];
//                }
//                m += 1;
//            }
//        }
//        // Then the holes
//        for (int p = 0; p < aholepi[h]; ++p){
//            for (int q = 0; q < nso; ++q){
//                T_h[q][m] = Ch_h[q][p];
//            }
//            m += 1;
//        }
//    }
//    Ca_->copy(TempMatrix);
//    epsilon_a_->copy(TempVector.get());

//    // BETA
//    if(KS::options_.get_str("CDFT_EXC_METHOD") == "CHP"){
//        diagonalize_F(Fb_, Cb_, epsilon_b_);
//    }else{
//        // Unrelaxed procedure, but still find MOs which diagonalize the occupied block
//        // Transform Fb to the MO basis of the ground state
//        TempMatrix->transform(Fb_,dets[0]->Cb());
//        // Grab the occ block of Fb
//        extract_square_subblock(TempMatrix,PoFaPo_,true,dets[0]->nbetapi(),1.0e9);
//        // Grab the vir block of Fa
//        extract_square_subblock(TempMatrix,PvFaPv_,false,dets[0]->nbetapi(),1.0e9);
//        // Diagonalize the hole block
//        PoFaPo_->diagonalize(Ua_o_,lambda_a_o_);
//        // Diagonalize the particle block
//        PvFaPv_->diagonalize(Ua_v_,lambda_a_v_);
//        // Form the transformation matrix that diagonalizes the PoFaPo and PvFaPv blocks
//        // |----|----|
//        // | Uo | 0  |
//        // |----|----|
//        // | 0  | Uv |
//        // |----|----|
//        TempMatrix->zero();
//        for (int h = 0; h < nirrep_; ++h){
//            int nocc = dets[0]->nbetapi()[h];
//            int nvir = nmopi_[h] - nocc;
//            if (nocc != 0){
//                double** Temp_h = TempMatrix->pointer(h);
//                double** Uo_h = Ua_o_->pointer(h);
//                for (int i = 0; i < nocc; ++i){
//                    epsilon_b_->set(h,i,lambda_a_o_->get(h,i));
//                    for (int j = 0; j < nocc; ++j){
//                        Temp_h[i][j] = Uo_h[i][j];
//                    }
//                }
//            }
//            if (nvir != 0){
//                double** Temp_h = TempMatrix->pointer(h);
//                double** Uv_h = Ua_v_->pointer(h);
//                for (int i = 0; i < nvir; ++i){
//                    epsilon_b_->set(h,i + nocc,lambda_a_v_->get(h,i));
//                    for (int j = 0; j < nvir; ++j){
//                        Temp_h[i + nocc][j + nocc] = Uv_h[i][j];
//                    }
//                }
//            }
//        }
//        // Get the excited state orbitals: Cb(ex) = Cb(gs) * (Uo | Uv)
//        Cb_->gemm(false,false,1.0,dets[0]->Cb(),TempMatrix,0.0);
//    }
//    if (debug_) {
//        Ca_->print(outfile);
//        Cb_->print(outfile);
//    }
}


void UCKS::form_C_beta()
{
    // BETA
    if(KS::options_.get_str("CDFT_EXC_METHOD") != "CHP-FB"){
        diagonalize_F(Fb_, Cb_, epsilon_b_);
    }else{
        fprintf(outfile,"\n  Frozen beta algorithm\n");
        if(! PoFbPo_){
            PoFbPo_ = SharedMatrix(new Matrix("PoFbPo",gs_nbetapi_,gs_nbetapi_));
            PvFbPv_ = SharedMatrix(new Matrix("PvFbPo",gs_nbvirpi_,gs_nbvirpi_));
            Ub_o_ = SharedMatrix(new Matrix("Ub_o_",gs_nbetapi_,gs_nbetapi_));
            Ub_v_ = SharedMatrix(new Matrix("Ub_v_",gs_nbvirpi_,gs_nbvirpi_));
            lambda_b_o_ = SharedVector(new Vector("lambda_b_o_",gs_nbetapi_));
            lambda_b_v_ = SharedVector(new Vector("lambda_b_v_",gs_nbvirpi_));
            fprintf(outfile,"\n  Allocated beta matrices!!!\n");
        }

        // Unrelaxed procedure, but still find MOs which diagonalize the occupied block
        // Transform Fb to the MO basis of the ground state
        TempMatrix->transform(Fb_,dets[0]->Cb());

        // Grab the occ block of Fb
        copy_block(TempMatrix,1.0,PoFbPo_,0.0,gs_nbetapi_,gs_nbetapi_);

        // Diagonalize the occ block
        PoFbPo_->diagonalize(Ub_o_,lambda_b_o_);

        // Grab the vir block of Fb
        copy_block(TempMatrix,1.0,PvFbPv_,0.0,gs_nbvirpi_,gs_nbvirpi_,gs_nbetapi_,gs_nbetapi_);

        // Diagonalize the vir block
        PvFbPv_->diagonalize(Ub_v_,lambda_b_v_);

        TempMatrix->zero();
        copy_block(Ub_o_,1.0,TempMatrix,0.0,gs_nbetapi_,gs_nbetapi_);
        copy_block(Ub_v_,1.0,TempMatrix,0.0,gs_nbvirpi_,gs_nbvirpi_,Dimension(nirrep_),Dimension(nirrep_),gs_nbetapi_,gs_nbetapi_);

        // Get the excited state orbitals: Cb(ex) = Cb(gs) * (Uo | Uv)
        Cb_->gemm(false,false,1.0,dets[0]->Cb(),TempMatrix,0.0);
    }
}

/// Gradient of W
///
/// Implements Eq. (6) of Phys. Rev. A 72, 024502 (2005) and estimates a correction to the
/// gradient from the orbital response
void UCKS::gradient_of_W()
{
    for (int c = 0; c < nconstraints; ++c){
        double grad = 0.0;
        grad  = constraints[c]->weight_alpha() * Da_->vector_dot(constraints[c]->W_so());
        grad += constraints[c]->weight_beta()  * Db_->vector_dot(constraints[c]->W_so());
        grad -= constraints[c]->Nc();
        gradW->set(c,grad);
    }
    for (int c = 0; c < nconstraints; ++c){
        double grad = 0.0;
        // Transform W_so to the MO basis (alpha)
        TempMatrix->transform(constraints[c]->W_so(),Ca_);
        TempMatrix->scale(constraints[c]->weight_alpha());
        // Transform Fa_ to the MO basis
        TempMatrix2->transform(Fa_,Ca_);
        for (int h = 0; h < nirrep_; h++) {
            int nmo  = nmopi_[h];
            int nocc = nalphapi_[h];
            int nvir = nmo - nocc;
            if (nvir == 0 or nocc == 0) continue;
            double** Temp_h = TempMatrix->pointer(h);
            double** Temp2_h = TempMatrix2->pointer(h);
            for (int i = 0; i < nocc; ++i){
                for (int a = nocc; a < nmo; ++a){
                    grad += 2.0 * Temp_h[a][i] * Temp2_h[a][i] / (Temp2_h[i][i] - Temp2_h[a][a]);
                }
            }
        }
        // Transform W_so to the MO basis (beta)
        TempMatrix->transform(constraints[c]->W_so(),Cb_);
        TempMatrix->scale(constraints[c]->weight_beta());
        // Transform Fb_ to the MO basis
        TempMatrix2->transform(Fb_,Cb_);
        for (int h = 0; h < nirrep_; h++) {
            int nmo  = nmopi_[h];
            int nocc = nbetapi_[h];
            int nvir = nmo - nocc;
            if (nvir == 0 or nocc == 0) continue;
            double** Temp_h = TempMatrix->pointer(h);
            double** Temp2_h = TempMatrix2->pointer(h);
            for (int i = 0; i < nocc; ++i){
                for (int a = nocc; a < nmo; ++a){
                    grad += 2.0 * Temp_h[a][i] * Temp2_h[a][i] / (Temp2_h[i][i] - Temp2_h[a][a]);
                }
            }
        }
        gradW_mo_resp->set(c,grad);
    }
    if(nconstraints > 0){
        for (int c = 0; c < nconstraints; ++c){
            fprintf(outfile,"   %-10s: grad = %10.7f    grad (resp) = %10.7f    Vc = %10.7f\n",constraints[c]->type().c_str(),
                    gradW->get(c),gradW_mo_resp->get(c),Vc->get(c));
        }
    }
}

/// Hessian of W
///
/// Implements Eq. (7) of Phys. Rev. A 72, 024502 (2005).
void UCKS::hessian_of_W()
{
    fprintf(outfile,"\n  COMPUTE THE HESSIAN\n\n");
    for (int c1 = 0; c1 < nconstraints; ++c1){
        for (int c2 = 0; c2 <= c1; ++c2){
            double hess = 0.0;
            // Transform W_so to the MO basis (alpha)
            TempMatrix->transform(constraints[c1]->W_so(),Ca_);
            TempMatrix->scale(constraints[c1]->weight_alpha());
            // Transform W_so to the MO basis (alpha)
            TempMatrix2->transform(constraints[c2]->W_so(),Ca_);
            TempMatrix2->scale(constraints[c2]->weight_alpha());
            for (int h = 0; h < nirrep_; h++) {
                int nmo  = nmopi_[h];
                int nocc = nalphapi_[h];
                int nvir = nmo - nocc;
                if (nvir == 0 or nocc == 0) continue;
                double** Temp_h = TempMatrix->pointer(h);
                double** Temp2_h = TempMatrix2->pointer(h);
                double* eps = epsilon_a_->pointer(h);
                for (int i = 0; i < nocc; ++i){
                    for (int a = nocc; a < nmo; ++a){
                        hess += 2.0 * Temp_h[i][a] * Temp2_h[a][i] /  (eps[i] - eps[a]);
                    }
                }
            }
            // Transform W_so to the MO basis (beta)
            TempMatrix->transform(constraints[c1]->W_so(),Cb_);
            TempMatrix->scale(constraints[c1]->weight_beta());
            // Transform W_so to the MO basis (beta)
            TempMatrix2->transform(constraints[c2]->W_so(),Cb_);
            TempMatrix2->scale(constraints[c2]->weight_beta());
            for (int h = 0; h < nirrep_; h++) {
                int nmo  = nmopi_[h];
                int nocc = nbetapi_[h];
                int nvir = nmo - nocc;
                if (nvir == 0 or nocc == 0) continue;
                double** Temp_h = TempMatrix->pointer(h);
                double** Temp2_h = TempMatrix2->pointer(h);
                double* eps = epsilon_b_->pointer(h);
                for (int i = 0; i < nocc; ++i){
                    for (int a = nocc; a < nmo; ++a){
                        hess += 2.0 * Temp_h[i][a] * Temp2_h[a][i] /  (eps[i] - eps[a]);
                    }
                }
            }
            hessW->set(c1,c2,hess);
            hessW->set(c2,c1,hess);
        }
    }
}

/// Apply the BFGS update to the approximate Hessian h[][].
/// h[][] = Hessian matrix from previous iteration
/// dx[]  = Step from previous iteration (dx[] = x[] - xp[] where xp[] is docc previous point)
/// dg[]  = gradient difference (dg = g - gp)
void UCKS::hessian_update(SharedMatrix h, SharedVector dx, SharedVector dg)
{
    SharedVector hdx = SharedVector(new Vector("hdx",nconstraints));
    hdx->gemv(false, 1.0, h.get(), dx.get(), 0.0);
    double dxhdx = dx->dot(hdx.get());
    double dxdx  = dx->dot(dx.get());
    double dxdg  = dx->dot(dg.get());
    double dgdg  = dg->dot(dg.get());

    if ( (dxdx > 0.0) && (dgdg > 0.0) && (std::fabs(dxdg / std::sqrt(dxdx * dgdg)) > 1.e-8) ) {
        for (int i = 0; i < nconstraints; ++i) {
            for (int j = 0; j < nconstraints; ++j) {
                h->add(i,j,dg->get(i) * dg->get(j) / dxdg - hdx->get(i) * hdx->get(j) / dxhdx);
            }
        }
    }
    else {
        fprintf(outfile,"  BFGS not updating dxdg (%e), dgdg (%e), dxhdx (%f), dxdx(%e)\n" , dxdg, dgdg, dxhdx, dxdx);
    }
}

/// Optimize the Lagrange multiplier
void UCKS::constraint_optimization()
{
    if(KS::options_.get_str("W_ALGORITHM") == "NEWTON"){
        // Optimize Vc once you have a good gradient and the gradient is not converged
        double max_error = 0.0;
        for (int c = 0; c < nconstraints; ++c){
            if(std::fabs(gradW->get(c)) > gradW_threshold_){
                max_error = std::max(max_error,std::fabs(gradW_mo_resp->get(c) / gradW->get(c)));
            }
        }
        // Make sure the component with the largest error is within 10% of the estimate
        if (max_error < 0.1 and gradW->norm() > gradW_threshold_){
            fprintf(outfile, "  ==> Constraint optimization <==\n");
            if(nW_opt > 0){
                SharedVector dVc = SharedVector(new Vector("dVc",nconstraints));
                dVc->copy(Vc.get());
                dVc->subtract(Vc_old);
                SharedVector dgradW = SharedVector(new Vector("dgradW",nconstraints));
                dgradW->copy(gradW.get());
                dgradW->subtract(gradW_old);
                hessian_update(hessW_BFGS, dVc, dgradW);
                fprintf(outfile, "  Hessian update.\n");
            }else{
                hessian_of_W();
                hessW_BFGS->copy(hessW);
            }
            hessW_BFGS->print();
            hessW->print();
            // First step, do a Newton with the hessW information
            Vc_old->copy(Vc.get());
            gradW_old->copy(gradW.get());
            // Use the Hessian to do a Newton step with trust radius
            bool warning;
            SharedMatrix hessW_inv = hessW_BFGS->pseudoinverse(1.0E-10, &warning);
            if (warning) {
                fprintf(outfile, "  Warning, the inverse Hessian had to be conditioned.\n\n");
            }
            SharedVector h_inv_g = SharedVector(new Vector("h_inv_g",nconstraints));
            h_inv_g->gemv(false, 1.0, hessW_inv.get(), gradW.get(), 0.0);

            // Apply trust radius
            double step_size = h_inv_g->norm();
            double threshold = 0.5;
            if(step_size > threshold){
                h_inv_g->scale(threshold / step_size);
            }
            Vc->subtract(h_inv_g);

            // Reset the DIIS subspace
            diis_manager_->reset_subspace();
            nW_opt += 1;
        }
    }
}

double UCKS::compute_E()
{
    // E_CDFT = 2.0 D*H + D*J - \alpha D*K + E_xc - Nc * Vc
    double one_electron_E = Da_->vector_dot(H_);
    one_electron_E += Db_->vector_dot(H_);
    for (int c = 0; c < nconstraints; ++c){
        one_electron_E -= Vc->get(c) * constraints[c]->Nc(); // Added the CDFT contribution that is not included in H_
    }
    double coulomb_E = Da_->vector_dot(J_);
    coulomb_E += Db_->vector_dot(J_);

    std::map<std::string, double>& quad = potential_->quadrature_values();
    double XC_E = quad["FUNCTIONAL"];
    double exchange_E = 0.0;
    double alpha = functional_->x_alpha();
    double beta = 1.0 - alpha;
    if (functional_->is_x_hybrid()) {
        exchange_E -= alpha*Da_->vector_dot(Ka_);
        exchange_E -= alpha*Db_->vector_dot(Kb_);
    }

    std::string functional_prefix = functional_->name().substr(0,2);
    if (functional_prefix == "sr"){
        exchange_E +=  alpha * Da_->vector_dot(wKa_);
        exchange_E +=  alpha * Db_->vector_dot(wKb_);
    }else{
        if (functional_->is_x_lrc()) {
            exchange_E -=  beta*Da_->vector_dot(wKa_);
            exchange_E -=  beta*Db_->vector_dot(wKb_);
        }
    }

    double dashD_E = 0.0;
    boost::shared_ptr<Dispersion> disp;
    if (disp) {
        dashD_E = disp->compute_energy(HF::molecule_);
    }

    double Etotal = 0.0;
    Etotal += nuclearrep_;
    Etotal += one_electron_E;
    Etotal += 0.5 * coulomb_E;
    Etotal += 0.5 * exchange_E;
    Etotal += XC_E;
    Etotal += dashD_E;

    if (debug_) {
        fprintf(outfile, "   => Energetics <=\n\n");
        fprintf(outfile, "    Nuclear Repulsion Energy = %24.14f\n", nuclearrep_);
        fprintf(outfile, "    One-Electron Energy =      %24.14f\n", one_electron_E);
        fprintf(outfile, "    Coulomb Energy =           %24.14f\n", 0.5 * coulomb_E);
        fprintf(outfile, "    Hybrid Exchange Energy =   %24.14f\n", 0.5 * exchange_E);
        fprintf(outfile, "    XC Functional Energy =     %24.14f\n", XC_E);
        fprintf(outfile, "    -D Energy =                %24.14f\n", dashD_E);
    }
    return Etotal;
}

void UCKS::damp_update()
{
    // Turn on damping only for excited state computations
    if(do_excitation){
        double damping = damping_percentage_;
        for(int h = 0; h < nirrep_; ++h){
            for(int row = 0; row < Da_->rowspi(h); ++row){
                for(int col = 0; col < Da_->colspi(h); ++col){
                    double Dolda = damping * Dolda_->get(h, row, col);
                    double Dnewa = (1.0 - damping) * Da_->get(h, row, col);
                    Da_->set(h, row, col, Dolda+Dnewa);
                    double Doldb = damping * Doldb_->get(h, row, col);
                    double Dnewb = (1.0 - damping) * Db_->get(h, row, col);
                    Db_->set(h, row, col, Doldb+Dnewb);
                }
            }
        }
        // Update Dt_
        Dt_->copy(Da_);
        Dt_->add(Db_);
    }
}

bool UCKS::test_convergency()
{
    // energy difference
    double ediff = E_ - Eold_;

    // RMS of the density
    Matrix D_rms;
    D_rms.copy(Dt_);
    D_rms.subtract(Dtold_);
    Drms_ = 0.5 * D_rms.rms();

    bool energy_test = fabs(ediff) < energy_threshold_;
    bool density_test = Drms_ < density_threshold_;
    bool cycle_test = iteration_ > 5;

    if(optimize_Vc){
        bool constraint_test = gradW->norm() < gradW_threshold_;
        constraint_optimization();
        if(energy_test and density_test and constraint_test and cycle_test){
            return true;
        }else{
            return false;
        }
    }else{
        if(energy_test and density_test and cycle_test){
            return true;
        }
        return false;
    }
}

void UCKS::save_information()
{

//    saved_naholepi_ = naholepi_;
//    saved_napartpi_ = napartpi_;
    dets.push_back(SharedDeterminant(new Determinant(E_,Ca_,Cb_,nalphapi_,nbetapi_)));
    if(do_excitation){
        double mixlet_exc_energy = E_ - ground_state_energy;
        fprintf(outfile,"  Excited mixed state   : excitation energy = %9.6f Eh = %8.4f eV = %9.1f cm**-1 \n",
                mixlet_exc_energy,mixlet_exc_energy * pc_hartree2ev, mixlet_exc_energy * pc_hartree2wavenumbers);
        if(KS::options_.get_bool("CDFT_SPIN_ADAPT_CI")){
            spin_adapt_mixed_excitation();
        }
        if(KS::options_.get_bool("CDFT_SPIN_ADAPT_SP")){
            compute_S_plus_triplet_correction();
        }

        if (do_project_out_holes){
            // Add the saved holes to the Ch_ matrix
            for (int h = 0; h < nirrep_; ++h){
                for (int i = 0; i < saved_naholepi_[h]; ++i){
                    Ch_->set_column(h,naholepi_[h] + i,saved_Ch_->get_column(h,i));
                }
            }
            naholepi_ += saved_naholepi_;
        }
        if (do_project_out_particles){
            // Add the saved particles to the Cp_ matrix
            for (int h = 0; h < nirrep_; ++h){
                for (int i = 0; i < saved_napartpi_[h]; ++i){
                    Cp_->set_column(h,napartpi_[h] + i,saved_Cp_->get_column(h,i));
                }
            }
            napartpi_ += saved_napartpi_;
        }
    }
    if(KS::options_.get_str("CDFT_EXC_METHOD") == "CIS")
        cis_excitation_energy();


//    if(KS::options_["CDFT_BREAK_SYMMETRY"].has_changed()){
//        spin_adapt_mixed_excitation();
//        compute_S_plus_triplet_correction();
//    }
}

//void UCKS::save_fock()
//{
//    if(not do_excitation){
//        UHF::save_fock();
//    }else{
//        if (initialized_diis_manager_ == false) {
//            diis_manager_ = boost::shared_ptr<DIISManager>(new DIISManager(max_diis_vectors_, "HF DIIS vector", DIISManager::LargestError, DIISManager::OnDisk));
//            diis_manager_->set_error_vector_size(2,
//                                                 DIISEntry::Matrix, Fa_.get(),
//                                                 DIISEntry::Matrix, Fb_.get());
//            diis_manager_->set_vector_size(2,
//                                           DIISEntry::Matrix, Fa_.get(),
//                                           DIISEntry::Matrix, Fb_.get());
//            initialized_diis_manager_ = true;
//        }

//        SharedMatrix errveca(moFeffa_);
//        errveca->zero_diagonal();
//        errveca->back_transform(Ca_);
//        SharedMatrix errvecb(moFeffb_);
//        errvecb->zero_diagonal();
//        errvecb->back_transform(Cb_);
//        diis_manager_->add_entry(4, errveca.get(), errvecb.get(), Fa_.get(), Fb_.get());
//    }
//}


void UCKS::compute_orbital_gradient(bool save_fock)
{
    if(not do_excitation){
        UHF::compute_orbital_gradient(save_fock);
    }else{
        SharedMatrix gradient_a = form_FDSmSDF(Fa_, Da_);
        SharedMatrix gradient_b = form_FDSmSDF(Fb_, Db_);
        Drms_ = 0.5*(gradient_a->rms() + gradient_b->rms());

//        if(save_fock){
//            if (initialized_diis_manager_ == false) {
//                diis_manager_ = boost::shared_ptr<DIISManager>(new DIISManager(max_diis_vectors_, "HF DIIS vector", DIISManager::LargestError, DIISManager::OnDisk));
//                diis_manager_->set_error_vector_size(2,
//                                                     DIISEntry::Matrix, gradient_a.get(),
//                                                     DIISEntry::Matrix, gradient_b.get());
//                diis_manager_->set_vector_size(2,
//                                               DIISEntry::Matrix, Fa_.get(),
//                                               DIISEntry::Matrix, Fb_.get());
//                initialized_diis_manager_ = true;
//            }

//            diis_manager_->add_entry(4, gradient_a.get(), gradient_b.get(), Fa_.get(), Fb_.get());
//        }

        if(save_fock){
            if (initialized_diis_manager_ == false) {
                diis_manager_ = boost::shared_ptr<DIISManager>(new DIISManager(max_diis_vectors_, "HF DIIS vector", DIISManager::LargestError, DIISManager::OnDisk));
                diis_manager_->set_error_vector_size(2,
                                                     DIISEntry::Matrix, Fa_.get(),
                                                     DIISEntry::Matrix, Fb_.get());
                diis_manager_->set_vector_size(2,
                                               DIISEntry::Matrix, Fa_.get(),
                                               DIISEntry::Matrix, Fb_.get());
                initialized_diis_manager_ = true;
            }
            SharedMatrix errveca(moFeffa_);
            errveca->zero_diagonal();
            errveca->back_transform(Ca_);
            SharedMatrix errvecb(moFeffb_);
            errvecb->zero_diagonal();
            errvecb->back_transform(Cb_);
            diis_manager_->add_entry(4, errveca.get(), errvecb.get(), Fa_.get(), Fb_.get());
        }
    }
}

void UCKS::spin_adapt_mixed_excitation()
{
    CharacterTable ct = KS::molecule_->point_group()->char_table();
    SharedDeterminant D1 = SharedDeterminant(new Determinant(E_,Ca_,Cb_,nalphapi_,nbetapi_));
    SharedDeterminant D2 = SharedDeterminant(new Determinant(E_,Cb_,Ca_,nbetapi_,nalphapi_));
    std::pair<double,double> M12 = matrix_element(D1,D2);
    double S12 = M12.first;
    double H12 = M12.second;
    double triplet_energy = (E_ - H12)/(1.0 - S12);
    double singlet_energy = (E_ + H12)/(1.0 + S12);
    double triplet_exc_energy = (E_ - H12)/(1.0 - S12) - ground_state_energy;
    double singlet_exc_energy = (E_ + H12)/(1.0 + S12) - ground_state_energy;
    fprintf(outfile,"\n\n  H12  %d-%s = %15.9f\n",
            state_ + (ground_state_symmetry_ == excited_state_symmetry_ ? 1 : 0),
            ct.gamma(excited_state_symmetry_).symbol(),H12);
    fprintf(outfile,"  S12  %d-%s = %15.9f\n\n",
            state_ + (ground_state_symmetry_ == excited_state_symmetry_ ? 1 : 0),
            ct.gamma(excited_state_symmetry_).symbol(),S12);

    fprintf(outfile,"\n  Triplet state energy (CI) %d-%s %20.9f Eh \n",
            state_ + (ground_state_symmetry_ == excited_state_symmetry_ ? 1 : 0),
            ct.gamma(excited_state_symmetry_).symbol(),triplet_energy);

    fprintf(outfile,"\n  Singlet state energy (CI) %d-%s %20.9f Eh \n",
            state_ + (ground_state_symmetry_ == excited_state_symmetry_ ? 1 : 0),
            ct.gamma(excited_state_symmetry_).symbol(),singlet_energy);

    fprintf(outfile,"\n  Excited triplet state %d-%s : excitation energy (CI) = %9.6f Eh = %8.4f eV = %9.1f cm**-1 \n",
            state_ + (ground_state_symmetry_ == excited_state_symmetry_ ? 1 : 0),
            ct.gamma(excited_state_symmetry_).symbol(),
            triplet_exc_energy,triplet_exc_energy * pc_hartree2ev, triplet_exc_energy * pc_hartree2wavenumbers);

    fprintf(outfile,"  Excited singlet state %d-%s : excitation energy (CI) = %9.6f Eh = %8.4f eV = %9.1f cm**-1 \n",
            state_ + (ground_state_symmetry_ == excited_state_symmetry_ ? 1 : 0),
            ct.gamma(excited_state_symmetry_).symbol(),
            singlet_exc_energy,singlet_exc_energy * pc_hartree2ev, singlet_exc_energy * pc_hartree2wavenumbers);
}

double UCKS::compute_triplet_correction()
{
    // I. Form the corresponding alpha and beta orbitals, this gives us a way
    // to identify the alpha and beta paired orbitals (singular value ~ 1)
    // and distinguish them from singly occupied MOs (singular value ~ 0).

    // Form <phi_b|S|phi_a>
    TempMatrix->gemm(false,false,1.0,S_,Ca_,0.0);
    TempMatrix2->gemm(true,false,1.0,Cb_,TempMatrix,0.0);

    // Scale it down to the occupied blocks only
    SharedMatrix Sba = SharedMatrix(new Matrix("Sba",nbetapi_,nalphapi_));
    nalphapi_.print();
    nbetapi_.print();
    for (int h = 0; h < nirrep_; ++h) {
        int nmo = nmopi_[h];
        int naocc = nalphapi_[h];
        int nbocc = nbetapi_[h];
        double** Sba_h = Sba->pointer(h);
        double** S_h = TempMatrix2->pointer(h);
        for (int i = 0; i < nbocc; ++i){
            for (int j = 0; j < naocc; ++j){
                Sba_h[i][j] = S_h[i][j];
            }
        }
    }

    // SVD <phi_b|S|phi_a>
    boost::tuple<SharedMatrix, SharedVector, SharedMatrix> UsV = Sba->svd_temps();
    SharedMatrix U = UsV.get<0>();
    SharedVector sigma = UsV.get<1>();
    SharedMatrix V = UsV.get<2>();
    Sba->svd(U,sigma,V);
    sigma->print();
    U->print();
    V->print();

    // II. Transform the occupied alpha and beta orbitals to the new representation
    // and compute the energy of the high-spin state.  The singly occupied MOs can
    // be used to guide the selection of the occupation numbers

    // Transform Ca_ with V (need to transpose V since svd returns V^T)
    TempMatrix->identity();
    for (int h = 0; h < nirrep_; ++h) {
        int rows = V->rowdim(h);
        int cols = V->coldim(h);
        double** V_h = V->pointer(h);
        double** T_h = TempMatrix->pointer(h);
        for (int i = 0; i < rows; ++i){
            for (int j = 0; j < cols; ++j){
                T_h[i][j] = V_h[i][j];
            }
        }
    }
    TempMatrix2->copy(Ca_);
    Ca_->gemm(false,true,1.0,TempMatrix2,TempMatrix,0.0);

    // Transform Cb_ with U
    TempMatrix->identity();
    for (int h = 0; h < nirrep_; ++h) {
        int rows = U->rowdim(h);
        int cols = U->coldim(h);
        double** U_h = U->pointer(h);
        double** T_h = TempMatrix->pointer(h);
        for (int i = 0; i < rows; ++i){
            for (int j = 0; j < cols; ++j){
                T_h[i][j] = U_h[i][j];
            }
        }
    }
    TempMatrix2->copy(Cb_);
    Cb_->gemm(false,false,1.0,TempMatrix2,TempMatrix,0.0);

    std::vector<std::pair<int,int> > noncoincidences;
    double noncoincidence_threshold = 1.0e-5;
    double Stilde = 1.0;
    // Compute the number of noncoincidences
    for (int h = 0; h < nirrep_; ++h){
        for (int p = 0; p < sigma->dim(h); ++p){
            if(std::fabs(sigma->get(h,p)) >= noncoincidence_threshold){
                Stilde *= sigma->get(h,p);
            }else{
                noncoincidences.push_back(std::make_pair(h,p));
            }
        }
    }
    int num_noncoincidences = static_cast<int>(noncoincidences.size());
    for (int k = 0; k < num_noncoincidences; ++k){
        int i_h = noncoincidences[k].first;
        int i_mo = noncoincidences[k].second;
        fprintf(outfile,"  Found a noncoincidence: irrep %d mo %d\n",i_h,i_mo);
    }
    fprintf(outfile,"  Stilde = %.6f\n",Stilde);
    double overlap = 1.0;
    if(num_noncoincidences == 0){
        throw FeatureNotImplemented("CKS", "Overlap in the case of zero noncoincidences", __FILE__, __LINE__);
    }

    int i_h = noncoincidences[0].first;
    int i_mo = noncoincidences[0].second;
    Dimension nnonc(1,"");
    nnonc[0] = 1;
    SharedMatrix Cnca(new Matrix("Cnca", nsopi_, nnonc));
    Cnca->set_column(0,0,Ca_->get_column(i_h,i_mo));
    SharedMatrix Cncb(new Matrix("Cncb", nsopi_, nnonc));
    Cncb->set_column(0,0,Cb_->get_column(i_h,i_mo));
    Cnca->print();
    Cncb->print();

    boost::shared_ptr<JK> jk = JK::build_JK();
    jk->initialize();
    std::vector<SharedMatrix>& C_left = jk->C_left();
    C_left.clear();
    C_left.push_back(Cncb);
    std::vector<SharedMatrix>& C_right = jk->C_right();
    C_right.clear();
    C_right.push_back(Cnca);
    jk->compute();
    SharedMatrix Jnew = jk->J()[0];

    double coupling = 0.0;
    for (int m = 0; m < nsopi_[0]; ++m){
        for (int n = 0; n < nsopi_[0]; ++n){
            double Dvalue = Cncb->get(0,m) * Cnca->get(0,n);
            double Jvalue = Jnew->get(m,n);
            coupling += Dvalue * Jvalue;
        }
    }
    fprintf(outfile,"  Matrix element from libfock = %20.12f\n",coupling);

    coupling *= Stilde * Stilde;

    jk->finalize();

    int maxi4 = INDEX4(nsopi_[0]+1,nsopi_[0]+1,nsopi_[0]+1,nsopi_[0]+1)+nsopi_[0]+1;
    double* integrals = new double[maxi4];
    for (int l = 0; l < maxi4; ++l){
        integrals[l] = 0.0;
    }

    IWL *iwl = new IWL(KS::psio_.get(), PSIF_SO_TEI, integral_threshold_, 1, 1);
    Label *lblptr = iwl->labels();
    Value *valptr = iwl->values();
    int labelIndex, pabs, qabs, rabs, sabs, prel, qrel, rrel, srel, psym, qsym, rsym, ssym;
    double value;
    bool lastBuffer;
    do{
        lastBuffer = iwl->last_buffer();
        for(int index = 0; index < iwl->buffer_count(); ++index){
            labelIndex = 4*index;
            pabs  = abs((int) lblptr[labelIndex++]);
            qabs  = (int) lblptr[labelIndex++];
            rabs  = (int) lblptr[labelIndex++];
            sabs  = (int) lblptr[labelIndex++];
            prel  = so2index_[pabs];
            qrel  = so2index_[qabs];
            rrel  = so2index_[rabs];
            srel  = so2index_[sabs];
            psym  = so2symblk_[pabs];
            qsym  = so2symblk_[qabs];
            rsym  = so2symblk_[rabs];
            ssym  = so2symblk_[sabs];
            value = (double) valptr[index];
            integrals[INDEX4(prel,qrel,rrel,srel)] = value;
        } /* end loop through current buffer */
        if(!lastBuffer) iwl->fetch();
    }while(!lastBuffer);
    iwl->set_keep_flag(1);
    delete iwl;
    double c2 = 0.0;
    double* Ci = Cncb->get_pointer();
    double* Cj = Cnca->get_pointer();
    double* Ck = Cnca->get_pointer();
    double* Cl = Cncb->get_pointer();
    for (int i = 0; i < nsopi_[0]; ++i){
        for (int j = 0; j < nsopi_[0]; ++j){
            for (int k = 0; k < nsopi_[0]; ++k){
                for (int l = 0; l < nsopi_[0]; ++l){
                    c2 += integrals[INDEX4(i,j,k,l)] * Ci[i] * Cj[j] * Ck[k] * Cl[l];
                }
            }
        }
    }
    delete[] integrals;
    fprintf(outfile,"  Matrix element from functor = %20.12f\n",c2);

    return coupling;
}

double UCKS::compute_S_plus_triplet_correction()
{
    fprintf(outfile,"\n  ==> Spin-adaptation correction using S+ <==\n");
    CharacterTable ct = KS::molecule_->point_group()->char_table();
    // A. Form the corresponding virtual alpha and occupied beta orbitals
    SharedMatrix Sba = SharedMatrix(new Matrix("Sba",nbetapi_,nmopi_ - nalphapi_));

    // Form <phi_b|S|phi_a>
    TempMatrix->gemm(false,false,1.0,S_,Ca_,0.0);
    TempMatrix2->gemm(true,false,1.0,Cb_,TempMatrix,0.0);

    // Grab the virtual alpha and occupied beta blocks
    for (int h = 0; h < nirrep_; ++h) {
        int nmo = nmopi_[h];
        int naocc = nalphapi_[h];
        int navir = nmo - naocc;
        int nbocc = nbetapi_[h];
        double** Sba_h = Sba->pointer(h);
        double** S_h = TempMatrix2->pointer(h);
        for (int i = 0; i < nbocc; ++i){
            for (int a = 0; a < navir; ++a){
                Sba_h[i][a] = S_h[i][a + naocc];
            }
        }
    }

    // SVD <phi_b|S|phi_a>
    boost::tuple<SharedMatrix, SharedVector, SharedMatrix> UsV = Sba->svd_a_temps();
    SharedMatrix U = UsV.get<0>();
    SharedVector sigma = UsV.get<1>();
    SharedMatrix V = UsV.get<2>();
    Sba->svd_a(U,sigma,V);

    // B. Find the corresponding alpha and beta orbitals
    std::vector<boost::tuple<double,int,int> > sorted_pair; // (singular value,irrep,mo in irrep)
    for (int h = 0; h < nirrep_; ++h){
        int npairs = sigma->dim(h);
        for (int p = 0; p < npairs; ++p){
            sorted_pair.push_back(boost::make_tuple(sigma->get(h,p),h,p));  // N.B. shifted wrt to full indexing
        }
    }
    std::sort(sorted_pair.begin(),sorted_pair.end(),std::greater<boost::tuple<double,int,int> >());

    // Print some useful information
    int npairs = std::min(10,static_cast<int>(sorted_pair.size()));
    fprintf(outfile,"  Most important corresponding occupied/virtual orbitals:\n\n");
    fprintf(outfile,"  Pair  Irrep  MO  <phi_b|phi_a>\n");
    for (int p = 0; p < npairs; ++p){
        fprintf(outfile,"    %2d     %3s %4d   %9.6f\n",p,ct.gamma(sorted_pair[p].get<1>()).symbol(),sorted_pair[p].get<2>(),sorted_pair[p].get<0>());
    }

    // C. Transform the alpha virtual and beta occupied orbitals to the new representation
    // Transform Ca_ with V (need to transpose V since svd returns V^T)
    TempMatrix->identity();
    for (int h = 0; h < nirrep_; ++h) {
        int rows = V->rowdim(h);
        int cols = V->coldim(h);
        int naocc = nalphapi_[h];
        double** V_h = V->pointer(h);
        double** T_h = TempMatrix->pointer(h);
        for (int i = 0; i < rows; ++i){
            for (int j = 0; j < cols; ++j){
                T_h[i + naocc][j + naocc] = V_h[i][j]; // Offset by the number of occupied MOs
            }
        }
    }
    TempMatrix2->copy(Ca_);
    Ca_->gemm(false,true,1.0,TempMatrix2,TempMatrix,0.0);

    // Transform Cb_ with U (reversing the order so that the corresponding orbital is the first to be excluded)
    TempMatrix->identity();
    for (int h = 0; h < nirrep_; ++h) {
        int rows = U->rowdim(h);
        int cols = U->coldim(h);
        double** U_h = U->pointer(h);
        double** T_h = TempMatrix->pointer(h);
        for (int i = 0; i < rows; ++i){
            for (int j = 0; j < cols; ++j){
                T_h[i][j] = U_h[i][rows - j - 1]; // invert the order
            }
        }
    }
    TempMatrix2->copy(Cb_);
    Cb_->gemm(false,false,1.0,TempMatrix2,TempMatrix,0.0);

    fprintf(outfile,"\n  Original occupation numbers:\n");
    fprintf(outfile, "\tNA   [ ");
    for(int h = 0; h < nirrep_-1; ++h) fprintf(outfile, " %4d,", nalphapi_[h]);
    fprintf(outfile, " %4d ]\n", nalphapi_[nirrep_-1]);
    fprintf(outfile, "\tNB   [ ");
    for(int h = 0; h < nirrep_-1; ++h) fprintf(outfile, " %4d,", nbetapi_[h]);
    fprintf(outfile, " %4d ]\n", nbetapi_[nirrep_-1]);
    int mo_h = sorted_pair[0].get<1>();

    fprintf(outfile,"\n  Final occupation numbers:\n");
    // Update the occupation numbers
    nalphapi_[mo_h] += 1;
    nbetapi_[mo_h]  -= 1;
    nalpha_ = 0;
    nbeta_ = 0;
    for (int h = 0; h < nirrep_; ++h) {
        soccpi_[h] = std::abs(nalphapi_[h] - nbetapi_[h]);
        doccpi_[h] = std::min(nalphapi_[h] , nbetapi_[h]);
        nalpha_ += nalphapi_[h];
        nbeta_ += nalphapi_[h];
    }
    fprintf(outfile, "\tNA   [ ");
    for(int h = 0; h < nirrep_-1; ++h) fprintf(outfile, " %4d,", nalphapi_[h]);
    fprintf(outfile, " %4d ]\n", nalphapi_[nirrep_-1]);
    fprintf(outfile, "\tNB   [ ");
    for(int h = 0; h < nirrep_-1; ++h) fprintf(outfile, " %4d,", nbetapi_[h]);
    fprintf(outfile, " %4d ]\n", nbetapi_[nirrep_-1]);

    // Compute the density matrices with the new occupation

    form_D();
    form_G();
    form_F();

    // Compute the triplet energy from the density matrices
    double triplet_energy = compute_E();
    double triplet_exc_energy = triplet_energy - ground_state_energy;

    fprintf(outfile,"\n  Triplet state energy (S+) %d-%s %20.9f Eh \n",
            state_ + (ground_state_symmetry_ == excited_state_symmetry_ ? 1 : 0),
            ct.gamma(excited_state_symmetry_).symbol(),triplet_energy);

    fprintf(outfile,"\n  Singlet state energy (S+) %d-%s %20.9f Eh \n",
            state_ + (ground_state_symmetry_ == excited_state_symmetry_ ? 1 : 0),
            ct.gamma(excited_state_symmetry_).symbol(),2.0 * E_ - triplet_energy);

    fprintf(outfile,"\n  Excited triplet state %d-%s : excitation energy (S+) = %9.6f Eh = %8.4f eV = %9.1f cm**-1 \n",
            state_ + (ground_state_symmetry_ == excited_state_symmetry_ ? 1 : 0),
            ct.gamma(excited_state_symmetry_).symbol(),
            triplet_exc_energy,triplet_exc_energy * pc_hartree2ev, triplet_exc_energy * pc_hartree2wavenumbers);
    double singlet_exc_energy = 2.0 * E_ - triplet_energy - ground_state_energy;
    fprintf(outfile,"  Excited singlet state %d-%s : excitation energy (S+) = %9.6f Eh = %8.4f eV = %9.1f cm**-1 \n",
            state_ + (ground_state_symmetry_ == excited_state_symmetry_ ? 1 : 0),
            ct.gamma(excited_state_symmetry_).symbol(),
            singlet_exc_energy,singlet_exc_energy * pc_hartree2ev, singlet_exc_energy * pc_hartree2wavenumbers);

    fprintf(outfile,"\n\n");
    compute_spin_contamination();
    fprintf(outfile,"\n");

    // Revert to the mixed state occupation numbers
    nalphapi_[mo_h] -= 1;
    nbetapi_[mo_h]  += 1;
    nalpha_ = 0;
    nbeta_ = 0;
    for (int h = 0; h < nirrep_; ++h) {
        soccpi_[h] = std::abs(nalphapi_[h] - nbetapi_[h]);
        doccpi_[h] = std::min(nalphapi_[h] , nbetapi_[h]);
        nalpha_ += nalphapi_[h];
        nbeta_ += nalphapi_[h];
    }
    Ca_->copy(dets.back()->Ca());
    Cb_->copy(dets.back()->Cb());
    return singlet_exc_energy;
}

void UCKS::cis_excitation_energy()
{

//    CharacterTable ct = KS::molecule_->point_group()->char_table();

//    int symmetry = 0;
//    SharedMatrix ra = SharedMatrix(new Matrix("r Amplitudes",nmopi_,nmopi_,symmetry));
//    SharedMatrix rb = SharedMatrix(new Matrix("r Amplitudes",nmopi_,nmopi_,symmetry));
//    SharedMatrix genDa = factory_->create_shared_matrix("genDa");
//    SharedMatrix genDb = factory_->create_shared_matrix("genDb");
//    // Determine the hole/particle pair to follow
//    // Compute the symmetry adapted hole/particle pairs
//    std::vector<boost::tuple<double,int,int,double,int,int,double> > sorted_hp_pairs;
//    for (int h = 0; h < nirrep_; ++h){
//        int h_i = h;
//        int h_a = h ^ symmetry;
//        int nocc_i = nalphapi_[h_i];
//        int nocc_a = nalphapi_[h_a];
//        int nvir_a = nmopi_[h_a] - nalphapi_[h_a];
//        for (int i = 0; i < nocc_i; ++i){
//            for (int a = 0; a < nvir_a; ++a){
//                double e_i = epsilon_a_->get(h_i,i);
//                double e_a = epsilon_a_->get(h_a,a + nocc_a);
//                double delta_ai = e_a - e_i;
//                sorted_hp_pairs.push_back(boost::make_tuple(delta_ai,h_i,i,e_i,h_a,a + nocc_a,e_a));
//            }
//        }
//    }

//    std::sort(sorted_hp_pairs.begin(),sorted_hp_pairs.end());
////    if(iteration_ == 0){
//    fprintf(outfile, "\n  Ground state symmetry: %s\n",ct.gamma(ground_state_symmetry_).symbol());
//    fprintf(outfile, "  Excited state symmetry: %s\n",ct.gamma(excited_state_symmetry_).symbol());
//    fprintf(outfile, "\n  Lowest energy excitations:\n");
//    fprintf(outfile, "  --------------------------------------\n");
//    fprintf(outfile, "    N   Occupied     Virtual     E(eV)  \n");
//    fprintf(outfile, "  --------------------------------------\n");
//    int maxstates = std::min(10,static_cast<int>(sorted_hp_pairs.size()));
//    for (int n = 0; n < maxstates; ++n){
//        double energy_hp = sorted_hp_pairs[n].get<6>() - sorted_hp_pairs[n].get<3>();
//        fprintf(outfile,"   %2d:  %4d%-3s  -> %4d%-3s   %9.3f\n",n + 1,
//                sorted_hp_pairs[n].get<2>() + 1,
//                ct.gamma(sorted_hp_pairs[n].get<1>()).symbol(),
//                sorted_hp_pairs[n].get<5>() + 1,
//                ct.gamma(sorted_hp_pairs[n].get<4>()).symbol(),
//                energy_hp * _hartree2ev);
//    }
//    fprintf(outfile, "  --------------------------------------\n");

//    int select_pair = 0;
//    aholes_h = sorted_hp_pairs[select_pair].get<1>();
//    aholes_mo = sorted_hp_pairs[select_pair].get<2>();
//    aparts_h = sorted_hp_pairs[select_pair].get<4>();
//    aparts_mo = sorted_hp_pairs[select_pair].get<5>();
//    ra->set(aholes_h,aholes_mo,aparts_mo,1.0 / std::sqrt(2.0));
//    rb->set(aholes_h,aholes_mo,aparts_mo,-1.0 / std::sqrt(2.0));

//    // Compute the density matrix
//    genDa->zero();
//    genDb->zero();

//    // C1 symmetry
//    {
//    int naocc = nalphapi_[0];
//    int navir = nmopi_[0] - nalphapi_[0];
//    int nmo = nmopi_[0];
//    double** ra_h = ra->pointer(0);
//    for (int i = 0; i < naocc; ++i){
//        for (int j = 0; j < naocc; ++j){
//            double da = (i == j ? 1.0 : 0.0);
//            for (int c = naocc; c < nmo; ++c){
//                da -= ra_h[i][c] * ra_h[j][c];
//            }
//            genDa->set(0,i,j,da);
//        }
//    }
//    for (int a = naocc; a < nmo; ++a){
//        for (int b = naocc; b < nmo; ++b){
//            double da = 0.0;
//            for (int k = 0; k < naocc; ++k){
//                da += ra_h[k][a] * ra_h[k][b];
//            }
//            genDa->set(0,a,b,da);
//        }
//    }


//    }

//    {
//    int nbocc = nbetapi_[0];
//    int nbvir = nmopi_[0] - nbetapi_[0];
//    int nmo = nmopi_[0];
//    double** rb_h = rb->pointer(0);
//    for (int i = 0; i < nbocc; ++i){
//        for (int j = 0; j < nbocc; ++j){
//            double db = (i == j ? 1.0 : 0.0);
//            for (int c = nbocc; c < nmo; ++c){
//                db -= rb_h[i][c] * rb_h[j][c];
//            }
//            genDb->set(0,i,j,db);
//        }
//    }
//    for (int a = nbocc; a < nmo; ++a){
//        for (int b = nbocc; b < nmo; ++b){
//            double db = 0.0;
//            for (int k = 0; k < nbocc; ++k){
//                db += rb_h[k][a] * rb_h[k][b];
//            }
//            genDb->set(0,a,b,db);
//        }
//    }
//    }

////    // Off-diagonal terms
////    for (int h = 0; h < nirrep_; ++h){
////        int g = h ^ symmetry;
////        int nmo_h = nmopi_[h];
////        int nmo_g = nmopi_[h];
////        double** ra_h = ra->pointer(h);
////        double** ra_g = ra->pointer(g);
////        int navir_g = nmopi_[g] - nalphapi_[g];
////        for (int p = 0; p < nmo; ++p){
////            for (int q = 0; q < nmo; ++q){
////                double da = 0.0;
////                for (int c = navir_g; c < nmo_; ++c){
////                    da -= ra_h[p][c] * ra_h[q][c];
////                }
////            }
////        }

////        int navir = nmopi_[h] - nalphapi_[h];
////        int h_a = h ^ symmetry;
////        double** ra_h = ra->pointer(h);
////        int nvir_ac = nmopi_[h_a] - nalphapi_[h_a];
////        int nocc_ak = nalphapi_[h_a];
////        for (int i = 0; i < naocc; ++i){
////            for (int j = 0; j < naocc; ++j){
////                double da = (i == j ? 1.0 : 0.0);
////                for (int c = 0; c < nvir_ac; ++c){
////                        da -= ra_h[i][c] * ra_h[j][c];
////                }
////                genDa->set(h,i,j,da);
////            }
////        }
////        double** ra_h_a = ra->pointer(h_a);
////        for (int a = 0; a < navir; ++a){
////            for (int b = 0; b < navir; ++b){
////                double da = 0.0;
////                for (int k = 0; k < nocc_ak; ++k){
////                        da += ra_h_a[k][a] * ra_h_a[k][b];
////                }
////                genDa->set(h,a + naocc,b + naocc,da);
////            }
////        }

////        int nbocc = nbetapi_[h];
////        int nbvir = nmopi_[h] - nbetapi_[h];
////        double** rb_h = rb->pointer(h);
////        int nvir_bc = nmopi_[h_a] - nbetapi_[h_a];
////        int nocc_bk = nbetapi_[h_a];
////        for (int i = 0; i < nbocc; ++i){
////            for (int j = 0; j < nbocc; ++j){
////                double db = (i == j ? 1.0 : 0.0);
////                for (int c = 0; c < nvir_bc; ++c){
////                        db -= rb_h[i][c] * rb_h[j][c];
////                }
////                genDb->set(h,i,j,db);
////            }
////        }
////        double** rb_h_a = rb->pointer(h_a);
////        for (int a = 0; a < nbvir; ++a){
////            for (int b = 0; b < nbvir; ++b){
////                double db = 0.0;
////                for (int k = 0; k < nocc_ak; ++k){
////                        db += rb_h_a[k][a] * rb_h_a[k][b];
////                }
////                genDb->set(h,a + nbocc,b + nbocc,db);
////            }
////        }
////    }

////    genDa->zero();
////    genDb->zero();
////    for (int n = 0; n < 5; ++n){

////    }


//    Da_->back_transform(genDa,Ca_);
//    Db_->back_transform(genDb,Cb_);
//    Dt_->copy(Da_);
//    Dt_->add(Db_);
////    nalphapi_[7] = 0;
////    nalphapi_[2] = 1;

////    form_D();
////    for (int h = 0; h < nirrep_; ++h) {
////        soccpi_[h] = std::abs(nalphapi_[h] - nbetapi_[h]);
////        doccpi_[h] = std::min(nalphapi_[h] , nbetapi_[h]);
////    }
//    form_G();
//    form_F();



//    // Compute the energy
//    double cis_energy = compute_E() - E_;
//    fprintf(outfile,"\n  CIS excited state = %9.6f Eh = %8.4f eV = %9.1f cm**-1 \n",
//            cis_energy,cis_energy * _hartree2ev, cis_energy * _hartree2wavenumbers);
}

//void UCKS::form_D_cis()
//{
//    for (int h = 0; h < nirrep_; ++h) {
//        int nso = nsopi_[h];
//        int nmo = nmopi_[h];
//        int na = nalphapi_[h];
//        int nb = nbetapi_[h];

//        if (nso == 0 || nmo == 0) continue;

//        double** Ca = Ca_->pointer(h);
//        double** Cb = Cb_->pointer(h);
//        double** Da = Da_->pointer(h);
//        double** Db = Db_->pointer(h);

//        if (na == 0)
//            ::memset(static_cast<void*>(Da[0]), '\0', sizeof(double)*nso*nso);
//        if (nb == 0)
//            ::memset(static_cast<void*>(Db[0]), '\0', sizeof(double)*nso*nso);

//        C_DGEMM('N','T',nso,nso,na,1.0,Ca[0],nmo,Ca[0],nmo,0.0,Da[0],nso);
//        C_DGEMM('N','T',nso,nso,nb,1.0,Cb[0],nmo,Cb[0],nmo,0.0,Db[0],nso);

//    }

//    Dt_->copy(Da_);
//    Dt_->add(Db_);

//    if (debug_) {
//        fprintf(outfile, "in UHF::form_D:\n");
//        Da_->print();
//        Db_->print();
//    }
//}

void UCKS::extract_square_subblock(SharedMatrix A, SharedMatrix B, bool occupied, Dimension npi, double diagonal_shift)
{
    // Set the diagonal of B to diagonal_shift
    B->identity();
    B->scale(diagonal_shift);

    // Copy the block from A
    for (int h = 0; h < nirrep_; ++h){
        int block_dim = occupied ? npi[h] : nmopi_[h] - npi[h];
        int block_shift = occupied ? 0 : npi[h];
        if (block_dim != 0){
            double** A_h = A->pointer(h);
            double** B_h = B->pointer(h);
            for (int i = 0; i < block_dim; ++i){
                for (int j = 0; j < block_dim; ++j){
                    B_h[i][j] = A_h[i + block_shift][j + block_shift];
                }
            }
        }
    }
}

void UCKS::copy_subblock(SharedMatrix A, SharedMatrix B, Dimension rowspi, Dimension colspi, bool occupied)
{
    for (int h = 0; h < nirrep_; ++h){
        int nrows = occupied ? rowspi[h] : nmopi_[h] - rowspi[h];
        int row_shift = occupied ? 0 : rowspi[h];
        int ncols = occupied ? colspi[h] : nmopi_[h] - colspi[h];
        int col_shift = occupied ? 0 : colspi[h];
        if (nrows * ncols != 0){
            double** A_h = A->pointer(h);
            double** B_h = B->pointer(h);
            for (int i = 0; i < nrows; ++i){
                for (int j = 0; j < ncols; ++j){
                    B_h[i][j] = A_h[i + row_shift][j + col_shift];
                }
            }
        }
    }
}

void UCKS::copy_block(SharedMatrix A, double alpha, SharedMatrix B, double beta, Dimension rowspi, Dimension colspi,
                      Dimension A_rows_offsetpi, Dimension A_cols_offsetpi,
                      Dimension B_rows_offsetpi, Dimension B_cols_offsetpi)
{
    for (int h = 0; h < nirrep_; ++h){
        int nrows = rowspi[h];
        int ncols = colspi[h];
        int A_row_offset = A_rows_offsetpi[h];
        int A_col_offset = A_cols_offsetpi[h];
        int B_row_offset = B_rows_offsetpi[h];
        int B_col_offset = B_cols_offsetpi[h];
        if (nrows * ncols != 0){
            double** A_h = A->pointer(h);
            double** B_h = B->pointer(h);
            for (int i = 0; i < nrows; ++i){
                for (int j = 0; j < ncols; ++j){
                    B_h[i + B_row_offset][j + B_col_offset] = alpha * A_h[i + A_row_offset][j + A_col_offset] + beta * B_h[i + B_row_offset][j + B_col_offset];
                }
            }
        }
    }
}

}} // Namespaces
