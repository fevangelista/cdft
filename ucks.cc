
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
  state_(state)
{
    init();
    init_excitation(ref_scf);
}

UCKS::UCKS(Options &options, boost::shared_ptr<PSIO> psio, boost::shared_ptr<Wavefunction> ref_scf, int state,int symmetry)
: UKS(options, psio),
  do_excitation(true),
  do_symmetry(true),
  optimize_Vc(false),
  gradW_threshold_(1.0e-9),
  nW_opt(0),
  ground_state_energy(0.0),
  ground_state_symmetry_(0),
  excited_state_symmetry_(symmetry),
  state_(state)
{
    init();
    init_excitation(ref_scf);
    ground_state_energy = dets[0]->energy();
    ground_state_symmetry_ = dets[0]->symmetry();
}

void UCKS::init()
{
    fprintf(outfile,"\n  ==> Constrained DFT (UCKS) <==\n\n");

    optimize_Vc = KS::options_.get_bool("OPTIMIZE_VC");
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
    Ua = factory_->create_shared_matrix("U alpha");
    Ub = factory_->create_shared_matrix("U beta");
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

    PoFPo_ = factory_->create_shared_matrix("PoFsPo");
    PvFPv_ = factory_->create_shared_matrix("PvFPv");
    QFQ_ = factory_->create_shared_matrix("QFQ");
    Ch_ = factory_->create_shared_matrix("Hole MOs");
    Cp_ = factory_->create_shared_matrix("Particle MOs");
    moFeffa_ = factory_->create_shared_matrix("MO alpha Feff");
    moFeffb_ = factory_->create_shared_matrix("MO beta Feff");
    Uo_ = factory_->create_shared_matrix("Uo");
    Uv_ = factory_->create_shared_matrix("Uv");
    lambda_o_ = factory_->create_shared_vector("lambda_o");
    lambda_v_ = factory_->create_shared_vector("lambda_v");

    // Save the reference state MOs and occupation numbers
    fprintf(outfile,"  Saving the reference orbitals for an excited state computation\n");
    UCKS* ucks_ptr = dynamic_cast<UCKS*>(ref_scf.get());
    if(do_symmetry and (state_ == 1)){  // If we are starting with a new irrep save only the ground state wfn
        dets.push_back(ucks_ptr->dets[0]);
    }else{
        dets = ucks_ptr->dets;
    }

    // Set the Fock matrix to the converged Fock matrix for the previous state
    Fa_->copy(ref_scf->Fa());
    Fb_->copy(ref_scf->Fb());
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
    if(Pa){
        Fa_->add(Pa);
    }

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
        TempMatrix->gemm(false,true,1.0,Ch_,Ch_,0.0);
        TempMatrix->gemm(false,true,1.0,Cp_,Cp_,1.0);
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

void UCKS:: form_C()
{
    if(not do_excitation){
        // Ground state: use the default form_C
        UKS::form_C();
    }else{
        // Excited state: use a special form_C
        if(KS::options_.get_str("CDFT_EXC_METHOD") == "CH"){
            form_C_CH_algorithm();
        }else if(KS::options_.get_str("CDFT_EXC_METHOD") == "CP"){
            form_C_CP_algorithm();
        }else if(KS::options_.get_str("CDFT_EXC_METHOD") == "CHP" or
                 KS::options_.get_str("CDFT_EXC_METHOD") == "CHP-F"){
            form_C_CHP_algorithm();
        }
    }
}

void UCKS::form_C_CH_algorithm()
{
    int nstate = static_cast<int>(dets.size());
    fprintf(outfile,"  Computing %d optimal hole orbitals\n",nstate);fflush(outfile);

    // Data structures to save the hole information
    Dimension aholepi(nirrep_,"Alpha holes per irrep");
    std::vector<SharedVector> holes_Ca;
    std::vector<int> holes_h;

    // Compute the hole states
    for (int m = 0; m < nstate; ++m){
        // Transform Fa to the MO basis of state m
        TempMatrix->transform(Fa_,dets[m]->Ca());
        // Grab the occ block of Fa
        extract_square_subblock(TempMatrix,PoFPo_,true,dets[m]->nalphapi(),1.0e9);
        PoFPo_->diagonalize(Uo_,lambda_o_);
        std::vector<boost::tuple<double,int,int> > sorted_holes; // (energy,irrep,mo in irrep)
        for (int h = 0; h < nirrep_; ++h){
            int nmo = nmopi_[h];
            for (int p = 0; p < nmo; ++p){
                if (lambda_o_->get(h,p) < 1.0e6){
                    sorted_holes.push_back(boost::make_tuple(lambda_o_->get(h,p),h,p));
                }
            }
        }
        std::sort(sorted_holes.begin(),sorted_holes.end());
        boost::tuple<double,int,int> hole;
        // Extract the hole alpha orbital according to an energy criteria (this needs a generalization)
        if (KS::options_.get_str("CDFT_EXC_HOLE") == "VALENCE"){
            // For valence excitations select the highest lying orbital (HOMO-like)
            hole = sorted_holes.back();
        }else if(KS::options_.get_str("CDFT_EXC_HOLE") == "CORE"){
            // For core excitations select the lowest lying orbital (1s-like)
            hole = sorted_holes.front();
        }
        double hole_energy = hole.get<0>();
        int hole_h = hole.get<1>();
        int hole_mo = hole.get<2>();
        fprintf(outfile,"   constrained hole %d :(irrep = %d,mo = %d,energy = %.6f)\n",
                        m,hole_h,hole_mo,hole_energy);

        // Compute the hole orbital
        SharedVector hole_Ca = factory_->create_shared_vector("Hole");
        for (int p = 0; p < nsopi_[hole_h]; ++p){
            double c_p = 0.0;
            for (int i = 0; i < dets[m]->nalphapi()[hole_h]; ++i){
                c_p += dets[m]->Ca()->get(hole_h,p,i) * Uo_->get(hole_h,i,hole_mo) ;
            }
            hole_Ca->set(hole_h,p,c_p);
        }
        holes_Ca.push_back(hole_Ca);
        holes_h.push_back(hole_h);
        aholepi[hole_h] += 1;
    }

    // Put the hole orbitals in Ch
    SharedMatrix Ch = SharedMatrix(new Matrix("Ch",nsopi_,aholepi));
    SharedMatrix Cho = SharedMatrix(new Matrix("Cho",nsopi_,aholepi));
    std::vector<int> offset(nirrep_,0);
    for (int m = 0; m < nstate; ++m){
        //int h = current_excited_state->ah_sym(m);
        int h = holes_h[m];
        Ch->set_column(h,offset[h],holes_Ca[m]);
        offset[h] += 1;
    }

    // Orthogonalize the hole orbitals
    SharedMatrix Spp = SharedMatrix(new Matrix("Spp",aholepi,aholepi));
    SharedMatrix Upp = SharedMatrix(new Matrix("Upp",aholepi,aholepi));
    SharedVector spp = SharedVector(new Vector("spp",aholepi));
    Spp->transform(S_,Ch);
    Spp->diagonalize(Upp,spp);
    double S_cutoff = 1.0e-3;
    // Form the transformation matrix X (in place of Upp)
    for (int h = 0; h < nirrep_; ++h) {
        //in each irrep, scale significant cols i by 1.0/sqrt(s_i)
        for (int i = 0; i < aholepi[h]; ++i) {
            if (std::fabs(spp->get(h,i)) > S_cutoff) {
                double scale = 1.0 / std::sqrt(spp->get(h,i));
                Upp->scale_column(h,i,scale);
            } else {
                throw FeatureNotImplemented("CKS", "Cannot yet deal with linear dependent particle orbitals", __FILE__, __LINE__);
            }
        }
    }
    Cho->zero();
    Cho->gemm(false,false,1.0,Ch,Upp,0.0);
    Ch_->zero();
    copy_block(Cho,Ch_,nsopi_,aholepi);

    // Form the projector onto the orbitals orthogonal to the particles in the ground state mo representation
    TempMatrix->zero();
    TempMatrix->gemm(false,true,1.0,Cho,Cho,0.0);
    TempMatrix->transform(S_);
    TempMatrix->transform(dets[0]->Ca());
    TempMatrix2->identity();
    TempMatrix2->subtract(TempMatrix);

    // Form the Fock matrix in the GS basis, project out the holes, diagonalize it, and transform the MO coefficients
    TempMatrix->transform(Fa_,dets[0]->Ca());
    TempMatrix->transform(TempMatrix2);

    TempMatrix->diagonalize(TempMatrix2,epsilon_a_);
    Ca_->zero();
    Ca_->gemm(false,false,1.0,dets[0]->Ca(),TempMatrix2,0.0);

//    epsilon_a_->print();

    std::vector<boost::tuple<double,int,int> > sorted_spectators;
    for (int h = 0; h < nirrep_; ++h){
        for (int p = 0; p < nmopi_[h]; ++p){
            sorted_spectators.push_back(boost::make_tuple(epsilon_a_->get(h,p),h,p));
        }
    }
    std::sort(sorted_spectators.begin(),sorted_spectators.end());

    for (int h = 0; h < nirrep_; ++h){
        nalphapi_[h] = 0;
    }
    nbetapi_ = dets[0]->nbetapi();
    int assigned = 0;
    for (int p = 0; p < nmo_; ++p){
        if (assigned < nalpha_){
            if(std::fabs(sorted_spectators[p].get<0>()) > 1.0e-6){  // !!! Check this out NB WARNING
                int h = sorted_spectators[p].get<1>();
                nalphapi_[h] += 1;
                assigned += 1;
            }
        }
    }
    // At this point the orbitals are sorted according to the energy but we
    // want to make sure that the hole MO appear where they should, that is
    // the holes in the virtual space.
    // |(1) (2) ... (hole) | ...> will become
    // |(particle) (1) (2) ... | ... (hole)>
    TempMatrix->zero();
    TempVector->zero();
    for (int h = 0; h < nirrep_; ++h){
        int nso = nsopi_[h];
        int nmo = nmopi_[h];
        double** T_h = TempMatrix->pointer(h);
        double** C_h = Ca_->pointer(h);
        double** Cho_h = Cho->pointer(h);
        // First place the holes
        int m = 0;
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
        for (int p = 0; p < aholepi[h]; ++p){
            for (int q = 0; q < nso; ++q){
                T_h[q][m] = Cho_h[q][p];
            }
            m += 1;
        }
    }

    Ca_->copy(TempMatrix);
    epsilon_a_->copy(TempVector.get());

//    Ca_->print();
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

    // BETA
    diagonalize_F(Fb_, Cb_, epsilon_b_);

    if (debug_) {
        Ca_->print(outfile);
        Cb_->print(outfile);
    }
}

void UCKS::form_C_CP_algorithm()
{
    int nstate = static_cast<int>(dets.size());
    fprintf(outfile,"  Computing %d optimal particle orbitals\n",nstate);

    // Data structures to save the particle information
    Dimension apartpi(nirrep_,"Alpha particles per irrep");
    std::vector<SharedVector> parts_Ca;
    std::vector<int> parts_h;

    // Compute the particle states
    for (int m = 0; m < nstate; ++m){
        // Transform Fa to the MO basis of state m
        TempMatrix->transform(Fa_,dets[m]->Ca());
        // Grab the vir block of Fa
        extract_square_subblock(TempMatrix,PvFPv_,false,dets[m]->nalphapi(),1.0e9);
        PvFPv_->diagonalize(Uv_,lambda_v_);
        std::vector<boost::tuple<double,int,int> > sorted_vir; // (energy,irrep,mo in irrep)
        for (int h = 0; h < nirrep_; ++h){
            int nmo = nmopi_[h];
            for (int p = 0; p < nmo; ++p){
                sorted_vir.push_back(boost::make_tuple(lambda_v_->get(h,p),h,p));  // N.B. shifted to full indexing
            }
        }
        std::sort(sorted_vir.begin(),sorted_vir.end());
        // In the case of particle, we assume that we are always interested in the lowest lying orbitals
        boost::tuple<double,int,int> particle = sorted_vir.front();
        int part_h = particle.get<1>();
        int part_mo = particle.get<2>();
        fprintf(outfile,"   constrained particle %d :(irrep = %d,mo = %d,energy = %.6f)\n",
                m,particle.get<1>(),particle.get<2>(),particle.get<0>());

        // Compute the particle orbital
        SharedVector part_Ca = factory_->create_shared_vector("Particle");
        for (int p = 0; p < nsopi_[part_h]; ++p){
            double c_p = 0.0;
            int maxa = nmopi_[part_h] - dets[m]->nalphapi()[part_h];
            for (int a = 0; a < maxa; ++a){
                c_p += dets[m]->Ca()->get(part_h,p,dets[m]->nalphapi()[part_h] + a) * Uv_->get(part_h,a,part_mo) ;
            }
            part_Ca->set(part_h,p,c_p);
        }
        parts_Ca.push_back(part_Ca);
        parts_h.push_back(part_h);
        apartpi[part_h] += 1;
    }

    // Put the particle orbitals in Cp
    SharedMatrix Cp = SharedMatrix(new Matrix("Cp",nsopi_,apartpi));
    SharedMatrix Cpo = SharedMatrix(new Matrix("Cpo",nsopi_,apartpi));
    std::vector<int> offset(nirrep_,0);
    for (int m = 0; m < nstate; ++m){
        int h = parts_h[m];
        Cp->set_column(h,offset[h],parts_Ca[m]);
        offset[h] += 1;
    }
    SharedMatrix Spp = SharedMatrix(new Matrix("Spp",apartpi,apartpi));
    SharedMatrix Upp = SharedMatrix(new Matrix("Upp",apartpi,apartpi));
    SharedVector spp = SharedVector(new Vector("spp",apartpi));
    Spp->transform(S_,Cp);
    Spp->print();
    Spp->diagonalize(Upp,spp);

    double S_cutoff = 1.0e-2;
    // Form the transformation matrix X (in place of Upp)
    for (int h = 0; h < nirrep_; ++h) {
        //in each irrep, scale significant cols i  by 1.0/sqrt(s_i)
        for (int i = 0; i < apartpi[h]; ++i) {
            if (S_cutoff  < spp->get(h,i)) {
                double scale = 1.0 / sqrt(spp->get(h, i));
                Upp->scale_column(h, i, scale);
            } else {
                throw FeatureNotImplemented("CKS", "Cannot yet deal with linear dependent particle orbitals", __FILE__, __LINE__);
            }
        }
    }
    Cpo->zero();
    Cpo->gemm(false,false,1.0,Cp,Upp,0.0);
    Cp_->zero();
    copy_block(Cpo,Cp_,nsopi_,apartpi);

    // Form the projector onto the orbitals orthogonal to the particles in the ground state mo representation
    TempMatrix->zero();
    TempMatrix->gemm(false,true,1.0,Cpo,Cpo,0.0);
    TempMatrix->transform(S_);
    TempMatrix->transform(dets[0]->Ca());
    TempMatrix2->identity();
    TempMatrix2->subtract(TempMatrix);

    // Form the Fock matrix in the GS basis, diagonalize it, and transform the MO coefficients
    TempMatrix->transform(Fa_,dets[0]->Ca());
    TempMatrix->transform(TempMatrix2);
    TempMatrix->diagonalize(TempMatrix2,epsilon_a_);
    Ca_->gemm(false,false,1.0,dets[0]->Ca(),TempMatrix2,0.0);

    std::vector<boost::tuple<double,int,int> > sorted_spectators;
    for (int h = 0; h < nirrep_; ++h){
        for (int p = 0; p < nmopi_[h]; ++p){
            sorted_spectators.push_back(boost::make_tuple(epsilon_a_->get(h,p),h,p));
        }
    }
    std::sort(sorted_spectators.begin(),sorted_spectators.end());

    for (int h = 0; h < nirrep_; ++h){
        nalphapi_[h] = apartpi[h];
    }
    nbetapi_ = dets[0]->nbetapi();
    int assigned = 0;
    for (int p = 0; p < nmo_; ++p){
        if (assigned < nalpha_ - nstate){
            if(std::fabs(sorted_spectators[p].get<0>()) > 1.0e-6){  // !!! Check this out NB WARNING
                int h = sorted_spectators[p].get<1>();
                nalphapi_[h] += 1;
                assigned += 1;
            }
        }
    }

    // At this point the orbitals are sorted according to the energy but we
    // want to make sure that the hole and the particle MO appear where they
    // should, that is the holes in the virtual space and the particles in
    // the occupied space.
    // |(1) (2) ... (hole) | (particle) ...> will become
    // |(particle) (1) (2) ...  | ... (hole)>
    TempMatrix->zero();
    TempVector->zero();
    for (int h = 0; h < nirrep_; ++h){
        int m = apartpi[h];  // Offset by the number of holes
        int nso = nsopi_[h];
        int nmo = nmopi_[h];
        double** T_h = TempMatrix->pointer(h);
        double** C_h = Ca_->pointer(h);
        double** Cpo_h = Cpo->pointer(h);
        for (int p = 0; p < m; ++p){
            for (int q = 0; q < nso; ++q){
                T_h[q][p] = Cpo_h[q][p];
            }
        }
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
    }

    Ca_->copy(TempMatrix);
    epsilon_a_->copy(TempVector.get());

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

    // BETA
    diagonalize_F(Fb_, Cb_, epsilon_b_);

    if (debug_) {
        Ca_->print(outfile);
        Cb_->print(outfile);
    }
}

void UCKS::form_C_CHP_algorithm()
{
    int nstate = static_cast<int>(dets.size());
    if (nstate > 1)
        throw FeatureNotImplemented("CKS", "Cannot treat more than one excited state in the CHP method", __FILE__, __LINE__);

    // Data structures to save the hole information
    Dimension aholepi(nirrep_,"Alpha holes per irrep");
    std::vector<SharedVector> holes_Ca;
    std::vector<int> holes_h;
    std::vector<double> holes_energy;

    // Data structures to save the particle information
    Dimension apartpi(nirrep_,"Alpha particles per irrep");
    std::vector<SharedVector> parts_Ca;
    std::vector<int> parts_h;
    std::vector<double> parts_energy;

    // Compute the hole and particle states
    for (int m = 0; m < nstate; ++m){
        // Transform Fa to the MO basis of state m
        TempMatrix->transform(Fa_,dets[m]->Ca());
        // Grab the occ and vir blocks
        // |--------|--------|
        // |        |        |
        // | PoFaPo |        |
        // |        |        |
        // |--------|--------|
        // |        |        |
        // |        | PvFaPv |
        // |        |        |
        // |--------|--------|
        // Grab the occ block of Fa
        extract_square_subblock(TempMatrix,PoFPo_,true,dets[m]->nalphapi(),1.0e9);
        // Grab the vir block of Fa
        extract_square_subblock(TempMatrix,PvFPv_,false,dets[m]->nalphapi(),1.0e9);

        // Diagonalize the hole block
        PoFPo_->diagonalize(Uo_,lambda_o_);
        std::vector<boost::tuple<double,int,int> > sorted_holes; // (energy,irrep,mo in irrep)
        for (int h = 0; h < nirrep_; ++h){
            int nmo = nmopi_[h];
            for (int p = 0; p < nmo; ++p){
                if (lambda_o_->get(h,p) < 1.0e6){
                    sorted_holes.push_back(boost::make_tuple(lambda_o_->get(h,p),h,p));
                }
            }
        }
        std::sort(sorted_holes.begin(),sorted_holes.end());

        // Diagonalize the particle block
        PvFPv_->diagonalize(Uv_,lambda_v_);
        std::vector<boost::tuple<double,int,int> > sorted_vir; // (energy,irrep,mo in irrep)
        for (int h = 0; h < nirrep_; ++h){
            int nmo = nmopi_[h];
            for (int p = 0; p < nmo; ++p){
                sorted_vir.push_back(boost::make_tuple(lambda_v_->get(h,p),h,p));  // N.B. shifted wrt to full indexing
            }
        }
        std::sort(sorted_vir.begin(),sorted_vir.end());

        boost::tuple<double,int,int> hole;
        boost::tuple<double,int,int> particle;
        std::vector<boost::tuple<double,int,int,double,int,int,double> > sorted_hp_pairs;

        // Extract the hole alpha orbital according to an energy criteria (this needs a generalization)
        bool do_core_excitation = false;
        double hole_energy_shift = 0.0;
        if (KS::options_.get_str("CDFT_EXC_HOLE") == "VALENCE"){
            // For valence excitations select the highest lying orbital (HOMO-like)
        }else if(KS::options_.get_str("CDFT_EXC_HOLE") == "CORE"){
            do_core_excitation = true;
            // Get the energy of the lowest lying orbital (1s-like)
            hole_energy_shift = sorted_holes.front().get<0>();
        }
        CharacterTable ct = KS::molecule_->point_group()->char_table();

        // Determine the hole/particle pair to follow
        // Compute the symmetry adapted hole/particle pairs
        for (int h_h = 0; h_h < nirrep_; ++h_h){
            int nmo_h = nmopi_[h_h];
            for (int h = 0; h < nmo_h; ++h){
                double e_h = lambda_o_->get(h_h,h);
                for (int h_p = 0; h_p < nirrep_; ++h_p){
                    int nmo_p = nmopi_[h_p];
                    for (int p = 0; p < nmo_p; ++p){
                        double e_p = lambda_v_->get(h_p,p);
                        if ((e_h < 1.0e6) and (e_p < 1.0e6)){  // Test to eliminate the fake eigenvalues added to the PFP matrices
                            double e_hp = do_core_excitation ? (e_p + e_h - hole_energy_shift) : (e_p - e_h);
                            int symm = h_h ^ h_p ^ ground_state_symmetry_;
                            if(not do_symmetry or (symm == excited_state_symmetry_)){ // Test for symmetry
                                sorted_hp_pairs.push_back(boost::make_tuple(e_hp,h_h,h,e_h,h_p,p,e_p));  // N.B. shifted wrt to full indexing
//                                fprintf(outfile, "  %s  gamma(h) = %s, gamma(p) = %s, gamma(hp) = %s, gamma(Phi-hp) = %s \n",do_symmetry ? "true" : "false",
//                                        ct.gamma(h_h).symbol(),ct.gamma(h_p).symbol(),ct.gamma(h_h ^ h_p).symbol(),
//                                        ct.gamma(symm).symbol());
                            }
                        }
                    }
                }
            }
        }

        fprintf(outfile, "\n  Ground state symmetry: %s\n",ct.gamma(ground_state_symmetry_).symbol());
        fprintf(outfile, "  Excited state symmetry: %s\n",ct.gamma(excited_state_symmetry_).symbol());
        std::sort(sorted_hp_pairs.begin(),sorted_hp_pairs.end());
        if(iteration_ == 0){
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
                        dets[m]->nalphapi()[sorted_hp_pairs[n].get<4>()] + sorted_hp_pairs[n].get<5>() + 1,
                        ct.gamma(sorted_hp_pairs[n].get<4>()).symbol(),
                        energy_hp * _hartree2ev);
            }
            fprintf(outfile, "  --------------------------------------\n");
        }

        int hole_h = sorted_hp_pairs[0].get<1>();
        int hole_mo = sorted_hp_pairs[0].get<2>();
        double hole_energy = sorted_hp_pairs[0].get<3>();

        int part_h = sorted_hp_pairs[0].get<4>();
        int part_mo = sorted_hp_pairs[0].get<5>();
        double part_energy = sorted_hp_pairs[0].get<6>();

        fprintf(outfile,"   constrained hole     %d :(irrep = %d,mo = %d,energy = %.6f)\n",
                        m,hole_h,hole_mo,hole_energy);
        fprintf(outfile,"   constrained particle %d :(irrep = %d,mo = %d,energy = %.6f)\n",
                m,part_h,part_mo + dets[m]->nalphapi()[part_h],part_energy);

        // Compute the hole orbital
        SharedVector hole_Ca = factory_->create_shared_vector("Hole");
        for (int p = 0; p < nsopi_[hole_h]; ++p){
            double c_p = 0.0;
            for (int i = 0; i < dets[m]->nalphapi()[hole_h]; ++i){
                c_p += dets[m]->Ca()->get(hole_h,p,i) * Uo_->get(hole_h,i,hole_mo) ;
            }
            hole_Ca->set(hole_h,p,c_p);
        }
        holes_Ca.push_back(hole_Ca);
        holes_h.push_back(hole_h);
        holes_energy.push_back(hole_energy);
        aholepi[hole_h] += 1;

        // Compute the particle orbital
        SharedVector part_Ca = factory_->create_shared_vector("Particle");
        for (int p = 0; p < nsopi_[part_h]; ++p){
            double c_p = 0.0;
            int maxa = nmopi_[part_h] - dets[m]->nalphapi()[part_h];
            for (int a = 0; a < maxa; ++a){
                c_p += dets[m]->Ca()->get(part_h,p,dets[m]->nalphapi()[part_h] + a) * Uv_->get(part_h,a,part_mo) ;
            }
            part_Ca->set(part_h,p,c_p);
        }
        parts_Ca.push_back(part_Ca);
        parts_h.push_back(part_h);
        parts_energy.push_back(part_energy);
        apartpi[part_h] += 1;
    }

    // Put the hole and particle orbitals in Ch_ and Cp_
    std::vector<int> hole_offset(nirrep_,0);
    std::vector<int> part_offset(nirrep_,0);
    Cp_->zero();
    Ch_->zero();
    for (int m = 0; m < nstate; ++m){
        int hole_h = holes_h[m];
        Ch_->set_column(hole_h,hole_offset[hole_h],holes_Ca[m]);
        hole_offset[hole_h] += 1;
        int part_h = parts_h[m];
        Cp_->set_column(part_h,part_offset[part_h],parts_Ca[m]);
        part_offset[part_h] += 1;
    }

    // Frozen spectator orbital algorithm
    // Transform the ground state orbitals to the representation which diagonalizes the
    // the PoFaPo and PvFaPv blocks
    // Form the transformation matrix that diagonalizes the PoFaPo and PvFaPv blocks
    // |----|----|
    // | Uo | 0  |
    // |----|----|
    // | 0  | Uv |
    // |----|----|
    TempMatrix->zero();
    for (int h = 0; h < nirrep_; ++h){
        int nocc = dets[0]->nalphapi()[h];
        int nvir = nmopi_[h] - nocc;
        if (nocc != 0){
            double** Temp_h = TempMatrix->pointer(h);
            double** Uo_h = Uo_->pointer(h);
            for (int i = 0; i < nocc; ++i){
                epsilon_a_->set(h,i,lambda_o_->get(h,i));
                for (int j = 0; j < nocc; ++j){
                    Temp_h[i][j] = Uo_h[i][j];
                }
            }
        }
        if (nvir != 0){
            double** Temp_h = TempMatrix->pointer(h);
            double** Uv_h = Uv_->pointer(h);
            for (int i = 0; i < nvir; ++i){
                epsilon_a_->set(h,i + nocc,lambda_v_->get(h,i));
                for (int j = 0; j < nvir; ++j){
                    Temp_h[i + nocc][j + nocc] = Uv_h[i][j];
                }
            }
        }
    }
    // Get the excited state orbitals: Ca(ex) = Ca(gs) * (Uo | Uv)
    Ca_->gemm(false,false,1.0,dets[0]->Ca(),TempMatrix,0.0);

    // Form the projector onto the orbitals orthogonal to the holes and particles in the excited state mo representation
    TempMatrix->zero();
    TempMatrix->gemm(false,true,1.0,Ch_,Ch_,0.0);
    TempMatrix->gemm(false,true,1.0,Cp_,Cp_,1.0);
    TempMatrix->transform(S_);
    TempMatrix->transform(Ca_);
    TempMatrix2->identity();
    TempMatrix2->subtract(TempMatrix);

    // Form the Fock matrix in the excited state basis, project out the h/p
    TempMatrix->transform(Fa_,Ca_);
    TempMatrix->transform(TempMatrix2);
    // If we want the relaxed orbitals diagonalize the Fock matrix and transform the MO coefficients
    if(KS::options_.get_str("CDFT_EXC_METHOD") == "CHP"){
        TempMatrix->diagonalize(TempMatrix2,epsilon_a_);
        TempMatrix->zero();
        TempMatrix->gemm(false,false,1.0,Ca_,TempMatrix2,0.0);
        Ca_->copy(TempMatrix);
    }else{
        // The orbitals don't change, but make sure that epsilon_a_ has the correct eigenvalues (some which are zero)
        for (int h = 0; h < nirrep_; ++h){
            for (int p = 0; p < nmopi_[h]; ++p){
                epsilon_a_->set(h,p,TempMatrix->get(h,p,p));
            }
        }
    }

    std::vector<boost::tuple<double,int,int> > sorted_spectators;
    for (int h = 0; h < nirrep_; ++h){
        for (int p = 0; p < nmopi_[h]; ++p){
            sorted_spectators.push_back(boost::make_tuple(epsilon_a_->get(h,p),h,p));
        }
    }
    std::sort(sorted_spectators.begin(),sorted_spectators.end());

    // Find the alpha occupation
    int assigned = 0;
    for (int h = 0; h < nirrep_; ++h){
        nalphapi_[h] = apartpi[h];
        assigned += apartpi[h];
    }
    for (int p = 0; p < nmo_; ++p){
        if (assigned < nalpha_){
            if(std::fabs(sorted_spectators[p].get<0>()) > 1.0e-6){  // !!! Check this out NB WARNING
                int h = sorted_spectators[p].get<1>();
                nalphapi_[h] += 1;
                assigned += 1;
            }
        }
    }
    nbetapi_ = dets[0]->nbetapi();
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
        double** T_h = TempMatrix->pointer(h);
        double** C_h = Ca_->pointer(h);
        double** Cp_h = Cp_->pointer(h);
        double** Ch_h = Ch_->pointer(h);
        // First place the particles
        int m = 0;
        for (int p = 0; p < apartpi[h]; ++p){
            for (int q = 0; q < nso; ++q){
                T_h[q][m] = Cp_h[q][p];
            }
            m += 1;
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
        for (int p = 0; p < aholepi[h]; ++p){
            for (int q = 0; q < nso; ++q){
                T_h[q][m] = Ch_h[q][p];
            }
            m += 1;
        }
    }
    Ca_->copy(TempMatrix);
    epsilon_a_->copy(TempVector.get());

    // BETA
    if(KS::options_.get_str("CDFT_EXC_METHOD") == "CHP"){
        diagonalize_F(Fb_, Cb_, epsilon_b_);
    }else{
        // Unrelaxed procedure, but still find MOs which diagonalize the occupied block
        // Transform Fb to the MO basis of the ground state
        TempMatrix->transform(Fb_,dets[0]->Cb());
        // Grab the occ block of Fb
        extract_square_subblock(TempMatrix,PoFPo_,true,dets[0]->nbetapi(),1.0e9);
        // Grab the vir block of Fa
        extract_square_subblock(TempMatrix,PvFPv_,false,dets[0]->nbetapi(),1.0e9);
        // Diagonalize the hole block
        PoFPo_->diagonalize(Uo_,lambda_o_);
        // Diagonalize the particle block
        PvFPv_->diagonalize(Uv_,lambda_v_);
        // Form the transformation matrix that diagonalizes the PoFaPo and PvFaPv blocks
        // |----|----|
        // | Uo | 0  |
        // |----|----|
        // | 0  | Uv |
        // |----|----|
        TempMatrix->zero();
        for (int h = 0; h < nirrep_; ++h){
            int nocc = dets[0]->nbetapi()[h];
            int nvir = nmopi_[h] - nocc;
            if (nocc != 0){
                double** Temp_h = TempMatrix->pointer(h);
                double** Uo_h = Uo_->pointer(h);
                for (int i = 0; i < nocc; ++i){
                    epsilon_b_->set(h,i,lambda_o_->get(h,i));
                    for (int j = 0; j < nocc; ++j){
                        Temp_h[i][j] = Uo_h[i][j];
                    }
                }
            }
            if (nvir != 0){
                double** Temp_h = TempMatrix->pointer(h);
                double** Uv_h = Uv_->pointer(h);
                for (int i = 0; i < nvir; ++i){
                    epsilon_b_->set(h,i + nocc,lambda_v_->get(h,i));
                    for (int j = 0; j < nvir; ++j){
                        Temp_h[i + nocc][j + nocc] = Uv_h[i][j];
                    }
                }
            }
        }
        // Get the excited state orbitals: Cb(ex) = Cb(gs) * (Uo | Uv)
        Cb_->gemm(false,false,1.0,dets[0]->Cb(),TempMatrix,0.0);
    }
    if (debug_) {
        Ca_->print(outfile);
        Cb_->print(outfile);
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
    if (functional_->is_x_lrc()) {
        exchange_E -=  beta*Da_->vector_dot(wKa_);
        exchange_E -=  beta*Db_->vector_dot(wKb_);
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
//            if(do_excitation){
////                double E_T = compute_triplet_correction();
////                double exc_energy = E_ - E_T - ground_state_energy;
////                fprintf(outfile,"  Excited triplet state : excitation energy = %9.6f Eh = %8.4f eV = %9.1f cm**-1 \n",
////                        exc_energy,exc_energy * _hartree2ev, exc_energy * _hartree2wavenumbers);
////                exc_energy = E_ + E_T - ground_state_energy;
////                fprintf(outfile,"  Excited singlet state : excitation energy = %9.6f Eh = %8.4f eV = %9.1f cm**-1 \n",
////                        exc_energy,exc_energy * _hartree2ev, exc_energy * _hartree2wavenumbers);
//            }
            return true;
        }else{
            return false;
        }
    }else{
        if(energy_test and density_test and cycle_test){
//            if(do_excitation){
//                double E_T = compute_triplet_correction();
//                fprintf(outfile,"  Energy corrected for triplet component = %20.12f (%.12f)",2.0 * E_ - E_T,E_T);
//            }
            return true;
        }
        return false;
    }
}

void UCKS::save_information()
{
    dets.push_back(SharedDeterminant(new Determinant(E_,Ca_,Cb_,nalphapi_,nbetapi_)));
    if(do_excitation){
        double mixlet_exc_energy = E_ - ground_state_energy;
        fprintf(outfile,"  Excited mixed state   : excitation energy = %9.6f Eh = %8.4f eV = %9.1f cm**-1 \n",
                mixlet_exc_energy,mixlet_exc_energy * _hartree2ev, mixlet_exc_energy * _hartree2wavenumbers);

        if(KS::options_.get_bool("CDFT_SPIN_ADAPT")){
            spin_adapt_mixed_excitation();
        }
    }
}

void UCKS::save_fock()
{
    if(not do_excitation){
        UHF::save_fock();
    }else{
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

void UCKS::spin_adapt_mixed_excitation()
{
    SharedDeterminant D1 = SharedDeterminant(new Determinant(E_,Ca_,Cb_,nalphapi_,nbetapi_));
    SharedDeterminant D2 = SharedDeterminant(new Determinant(E_,Cb_,Ca_,nbetapi_,nalphapi_));
    std::pair<double,double> M12 = matrix_element(D1,D2);
    double S12 = M12.first;
    double H12 = M12.second;
    double triplet_exc_energy = E_ - H12 - ground_state_energy;
    fprintf(outfile,"  Excited triplet state : excitation energy = %9.6f Eh = %8.4f eV = %9.1f cm**-1 \n",
            triplet_exc_energy,triplet_exc_energy * _hartree2ev, triplet_exc_energy * _hartree2wavenumbers);
    double singlet_exc_energy = E_ + H12 - ground_state_energy;
    fprintf(outfile,"  Excited singlet state : excitation energy = %9.6f Eh = %8.4f eV = %9.1f cm**-1 \n",
            singlet_exc_energy,singlet_exc_energy * _hartree2ev, singlet_exc_energy * _hartree2wavenumbers);
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

void UCKS::copy_block(SharedMatrix A, SharedMatrix B, Dimension rowspi, Dimension colspi,
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
                    B_h[i + B_row_offset][j + B_col_offset] = A_h[i + A_row_offset][j + A_col_offset];
                }
            }
        }
    }
}

}} // Namespaces
