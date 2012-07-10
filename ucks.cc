
#include <ucks.h>
#include <libmints/view.h>
#include <libmints/mints.h>
#include <libfock/apps.h>
#include <libfock/v.h>
#include <libfock/jk.h>
#include <libdisp/dispersion.h>
#include <liboptions/liboptions.h>
#include <libciomr/libciomr.h>
#include <libqt/qt.h>
#include "boost/tuple/tuple.hpp"
#include "boost/tuple/tuple_comparison.hpp"

using namespace psi;

namespace psi{ namespace scf{

UCKS::UCKS(Options &options, boost::shared_ptr<PSIO> psio)
    : UKS(options, psio), optimize_Vc(false), gradW_threshold_(1.0e-9),nW_opt(0)
{
    do_excitation = false;
    nexclude_occ = 0;
    nexclude_vir = 0;
    init();
}

UCKS::UCKS(Options &options, boost::shared_ptr<PSIO> psio, boost::shared_ptr<UCKS> ref_scf)
    : UKS(options, psio), optimize_Vc(false), gradW_threshold_(1.0e-9),nW_opt(0), ref_scf_(ref_scf)
{
    do_excitation = true;
    nexclude_occ = ref_scf->nexclude_occ;
    nexclude_vir = ref_scf->nexclude_vir;
    init();
}

void UCKS::init()
{
    fprintf(outfile,"\n  ==> Constrained DFT (UCKS) <==\n\n");

    optimize_Vc = KS::options_.get_bool("OPTIMIZE_VC");
    gradW_threshold_ = KS::options_.get_double("W_CONVERGENCE");
    fprintf(outfile,"  gradW threshold = :%9.2e\n",gradW_threshold_);
    nfrag = basisset()->molecule()->nfragments();
    fprintf(outfile,"  Number of fragments: %d\n",nfrag);

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
    aocc_num_ = factory_->create_shared_vector("Alpha occupation number");
    bocc_num_ = factory_->create_shared_vector("Beta occupation number");
    svds = factory_->create_shared_vector("SVD sigma");
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
    svdV = factory_->create_shared_matrix("SVD V");
    svdU = factory_->create_shared_matrix("SVD U");
    Ua = factory_->create_shared_matrix("U alpha");
    Ub = factory_->create_shared_matrix("U beta");
    hessW = SharedMatrix(new Matrix("hessW",nconstraints,nconstraints));
    hessW_BFGS = SharedMatrix(new Matrix("hessW_BFGS",nconstraints,nconstraints));

    if(do_excitation){
        fprintf(outfile,"  Saving the reference orbitals for an excited state computation\n");

        Dimension nocc_alphapi = ref_scf_->nalphapi_;
        Dimension nvir_alphapi = ref_scf_->nmopi_ - nocc_alphapi;
        PoFaPo_ = SharedMatrix(new Matrix("PoFaPo",nocc_alphapi,nocc_alphapi));
        PvFaPv_ = SharedMatrix(new Matrix("PvFaPv",nvir_alphapi,nvir_alphapi));
        Uo = SharedMatrix(new Matrix("PoFaPo",nocc_alphapi,nocc_alphapi));
        Uv = SharedMatrix(new Matrix("PvFaPv",nvir_alphapi,nvir_alphapi));
        lambda_o = SharedVector(new Vector("lambda_o",nocc_alphapi));
        lambda_v = SharedVector(new Vector("lambda_v",nvir_alphapi));
        Dimension nocc_betapi = ref_scf_->nbetapi_;
        Dimension nvir_betapi = ref_scf_->nmopi_ - nocc_betapi;
        PoFbPo_ = SharedMatrix(new Matrix("PoFbPo",nocc_betapi,nocc_betapi));
        PvFbPv_ = SharedMatrix(new Matrix("PvFbPv",nvir_betapi,nvir_betapi));
        Uob = SharedMatrix(new Matrix("Uob",nocc_betapi,nocc_betapi));
        Uvb = SharedMatrix(new Matrix("Uvb",nvir_betapi,nvir_betapi));
        lambda_ob = SharedVector(new Vector("lambda_o",nocc_betapi));
        lambda_vb = SharedVector(new Vector("lambda_v",nvir_betapi));
        // Save the reference state MOs and density matrix
        state_epsilon_a.push_back(SharedVector(ref_scf_->epsilon_a_->clone()));
        state_Ca.push_back(ref_scf_->Ca_->clone());
        state_Cb.push_back(ref_scf_->Cb_->clone());
        state_Da.push_back(ref_scf_->Da_->clone());
        state_Db.push_back(ref_scf_->Db_->clone());
        state_nalphapi.push_back(ref_scf_->nalphapi_);
        state_nbetapi.push_back(ref_scf_->nbetapi_);
        Fa_->copy(ref_scf_->Fa_);
        Fb_->copy(ref_scf_->Fb_);

//        if(nexclude_occ == 0 and nexclude_vir){
//            // Find the lowest single excitations
//            std::vector<boost::tuple<double,int,int,int,int> > sorted_exc;
//            // Loop over occupied MOs
//            for (int hi = 0; hi < nirrep_; ++hi){
//                int nocci = ref_scf_->nalphapi_[0][hi];
//                for (int i = 0; i < nocc; ++i){
//                    for (int ha = 0; ha < nirrep_; ++ha){
//                        int nocca = ref_scf_->nalphapi_[0][ha];
//                        for (int a = nocca; a < nmopi_[ha]; ++a){
//                            sorted_exc.push_back(boost::make_tuple(
//                        }
//                    }
//                    int nocc = state_nalphapi[0][h];

//                int nvir = nmopi_[h] - nocc;
//                for (int i = 0; i < nocc; ++i){
//                    sorted_occ.push_back(boost::make_tuple(lambda_o->get(h,i),h,i));
//                }
//                for (int i = 0; i < nvir; ++i){
//                    sorted_vir.push_back(boost::make_tuple(lambda_v->get(h,i),h,i));
//                }
//            }
//            std::sort(sorted_occ.begin(),sorted_occ.end());
//            std::sort(sorted_vir.begin(),sorted_vir.end());
//        }
    }

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

UCKS::~UCKS()
{
}

void UCKS::guess()
{
    if(do_excitation){
        form_initial_C();
        find_occupation();
        form_D();
        E_ = compute_initial_E();
    }else{
        HF::guess();
    }
}

//void UCKS::find_occupation()
//{
//    if(not do_excitation){
//        // Find or read the ground state occupation
//        std::vector<std::pair<double, int> > pairs_a;
//        std::vector<std::pair<double, int> > pairs_b;
//        for (int h = 0; h < epsilon_a_->nirrep(); ++h) {
//            for (int i = 0; i < epsilon_a_->dimpi()[h]; ++i)
//                pairs_a.push_back(std::make_pair(epsilon_a_->get(h, i), h));
//        }
//        for (int h = 0; h < epsilon_b_->nirrep(); ++h) {
//            for (int i = 0; i < epsilon_b_->dimpi()[h]; ++i)
//                pairs_b.push_back(std::make_pair(epsilon_b_->get(h, i), h));
//        }
//        sort(pairs_a.begin(),pairs_a.end());
//        sort(pairs_b.begin(),pairs_b.end());

//        if(!input_docc_ and !input_socc_){
//            memset(nalphapi_, 0, sizeof(int) * epsilon_a_->nirrep());
//            for (int i=0; i<nalpha_; ++i)
//                nalphapi_[pairs_a[i].second]++;
//        }
//        if(!input_docc_ and !input_socc_){
//            memset(nbetapi_, 0, sizeof(int) * epsilon_b_->nirrep());
//            for (int i=0; i<nbeta_; ++i)
//                nbetapi_[pairs_b[i].second]++;
//        }

//        int old_socc[8];
//        int old_docc[8];
//        for(int h = 0; h < nirrep_; ++h){
//            old_socc[h] = soccpi_[h];
//            old_docc[h] = doccpi_[h];
//        }

//        for (int h = 0; h < nirrep_; ++h) {
//            soccpi_[h] = std::abs(nalphapi_[h] - nbetapi_[h]);
//            doccpi_[h] = std::min(nalphapi_[h] , nbetapi_[h]);
//        }

//        bool occ_changed = false;
//        for(int h = 0; h < nirrep_; ++h){
//            if( old_socc[h] != soccpi_[h] or old_docc[h] != doccpi_[h]){
//                occ_changed = true;
//                break;
//            }
//        }

//        // If print > 2 (diagnostics), print always
//        if((print_ > 2 or (print_ and occ_changed)) and iteration_ > 0){
//            if (Communicator::world->me() == 0)
//                fprintf(outfile, "\tOccupation by irrep:\n");
//            print_occupation();
//        }
//    }
//}

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
    }else{
        // Excited state: use specialized code

        // Transform Fa to the ground state MO basis
        TempMatrix->transform(Fa_,state_Ca[0]);

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
        for (int h = 0; h < nirrep_; ++h){
            int nocc = state_nalphapi[0][h];
            int nvir = nmopi_[h] - nocc;
            if (nocc != 0){
                double** Temp_h = TempMatrix->pointer(h);
                double** PoFaPo_h = PoFaPo_->pointer(h);
                for (int i = 0; i < nocc; ++i){
                    for (int j = 0; j < nocc; ++j){
                        PoFaPo_h[i][j] = Temp_h[i][j];
                    }
                }
            }
            if (nvir != 0){
                double** Temp_h = TempMatrix->pointer(h);
                double** PvFaPv_h = PvFaPv_->pointer(h);
                for (int i = 0; i < nvir; ++i){
                    for (int j = 0; j < nvir; ++j){
                        PvFaPv_h[i][j] = Temp_h[i + nocc][j + nocc];
                    }
                }
            }
        }
        PoFaPo_->diagonalize(Uo,lambda_o);
        PvFaPv_->diagonalize(Uv,lambda_v);

        // Sort the orbitals according to the eigenvalues of PoFaPo and PvFaPv
        std::vector<boost::tuple<double,int,int> > sorted_occ;
        std::vector<boost::tuple<double,int,int> > sorted_vir;
        for (int h = 0; h < nirrep_; ++h){
            int nocc = state_nalphapi[0][h];
            int nvir = nmopi_[h] - nocc;
            for (int i = 0; i < nocc; ++i){
                sorted_occ.push_back(boost::make_tuple(lambda_o->get(h,i),h,i));
            }
            for (int i = 0; i < nvir; ++i){
                sorted_vir.push_back(boost::make_tuple(lambda_v->get(h,i),h,i + nocc));  // N.B. shifted to full indexing
            }
        }
        std::sort(sorted_occ.begin(),sorted_occ.end());
        std::sort(sorted_vir.begin(),sorted_vir.end());

        // Extract the hole and particle alpha orbitals according to an energy criteria (this needs a generalization)
        boost::tuple<double,int,int> hole;
        if (KS::options_.get_str("CDFT_EXC_HOLE") == "VALENCE"){
            // For valence excitations select the highest lying orbital (HOMO-like)
            hole = sorted_occ.back();
        }else if(KS::options_.get_str("CDFT_EXC_HOLE") == "CORE"){
            // For core excitations select the lowest lying orbital (1s-like)
            hole = sorted_occ.front();
        }
        // In the case of particle, we assume that we are always interested in the lowest lying orbitals
        boost::tuple<double,int,int> particle = sorted_vir.front();

        // Form the matrix that diagonalizes the PoFaPo and PvFaPv blocks
        // |----|----|
        // | Uo | 0  |
        // |----|----|
        // | 0  | Uv |
        // |----|----|
        TempMatrix->zero();
        for (int h = 0; h < nirrep_; ++h){
            int nocc = state_nalphapi[0][h];
            int nvir = nmopi_[h] - nocc;
            if (nocc != 0){
                double** Temp_h = TempMatrix->pointer(h);
                double** Uo_h = Uo->pointer(h);
                for (int i = 0; i < nocc; ++i){
                    epsilon_a_->set(h,i,lambda_o->get(h,i));
                    for (int j = 0; j < nocc; ++j){
                        Temp_h[i][j] = Uo_h[i][j];
                    }
                }
            }
            if (nvir != 0){
                double** Temp_h = TempMatrix->pointer(h);
                double** Uv_h = Uv->pointer(h);
                for (int i = 0; i < nvir; ++i){
                    epsilon_a_->set(h,i + nocc,lambda_v->get(h,i));
                    for (int j = 0; j < nvir; ++j){
                        Temp_h[i + nocc][j + nocc] = Uv_h[i][j];
                    }
                }
            }
        }

        // Get the excited state orbitals: Ca(ex) = Ca(gs) * (Uo | Uv)
        Ca_->gemm(false,false,1.0,state_Ca[0],TempMatrix,0.0);
        fprintf(outfile,"   hole/particle pair :(irrep = %d,mo = %d,energy = %.6f) -> (irrep = %d,mo = %d,energy = %.6f)\n",
                        hole.get<1>(),hole.get<2>(),hole.get<0>(),
                        particle.get<1>(),particle.get<2>(),particle.get<0>());

        // Save the hole and particle information and at the same time zero the columns in Ca_
        current_excited_state = SharedExcitedState(new ExcitedState);
        SharedVector hole_mo = Ca_->get_column(hole.get<1>(),hole.get<2>());
        Ca_->zero_column(hole.get<1>(),hole.get<2>());
        epsilon_a_->set(hole.get<1>(),hole.get<2>(),0.0);
        SharedVector particle_mo = Ca_->get_column(particle.get<1>(),particle.get<2>());
        Ca_->zero_column(particle.get<1>(),particle.get<2>());
        epsilon_a_->set(particle.get<1>(),particle.get<2>(),0.0);
        current_excited_state->add_hole(hole.get<1>(),hole_mo,true);
        current_excited_state->add_particle(particle.get<1>(),particle_mo,true);

        // Adjust the occupation (nalphapi_,nbetapi_)
        for (int h = 0; h < nirrep_; ++h){
            nalphapi_[h] = state_nalphapi[0][h];
            nbetapi_[h] = state_nbetapi[0][h];
        }
        nalphapi_[hole.get<1>()] -= 1;
        nalphapi_[particle.get<1>()] += 1;

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

        // If print > 2 (diagnostics), print always
        if((print_ > 2 || (print_ && occ_changed)) && iteration_ > 0){
            if (Communicator::world->me() == 0)
                fprintf(outfile, "\tOccupation by irrep:\n");
            print_occupation();
        }

        // Optionally, include relaxation effects
        if(KS::options_.get_str("CDFT_EXC_METHOD") == "CEE-R"){
            // Transform Fa to the excited state MO basis, this includes the hole and particle states
            TempMatrix->transform(Fa_,Ca_);

            // Zero the terms that couple the hole, particle, and the rest of the orbitals
            // |--------|--------|
            // |       0|0       |
            // |       0|0       |
            // |00000000|00000000|
            // |--------|--------|
            // |00000000|00000000|
            // |       0|0       |
            // |       0|0       |
            // |--------|--------|
            {
                // Zero the hole couplings
                int h = hole.get<1>();
                int i = hole.get<2>();
                int nmo = nmopi_[h];
                if (nmo != 0){
                    double** Temp_h = TempMatrix->pointer(h);
                    for (int p = 0; p < nmo; ++p){
                        if(p != i){
                            Temp_h[i][p] = Temp_h[p][i] = 0.0;
                        }
                    }
                }

            }
            {
                // Zero the LUMO couplings
                int h = particle.get<1>();
                int i = particle.get<2>();
                int nmo = nmopi_[h];
                if (nmo != 0){
                    double** Temp_h = TempMatrix->pointer(h);
                    for (int p = 0; p < nmo; ++p){
                        if(p != i){
                            Temp_h[i][p] = Temp_h[p][i] = 0.0;
                        }
                    }
                }
            }

            TempMatrix->diagonalize(TempMatrix2,epsilon_a_);
            TempMatrix->copy(Ca_);
            Ca_->gemm(false,false,1.0,TempMatrix,TempMatrix2,0.0);
        }
        // At this point the orbitals are sorted according to the energy but we
        // want to make sure that the hole and the particle MO appear where they
        // should, that is the holes in the virtual space and the particles in
        // the occupied space.
        // |(1) (2) ... (hole) | (particle) ...> will become
        // |(particle) (1) (2) ...  | ... (hole)>
        std::vector<int> naholepi = current_excited_state->aholepi();
        std::vector<int> napartpi = current_excited_state->apartpi();
        TempMatrix->zero();
        TempVector->zero();
        for (int h = 0; h < nirrep_; ++h){
            int m = naholepi[h];  // Offset by the number of holes
            int nso = nsopi_[h];
            int nmo = nmopi_[h];
            double** T_h = TempMatrix->pointer();
            double** C_h = Ca_->pointer();
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
        // Place the hole orbital in the last MO of its irrep
        TempMatrix->set_column(hole.get<1>(),nmopi_[hole.get<1>()],hole_mo);
        TempVector->set(hole.get<1>(),nmopi_[hole.get<1>()],hole.get<0>());
        // Place the particle orbital in the first MO of its irrep
        TempMatrix->set_column(particle.get<1>(),0,particle_mo);
        TempVector->set(particle.get<1>(),0,particle.get<0>());
        Ca_->copy(TempMatrix);
        epsilon_a_->copy(TempVector.get());

        // BETA
        diagonalize_F(Fb_, Cb_, epsilon_b_);

        //find_occupation();

        if (debug_) {
            Ca_->print(outfile);
            Cb_->print(outfile);
        }
    }
}

void UCKS::form_D()
{
    for (int h = 0; h < nirrep_; ++h) {
        int nso = nsopi_[h];
        int nmo = nmopi_[h];
        int na = nalphapi_[h];
        int nb = nbetapi_[h];

        if (nso == 0 || nmo == 0) continue;

        double* aocc_num_h = aocc_num_->pointer(h);
        double* bocc_num_h = bocc_num_->pointer(h);
        double** Ca = Ca_->pointer(h);
        double** Cb = Cb_->pointer(h);
        double** Da = Da_->pointer(h);
        double** Db = Db_->pointer(h);

        if (na == 0)
            ::memset(static_cast<void*>(Da[0]), '\0', sizeof(double)*nso*nso);
        if (nb == 0)
            ::memset(static_cast<void*>(Db[0]), '\0', sizeof(double)*nso*nso);
        for (int mu = 0; mu < nso; ++mu){
            for (int nu = 0; nu < nso; ++nu){
                for (int p = 0; p < nmo; ++p){
                    Da[mu][nu] += Ca[mu][p] * Ca[nu][p] * aocc_num_h[p];
                    Db[mu][nu] += Cb[mu][p] * Cb[nu][p] * bocc_num_h[p];
                }
            }
        }
    }

    Dt_->copy(Da_);
    Dt_->add(Db_);

    if (debug_) {
        fprintf(outfile, "in UCKS::form_D:\n");
        Da_->print();
        Db_->print();
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

    if(optimize_Vc){
        bool constraint_test = gradW->norm() < gradW_threshold_;
        constraint_optimization();
        if(energy_test and density_test and constraint_test){
            return true;
        }else{
            return false;
        }
    }else{
        if(energy_test and density_test){
            if(do_excitation){
                double E_T = compute_triplet_correction();
                fprintf(outfile,"  Energy corrected for triplet component = %20.12f (%.12f)",2.0 * E_ - E_T,E_T);
            }
            return true;
        }
        return false;
    }
}

double UCKS::compute_triplet_correction()
{
    corresponding_ab_mos();
    // Modify the occupation numbers to get a triplet
    for (int h = 0; h < nirrep_; ++h) {
        int na = nalphapi_[h];
        int nb = nbetapi_[h];
        nalphapi_[h] = std::max(na,nb);
        nbetapi_[h]  = std::min(na,nb);
    }
    fprintf(outfile, "\tNA   [ ");
    for(int h = 0; h < nirrep_-1; ++h) fprintf(outfile, " %4d,", nalphapi_[h]);
    fprintf(outfile, " %4d ]\n", nalphapi_[nirrep_-1]);
    fprintf(outfile, "\tNB   [ ");
    for(int h = 0; h < nirrep_-1; ++h) fprintf(outfile, " %4d,", nbetapi_[h]);
    fprintf(outfile, " %4d ]\n", nbetapi_[nirrep_-1]);
    // Compute the density matrices with the new occupation
    form_D();
    // Compute the triplet energy from the density matrices
    double triplet_energy_aa = compute_E();
    // Modify the occupation numbers to get a triplet
    for (int h = 0; h < nirrep_; ++h) {
        int na = nalphapi_[h];
        int nb = nbetapi_[h];
        nalphapi_[h] = std::max(na,nb);
        nbetapi_[h]  = std::min(na,nb);
    }
    fprintf(outfile, "\tNA   [ ");
    for(int h = 0; h < nirrep_-1; ++h) fprintf(outfile, " %4d,", nalphapi_[h]);
    fprintf(outfile, " %4d ]\n", nalphapi_[nirrep_-1]);
    fprintf(outfile, "\tNB   [ ");
    for(int h = 0; h < nirrep_-1; ++h) fprintf(outfile, " %4d,", nbetapi_[h]);
    fprintf(outfile, " %4d ]\n", nbetapi_[nirrep_-1]);
    // Compute the density matrices with the new occupation
    form_D();
    // Compute the triplet energy from the density matrices
    double triplet_energy_bb = compute_E();
    return 0.5 * (triplet_energy_aa + triplet_energy_bb);
}

double UCKS::compute_overlap(int n)
{
//    // Orthogonality test
//    Temp->gemm(false,false,1.0,state_Da[0],S_,0.0);
//    // DSC'
//    Ua->gemm(false,false,1.0,Temp,Ca_,0.0);
//    Ua->print();

//    // Temp = 1 - DS
//    Temp2->identity();
//    Temp2->subtract(Temp);
//    // (1 - DS)C'
//    Ua->gemm(false,false,1.0,Temp2,Ca_,0.0);
//    Ua->print();


//    Temp->gemm(false,false,1.0,S_,Ca_,0.0);
//    Ua->gemm(true,false,1.0,state_Da[n],Temp,0.0);
//    Ua->print();
//    // Orthogonality test
//    Temp->gemm(false,false,1.0,PvFaPv_,Ca_,0.0);
//    Ua->gemm(true,false,1.0,Ca_,Temp,0.0);
//    Ua->print();

    // Alpha block
    TempMatrix->gemm(false,false,1.0,S_,Ca_,0.0);
    Ua->gemm(true,false,1.0,state_Ca[n],TempMatrix,0.0);
    //Ua->print();
    // Grab S_aa from Ua
    SharedMatrix S_aa = SharedMatrix(new Matrix("S_aa",state_nalphapi[n],nalphapi_));
    for (int h = 0; h < nirrep_; ++h) {
        int ngs_occ = state_nalphapi[0][h];
        int nex_occ = nalphapi_[h];
        if (ngs_occ == 0 or nex_occ == 0) continue;
        double** Ua_h = Ua->pointer(h);
        double** S_aa_h = S_aa->pointer(h);
        for (int i = 0; i < ngs_occ; ++i){
            for (int j = 0; j < nex_occ; ++j){
                S_aa_h[i][j] = Ua_h[i][j];
            }
        }
    }

    double detS_aa = 1.0;
    double traceS2_aa = 0.0;
    {
        boost::tuple<SharedMatrix, SharedVector, SharedMatrix> UsV = S_aa->svd_temps();
        S_aa->svd(UsV.get<0>(),UsV.get<1>(),UsV.get<2>());
        if(state_nalphapi[0] == nalphapi_){
            for (int h = 0; h < nirrep_; ++h) {
                for (int i = 0; i < UsV.get<1>()->dim(h); ++i){
                    detS_aa *= UsV.get<1>()->get(h,i);
                }
            }
        }else{
            detS_aa = 0.0;
        }
        for (int h = 0; h < nirrep_; ++h) {
            for (int i = 0; i < UsV.get<1>()->dim(h); ++i){
                traceS2_aa += std::pow(UsV.get<1>()->get(h,i),2.0);
            }
        }
    }

    // Beta block
    TempMatrix->gemm(false,false,1.0,S_,Cb_,0.0);
    Ub->gemm(true,false,1.0,state_Cb[n],TempMatrix,0.0);

    // Grab S_bb from Ub
    SharedMatrix S_bb = SharedMatrix(new Matrix("S_bb",state_nbetapi[n],nbetapi_));

    for (int h = 0; h < nirrep_; ++h) {
        int ngs_occ = state_nbetapi[0][h];
        int nex_occ = nbetapi_[h];
        if (ngs_occ == 0 or nex_occ == 0) continue;
        double** Ub_h = Ub->pointer(h);
        double** S_bb_h = S_bb->pointer(h);
        for (int i = 0; i < ngs_occ; ++i){
            for (int j = 0; j < nex_occ; ++j){
                S_bb_h[i][j] = Ub_h[i][j];
            }
        }
    }
    double detS_bb = 1.0;
    double traceS2_bb = 0.0;
    {
        boost::tuple<SharedMatrix, SharedVector, SharedMatrix> UsV = S_bb->svd_temps();
        S_bb->svd(UsV.get<0>(),UsV.get<1>(),UsV.get<2>());
        if(state_nbetapi[0] == nbetapi_){
            for (int h = 0; h < nirrep_; ++h) {
                for (int i = 0; i < UsV.get<1>()->dim(h); ++i){
                    detS_bb *= UsV.get<1>()->get(h,i);
                }
            }
        }else{
            detS_bb = 0.0;
        }
        for (int h = 0; h < nirrep_; ++h) {
            for (int i = 0; i < UsV.get<1>()->dim(h); ++i){
                traceS2_bb += std::pow(UsV.get<1>()->get(h,i),2.0);
            }
        }
    }
    fprintf(outfile,"   det(S_aa) = %.6f det(S_bb) = %.6f  <Phi|Phi'> = %.6f\n",detS_aa,detS_bb,detS_aa * detS_bb);
    fprintf(outfile,"   <Phi'|Poa|Phi'> = %.6f  <Phi'|Pob|Phi'> = %.6f  <Phi'|Po|Phi'> = %.6f\n",nalpha_ - traceS2_aa,nbeta_ - traceS2_bb,nalpha_ - traceS2_aa + nbeta_ - traceS2_bb);
    TempMatrix->transform(state_Da[0],S_);
    TempMatrix2->transform(TempMatrix,Ca_);
    return (detS_aa * detS_bb);
}

void UCKS::corresponding_ab_mos()
{
    // Form <phi_b|S|phi_a>
    TempMatrix->gemm(false,false,1.0,S_,Ca_,0.0);
    TempMatrix2->gemm(true,false,1.0,Cb_,TempMatrix,0.0);
    TempMatrix2->print();
    TempMatrix->zero();
    for (int h = 0; h < nirrep_; ++h) {
        int naocc = nalphapi_[h];
        int nbocc = nbetapi_[h];
        if (naocc == 0 or nbocc == 0) continue;
        double** S_occ_h = TempMatrix->pointer(h);
        double** S_h = TempMatrix2->pointer(h);
        for (int i = 0; i < naocc; ++i){
            for (int j = 0; j < nbocc; ++j){
                S_occ_h[i][j] = S_h[i][j];
            }
        }
    }
    TempMatrix->print();
    TempMatrix->svd(svdU,svds,svdV);
    svds->print();
}

}} // Namespaces



//        if(do_penalty){
//            // Find the alpha HOMO of the ground state wave function
//            int homo_h = 0;
//            int homo_p = 0;
//            double homo_e = -1.0e9;
//            for (int h = 0; h < nirrep_; ++h) {
//                int nocc = state_nalphapi[0][h] - 1;
//                if (nocc < 0) continue;
//                if(state_epsilon_a[0]->get(h,nocc) > homo_e){
//                    homo_h = h;
//                    homo_p = nocc;
//                    homo_e = state_epsilon_a[0]->get(h,nocc);
//                }
//            }
//            fprintf(outfile,"  The HOMO orbital has energy %.9f and is %d of irrep %d.\n",homo_e,homo_p,homo_h);
//            Pa = SharedMatrix(factory_->create_matrix("Penalty matrix alpha"));
//            for (int mu = 0; mu < nsopi_[homo_h]; ++mu){
//                for (int nu = 0; nu < nsopi_[homo_h]; ++nu){
//                    double P_mn = 1000000.0 * state_Ca[0]->get(homo_h,mu,homo_p) * state_Ca[0]->get(homo_h,nu,homo_p);
//                    Pa->set(homo_h,mu,nu,P_mn);
//                }
//            }
//            Pa->transform(S_);
//        }



//    if(gs_scf_ and do_excitation){
//        // Form the projected Fock matrices
//        // Po = DS
//        Temp->gemm(false,false,1.0,state_Da[0],S_,0.0);
//        // SDFDS
//        PoFaPo_->transform(Fa_,Temp);
//        // Temp = 1 - DS
//        Temp2->identity();
//        Temp2->subtract(Temp);
//        PvFaPv_->transform(Fa_,Temp2);
//    }



////// EXCITED STATES
//        Temp->copy(Fa_);
//        Temp->transform(Ca_);

//        Temp2->copy(Fb_);
//        Temp2->transform(Ca_);

//        Ub->copy(Temp);
//        Ub->add(Temp2);
//        Ub->scale(0.5);

//        for (int h = 0; h < nirrep_; ++h) {
//          // CO
//          for (int i = 0; i < doccpi_[h]; ++i) {
//            for (int j = doccpi_[h]; j < doccpi_[h] + soccpi_[h]; ++j) {
//                Ub->set(h,i,j,Temp2->get(h,i,j));
//                Ub->set(h,j,i,Temp2->get(h,j,i));
//            }
//          }
//          for (int i = doccpi_[h]; i < doccpi_[h] + soccpi_[h]; ++i) {
//            for (int j = doccpi_[h] + soccpi_[h]; j < nmopi_[h]; ++j) {
//                Ub->set(h,i,j,Temp->get(h,i,j));
//                Ub->set(h,j,i,Temp->get(h,j,i));
//            }
//          }
//        }
//        Ub->diagonalize(Ua,epsilon_a_);
//        Temp->copy(Ca_);
//        Ca_->gemm(false,false,1.0,Temp,Ua,0.0);
//        Cb_->copy(Ca_);
//        epsilon_b_->copy(epsilon_a_.get());

//        Temp->diagonalize(Ua,epsilon_a_);
//        Temp2->diagonalize(Ub,epsilon_b_);
//        Temp->copy(Ca_);
//        Temp2->copy(Cb_);
//        Ca_->gemm(false,false,1.0,Temp,Ua,0.0);
//        Cb_->gemm(false,false,1.0,Temp2,Ub,0.0);


//        diagonalize_F(Fa_, Ca_, epsilon_a_);
//        Temp->copy(Fb_);
//        Temp->transform(Ca_);
//        for (int h = 0; h < nirrep_; ++h){
//            int nocc = doccpi_[h];
//            int nvir = nmopi_[h] - nocc;
//            if (nocc != 0 and nvir!= 0){
//                double** Temp_h = Temp->pointer(h);
//                for (int i = 0; i < nocc; ++i){
////                    for (int j = 0; j < nvir; ++j){
////                        Temp_h[i][j + nocc] = Temp_h[j + nocc][i] = 0.0;
//                    for (int j = doccpi_[h] + soccpi_[h]; j < nmopi_[h]; ++j){
//                        Temp_h[i][j] = Temp_h[j][i] = 0.0;
//                    }
//                }
//            }
//        }
//        Temp->diagonalize(Temp2,epsilon_b_);
//        Temp->copy(Ca_);
//        Cb_->gemm(false,false,1.0,Temp,Temp2,0.0);

//        for (int h = 0; h < nirrep_; ++h){
//            int nso = nsopi_[h];
//            if (nso != 0){
//                double** Temp_h = Temp->pointer(h);
//                for (int p = 0; p < nso; ++p){
//                    epsilon_b_->set(h,p,Temp_h[p][p]);
//                }
//            }
//        }


//        // Transform Fa to the ground state MO basis
//        Temp->transform(Fb_,state_Cb[0]);
//        // Grab the occ and vir blocks
//        for (int h = 0; h < nirrep_; ++h){
//            int nocc = state_nbetapi[0][h];
//            int nvir = nmopi_[h] - nocc;
//            if (nocc != 0){
//                double** Temp_h = Temp->pointer(h);
//                double** PoFbPo_h = PoFbPo_->pointer(h);
//                for (int i = 0; i < nocc; ++i){
//                    for (int j = 0; j < nocc; ++j){
//                        PoFbPo_h[i][j] = Temp_h[i][j];
//                    }
//                }
//            }
//            if (nvir != 0){
//                double** Temp_h = Temp->pointer(h);
//                double** PvFbPv_h = PvFbPv_->pointer(h);
//                for (int i = 0; i < nvir; ++i){
//                    for (int j = 0; j < nvir; ++j){
//                        PvFbPv_h[i][j] = Temp_h[i + nocc][j + nocc];
//                    }
//                }
//            }
//        }
//        PoFbPo_->diagonalize(Uob,lambda_ob);
//        PvFbPv_->diagonalize(Uvb,lambda_vb);

//        Temp->zero();
//        for (int h = 0; h < nirrep_; ++h){
//            int nocc = state_nbetapi[0][h];
//            int nvir = nmopi_[h] - nocc;
//            if (nocc != 0){
//                double** Temp_h = Temp->pointer(h);
//                double** Uob_h = Uob->pointer(h);
//                for (int i = 0; i < nocc; ++i){
//                    epsilon_b_->set(h,i,lambda_ob->get(h,i));
//                    for (int j = 0; j < nocc; ++j){
//                        Temp_h[i][j] = Uob_h[i][j];
//                    }
//                }
//            }
//            if (nvir != 0){
//                double** Temp_h = Temp->pointer(h);
//                double** Uvb_h = Uvb->pointer(h);
//                for (int i = 0; i < nvir; ++i){
//                    epsilon_b_->set(h,i + nocc,lambda_vb->get(h,i));
//                    for (int j = 0; j < nvir; ++j){
//                        Temp_h[i + nocc][j + nocc] = Uvb_h[i][j];
//                    }
//                }
//            }
//        }
//        // Get the new orbitals
//        Cb_->gemm(false,false,1.0,state_Cb[0],Temp,0.0);
//        Temp->transform(Fb_,Cb_);
//        Temp->print();



//        // Grab the occ and vir blocks
//        for (int h = 0; h < nirrep_; ++h){
//            int nocc = new_occupation[h];
//            int nvir = nmopi_[h] - nocc;
//            if (nocc != 0){
//                double** Temp_h = Temp->pointer(h);
//                double** Fcan_h = Fcan->pointer(h);
//                for (int i = 0; i < nocc; ++i){
//                    for (int j = 0; j < nocc; ++j){
//                        Fcan_h[i][j] = Temp_h[i][j];
//                    }
//                }
//            }
//        }
//        Fcan->diagonalize(Ucan,lambda_can);
//        Temp->zero();
//        for (int h = 0; h < nirrep_; ++h){
//            int nocc = new_occupation[h];
//            int nvir = nmopi_[h] - nocc;
//            if (nocc != 0){
//                double** Temp_h = Temp->pointer(h);
//                double** Ucan_h = Ucan->pointer(h);
//                for (int i = 0; i < nocc; ++i){
//                    epsilon_a_->set(h,i,lambda_can->get(h,i));
//                    for (int j = 0; j < nocc; ++j){
//                        Temp_h[i][j] = Ucan_h[i][j];
//                    }
//                }
//            }
//            if (nvir != 0){
//                double** Temp_h = Temp->pointer(h);
//                for (int i = 0; i < nvir; ++i){
//                    Temp_h[i + nocc][i + nocc] = 1.0;
//                }
//            }
//        }

//        Temp->transform(Fa_,Ca_);
//        // Get the new orbitals
//        Temp2->copy(Ca_);
//        Ca_->gemm(false,false,1.0,Temp2,Temp,0.0);

//        Dimension new
//        SharedVector epsilon_ao_ = SharedVector(factory_->create_vector());
//        SharedVector epsilon_av_ = SharedVector(factory_->create_vector());
//        diagonalize_F(PoFaPo_, Temp,  epsilon_ao_);
//        diagonalize_F(PvFaPv_, Temp2, epsilon_av_);
//        int homo_h = 0;
//        int homo_p = 0;
//        double homo_energy = -1.0e10;
//        int lumo_h = 0;
//        int lumo_p = 0;
//        double lumo_energy = +1.0e10;
//        for (int h = 0; h < nirrep_; ++h){
//            int nmo  = nmopi_[h];
//            int nso  = nsopi_[h];
//            if (nmo == 0 or nso == 0) continue;
//            double** Ca_h  = Ca_->pointer(h);
//            double** Cao_h = Temp->pointer(h);
//            double** Cav_h = Temp2->pointer(h);
//            int no = 0;
//            for (int p = 0; p < nmo; ++p){
//                if(std::fabs(epsilon_ao_->get(h,p)) > 1.0e-6 ){
//                    for (int mu = 0; mu < nmo; ++mu){
//                        Ca_h[mu][no] = Cao_h[mu][p];
//                    }
//                    epsilon_a_->set(h,no,epsilon_ao_->get(h,p));
//                    if (epsilon_ao_->get(h,p) > homo_energy){
//                        homo_energy = epsilon_ao_->get(h,p);
//                        homo_h = h;
//                        homo_p = no;
//                    }
//                    no++;
//                }
//            }
//            for (int p = 0; p < nmo; ++p){
//                if(std::fabs(epsilon_av_->get(h,p)) > 1.0e-6 ){
//                    for (int mu = 0; mu < nmo; ++mu){
//                        Ca_h[mu][no] = Cav_h[mu][p];
//                    }
//                    epsilon_a_->set(h,no,epsilon_av_->get(h,p));
//                    if (epsilon_av_->get(h,p) < lumo_energy){
//                        lumo_energy = epsilon_av_->get(h,p);
//                        lumo_h = h;
//                        lumo_p = no;
//                    }
//                    no++;
//                }
//            }
//        }
//        // Shift the HOMO orbital in the occupied space
