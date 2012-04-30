
#include <cks.h>
#include <libmints/view.h>
#include <libmints/mints.h>
#include <libfock/apps.h>
#include <libfock/v.h>
#include <libfock/jk.h>
#include <libdisp/dispersion.h>
#include <liboptions/liboptions.h>
#include <libciomr/libciomr.h>
#include <libqt/qt.h>

using namespace psi;

namespace psi{ namespace scf{

UCKS::UCKS(Options &options, boost::shared_ptr<PSIO> psio)
    : UKS(options, psio), optimize_Vc(false), gradW_threshold_(1.0e-9),nW_opt(0)
{
    init(options);
}

UCKS::UCKS(Options &options, boost::shared_ptr<PSIO> psio, boost::shared_ptr<UCKS> gs_scf)
    : UKS(options, psio), optimize_Vc(false), gradW_threshold_(1.0e-9),nW_opt(0), gs_scf_(gs_scf)
{
    init(options);
}

void UCKS::init(Options &options)
{
    fprintf(outfile,"\n  ==> Constrained DFT (UCKS) <==\n\n");

    optimize_Vc = options.get_bool("OPTIMIZE_VC");
    gradW_threshold_ = options.get_double("W_CONVERGENCE");
    fprintf(outfile,"  gradW threshold = :%9.2e\n",gradW_threshold_);
    nfrag = basisset()->molecule()->nfragments();
    fprintf(outfile,"  Number of fragments: %d\n",nfrag);

    do_excitation = (options["FRAG_EXCITATION"].size() > 0);
    do_penalty = options.get_bool("HOMO_PENALTY");


    build_W_frag();

    // Check the option CHARGE, if it is defined use it to define the constrained charges, "-" skips the constraint
    for (int f = 0; f < int(options["CHARGE"].size()); ++f){
        if(options["CHARGE"][f].to_string() != "-"){
            double constrained_charge = options["CHARGE"][f].to_double();
            double Nc = frag_nuclear_charge[f] - constrained_charge;
            SharedConstraint constraint(new Constraint(W_frag[f],Nc,1.0,1.0,"charge(" + to_string(f) + ")"));
            constraints.push_back(constraint);
            fprintf(outfile,"  Fragment %d: constrained charge = %f .\n",f,constrained_charge);
        }else{
            fprintf(outfile,"  Fragment %d: no charge constraint specified .\n",f);
        }
    }
    // Check the option SPIN, if it is defined use it to define the constrained spins, "-" skips the constraint
    for (int f = 0; f < int(options["SPIN"].size()); ++f){
        if(options["SPIN"][f].to_string() != "-"){
            double constrained_spin = options["SPIN"][f].to_double();
            double Nc = constrained_spin;
            SharedConstraint constraint(new Constraint(W_frag[f],Nc,0.5,-0.5,"spin(" + to_string(f) + ")"));
            constraints.push_back(constraint);
            fprintf(outfile,"  Fragment %d: constrained spin   = %f .\n",f,constrained_spin);
        }else{
            fprintf(outfile,"  Fragment %d: no spin constraint specified .\n",f);
        }
    }

    nconstraints = static_cast<int>(constraints.size());
    gradW = SharedVector(new Vector("gradW",nconstraints));
    gradW_old = SharedVector(new Vector("gradW_old",nconstraints));
    gradW_mo_resp = SharedVector(new Vector("gradW_mo_resp",nconstraints));
    Vc = SharedVector(new Vector("Vc",nconstraints));
    Vc_old = SharedVector(new Vector("Vc_old",nconstraints));
    hessW = SharedMatrix(new Matrix("hessW",nconstraints,nconstraints));
    hessW_BFGS = SharedMatrix(new Matrix("hessW_BFGS",nconstraints,nconstraints));
    H_copy = SharedMatrix(factory_->create_matrix("H_copy"));
    Temp = SharedMatrix(factory_->create_matrix("Temp"));
    Temp2 = SharedMatrix(factory_->create_matrix("Temp2"));
    Ua = SharedMatrix(factory_->create_matrix("U alpha"));
    Ub = SharedMatrix(factory_->create_matrix("U beta"));

    if(gs_scf_){
        if(do_excitation)
            fprintf(outfile,"  Saving the ground orbitals for an excited state computation\n");
        if(do_penalty)
            fprintf(outfile,"  Saving the ground orbitals for a homo projection computation\n");

        PoFaPo_ = SharedMatrix(factory_->create_matrix("PoFaPo"));
        PvFaPv_ = SharedMatrix(factory_->create_matrix("PvFaPv"));
        // Save the ground state MOs and density matrix
        state_epsilon_a.push_back(SharedVector(gs_scf_->epsilon_a_->clone()));
        state_Ca.push_back(gs_scf_->Ca_->clone());
        state_Cb.push_back(gs_scf_->Cb_->clone());
        state_Da.push_back(gs_scf_->Da_->clone());
        state_Db.push_back(gs_scf_->Db_->clone());
        state_nalphapi.push_back(gs_scf_->nalphapi_);
        state_nbetapi.push_back(gs_scf_->nbetapi_);
        Fa_->copy(gs_scf_->Fa_);
        Fb_->copy(gs_scf_->Fb_);

        if(do_penalty){
            // Find the alpha HOMO of the ground state wave function
            int homo_h = 0;
            int homo_p = 0;
            double homo_e = -1.0e9;
            for (int h = 0; h < nirrep_; ++h) {
                int nocc = state_nalphapi[0][h] - 1;
                if (nocc < 0) continue;
                if(state_epsilon_a[0]->get(h,nocc) > homo_e){
                    homo_h = h;
                    homo_p = nocc;
                    homo_e = state_epsilon_a[0]->get(h,nocc);
                }
            }
            fprintf(outfile,"  The HOMO orbital has energy %.9f and is %d of irrep %d.\n",homo_e,homo_p,homo_h);
            Pa = SharedMatrix(factory_->create_matrix("Penalty matrix alpha"));
            for (int mu = 0; mu < nsopi_[homo_h]; ++mu){
                for (int nu = 0; nu < nsopi_[homo_h]; ++nu){
                    double P_mn = 1000.0 * state_Ca[0]->get(homo_h,mu,homo_p) * state_Ca[0]->get(homo_h,nu,homo_p);
                    Pa->set(homo_h,mu,nu,P_mn);
                }
            }
            Pa->transform(S_);
        }
    }

    for (int f = 0; f < std::min(int(options["VC"].size()),nconstraints); ++f){
        if(options["VC"][f].to_string() != "-"){
            Vc->set(f,options["VC"][f].to_double());
        }else{
            Vc->set(f,0.0);
        }
        fprintf(outfile,"  The Lagrange multiplier for constraint %d will be initialized to Vc = %f .\n",f,options["VC"][f].to_double());
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
        Temp->copy(constraints[c]->W_so());
        Temp->scale(Vc->get(c) * constraints[c]->weight_alpha());
        H_->add(Temp);
    }
    Fa_->copy(H_);
    Fa_->add(Ga_);
    if(Pa){
        Fa_->add(Pa);
    }

    H_->copy(H_copy);
    for (int c = 0; c < nconstraints; ++c){
        Temp->copy(constraints[c]->W_so());
        Temp->scale(Vc->get(c) * constraints[c]->weight_beta());
        H_->add(Temp);
    }
    Fb_->copy(H_);
    Fb_->add(Gb_);

    if(gs_scf_ and do_excitation){
        // Form the projected Fock matrices
        // Temp = DS
        Temp->gemm(false,false,1.0,state_Da[0],S_,0.0);
        PoFaPo_->transform(Fa_,Temp);
        // Temp = 1 - DS
        Temp2->identity();
        Temp2->subtract(Temp);
        PvFaPv_->transform(Fa_,Temp2);
    }

    gradient_of_W();

    if (debug_) {
        Fa_->print(outfile);
        Fb_->print(outfile);
    }
}

void UCKS::form_C()
{
    if(gs_scf_ and (PoFaPo_->rms() > 0.0)){
        SharedVector epsilon_ao_ = SharedVector(factory_->create_vector());
        SharedVector epsilon_av_ = SharedVector(factory_->create_vector());
        diagonalize_F(PoFaPo_, Temp,  epsilon_ao_);
        diagonalize_F(PvFaPv_, Temp2, epsilon_av_);
        int homo_h = 0;
        int homo_p = 0;
        double homo_energy = -1.0e10;
        int lumo_h = 0;
        int lumo_p = 0;
        double lumo_energy = +1.0e10;
        for (int h = 0; h < nirrep_; ++h){
            int nmo  = nmopi_[h];
            int nso  = nsopi_[h];
            if (nmo == 0 or nso == 0) continue;
            double** Ca_h  = Ca_->pointer(h);
            double** Cao_h = Temp->pointer(h);
            double** Cav_h = Temp2->pointer(h);
            int no = 0;
            for (int p = 0; p < nmo; ++p){
                if(std::fabs(epsilon_ao_->get(h,p)) > 1.0e-6 ){
                    for (int mu = 0; mu < nmo; ++mu){
                        Ca_h[mu][no] = Cao_h[mu][p];
                    }
                    epsilon_a_->set(h,no,epsilon_ao_->get(h,p));
                    if (epsilon_ao_->get(h,p) > homo_energy){
                        homo_energy = epsilon_ao_->get(h,p);
                        homo_h = h;
                        homo_p = no;
                    }
                    no++;
                }
            }
            for (int p = 0; p < nmo; ++p){
                if(std::fabs(epsilon_av_->get(h,p)) > 1.0e-6 ){
                    for (int mu = 0; mu < nmo; ++mu){
                        Ca_h[mu][no] = Cav_h[mu][p];
                    }
                    epsilon_a_->set(h,no,epsilon_av_->get(h,p));
                    if (epsilon_av_->get(h,p) < lumo_energy){
                        lumo_energy = epsilon_av_->get(h,p);
                        lumo_h = h;
                        lumo_p = no;
                    }
                    no++;
                }
            }
        }
        // Shift the HOMO orbital in the occupied space
        fprintf(outfile,"  homo_h = %d, homo_p = %d, homo_energy = %f\n",homo_h,homo_p,homo_energy);
        fprintf(outfile,"  lumo_h = %d, lumo_p = %d, lumo_energy = %f\n",lumo_h,lumo_p,lumo_energy);
        if(homo_h == lumo_h){
            Ca_->swap_columns(homo_h,homo_p,lumo_p);
            epsilon_a_->set(homo_h,homo_p,lumo_energy);
            epsilon_a_->set(lumo_h,lumo_p,homo_energy);
        }else{
            epsilon_a_->set(homo_h,homo_p,1000.0);
        }
        diagonalize_F(Fb_, Cb_, epsilon_b_);
        compute_overlap(0);
    }else{
        diagonalize_F(Fa_, Ca_, epsilon_a_);
        diagonalize_F(Fb_, Cb_, epsilon_b_);
    }

    find_occupation();

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
        Temp->transform(constraints[c]->W_so(),Ca_);
        Temp->scale(constraints[c]->weight_alpha());
        // Transform Fa_ to the MO basis
        Temp2->transform(Fa_,Ca_);
        for (int h = 0; h < nirrep_; h++) {
            int nmo  = nmopi_[h];
            int nocc = doccpi_[h] + soccpi_[h];
            int nvir = nmo - nocc;
            if (nvir == 0 or nocc == 0) continue;
            double** Temp_h = Temp->pointer(h);
            double** Temp2_h = Temp2->pointer(h);
            for (int i = 0; i < nocc; ++i){
                for (int a = nocc; a < nmo; ++a){
                    grad += 2.0 * Temp_h[a][i] * Temp2_h[a][i] / (Temp2_h[i][i] - Temp2_h[a][a]);
                }
            }
        }
        // Transform W_so to the MO basis (beta)
        Temp->transform(constraints[c]->W_so(),Cb_);
        Temp->scale(constraints[c]->weight_beta());
        // Transform Fb_ to the MO basis
        Temp2->transform(Fb_,Cb_);
        for (int h = 0; h < nirrep_; h++) {
            int nmo  = nmopi_[h];
            int nocc = doccpi_[h];
            int nvir = nmo - nocc;
            if (nvir == 0 or nocc == 0) continue;
            double** Temp_h = Temp->pointer(h);
            double** Temp2_h = Temp2->pointer(h);
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
            Temp->transform(constraints[c1]->W_so(),Ca_);
            Temp->scale(constraints[c1]->weight_alpha());
            // Transform W_so to the MO basis (alpha)
            Temp2->transform(constraints[c2]->W_so(),Ca_);
            Temp2->scale(constraints[c2]->weight_alpha());
            for (int h = 0; h < nirrep_; h++) {
                int nmo  = nmopi_[h];
                int nocc = doccpi_[h] + soccpi_[h];
                int nvir = nmo - nocc;
                if (nvir == 0 or nocc == 0) continue;
                double** Temp_h = Temp->pointer(h);
                double** Temp2_h = Temp2->pointer(h);
                double* eps = epsilon_a_->pointer(h);
                for (int i = 0; i < nocc; ++i){
                    for (int a = nocc; a < nmo; ++a){
                        hess += 2.0 * Temp_h[i][a] * Temp2_h[a][i] /  (eps[i] - eps[a]);
                    }
                }
            }
            // Transform W_so to the MO basis (beta)
            Temp->transform(constraints[c1]->W_so(),Cb_);
            Temp->scale(constraints[c1]->weight_beta());
            // Transform W_so to the MO basis (beta)
            Temp2->transform(constraints[c2]->W_so(),Cb_);
            Temp2->scale(constraints[c2]->weight_beta());
            for (int h = 0; h < nirrep_; h++) {
                int nmo  = nmopi_[h];
                int nocc = doccpi_[h];
                int nvir = nmo - nocc;
                if (nvir == 0 or nocc == 0) continue;
                double** Temp_h = Temp->pointer(h);
                double** Temp2_h = Temp2->pointer(h);
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
/// dx[]  = Step from previous iteration (dx[] = x[] - xp[] where xp[] is the previous point)
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
    if(psi::scf::KS::options_.get_str("W_ALGORITHM") == "NEWTON"){
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

    if(optimize_Vc){
        constraint_optimization();
        if(fabs(ediff) < energy_threshold_ and Drms_ < density_threshold_ and gradW->norm() < gradW_threshold_){
            return true;
        }else{
            return false;
        }
    }else{
        return (fabs(ediff) < energy_threshold_ and Drms_ < density_threshold_);
    }
}

/// Compute the overlap of the ground state to the current state
double UCKS::compute_overlap(int n)
{
    // Alpha block
    Temp->gemm(false,false,1.0,S_,Ca_,0.0);
    Ua->gemm(true,false,1.0,state_Ca[n],Temp,0.0);
    // Grab S_aa from Ua
    SharedMatrix S_aa = SharedMatrix(new Matrix("S_aa",state_nalphapi[0],nalphapi_));
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
    }

    // Beta block
    Temp->gemm(false,false,1.0,S_,Cb_,0.0);
    Ub->gemm(true,false,1.0,state_Cb[n],Temp,0.0);

    // Grab S_bb from Ub
    SharedMatrix S_bb = SharedMatrix(new Matrix("S_bb",state_nbetapi[0],nbetapi_));

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
    }
    fprintf(outfile,"   det(S_aa) = %.6f det(S_bb) = %.6f  <Phi|Phi'> = %.6f\n",detS_aa,detS_bb,detS_aa * detS_bb);
    return (detS_aa * detS_bb);
}

}} // Namespaces
