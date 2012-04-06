
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

UCKS::UCKS(Options& options, boost::shared_ptr<PSIO> psio)
    : UKS(options, psio), optimize_Vc(false), gradW_threshold_(1.0e-9),nW_opt(0)
{
    fprintf(outfile,"\n  ==> Constrained DFT (UCKS) <==\n\n");

    optimize_Vc = options.get_bool("OPTIMIZE_VC");

    if(optimize_Vc){
        fprintf(outfile,"  The constraint will be optimized.\n");
    }

    gradW_threshold_ = options.get_double("W_CONVERGENCE");
    fprintf(outfile,"  gradW threshold = :%9.2e\n",gradW_threshold_);
    nfrag = basisset()->molecule()->nfragments();
    fprintf(outfile,"  Number of fragments: %d\n",nfrag);

    build_W_so();

    // Check the option CHARGE, if it is defined use it to define the constrained charges, "-" skips the constraint
    for (int f = 0; f < int(options["CHARGE"].size()); ++f){
        if(options["CHARGE"][f].to_string() != "-"){
            double constrained_charge = options["CHARGE"][f].to_double();
            double Nc = frag_nuclear_charge[f] - constrained_charge;
            SharedConstraint constraint(new Constraint(W_so[f],Nc,1.0,1.0,"charge(" + to_string(f) + ")"));
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
            SharedConstraint constraint(new Constraint(W_so[f],Nc,0.5,-0.5,"spin(" + to_string(f) + ")"));
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
    Vc_old = SharedVector(new Vector("Vc",nconstraints));
    hessW = SharedMatrix(new Matrix("hessW",nconstraints,nconstraints));
    hessW_BFGS = SharedMatrix(new Matrix("hessW_BFGS",nconstraints,nconstraints));

    H_copy = SharedMatrix(factory_->create_matrix("H_copy"));
    Temp = SharedMatrix(factory_->create_matrix("Temp"));
    Temp2 = SharedMatrix(factory_->create_matrix("Temp2"));

    save_H_ = true;
}

UCKS::~UCKS()
{
}

void UCKS::build_W_so()
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
        W_so.push_back(W_f_so);
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

    H_->copy(H_copy);
    for (int c = 0; c < nconstraints; ++c){
        Temp->copy(constraints[c]->W_so());
        Temp->scale(Vc->get(c) * constraints[c]->weight_beta());
        H_->add(Temp);
    }
    Fb_->copy(H_);
    Fb_->add(Gb_);

    gradient_of_W();
    if (debug_) {
        Fa_->print(outfile);
        Fb_->print(outfile);
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
    fprintf(outfile,"  Constraint         ");
    for (int c = 0; c < nconstraints; ++c){
        fprintf(outfile,"  %-10s",constraints[c]->type().c_str());
    }
    fprintf(outfile,"\n");
    fprintf(outfile,"  grad(W)            ");
    for (int c = 0; c < nconstraints; ++c){
        fprintf(outfile," %10.7f",gradW->get(c));
    }
    fprintf(outfile,"\n");
    fprintf(outfile,"  grad(W) (response) ");
    for (int c = 0; c < nconstraints; ++c){
        fprintf(outfile," %10.7f",gradW_mo_resp->get(c));
    }
    fprintf(outfile,"\n");
    fprintf(outfile,"  Vc                 ");
    for (int c = 0; c < nconstraints; ++c){
        fprintf(outfile," %10.7f",Vc->get(c));
    }
    fprintf(outfile,"\n");
}

/// Hessian of W
///
/// Implements Eq. (7) of Phys. Rev. A 72, 024502 (2005).
void UCKS::hessian_of_W()
{
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
                dVc->add(Vc_old,-1.0);
                SharedVector dgradW = SharedVector(new Vector("dgradW",nconstraints));
                dgradW->copy(gradW.get());
                dgradW->add(gradW_old,-1.0);
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
            Vc->add(h_inv_g,-1.0);

            // Reset the DIIS subspace
            diis_manager_->reset_subspace();
        }
    }
    nW_opt += 1;
}
//if(std::fabs(Vc - old_Vc) > 1.0e-3){
//    // We landed somewhere else, update the Hessian
//    BFGS_hessW = (gradW - old_gradW) / (Vc - old_Vc);
//    fprintf(outfile, "  BFGS_hessW = %f\n",BFGS_hessW);
//    old_Vc = Vc;
//    old_gradW = gradW;
//}
//double new_Vc = Vc - gradW / BFGS_hessW;
//Vc = new_Vc;

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
        Ca_->print();
        Cb_->print();
        return (fabs(ediff) < energy_threshold_ and Drms_ < density_threshold_ and gradW->norm() < gradW_threshold_);
    }else{
        return (fabs(ediff) < energy_threshold_ and Drms_ < density_threshold_);
    }
}


//void UCKS::Lowdin2()
//{
//    fprintf(outfile, "  ==> Lowdin Charges <==\n\n");
//    for (int f = 0; f < nfrag; ++f){
//        double Q_f = -2.0 * D_->vector_dot(W_so[f]);
//        Q_f += frag_nuclear_charge[f];
//        fprintf(outfile, "  Fragment %d: charge = %.6f\n",f,Q_f);
//    }
//}

//void UCKS::Lowdin()
//{
//    //    Compute the overlap matrix
//    boost::shared_ptr<BasisSet> basisset_ = basisset();
//    boost::shared_ptr<OneBodyAOInt> overlap(integral_->ao_overlap());
//    SharedMatrix S_ao(new Matrix("S_ao",basisset_->nbf(),basisset_->nbf()));
//    SharedMatrix D_ao(new Matrix("D_ao",basisset_->nbf(),basisset_->nbf()));
//    SharedMatrix L_ao(new Matrix("L_ao",basisset_->nbf(),basisset_->nbf()));
//    overlap->compute(S_ao);

//    //    Form the S^(1/2) matrix
//    S_ao->power(1.0/2.0);

//    boost::shared_ptr<PetiteList> pet(new PetiteList(basisset_, integral_));
//    SharedMatrix SO2AO_ = pet->sotoao();
//    D_ao->remove_symmetry(D_,SO2AO_);
//    L_ao->transform(D_ao,S_ao);
//    L_ao->print();

//    boost::shared_ptr<Molecule> mol = basisset_->molecule();
//    SharedVector Qa(new Vector(mol->natom()));
//    double* Qa_pointer = Qa->pointer();


//    for (int a = 0; a < mol->natom(); ++a){
//        Qa->set(a,mol->Z(a));
//    }

//    for (int mu = 0; mu < basisset_->nbf(); mu++) {
//        double charge = L_ao->get(0,mu,mu);
//        int shell = basisset_->function_to_shell(mu);
//        int A = basisset_->shell_to_center(shell);

//        Qa_pointer[A] -= 2.0 * charge;
//      }
//    Qa->print();

//    int nfrag = mol->nfragments();
//    fprintf(outfile, "\n  There are %d fragments in this molecule\n", nfrag);
//    int a = 0;
//    for (int f = 0; f < nfrag; ++f){
//        std::vector<int> flist;
//        std::vector<int> glist;
//        flist.push_back(f);
//        boost::shared_ptr<Molecule> frag = mol->extract_subsets(flist,glist);
//        double fcharge = 0.0;
//        for (int n = 0; n < frag->natom(); ++n){
//            fcharge += Qa_pointer[a];
//            ++a;
//          }
//        fprintf(outfile,"  Fragment %d, charge = %.8f, constrained charge = %.8f:\n",f,fcharge,double(frag->molecular_charge()));
//    }
//}

}} // Namespaces
