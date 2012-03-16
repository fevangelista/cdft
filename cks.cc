#include <cks.h>
#include <libmints/view.h>
#include <libmints/mints.h>
#include <libfock/apps.h>
#include <libfock/v.h>
#include <libfock/jk.h>
#include <liboptions/liboptions.h>
#include <libciomr/libciomr.h>
#include <libqt/qt.h>

using namespace psi;

namespace psi{ namespace scf{

RCKS::RCKS(Options& options, boost::shared_ptr<PSIO> psio)
    : RKS(options, psio), Vc(0.0), optimize_Vc(false), gradW_threshold_(1.0e-9)
{
    fprintf(outfile,"\n  ==> Constrained DFT <==\n\n");

    Vc = options.get_double("VC");
    optimize_Vc = options.get_bool("OPTIMIZE_VC");

    if(optimize_Vc){
        fprintf(outfile,"  The constraint will be optimized.\n");
    }else{
        fprintf(outfile,"  The Lagrange multiplier for the constraint will be fixed: Vc = %f .\n",Vc);
    }

    gradW_threshold_ = options.get_double("W_CONVERGENCE");
    fprintf(outfile,"  gradW threshold = :%9.2e\n",gradW_threshold_);
    nfrag = basisset()->molecule()->nfragments();
    fprintf(outfile,"  Number of fragments: %d\n",nfrag);

    // Check the option CHARGE, if it is defined use it to define the constrained charges
    int charge_size = options["CHARGE"].size();
    if (charge_size > 0){
        if (charge_size == nfrag){
            for (int n = 0; n < charge_size; ++n){
                constrained_charges.push_back(options["CHARGE"][n].to_double());
            }
        }else{
            throw InputException("The number of charge constraints does not match the number of fragments", "CHARGE", __FILE__, __LINE__);
        }
    }

    build_W_so();

    Temp = SharedMatrix(factory_->create_matrix("Temp"));
    save_H_ = true;

    for (int f = 0; f < nfrag; ++f){
        fprintf(outfile,"  Fragment %d: constrained charge = %f .\n",f,constrained_charges[f]);
    }
    Nc  = frag_nuclear_charge[0] - constrained_charges[0];
    Nc -= frag_nuclear_charge[1] - constrained_charges[1];
    fprintf(outfile,"  Constraining Tr[w rho] = Nc = %f .\n\n",Nc);
}


RCKS::~RCKS()
{
}

void RCKS::build_W_so()
{
    // Allocate the total constraint matrix
    W_tot = SharedMatrix(factory_->create_matrix("W_tot"));

    //    Compute the overlap matrix
    boost::shared_ptr<BasisSet> basisset_ = basisset();
    boost::shared_ptr<Molecule> mol = basisset_->molecule();
    boost::shared_ptr<OneBodyAOInt> overlap(integral_->ao_overlap());
    SharedMatrix S_ao(new Matrix("S_ao",basisset_->nbf(),basisset_->nbf()));
    overlap->compute(S_ao);

    //    Form the S^(1/2) matrix
    S_ao->power(1.0/2.0);

    boost::shared_ptr<PetiteList> pet(new PetiteList(basisset_, integral_));
    SharedMatrix AO2SO_ = pet->aotoso();

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

        constrained_charges.push_back(double(frag->molecular_charge())); // TODO Remove the factor of 10.0!!!

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

        // If CHARGE is not defined, then read the charges from the fragment input
        if(constrained_charges.size() != nfrag){
            constrained_charges.push_back(double(frag->molecular_charge()));
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
    W_tot->zero();
    W_tot->add(W_so[0]);
    W_tot->subtract(W_so[1]);
}

void RCKS::form_F()
{
    // On the first iteration save H_
    if (save_H_){
        H_copy = SharedMatrix(factory_->create_matrix("H_copy"));
        H_copy->copy(H_);
        save_H_ = false;
    }

    // Augement the one-electron potential (H_) with the CDFT terms
    H_->copy(H_copy);
    Temp->copy(W_tot);
    Temp->scale(Vc);
    H_->add(Temp);  // Temp = Vc * W_tot

    Fa_->copy(H_);
    Fa_->add(G_);

    gradient_of_W();
    hessian_of_W();

    fprintf(outfile,"   @CDFT                Vc = %.7f  gradW = %.7f  hessW = %.7f\n",Vc,gradW,hessW);

    if (debug_) {
        Fa_->print(outfile);
        J_->print();
        K_->print();
        G_->print();
    }
}

double RCKS::compute_E()
{
    // E_CDFT = 2.0 D*H + 2.0 D*J - \alpha D*K + E_xc - Vc * Nc
    double one_electron_E = 2.0 * D_->vector_dot(H_) - Vc * Nc;  // Added the CDFT contribution that is not included in H_
    double coulomb_E = D_->vector_dot(J_);

    std::map<std::string, double>& quad = potential_->quadrature_values();
    double XC_E = quad["FUNCTIONAL"];
    double exchange_E = 0.0;
    double alpha = functional_->getExactExchange();
    double beta = 1.0 - alpha;
    if (functional_->isHybrid()) {
        exchange_E -= alpha*Da_->vector_dot(K_);
    }
    if (functional_->isRangeCorrected()) {
        exchange_E -=  beta*Da_->vector_dot(wK_);
    }

    double Etotal = 0.0;
    Etotal += nuclearrep_;
    Etotal += one_electron_E;
    Etotal += coulomb_E;
    Etotal += exchange_E;
    Etotal += XC_E;
    double dashD_E = 0.0;
    if (functional_->isDashD()) {
        dashD_E = functional_->getDashD()->computeEnergy(HF::molecule_);
    }
    Etotal += dashD_E;

    if (debug_) {
        fprintf(outfile, "   => Energetics <=\n\n");
        fprintf(outfile, "    Nuclear Repulsion Energy = %24.14f\n", nuclearrep_);
        fprintf(outfile, "    One-Electron Energy =      %24.14f\n", one_electron_E);
        fprintf(outfile, "    Coulomb Energy =           %24.14f\n", coulomb_E);
        fprintf(outfile, "    Hybrid Exchange Energy =   %24.14f\n", exchange_E);
        fprintf(outfile, "    XC Functional Energy =     %24.14f\n", XC_E);
        fprintf(outfile, "    -D Energy =                %24.14f\n\n", dashD_E);
    }

    return Etotal;
}

bool RCKS::test_convergency()
{
    // energy difference
    double ediff = E_ - Eold_;

    // RMS of the density
    Matrix D_rms;
    D_rms.copy(D_);
    D_rms.subtract(Dold_);
    Drms_ = D_rms.rms();

    if (fabs(ediff) < 1.0e-6){
        if(optimize_Vc){
            constraint_optimization();
            diis_manager_->reset_subspace();
        }
    }

    if (fabs(ediff) < energy_threshold_ && Drms_ < density_threshold_)
        if(optimize_Vc){
            return (std::fabs(gradW) < gradW_threshold_);
        }else{
            return true;
        }
    else
        return false;

//    if (fabs(ediff) < energy_threshold_ && Drms_ < density_threshold_)
//        if(optimize_Vc){
//            constraint_optimization();
//            diis_manager_->reset_subspace();
//            return (std::fabs(gradW) < gradW_threshold_);
//        }else{
//            return true;
//        }
//    else
//        return false;
}

/// Gradient of W
///
/// Implements Eq. (6) of Phys. Rev. A 72, 024502 (2005).
void RCKS::gradient_of_W()
{
//    gradW  = 2.0 * D_->vector_dot(W_so[0]) - (frag_nuclear_charge[0] - constrained_charges[0]);
//    gradW -= 2.0 * D_->vector_dot(W_so[1]) - (frag_nuclear_charge[1] - constrained_charges[1]);
    gradW = 2.0 * D_->vector_dot(W_tot) - Nc;
}

/// Hessian of W
///
/// Implements Eq. (7) of Phys. Rev. A 72, 024502 (2005).
void RCKS::hessian_of_W()
{
    hessW = 0.0;
    // Transform W_tot to the MO basis
    Temp->transform(W_tot,Ca_);

    //    Ca_->print();
    //    W_tot->print();
    //    Temp->print();
    for (int h = 0; h < nirrep_; h++) {
        int nmo = nmopi_[h];
        int nvir = nmopi_[h]-doccpi_[h];
        int nocc = doccpi_[h];
//        fprintf(outfile,"h = %d, nmo = %d, nvir = %d, nocc = %d\n",h,nmo,nvir,nocc);
        if (nvir == 0 or nocc == 0) continue;
        double** Temp_h = Temp->pointer(h);
        double* eps = epsilon_a_->pointer(h);
        for (int i = 0; i < nocc; ++i){
            for (int a = nocc; a < nmo; ++a){
                //fprintf(outfile,"  -> (%d,%d): (%f)^2 / (%f - %f) = %f\n",i,a,Temp_h[a][i],eps[i],eps[a],std::pow(Temp_h[a][i],2.0) /  (eps[i] - eps[a]));
                hessW += Temp_h[a][i] * Temp_h[a][i] /  (eps[i] - eps[a]);
            }
        }
    }
    hessW *= 4.0; // 2 for the complex conjugate and 2 for the spin cases
//    Temp->transform(Fa_,Ca_);
//    Temp->print();
}

/// Optimize the Lagrange multiplier
void RCKS::constraint_optimization()
{
    fprintf(outfile, "  ==> Constraint optimization <==\n");
    double threshold = 0.1;
    double new_Vc = Vc - gradW / hessW;
    if(std::fabs(new_Vc - Vc) < threshold){
        Vc = new_Vc;
    }else{
        Vc += (new_Vc > Vc ? threshold : -threshold);
    }
}

void RCKS::Lowdin2()
{
    fprintf(outfile, "  ==> Lowdin Charges <==\n\n");
    for (int f = 0; f < nfrag; ++f){
        double Q_f = -2.0 * D_->vector_dot(W_so[f]);
        Q_f += frag_nuclear_charge[f];
        fprintf(outfile, "  Fragment %d: charge = %.6f\n",f,Q_f);
    }
}

void RCKS::Lowdin()
{
    //    Compute the overlap matrix
    boost::shared_ptr<BasisSet> basisset_ = basisset();
    boost::shared_ptr<OneBodyAOInt> overlap(integral_->ao_overlap());
    SharedMatrix S_ao(new Matrix("S_ao",basisset_->nbf(),basisset_->nbf()));
    SharedMatrix D_ao(new Matrix("D_ao",basisset_->nbf(),basisset_->nbf()));
    SharedMatrix L_ao(new Matrix("L_ao",basisset_->nbf(),basisset_->nbf()));
    overlap->compute(S_ao);

    //    Form the S^(1/2) matrix
    S_ao->power(1.0/2.0);

    boost::shared_ptr<PetiteList> pet(new PetiteList(basisset_, integral_));
    SharedMatrix SO2AO_ = pet->sotoao();
    D_ao->remove_symmetry(D_,SO2AO_);
    L_ao->transform(D_ao,S_ao);
    L_ao->print();

    boost::shared_ptr<Molecule> mol = basisset_->molecule();
    SharedVector Qa(new Vector(mol->natom()));
    double* Qa_pointer = Qa->pointer();


    for (int a = 0; a < mol->natom(); ++a){
        Qa->set(a,mol->Z(a));
    }

    for (int mu = 0; mu < basisset_->nbf(); mu++) {
        double charge = L_ao->get(0,mu,mu);
        int shell = basisset_->function_to_shell(mu);
        int A = basisset_->shell_to_center(shell);

        Qa_pointer[A] -= 2.0 * charge;
      }
    Qa->print();

    int nfrag = mol->nfragments();
    fprintf(outfile, "\n  There are %d fragments in this molecule\n", nfrag);
    int a = 0;
    for (int f = 0; f < nfrag; ++f){
        std::vector<int> flist;
        std::vector<int> glist;
        flist.push_back(f);
        boost::shared_ptr<Molecule> frag = mol->extract_subsets(flist,glist);
        double fcharge = 0.0;
        for (int n = 0; n < frag->natom(); ++n){
            fcharge += Qa_pointer[a];
            ++a;
          }
        fprintf(outfile,"  Fragment %d, charge = %.8f, constrained charge = %.8f:\n",f,fcharge,double(frag->molecular_charge()));
    }
}

///**
// * Given the unique (occ X vir) elements of the orbital rotation matrix, R,
// * performs a unitary rotation of the orbitals, rigorously maintaining orthogonality.
// *
// * @param X The occ X vir unique orbital rotation parameters
// */
//void RCKS::rotate_orbitals(SharedMatrix X)
//{
//#if 1
//    // Transform using
//    // U = 1 + R + 0.5 RR
//    // then Schmidt orthogonalize U
//    SharedMatrix R(new Matrix("R", nmopi_, nmopi_));
//    for (int h = 0; h < nirrep_; ++h) {
//        int ndocc = doccpi_[h];
//        int nvirt = nmopi_[h] - doccpi_[h];
//        if (!ndocc || !nvirt) continue;
//        double** pR = R->pointer(h);
//        double** pX = X->pointer(h);
//        for (int i = 0; i < ndocc; ++i) {
//            for(int a = 0; a < nvirt; ++a){
//                pR[i][a + doccpi_[h]] = -pX[i][a];
//                pR[a + doccpi_[h]][i] = pX[i][a];
//            }
//        }
//    }
//    SharedMatrix RR(new Matrix("RR", nmopi_, nmopi_));
//    RR->gemm(false, false, 0.5, R, R, 0.0);
//    R->add(RR);
//    for(int h = 0; h < nirrep_; ++h){
//        int dim = nmopi_[h];
//        // Add the identity part in there
//        for(int n = 0; n < nmopi_[h]; ++n) R->add(h, n, n, 1.0);
//        schmidt(R->pointer(h), dim, dim, outfile);
//    }
//    // Rotate the orbitals
//    SharedMatrix Cnew(new Matrix("C new", nsopi_, nmopi_));
//    Cnew->gemm(false, false, 1.0, Ca_, R, 0.0);
//    Ca_->copy(Cnew);
//#else
//    // Transform using
//    // U = 1 + R
//    // then iteratively orthonormalize the resulting C.  This looks like a bad method, so far.
//    SharedMatrix R(new Matrix("R", nmopi_, nmopi_));
//    R->identity();
//    for (int h = 0; h < nirrep_; ++h) {
//        int ndocc = doccpi_[h];
//        int nvirt = nmopi_[h] - doccpi_[h];
//        if (!ndocc || !nvirt) continue;
//        double** pR = R->pointer(h);
//        double** pX = X->pointer(h);
//        for (int i = 0; i < ndocc; ++i) {
//            for(int a = 0; a < nvirt; ++a){
//                pR[i][a + doccpi_[h]] = -pX[i][a];
//                pR[a + doccpi_[h]][i] = pX[a][i];
//            }
//        }
//    }
//    /*
//     * Rotate the orbitals: Cnew = Ca (I + R)
//     */
//    SharedMatrix Cnew(new Matrix("C new", nsopi_, nmopi_));
//    Cnew->gemm(false, false, 1.0, Ca_, R, 0.0);
//    Ca_->copy(Cnew);

//    /*
//     * Purify the transformation, to restore orthogonality
//     */
//    double error = 0.0;
//    do{
//        // Ca(new) = 3/2 Ca(old)
//        Cnew->copy(Ca_);
//        Cnew->scale(1.5);
//        // Ca(new) -= 0.5 C Ct S C
//        // Recycle the R matrix from above to store the MO overlap
//        R->transform(S_, Ca_);
//        Cnew->gemm(false, false, -0.5, Ca_, R, 1.0);
//        SharedMatrix delta(Cnew->clone());
//        delta->set_name("Delta C");
//        delta->subtract(Ca_);
//        error = delta->rms();
//        Ca_->copy(Cnew);
//        fprintf(outfile, "\t\tOrthogonalization error is %16.10f (%16.10f)\n",
//                error, density_threshold_);
//    } while(error > density_threshold_);
//#endif
//}

//double RCKS::compute_energy()
//{
//    std::string reference = options_.get_str("REFERENCE");

//    /*
//     * Cheesy guess at an orthogonal set of MOs
//     */
//    form_H();
//    form_Shalf();
//    integrals();
//    if(options_.get_str("GUESS") != "READ")
//        guess();

//    /*
//     * Set up the CG solver
//     */
//    boost::shared_ptr<RCPHF> cphf(new RCPHF());
//    cphf->preiterations();
//    cphf->set_print(0);
//    cphf->jk()->set_print(0);
//    std::map<std::string, SharedMatrix>& tasks  = cphf->b();
//    std::map<std::string, SharedMatrix>& results = cphf->x();

//    iteration_ = 1;
//    bool converged = false;
//    if (Communicator::world->me() == 0) {
//        fprintf(outfile, "  ==> Iterations <==\n\n");
//        fprintf(outfile, "                        Total Energy        Delta E     Density RMS\n\n");
//    }
//    fflush(outfile);

//    Drms_ = 0.1;
//    /*
//     * Build the Fock matrix and assign the diagonal elements to epsilon
//     */
//    form_D();
//    form_G();
//    form_F();
//    E_ = compute_E();
//    do{
//        /*
//         * Compute the gradient (Fia)
//         */
//        Dimension zerodim = Dimension(nirrep_);
//        Dimension virtpi = nmopi_ - doccpi_;
//#if 1
//        // Obtain the eigenvalues exactly, and transform only the OV block
//        // of the Fock matrix to compute the MO basis orbital gradient
//        View oview(Ca_, nsopi_, doccpi_);
//        View vview(Ca_, nsopi_, virtpi, zerodim, doccpi_);
//        SharedMatrix Co = oview();
//        Co->set_name("Occupied MO Coefficients");
//        SharedMatrix Cv = vview();
//        Cv->set_name("Virtual MO Coefficients");
//        SharedMatrix temp(new Matrix("Temp doccXnso", doccpi_, nsopi_));
//        temp->gemm(true, false, 1.0, Co, Fa_, 0.0);
//        SharedMatrix grad(new Matrix("Orbital Gradient", doccpi_, virtpi));
//        grad->gemm(false, false, 1.0, temp, Cv, 0.0);
//        diag_F_temp_->transform(Fa_, X_);
//        // Diagonalize, but get only the eigenvalues
//        diag_F_temp_->diagonalize(temp, epsilon_a_, Matrix::EvalsOnlyAscending);
//#else
//        // Obtain the eigenvalues as the diagonals of the fock matrix
//        SharedMatrix moF(new Matrix("F (MO)", nmopi_, nmopi_));
//        moF->transform(Fa_, Ca_);
//        for(int h = 0; h < nirrep_; ++h)
//            for(int p = 0; p < nmopi_[h]; ++p)
//                epsilon_a_->set(h, p, moF->get(h, p, p));
//        View gradview(moF, doccpi_, virtpi, zerodim, doccpi_);
//        SharedMatrix grad = gradview();
//#endif

//        /*
//         * Compute/contract the hessian
//         */
//        grad->scale(-1.0);
//        // Task and solve linear Equations
//        tasks["Orbital Gradient"] = grad;
//        cphf->set_convergence(Drms_ / 10.0);
//        cphf->set_reference(Process::environment.reference_wavefunction());
//        cphf->compute_energy();
//        SharedMatrix X = results["Orbital Gradient"];
//        rotate_orbitals(X);

//        form_D();
//        form_G();
//        form_F();

//        Drms_ = grad->rms();
//        Eold_ = E_;
//        E_ = compute_E();

//        std::string status;
//        if (Communicator::world->me() == 0) {
//            fprintf(outfile, "   @%s iter %3d: %20.14f   %12.5e   %-11.5e %s\n",
//                              reference.c_str(), iteration_, E_, E_ - Eold_, Drms_, status.c_str());
//            fflush(outfile);
//        }
//        converged = Drms_ < density_threshold_ && fabs(E_ - Eold_) < energy_threshold_;
//        ++iteration_;
//    }while (!converged && iteration_ < maxiter_ );

//    if (converged) {
//        fprintf(outfile, "\n  Energy converged.\n");
//        fprintf(outfile, "\n  @%s Final Energy: %20.14f",reference.c_str(), E_);
//    }else{
//        fprintf(outfile, "\n Energy did not converge!\n");
//    }

//    // We're not guaranteed canonical orbitals, because o-o and v-v rotations are not considered.
//    // This can be done by calling semicanonicalize() to diagonalize only subblocks, but for now
//    // I'll just diagonalize the full matrix.
//    Fa_->diagonalize(Ca_, epsilon_a_, Matrix::Ascending);

//    return E_;
//}

}} // Namespaces
