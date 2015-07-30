#include <physconst.h>
#include <psifiles.h>
#include <libmints/mints.h>
#include <libfock/apps.h>
#include <libfock/v.h>
#include <libfock/jk.h>
#include <liboptions/liboptions.h>

#include "noci.h"

#define DEBUG_NOCI 0

#define DEBUG_THIS2(EXP) \
    outfile->Printf("\n  Starting " #EXP " ..."); fflush(outfile); \
    EXP \
    outfile->Printf("  done."); fflush(outfile); \

using namespace psi;

namespace psi{ namespace scf{

NOCI::NOCI(Options &options, boost::shared_ptr<PSIO> psio)
: UKS(options, psio),do_noci(false),state_(0)
{
    init();
    gs_Fa_ = Fa_;
    gs_Fb_ = Fb_;
}

NOCI::NOCI(Options &options, boost::shared_ptr<PSIO> psio, boost::shared_ptr<Wavefunction> ref_scf, int state)
: UKS(options, psio),do_noci(true), state_(state),occ_(1)
{
    init();
    init_excitation(ref_scf);
    outfile->Printf("\n  ==> first excitation CI <==\n\n");
    ground_state_energy = dets[0]->energy();
}

NOCI::NOCI(Options &options, boost::shared_ptr<PSIO> psio, boost::shared_ptr<Wavefunction> ref_scf, int state,int symmetry)
: UKS(options, psio),do_noci(true)

{
    init();
    init_excitation(ref_scf);

 //   ground_state_energy = dets[0]->energy();
 //   ground_state_symmetry_ = dets[0]->symmetry();
}

void NOCI::init()
{
    outfile->Printf("\n  ==> NON Orthogonality CI <==\n\n");


    zero_dim_ = Dimension(nirrep_);



    // Allocate matrices
    H_copy = factory_->create_shared_matrix("H_copy");
    TempMatrix = factory_->create_shared_matrix("Temp");
    TempMatrix2 = factory_->create_shared_matrix("Temp2");
    Dolda_ = factory_->create_shared_matrix("Dold alpha");
    Doldb_ = factory_->create_shared_matrix("Dold beta");

    Fpq_a = factory_->create_shared_matrix("Fpq_a");
    Fpq_b = factory_->create_shared_matrix("Fpq_b");
    save_H_ = true;
}

NOCI::~NOCI()
{
}

void NOCI::init_excitation( boost::shared_ptr<Wavefunction> ref_scf)
{
        NOCI* ucks_ptr = dynamic_cast<NOCI*>(ref_scf.get());

        naholepi_ = Dimension(nirrep_,"Number of holes per irrep");
        napartpi_ = Dimension(nirrep_,"Number of particles per irrep");

        nbholepi_ = Dimension(nirrep_,"Number of holes per irrep");
        nbpartpi_ = Dimension(nirrep_,"Number of particles per irrep");

        gs_nalphapi_ = ucks_ptr->dets[0]->nalphapi();
        gs_navirpi_  = nmopi_ - gs_nalphapi_;
        gs_nbetapi_  = ucks_ptr->dets[0]->nbetapi();
        gs_nbvirpi_  = nmopi_ - gs_nbetapi_;

        Cocc_a0 = SharedMatrix(new Matrix("Cocc_a0",nsopi_,gs_nalphapi_));
        Cvrt_a0 = SharedMatrix(new Matrix("Cvrt_a0",nsopi_,gs_navirpi_));

        Cocc_b0 = SharedMatrix(new Matrix("Cocc_b0",nsopi_,gs_nbetapi_));
        Cvrt_b0 = SharedMatrix(new Matrix("Cvrt_b0",nsopi_,gs_nbvirpi_));

        C_aN = SharedMatrix(new Matrix("C_aN",nsopi_,nmopi_));
        C_bN = SharedMatrix(new Matrix("C_bN",nsopi_,nmopi_));

        if (state_ == 1){
                dets.push_back(ucks_ptr->dets[0]);
            }else{
                dets = ucks_ptr->dets;
            }

}


void NOCI::guess()
{
    if(do_noci){
        iteration_ = 0;
//        Ca_ = dets[0]->Ca();
//        Cb_ = dets[0]->Cb();
        form_initial_C();
        form_D();
        E_ = compute_initial_E();
    }else{
        UKS::guess();
    }
}

void NOCI::save_density_and_energy()
{
    Dtold_->copy(Dt_);
    Dolda_->copy(Da_);
    Doldb_->copy(Db_);
    Eold_ = E_;
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


    Ka_->print();
    Kb_->print();

    J[0]->print();
    J[1]->print();

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
        UKS::form_C();
     }
   else{
      form_C_noci();
      UKS::form_C();
  }
}


void NOCI::form_C_noci()
{
    //step 1: get the ground state determinant C vectors
    SharedMatrix Ca0 = dets[0]->Ca();
    SharedMatrix Cb0 = dets[0]->Cb();
    int occ=0;
    double Cij=0.0;
    for (int h=0; h< nirrep_;++h){
        int nso = nsopi_[h];
        int nmo = nmopi_[h];
        for (int i=0; i < nso; ++i){
            for (int j=0; j <nmo; ++j){
                if(j==occ){
                    Cij=0.0;
                   C_aN->set(i,j,Cij);
                }
                else{
                    Cij = Ca0->get(i,j);
                    C_aN->set(i,j,Cij);
                    }
            }
        }
    }
C_aN->print();
Ca0->print();
}

void NOCI::form_D()
{
    for (int h = 0; h < nirrep_; ++h) {
        int nso = nsopi_[h];
        int nmo = nmopi_[h];
        int na = nalphapi_[h];
        if(state_==1) na=nalphapi_[h]-occ_;
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

    if (debug_) {
        outfile->Printf( "in UHF::form_D:\n");
        Da_->print();
        Db_->print();
    }
}

void NOCI::save_information()
{
//    saved_naholepi_ = naholepi_;
//    saved_napartpi_ = napartpi_;



    dets.push_back(SharedDeterminant(new Determinant(E_,Ca_,Cb_,nalphapi_,nbetapi_)));
    Ca_->print();
    Fa_->print();
    Fpq_a->transform(Fa_,Ca_);
    Fpq_a->print();
    Fpq_a->transform(S_);
    Fpq_a->back_transform(Ca_);
    //TempMatrix->print();
    Fpq_a->print();
//    Fpq_a->transform(Ca_);
//     Fpq_a->print();
//    Fpq_b->transform(Fb_,dets[0]->Cb());
}

double NOCI::compute_E()
{

    outfile->Printf(" PV this is HF energy \n");
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


    //if (debug_) {
        outfile->Printf( "   => Energetics <=\n\n");
        outfile->Printf( "    Nuclear Repulsion Energy = %24.14f\n", nuclearrep_);
        outfile->Printf( "    One-Electron Energy =      %24.14f\n", one_electron_E);
        outfile->Printf( "    Coulomb Energy =           %24.14f\n", 0.5 * coulomb_E);
        outfile->Printf( "    Hybrid Exchange Energy =   %24.14f\n", 0.5 * exchange_E);
         outfile->Printf( "    Coulomb EnergyA =           %24.14f\n", 0.5 * Db_->vector_dot(J_));
    //}
    return Etotal;
}
}} // Namespaces

