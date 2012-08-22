
#include <ucks.h>
#include <physconst.h>
#include <libmints/view.h>
#include <libmints/mints.h>
#include <libtrans/integraltransform.h>
#include <libfock/apps.h>
#include <libfock/v.h>
#include <libfock/jk.h>
#include <libdisp/dispersion.h>
#include <liboptions/liboptions.h>
#include <libciomr/libciomr.h>
#include <libqt/qt.h>
#include "boost/tuple/tuple.hpp"
#include "boost/tuple/tuple_comparison.hpp"
#include <libiwl/iwl.hpp>
#include <psifiles.h>

using namespace psi;

namespace psi{ namespace scf{

std::pair<double,double> UCKS::matrix_element(SharedDeterminant A, SharedDeterminant B)
{
    double overlap = 0.0;
    double hamiltonian = 0.0;

    // I. Form the corresponding alpha and beta orbitals
    boost::tuple<SharedMatrix,SharedMatrix,SharedVector,double> calpha = corresponding_orbitals(A->Ca(),B->Ca(),A->nalphapi(),B->nalphapi());
    boost::tuple<SharedMatrix,SharedMatrix,SharedVector,double> cbeta  = corresponding_orbitals(A->Cb(),B->Cb(),A->nbetapi(),B->nbetapi());
    SharedMatrix ACa = calpha.get<0>();
    SharedMatrix BCa = calpha.get<1>();
    SharedMatrix ACb = cbeta.get<0>();
    SharedMatrix BCb = cbeta.get<1>();
    double detUValpha = calpha.get<3>();
    double detUVbeta = cbeta.get<3>();
    SharedVector s_a = calpha.get<2>();
    SharedVector s_b = cbeta.get<2>();

    // Compute the number of noncoincidences
    double noncoincidence_threshold = 1.0e-4;

    std::vector<std::pair<int,int> > Aalpha_nonc;
    std::vector<std::pair<int,int> > Balpha_nonc;
    double Sta = 1.0;
    for (int h = 0; h < nirrep_; ++h){
        // Count all the numerical noncoincidences
        int nmin = std::min(A->nalphapi()[h],B->nalphapi()[h]);
        for (int p = 0; p < nmin; ++p){
            if(std::fabs(s_a->get(h,p)) >= noncoincidence_threshold){
                Sta *= s_a->get(h,p);
            }else{
                Aalpha_nonc.push_back(std::make_pair(h,p));
                Balpha_nonc.push_back(std::make_pair(h,p));
            }
        }
        // Count all the symmetry noncoincidences
        int nmax = std::max(A->nalphapi()[h],B->nalphapi()[h]);
        bool AgeB = A->nalphapi()[h] >= B->nalphapi()[h] ? true : false;
        for (int p = nmin; p < nmax; ++p){
            if(AgeB){
                Aalpha_nonc.push_back(std::make_pair(h,p));
            }else{
                Balpha_nonc.push_back(std::make_pair(h,p));
            }
        }
    }

    std::vector<std::pair<int,int> > Abeta_nonc;
    std::vector<std::pair<int,int> > Bbeta_nonc;
    double Stb = 1.0;
    for (int h = 0; h < nirrep_; ++h){
        // Count all the numerical noncoincidences
        int nmin = std::min(A->nbetapi()[h],B->nbetapi()[h]);
        for (int p = 0; p < nmin; ++p){
            if(std::fabs(s_b->get(h,p)) >= noncoincidence_threshold){
                Stb *= s_b->get(h,p);
            }else{
                Abeta_nonc.push_back(std::make_pair(h,p));
                Bbeta_nonc.push_back(std::make_pair(h,p));
            }
        }
        // Count all the symmetry noncoincidences
        int nmax = std::max(A->nbetapi()[h],B->nbetapi()[h]);
        bool AgeB = A->nbetapi()[h] >= B->nbetapi()[h] ? true : false;
        for (int p = nmin; p < nmax; ++p){
            if(AgeB){
                Abeta_nonc.push_back(std::make_pair(h,p));
            }else{
                Bbeta_nonc.push_back(std::make_pair(h,p));
            }
        }
    }
    fprintf(outfile,"\n  Corresponding orbitals:\n");
    fprintf(outfile,"  A(alpha): ");
    for (size_t k = 0; k < Aalpha_nonc.size(); ++k){
        int i_h = Aalpha_nonc[k].first;
        int i_mo = Aalpha_nonc[k].second;
        fprintf(outfile," (%1d,%2d)",i_h,i_mo);
    }
    fprintf(outfile,"\n  B(alpha): ");
    for (size_t k = 0; k < Balpha_nonc.size(); ++k){
        int i_h = Balpha_nonc[k].first;
        int i_mo = Balpha_nonc[k].second;
        fprintf(outfile," (%1d,%2d)",i_h,i_mo);
    }
    fprintf(outfile,"\n  A(beta):  ");
    for (size_t k = 0; k < Abeta_nonc.size(); ++k){
        int i_h = Abeta_nonc[k].first;
        int i_mo = Abeta_nonc[k].second;
        fprintf(outfile," (%1d,%2d)",i_h,i_mo);
    }
    fprintf(outfile,"\n  B(beta):  ");
    for (size_t k = 0; k < Bbeta_nonc.size(); ++k){
        int i_h = Bbeta_nonc[k].first;
        int i_mo = Bbeta_nonc[k].second;
        fprintf(outfile," (%1d,%2d)",i_h,i_mo);
    }

    double Stilde = Sta * Stb * detUValpha * detUVbeta;
    fprintf(outfile,"\n  Stilde = %.6f\n",Stilde);

    boost::shared_ptr<JK> jk = JK::build_JK();
//    boost::shared_ptr<BasisSetParser> parser(new Gaussian94BasisSetParser());
//    boost::shared_ptr<BasisSet> primary = BasisSet::construct(parser, Process::environment.molecule(), "BASIS");
//    boost::shared_ptr<BasisSet> auxiliary = BasisSet::construct(parser, primary->molecule(), "DF_BASIS_SCF");
//    boost::shared_ptr<JK> jk(new DFJK(primary,auxiliary));
//    // 800 MB Memory, 100 M doubles
//    jk->set_memory(100000000L);
//    // Cutoff of 1.0E-12
//    jk->set_cutoff(1.0E-12);
//    // Do J/K, Not wK (superfluous)
//    jk->set_do_J(true);
//    jk->set_do_K(true);
//    jk->set_do_wK(false);
    jk->initialize();

    int num_alpha_nonc = static_cast<int>(Aalpha_nonc.size());
    int num_beta_nonc = static_cast<int>(Abeta_nonc.size());
    if(num_alpha_nonc == 0 and num_beta_nonc == 0){
        overlap =Stilde;
        throw FeatureNotImplemented("CKS", "H in the case of zero noncoincidences", __FILE__, __LINE__);
    }else if(num_alpha_nonc == 1 and num_beta_nonc == 0){
        overlap = 0.0;
        throw FeatureNotImplemented("CKS", "H in the case of one noncoincidence", __FILE__, __LINE__);
    }else if(num_alpha_nonc == 0 and num_beta_nonc == 1){
        overlap = 0.0;
        throw FeatureNotImplemented("CKS", "H in the case of one noncoincidence", __FILE__, __LINE__);
    }else if(num_alpha_nonc == 1 and num_beta_nonc == 1){
        overlap = 0.0;
        int a_alpha_h  = Aalpha_nonc[0].first;
        int a_alpha_mo = Aalpha_nonc[0].second;
        int b_alpha_h  = Balpha_nonc[0].first;
        int b_alpha_mo = Balpha_nonc[0].second;
        int a_beta_h   = Abeta_nonc[0].first;
        int a_beta_mo  = Abeta_nonc[0].second;
        int b_beta_h   = Bbeta_nonc[0].first;
        int b_beta_mo  = Bbeta_nonc[0].second;

 // Version 1
//        Dimension A_beta_dim(nirrep_,"A_beta_dim");
//        A_beta_dim[0] = 1;
//        Dimension B_beta_dim(nirrep_,"B_beta_dim");
//        B_beta_dim[0] = 1;
//        SharedMatrix A_b(new Matrix("A_b", nsopi_, A_beta_dim,a_beta_h));
//        SharedMatrix B_b(new Matrix("B_b", nsopi_, B_beta_dim,b_beta_h));
//        A_b->print();
//        B_b->print();
//        fflush(outfile);
//        for (int m = 0; m < nsopi_[a_beta_h]; ++m){
//            A_b->set(a_beta_h,m,0,ACb->get(a_beta_h,m,a_beta_mo));
//        }
//        for (int m = 0; m < nsopi_[b_beta_h]; ++m){
//            B_b->set(b_beta_h,m,0,BCb->get(b_beta_h,m,b_beta_mo));
//        }


// Version 2
        // A_b absorbs the symmetry of B_b
        Dimension A_beta_dim(nirrep_,"A_beta_dim");
        A_beta_dim[b_beta_h] = 1;
        // B_b is total symmetric
        Dimension B_beta_dim(nirrep_,"B_beta_dim");
        B_beta_dim[b_beta_h] = 1;
        SharedMatrix A_b(new Matrix("A_b", nsopi_, A_beta_dim,a_beta_h ^ b_beta_h));
        SharedMatrix B_b(new Matrix("B_b", nsopi_, B_beta_dim));
        for (int m = 0; m < nsopi_[a_beta_h]; ++m){
            A_b->set(a_beta_h,m,0,ACb->get(a_beta_h,m,a_beta_mo));
        }
        for (int m = 0; m < nsopi_[b_beta_h]; ++m){
            B_b->set(b_beta_h,m,0,BCb->get(b_beta_h,m,b_beta_mo));
        }

        std::vector<SharedMatrix>& C_left = jk->C_left();
        C_left.clear();
        C_left.push_back(B_b);
        std::vector<SharedMatrix>& C_right = jk->C_right();
        C_right.clear();
        C_right.push_back(A_b);
        jk->compute();
        const std::vector<SharedMatrix >& Dn = jk->D();

        SharedMatrix Jnew = jk->J()[0];

        SharedMatrix D = SharedMatrix(new Matrix("D",nirrep_, nsopi_, nsopi_, a_alpha_h ^ b_alpha_h));
        D->zero();
        double** D_h = D->pointer(b_alpha_h);
        double* Dp = &(D_h[0][0]);
        for (int n = 0; n < nsopi_[a_alpha_h]; ++n){
            for (int m = 0; m < nsopi_[b_alpha_h]; ++m){
                D_h[m][n] = BCa->get(b_alpha_h,m,b_alpha_mo) * ACa->get(a_alpha_h,n,a_alpha_mo);
            }
        }
        double twoelint = Jnew->vector_dot(D);
        hamiltonian = twoelint * Stilde;
        fprintf(outfile,"  Matrix element from libfock = %20.12f\n",twoelint);
    }else if(num_alpha_nonc == 2 and num_beta_nonc == 0){
        overlap = 0.0;
        throw FeatureNotImplemented("CKS", "H in the case of two alpha noncoincidences", __FILE__, __LINE__);
    }else if(num_alpha_nonc == 0 and num_beta_nonc == 2){
        overlap = 0.0;
        throw FeatureNotImplemented("CKS", "H in the case of two beta noncoincidences", __FILE__, __LINE__);
    }
    jk->finalize();
    fflush(outfile);
    return std::make_pair(overlap,hamiltonian);
}

boost::tuple<SharedMatrix,SharedMatrix,SharedVector,double>
UCKS::corresponding_orbitals(SharedMatrix A, SharedMatrix B, Dimension dima, Dimension dimb)
{
    // Form <B|S|A>
    TempMatrix->gemm(false,false,1.0,S_,A,0.0);
    TempMatrix2->gemm(true,false,1.0,B,TempMatrix,0.0);

    // Extract the occupied blocks only
    SharedMatrix Sba = SharedMatrix(new Matrix("Sba",dimb,dima));
    for (int h = 0; h < nirrep_; ++h) {
        int naocc = dima[h];
        int nbocc = dimb[h];
        double** Sba_h = Sba->pointer(h);
        double** S_h = TempMatrix2->pointer(h);
        for (int i = 0; i < nbocc; ++i){
            for (int j = 0; j < naocc; ++j){
                Sba_h[i][j] = S_h[i][j];
            }
        }
    }
    // SVD <B|S|A>
    boost::tuple<SharedMatrix, SharedVector, SharedMatrix> UsV = Sba->svd_a_temps();
    SharedMatrix U = UsV.get<0>();
    SharedVector sigma = UsV.get<1>();
    SharedMatrix V = UsV.get<2>();
    Sba->svd_a(U,sigma,V);
//    sigma->print();
//    U->print();
//    V->print();

    // II. Transform the occupied orbitals to the new representation
    // Transform A with V (need to transpose V since svd returns V^T)
    // Extract the
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
    TempMatrix2->gemm(false,true,1.0,A,TempMatrix,0.0);
    SharedMatrix cA = SharedMatrix(new Matrix("Corresponding " + A->name(),A->rowspi(),dima));
    copy_subblock(TempMatrix2,cA,cA->rowspi(),dima,true);

    // Transform B with U
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
    TempMatrix2->gemm(false,false,1.0,B,TempMatrix,0.0);
    SharedMatrix cB = SharedMatrix(new Matrix("Corresponding " + B->name(),B->rowspi(),dimb));
    copy_subblock(TempMatrix2,cB,cB->rowspi(),dimb,true);


    // Find the product of the determinants of U and V
    double detU = 1.0;
    for (int h = 0; h < nirrep_; ++h) {
        int nmo = U->rowdim(h);
        if(nmo > 1){
            double d = 1.0;
            int* indx = new int[nmo];
            ludcmp(U->pointer(h),nmo,indx,&d);
            detU *= d;
            delete[] indx;
        }
    }
    double detV = 1.0;
    for (int h = 0; h < nirrep_; ++h) {
        int nmo = V->rowdim(h);
        if(nmo > 1){
            double d = 1.0;
            int* indx = new int[nmo];
            ludcmp(V->pointer(h),nmo,indx,&d);
            detV *= d;
            delete[] indx;
        }
    }
    fprintf(outfile,"\n det U = %f, det V = %f",detU,detV);
    double detUV = detU * detV;
    boost::tuple<SharedMatrix,SharedMatrix,SharedVector,double> result(cA,cB,sigma,detUV);
    return result;
}

}} // Namespaces



















//        boost::shared_ptr<PetiteList> pet(new PetiteList(KS::basisset_, integral_));
//        int nbf = KS::basisset_->nbf();
//        SharedMatrix SO2AO_ = pet->sotoao();
//        SharedMatrix Jnew_ao(new Matrix(nbf, nbf));
//        Dn[0]->print();
//        SO2AO_->print();
//        fflush(outfile);

//        Jnew_ao->remove_symmetry(Dn[0],SO2AO_);
//        Jnew_ao->print();
//        Jnew_ao->remove_symmetry(Jnew,SO2AO_);
//        Jnew_ao->print();
//        fflush(outfile);


//        int maxi4 = INDEX4(nsopi_[0]+1,nsopi_[0]+1,nsopi_[0]+1,nsopi_[0]+1)+nsopi_[0]+1;
//        double* integrals = new double[maxi4];
//        for (int l = 0; l < maxi4; ++l){
//            integrals[l] = 0.0;
//        }

//        IWL *iwl = new IWL(KS::psio_.get(), PSIF_SO_TEI, integral_threshold_, 1, 1);
//        Label *lblptr = iwl->labels();
//        Value *valptr = iwl->values();
//        int labelIndex, pabs, qabs, rabs, sabs, prel, qrel, rrel, srel, psym, qsym, rsym, ssym;
//        double value;
//        bool lastBuffer;
//        do{
//            lastBuffer = iwl->last_buffer();
//            for(int index = 0; index < iwl->buffer_count(); ++index){
//                labelIndex = 4*index;
//                pabs  = abs((int) lblptr[labelIndex++]);
//                qabs  = (int) lblptr[labelIndex++];
//                rabs  = (int) lblptr[labelIndex++];
//                sabs  = (int) lblptr[labelIndex++];
//                prel  = so2index_[pabs];
//                qrel  = so2index_[qabs];
//                rrel  = so2index_[rabs];
//                srel  = so2index_[sabs];
//                psym  = so2symblk_[pabs];
//                qsym  = so2symblk_[qabs];
//                rsym  = so2symblk_[rabs];
//                ssym  = so2symblk_[sabs];
//                value = (double) valptr[index];
//                integrals[INDEX4(prel,qrel,rrel,srel)] = value;
//            } /* end loop through current buffer */
//            if(!lastBuffer) iwl->fetch();
//        }while(!lastBuffer);
//        iwl->set_keep_flag(1);
//        delete iwl;
//        double c2 = 0.0;

//        SharedVector Ava = ACa->get_column(a_alpha_h,a_alpha_mo);
//        SharedVector Bva = BCa->get_column(b_alpha_h,b_alpha_mo);
//        SharedVector Avb = ACb->get_column(a_beta_h,a_beta_mo);
//        SharedVector Bvb = BCb->get_column(b_beta_h,b_beta_mo);


//        double* Ci = Bva->pointer();
//        double* Cj = Ava->pointer();
//        double* Ck = Bvb->pointer();
//        double* Cl = Avb->pointer();
//        for (int i = 0; i < nsopi_[0]; ++i){
//            for (int j = 0; j < nsopi_[0]; ++j){
//                for (int k = 0; k < nsopi_[0]; ++k){
//                    for (int l = 0; l < nsopi_[0]; ++l){
//                        c2 += integrals[INDEX4(i,j,k,l)] * Ci[i] * Cj[j] * Ck[k] * Cl[l];
//                    }
//                }
//            }
//        }
//        delete[] integrals;
//        fprintf(outfile,"  Matrix element from ints    = %20.12f\n",c2);

//        {
//        Dimension Aa_dim(nirrep_);
//        Dimension Ab_dim(nirrep_);
//        Dimension Ba_dim(nirrep_);
//        Dimension Bb_dim(nirrep_);
//        Aa_dim[a_alpha_h] = 1;
//        Ab_dim[a_beta_h] = 1;
//        Ba_dim[b_alpha_h] = 1;
//        Bb_dim[b_beta_h] = 1;

//        // Setting up and initialize the integraltransform object
//        std::vector<boost::shared_ptr<MOSpace> > spaces;
//        spaces.push_back(MOSpace::fzc);
//        spaces.push_back(MOSpace::occ);
//        spaces.push_back(MOSpace::vir);
//        spaces.push_back(MOSpace::fzv);
//        SharedMatrix Ama(new Matrix("F",nsopi_,Aa_dim));
//        SharedMatrix Amb(new Matrix("F",nsopi_,Ab_dim));
//        SharedMatrix Bma(new Matrix("F",nsopi_,Ba_dim));
//        SharedMatrix Bmb(new Matrix("F",nsopi_,Bb_dim));

//        SharedVector Ava = ACa->get_column(a_alpha_h,a_alpha_mo);
//        SharedVector Bva = BCa->get_column(b_alpha_h,b_alpha_mo);
//        SharedVector Avb = ACb->get_column(a_beta_h,a_beta_mo);
//        SharedVector Bvb = BCb->get_column(b_beta_h,b_beta_mo);

//        Ama->set_column(a_alpha_h,0,ACa->get_column(a_alpha_h,a_alpha_mo));
//        Bma->set_column(b_alpha_h,0,BCa->get_column(b_alpha_h,b_alpha_mo));
//        Amb->set_column(a_beta_h,0,ACb->get_column(a_beta_h,a_beta_mo));
//        Bmb->set_column(b_beta_h,0,BCb->get_column(b_beta_h,b_beta_mo));

//        Ama->print();
//        Bma->print();
//        Amb->print();
//        Bmb->print();
//        IntegralTransform* ints_ = new IntegralTransform(Bma,Ama,Bmb,Amb,spaces,
//                                                    IntegralTransform::Restricted,
//                                                    IntegralTransform::IWLOnly,
//                                                    IntegralTransform::QTOrder,
//                                                    IntegralTransform::None);
//        ints_->transform_tei(MOSpace::fzc, MOSpace::occ, MOSpace::vir, MOSpace::fzv);
//        delete ints_;

//        IWL *iwl = new IWL(KS::psio_.get(), PSIF_MO_TEI, integral_threshold_, 1, 1);
//        Label *lblptr = iwl->labels();
//        Value *valptr = iwl->values();
//        int labelIndex, pabs, qabs, rabs, sabs, prel, qrel, rrel, srel, psym, qsym, rsym, ssym;
//        double value;
//        bool lastBuffer;
//        do{
//            lastBuffer = iwl->last_buffer();
//            for(int index = 0; index < iwl->buffer_count(); ++index){
//                labelIndex = 4*index;
//                pabs  = abs((int) lblptr[labelIndex++]);
//                qabs  = (int) lblptr[labelIndex++];
//                rabs  = (int) lblptr[labelIndex++];
//                sabs  = (int) lblptr[labelIndex++];
//                prel  = so2index_[pabs];
//                qrel  = so2index_[qabs];
//                rrel  = so2index_[rabs];
//                srel  = so2index_[sabs];
//                psym  = so2symblk_[pabs];
//                qsym  = so2symblk_[qabs];
//                rsym  = so2symblk_[rabs];
//                ssym  = so2symblk_[sabs];
//                value = (double) valptr[index];
//                fprintf(outfile,"  (%2d %2d | %2d %2d) = %20.12f\n",pabs,qabs,rabs,sabs,value);
//            } /* end loop through current buffer */
//            if(!lastBuffer) iwl->fetch();
//        }while(!lastBuffer);
//        iwl->set_keep_flag(1);
//        delete iwl;

//        }
