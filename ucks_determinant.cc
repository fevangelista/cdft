
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
    boost::tuple<SharedMatrix,SharedMatrix,SharedVector,double> cbeta = corresponding_orbitals(A->Cb(),B->Cb(),A->nbetapi(),B->nbetapi());
    SharedMatrix ACa = calpha.get<0>();
    SharedMatrix BCa = calpha.get<1>();
    SharedMatrix ACb = cbeta.get<0>();
    SharedMatrix BCb = cbeta.get<1>();
    double detUValpha = calpha.get<3>();
    double detUVbeta = cbeta.get<3>();
    SharedVector s_a = calpha.get<2>();
    SharedVector s_b = cbeta.get<2>();
    s_a->print();
    s_b->print();
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
    fprintf(outfile,"\n A(alpha): ");
    for (size_t k = 0; k < Aalpha_nonc.size(); ++k){
        int i_h = Aalpha_nonc[k].first;
        int i_mo = Aalpha_nonc[k].second;
        fprintf(outfile,"%1d %3d",i_h,i_mo);
    }
    fprintf(outfile,"\n B(alpha): ");
    for (size_t k = 0; k < Balpha_nonc.size(); ++k){
        int i_h = Balpha_nonc[k].first;
        int i_mo = Balpha_nonc[k].second;
        fprintf(outfile,"%1d %3d",i_h,i_mo);
    }
    fprintf(outfile,"\n A(beta):  ");
    for (size_t k = 0; k < Abeta_nonc.size(); ++k){
        int i_h = Abeta_nonc[k].first;
        int i_mo = Abeta_nonc[k].second;
        fprintf(outfile,"%1d %3d",i_h,i_mo);
    }
    fprintf(outfile,"\n B(beta):  ");
    for (size_t k = 0; k < Bbeta_nonc.size(); ++k){
        int i_h = Bbeta_nonc[k].first;
        int i_mo = Bbeta_nonc[k].second;
        fprintf(outfile,"%1d %3d",i_h,i_mo);
    }

    double Stilde = Sta * Stb * detUValpha * detUVbeta;
    fprintf(outfile,"\n  Stilde = %.6f\n",Stilde);

    boost::shared_ptr<JK> jk = JK::build_JK();
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

        Dimension A_beta_dim(nirrep_,"A_beta_dim");
        A_beta_dim[a_beta_h] = 1;
        Dimension B_beta_dim(nirrep_,"B_beta_dim");
        B_beta_dim[b_beta_h] = 1;

        SharedMatrix A_b(new Matrix("A_b", nsopi_, A_beta_dim));
        A_b->set_column(a_beta_h,0,ACb->get_column(a_beta_h,a_beta_mo));
        SharedMatrix B_b(new Matrix("B_b", nsopi_, B_beta_dim));
        B_b->set_column(b_beta_h,0,BCb->get_column(b_beta_h,b_beta_mo));

        std::vector<SharedMatrix>& C_left = jk->C_left();
        C_left.clear();
        C_left.push_back(B_b);
        std::vector<SharedMatrix>& C_right = jk->C_right();
        C_right.clear();
        C_right.push_back(A_b);
        jk->compute();
        SharedMatrix Jnew = jk->J()[0];
        Jnew->print();

        SharedMatrix D = SharedMatrix(new Matrix("D",nirrep_, nsopi_, nsopi_, a_alpha_h * b_alpha_h));
        D->zero();
        double** D_h = D->pointer(b_alpha_h);
        for (int m = 0; m < nsopi_[b_alpha_h]; ++m){
            for (int n = 0; n < nsopi_[a_alpha_h]; ++n){
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
    boost::tuple<SharedMatrix, SharedVector, SharedMatrix> UsV = Sba->svd_temps();
    SharedMatrix U = UsV.get<0>();
    SharedVector sigma = UsV.get<1>();
    SharedMatrix V = UsV.get<2>();
    Sba->svd(U,sigma,V);
//    sigma->print();
    U->print();
    V->print();

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
        double d = 1.0;
        int nmo = U->rowdim(h);
        int* indx = new int[nmo];
        ludcmp(U->pointer(h),nmo,indx,&d);
        detU *= d;
        delete[] indx;
    }
    double detV = 1.0;
    for (int h = 0; h < nirrep_; ++h) {
        double d = 1.0;
        int nmo = V->rowdim(h);
        int* indx = new int[nmo];
        ludcmp(V->pointer(h),nmo,indx,&d);
        detV *= d;
        delete[] indx;
    }
    fprintf(outfile,"\n det U = %f, det V = %f",detU,detV);
    double detUV = detU * detV;

    return boost::tuple<SharedMatrix,SharedMatrix,SharedVector,double>(cA,cB,sigma,detUV);
}

}} // Namespaces


















