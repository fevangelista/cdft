
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

void UCKS::form_C_CH_algorithm()
{
    int nstate = static_cast<int>(dets.size());
    fprintf(outfile,"  Computing %d optimal hole orbitals\n",nstate);fflush(outfile);

    // Save the hole information in a ExcitedState object
    current_excited_state = SharedExcitedState(new ExcitedState(nirrep_));

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
        TempVector->zero();
        for (int p = 0; p < nsopi_[hole_h]; ++p){
            double c_p = 0.0;
            for (int i = 0; i < dets[m]->nalphapi()[hole_h]; ++i){
                c_p += dets[m]->Ca()->get(hole_h,p,i) * Uo_->get(hole_h,i,hole_mo) ;
            }
            TempVector->set(hole_h,p,c_p);
        }
        current_excited_state->add_hole(hole_h,TempVector,hole_energy,true);
    }

    // Put the hole orbitals in Ch
    std::vector<int> aholepi = current_excited_state->aholepi();
    SharedMatrix Ch = SharedMatrix(new Matrix("Ch",nsopi_,aholepi));
    SharedMatrix Cho = SharedMatrix(new Matrix("Cho",nsopi_,aholepi));
    std::vector<int> offset(nirrep_,0);
    for (int m = 0; m < nstate; ++m){
        int h = current_excited_state->ah_sym(m);
        Ch->set_column(h,offset[h],current_excited_state->get_hole(m,true));
        offset[h] += 1;
    }

    // Orthogonalize the hole orbitals
    SharedMatrix Spp = SharedMatrix(new Matrix("Spp",aholepi,aholepi));
    SharedMatrix Upp = SharedMatrix(new Matrix("Upp",aholepi,aholepi));
    SharedVector spp = SharedVector(new Vector("spp",aholepi));
    fprintf(outfile,"  -->> A <<--\n");fflush(outfile);
    S_->print();
    Ch->print();
    Spp->print();
    Spp->transform(S_,Ch);
    fprintf(outfile,"  -->> B <<--\n");fflush(outfile);
    Spp->print();
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
    Cho->gemm(false,false,1.0,Ch,Upp,0.0);
    // Form the projector onto the orbitals orthogonal to the particles in the ground state mo representation
    TempMatrix->gemm(false,true,1.0,Cho,Cho,0.0);
    TempMatrix->transform(S_);
    TempMatrix->transform(dets[0]->Ca());
    TempMatrix2->identity();
    TempMatrix2->subtract(TempMatrix);

    // Form the Fock matrix in the GS basis, project our the holes, diagonalize it, and transform the MO coefficients
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

}} // Namespaces

















