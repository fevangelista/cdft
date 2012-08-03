
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

void UCKS::form_C_CHP_algorithm()
{
    // Excited state: use specialized code

    // Transform Fa to the ground state MO basis
    TempMatrix->transform(Fa_,state_Ca[0]);

    // Set the orbital transformation matrices for the occ and vir blocks
    // equal to the identity so that if we decide to optimize only the hole
    // or the particle all is ok
    Uo_->identity();
    Uv_->identity();
    boost::tuple<double,int,int> hole;
    boost::tuple<double,int,int> particle;
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
    if(do_constrained_hole){
        extract_block(TempMatrix,PoFPo_,true,state_nalphapi[0],1.0e9);
        PoFPo_->diagonalize(Uo_,lambda_o_);
        // Sort the orbitals according to the eigenvalues of PoFaPo
        std::vector<boost::tuple<double,int,int> > sorted_occ;
        for (int h = 0; h < nirrep_; ++h){
            int nocc = state_nalphapi[0][h];
            for (int i = 0; i < nocc; ++i){
                sorted_occ.push_back(boost::make_tuple(lambda_o_->get(h,i),h,i));
            }
        }
        std::sort(sorted_occ.begin(),sorted_occ.end());
        // Extract the hole alpha orbital according to an energy criteria (this needs a generalization)
        if (KS::options_.get_str("CDFT_EXC_HOLE") == "VALENCE"){
            // For valence excitations select the highest lying orbital (HOMO-like)
            hole = sorted_occ.back();
        }else if(KS::options_.get_str("CDFT_EXC_HOLE") == "CORE"){
            // For core excitations select the lowest lying orbital (1s-like)
            hole = sorted_occ.front();
        }

    }

    if(do_constrained_part){
        extract_block(TempMatrix,PvFPv_,false,state_nalphapi[0],1.0e9);
        PvFPv_->diagonalize(Uv_,lambda_v_);
        // Sort the orbitals according to the eigenvalues of PvFaPv
        std::vector<boost::tuple<double,int,int> > sorted_vir;
        for (int h = 0; h < nirrep_; ++h){
            int nocc = state_nalphapi[0][h];
            int nvir = nmopi_[h] - nocc;
            for (int i = 0; i < nvir; ++i){
                sorted_vir.push_back(boost::make_tuple(lambda_v_->get(h,i),h,i + nocc));  // N.B. shifted to full indexing
            }
        }
        std::sort(sorted_vir.begin(),sorted_vir.end());
        // In the case of particle, we assume that we are always interested in the lowest lying orbitals
        particle = sorted_vir.front();
    }

    // Form the transformation matrix that diagonalizes the PoFaPo and PvFaPv blocks
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
    Ca_->gemm(false,false,1.0,state_Ca[0],TempMatrix,0.0);
    if(do_constrained_hole and do_constrained_part){
        fprintf(outfile,"   constrained hole/particle pair :(irrep = %d,mo = %d,energy = %.6f) -> (irrep = %d,mo = %d,energy = %.6f)\n",
                hole.get<1>(),hole.get<2>(),hole.get<0>(),
                particle.get<1>(),particle.get<2>(),particle.get<0>());
    }else if(do_constrained_hole and not do_constrained_part){
        fprintf(outfile,"   constrained hole :(irrep = %d,mo = %d,energy = %.6f)\n",
                hole.get<1>(),hole.get<2>(),hole.get<0>());
    }else if(not do_constrained_hole and do_constrained_part){
        fprintf(outfile,"   constrained particle :(irrep = %d,mo = %d,energy = %.6f)\n",
                particle.get<1>(),particle.get<2>(),particle.get<0>());
    }

    // Save the hole and particle information and at the same time zero the columns in Ca_
    current_excited_state = SharedExcitedState(new ExcitedState);
    if(do_constrained_hole){
        SharedVector hole_mo = Ca_->get_column(hole.get<1>(),hole.get<2>());
        Ca_->zero_column(hole.get<1>(),hole.get<2>());
        epsilon_a_->set(hole.get<1>(),hole.get<2>(),0.0);
        current_excited_state->add_hole(hole.get<1>(),hole_mo,hole.get<0>(),true);
    }
    if(do_constrained_part){
        SharedVector particle_mo = Ca_->get_column(particle.get<1>(),particle.get<2>());
        Ca_->zero_column(particle.get<1>(),particle.get<2>());
        epsilon_a_->set(particle.get<1>(),particle.get<2>(),0.0);
        current_excited_state->add_particle(particle.get<1>(),particle_mo,particle.get<0>(),true);
    }

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
    if(do_relax_spectators){
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
        if(do_constrained_hole){
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
        if(do_constrained_part){
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
        int m = napartpi[h];  // Offset by the number of holes
        int nso = nsopi_[h];
        int nmo = nmopi_[h];
        double** T_h = TempMatrix->pointer(h);
        double** C_h = Ca_->pointer(h);
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
    if(do_constrained_hole){
        // Place the hole orbital in the last MO of its irrep
        TempMatrix->set_column(hole.get<1>(),nmopi_[hole.get<1>()]-1,current_excited_state->get_hole(0,true));
        TempVector->set(hole.get<1>(),nmopi_[hole.get<1>()]-1,hole.get<0>());
    }
    if(do_constrained_part){
        // Place the particle orbital in the first MO of its irrep
        TempMatrix->set_column(particle.get<1>(),0,current_excited_state->get_particle(0,true));
        TempVector->set(particle.get<1>(),0,particle.get<0>());
    }
    Ca_->copy(TempMatrix);
    epsilon_a_->copy(TempVector.get());

    Ca_->print();
    epsilon_a_->print();

    // BETA
    diagonalize_F(Fb_, Cb_, epsilon_b_);

    //find_occupation();

    if (debug_) {
        Ca_->print(outfile);
        Cb_->print(outfile);
    }
}



}} // Namespaces


