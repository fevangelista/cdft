#include "excited_state.h"

using namespace psi;

namespace psi{ namespace scf{

ExcitedState::ExcitedState(int nirreps)
{
    // Set the number of holes/particles to zero
    nahole_ = 0;
    nbhole_ = 0;
    nbpart_ = 0;
    nbpart_ = 0;
    for(int h = 0; h < nirreps; ++h){
        aholepi_.push_back(0);
        bholepi_.push_back(0);
        apartpi_.push_back(0);
        bpartpi_.push_back(0);
    }
}

void ExcitedState::add_hole(int sym, SharedVector hole, double energy, bool alpha)
{
    if (alpha){
        nahole_ += 1;
        aholepi_[sym] += 1;
        ahole_.push_back(SharedVector(hole->clone()));
        ah_sym_.push_back(sym);
        epsilon_ah_.push_back(energy);
    }else{
        nbhole_ += 1;
        bholepi_[sym] += 1;
        bhole_.push_back(SharedVector(hole->clone()));
        bh_sym_.push_back(sym);
        epsilon_bh_.push_back(energy);
    }
}

void ExcitedState::add_particle(int sym, SharedVector particle, double energy, bool alpha)
{
    if (alpha){
        napart_ += 1;
        apartpi_[sym] += 1;
        apart_.push_back(SharedVector(particle->clone()));
        ap_sym_.push_back(sym);
        epsilon_ap_.push_back(energy);
    }else{
        nbpart_ += 1;
        bpartpi_[sym] += 1;
        bpart_.push_back(SharedVector(particle->clone()));
        bp_sym_.push_back(sym);
        epsilon_bp_.push_back(energy);
    }
}

SharedVector ExcitedState::get_hole(int n,bool alpha)
{
    if (alpha){
        return ahole_[n];
    }else{
        return bhole_[n];
    }
}

SharedVector ExcitedState::get_particle(int n,bool alpha)
{
    if (alpha){
        return apart_[n];
    }else{
        return bpart_[n];
    }
}

double ExcitedState::get_particle_energy(int n,bool alpha)
{
    if (alpha){
        return epsilon_ap_[n];
    }else{
        return epsilon_bp_[n];
    }
}

}} // Namespaces
