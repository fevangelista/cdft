#include "excited_state.h"

using namespace psi;

namespace psi{ namespace scf{

ExcitedState::ExcitedState()
{
    // Set the number of holes/particles to zero
    nahole_ = 0;
    nbhole_ = 0;
    nbpart_ = 0;
    nbpart_ = 0;
    for(int h = 0; h < 8; ++h){
        aholepi_.push_back(0);
        bholepi_.push_back(0);
        apartpi_.push_back(0);
        bpartpi_.push_back(0);
    }
}

void ExcitedState::add_hole(int sym,SharedVector hole,bool alpha)
{
    if (alpha){
        nahole_ += 1;
        aholepi_[sym] += 1;
        ahole_.push_back(hole);
        ah_sym_.push_back(sym);
    }else{
        nbhole_ += 1;
        bholepi_[sym] += 1;
        bhole_.push_back(hole);
        bh_sym_.push_back(sym);
    }
}

void ExcitedState::add_particle(int sym,SharedVector particle,bool alpha)
{
    if (alpha){
        napart_ += 1;
        apartpi_[sym] += 1;
        apart_.push_back(particle);
        ap_sym_.push_back(sym);
    }else{
        nbpart_ += 1;
        bpartpi_[sym] += 1;
        bpart_.push_back(particle);
        bp_sym_.push_back(sym);
    }
}

}} // Namespaces
