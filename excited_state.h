#ifndef EXCITED_STATE_H
#define EXCITED_STATE_H

#include <libscf_solver/ks.h>

namespace psi{
namespace scf{

class ExcitedState
{
public:
    ExcitedState();
    void add_hole(int sym,SharedVector hole,double energy,bool alpha);
    void add_particle(int sym,SharedVector particle,double energy,bool alpha);
    SharedVector get_hole(int n,bool alpha);
    SharedVector get_particle(int n,bool alpha);
    double get_hole_energy(int n,bool alpha);
    double get_particle_energy(int n,bool alpha);
    std::vector<int> aholepi() {return aholepi_;}
    std::vector<int> apartpi() {return apartpi_;}
    int ap_sym(int n) {return ap_sym_[n];}
    int ah_sym(int n) {return ah_sym_[n];}
protected:
    /// The number of alpha holes
    int nahole_;
    /// The number of beta holes
    int nbhole_;
    /// The number of alpha particles
    int napart_;
    /// The number of beta particles
    int nbpart_;
    /// The irrep of the alpha holes
    std::vector<int> ah_sym_;
    /// The irrep of the beta holes
    std::vector<int> bh_sym_;
    /// The irrep of the alpha particles
    std::vector<int> ap_sym_;
    /// The irrep of the beta particles
    std::vector<int> bp_sym_;
    /// The irrep of the alpha holes
    std::vector<int> aholepi_;
    /// The irrep of the beta holes
    std::vector<int> bholepi_;
    /// The irrep of the alpha particles
    std::vector<int> apartpi_;
    /// The irrep of the beta particles
    std::vector<int> bpartpi_;

    /// The energy of the alpha holes
    std::vector<double> epsilon_ah_;
    /// The energy of the beta holes
    std::vector<double> epsilon_bh_;
    /// The energy of the alpha particles
    std::vector<double> epsilon_ap_;
    /// The energy of the beta particles
    std::vector<double> epsilon_bp_;

    std::vector<SharedVector> ahole_;
    std::vector<SharedVector> apart_;
    std::vector<SharedVector> bhole_;
    std::vector<SharedVector> bpart_;
};

}
typedef boost::shared_ptr<psi::scf::ExcitedState> SharedExcitedState;
} // Namespaces

#endif // Header guard
