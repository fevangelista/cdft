#ifndef EXCITED_STATE_H
#define EXCITED_STATE_H

#include <libscf_solver/ks.h>

namespace psi{
namespace scf{

class ExcitedState
{
public:
    ExcitedState();
    void add_hole(int sym,SharedVector hole,bool alpha);
    void add_particle(int sym,SharedVector particle,bool alpha);
    std::vector<int> aholepi() {return aholepi_;}
    std::vector<int> apartpi() {return apartpi_;}
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

    std::vector<SharedVector> ahole_;
    std::vector<SharedVector> apart_;
    std::vector<SharedVector> bhole_;
    std::vector<SharedVector> bpart_;
};

}
typedef boost::shared_ptr<psi::scf::ExcitedState> SharedExcitedState;
} // Namespaces

#endif // Header guard
