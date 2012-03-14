#ifndef SRC_LIB_RCKS_H
#define SRC_LIB_RCKS_H

#include <libscf_solver/ks.h>

namespace psi{
    class Options;
//    class VBase;
namespace scf{

/// A class for restricted constrained Kohn-Sham theory
class RCKS : public RKS {
public:
     RCKS(Options &options, boost::shared_ptr<PSIO> psio);
     virtual ~RCKS();
     /// Compute the Lowdin charges
     virtual void Lowdin();
     virtual void Lowdin2();
protected:
     /// The constraint matrices in the SO basis
     std::vector<SharedMatrix> W_so;
     /// The Lagrange multiplier, Vc in Phys. Rev. A, 72, 024502 (2005).
     double Vc;
     /// Optimize the Lagrange multiplier
     bool optimize_Vc;
     /// The gradient of the constrained functional W
     double gradW;
     /// The hessian of the constrained functional W
     double hessW;
     /// The number of fragments
     int nfrag;
     /// The nuclear charge of a fragment
     std::vector<double> frag_nuclear_charge;
     /// The constrained charge on each fragment
     std::vector<double> constrained_charges;
     /// Build the constrain matrices in the AO basis
     void build_W_so();
     /// Compute the gradient with respect to the Lagrange multiplier
     void gradient_of_W();

     /// Form the Fock matrix augmented with the constraints
     virtual void form_F();
     /// Compute the value of the Lagrangian, at convergence it yields the energy
     virtual double compute_E();
};


}} // Namespaces

#endif // Header guard
