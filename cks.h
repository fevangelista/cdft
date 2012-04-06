#ifndef SRC_LIB_RCKS_H
#define SRC_LIB_RCKS_H

#include <libscf_solver/ks.h>
#include <constraint.h>

namespace psi{
    class Options;
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
     /// The fragment constraint matrices in the SO basis
     std::vector<SharedMatrix> W_so;
     /// The total constraint matrix
     SharedMatrix W_tot;
     /// A temporary matrix
     SharedMatrix Temp;
     /// A temporary matrix
     SharedMatrix Temp2;
     /// A copy of the one-electron potential
     SharedMatrix H_copy;
     /// The Lagrange multiplier, Vc in Phys. Rev. A, 72, 024502 (2005)
     double Vc;
     /// The constraint to enforce
     double Nc;
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
     /// Convergency threshold for the gradient of the constraint
     double gradW_threshold_;

     int nW_opt;
     double old_gradW;
     double gradW_mo_resp;
     double old_Vc;
     double BFGS_hessW;

     /// Build the constrain matrix in the SO basis
     void build_W_so();
     /// Compute the gradient of W with respect to the Lagrange multiplier
     void gradient_of_W();
     /// Compute the hessian of W with respect to the Lagrange multiplier
     void hessian_of_W();
     /// Optimize the constraint
     void constraint_optimization();
     /// Form the Fock matrix augmented with the constraints
     virtual void form_F();
     /// Compute the value of the Lagrangian, at convergence it yields the energy
     virtual double compute_E();
     /// Test the convergence of the CKS procedure
     virtual bool test_convergency();

     bool save_H_;
};

/// A class for unrestricted constrained Kohn-Sham theory
class UCKS : public UKS {
public:
     UCKS(Options &options, boost::shared_ptr<PSIO> psio);
     virtual ~UCKS();
//     /// Compute the Lowdin charges
//     virtual void Lowdin();
//     virtual void Lowdin2();
protected:
     /// The fragment constraint matrices in the SO basis
     std::vector<SharedMatrix> W_so;
     /// The constraint objects
     std::vector<SharedConstraint> constraints;
     /// A temporary matrix
     SharedMatrix Temp;
     /// A temporary matrix
     SharedMatrix Temp2;
     /// A copy of the one-electron potential
     SharedMatrix H_copy;
     /// The Lagrange multipliers, Vc in Phys. Rev. A, 72, 024502 (2005)
     SharedVector Vc;
     /// A copy of the Lagrange multipliers from the previous cycle
     SharedVector Vc_old;
     /// Optimize the Lagrange multiplier
     bool optimize_Vc;
     /// The number of constraints
     int nconstraints;
     /// The gradient of the constrained functional W
     SharedVector gradW;
     /// A copy of the gradient of W from the previous cycle
     SharedVector gradW_old;
     /// The MO response contribution to the gradient of W
     SharedVector gradW_mo_resp;
     /// The hessian of the constrained functional W
     SharedMatrix hessW;
     /// The hessian of the constrained functional W
     SharedMatrix hessW_BFGS;
     /// The number of fragments
     int nfrag;
     /// The nuclear charge of a fragment
     std::vector<double> frag_nuclear_charge;
     /// Convergency threshold for the gradient of the constraint
     double gradW_threshold_;

     int nW_opt;

     /// Build the constrain matrices in the SO basis
     void build_W_so();
     /// Compute the gradient of W with respect to the Lagrange multiplier
     void gradient_of_W();
     /// Compute the hessian of W with respect to the Lagrange multiplier
     void hessian_of_W();
     /// Update the hessian using the BFGS formula
     void hessian_update(SharedMatrix h, SharedVector dx, SharedVector dg);
     /// Optimize the constraint
     void constraint_optimization();
     /// Form the Fock matrix augmented with the constraints
     virtual void form_F();
     /// Compute the value of the Lagrangian, at convergence it yields the energy
     virtual double compute_E();
     /// Test the convergence of the CKS procedure
     virtual bool test_convergency();
     bool save_H_;
};

}} // Namespaces

#endif // Header guard
