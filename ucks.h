#ifndef SRC_LIB_UCKS_H
#define SRC_LIB_UCKS_H

#include <libscf_solver/ks.h>
#include <constraint.h>
#include <excited_state.h>

namespace psi{
    class Options;
namespace scf{

/// A class for unrestricted constrained Kohn-Sham theory
class UCKS : public UKS {
public:
     UCKS(Options &options, boost::shared_ptr<PSIO> psio);
     UCKS(Options &options, boost::shared_ptr<PSIO> psio, boost::shared_ptr<UCKS> ref_scf);
     virtual ~UCKS();
protected:
     /// The fragment constraint matrices in the SO basis
     std::vector<SharedMatrix> W_frag;

     // Excitation energy
     /// Compute an excited state as an optimal singly excited state
     bool do_excitation;
     /// Number of occupied orbitals to project out
     int nexclude_occ;
     /// Number of virtual orbitals to project out
     int nexclude_vir;
     /// The occupation of the alpha orbitals
     SharedVector aocc_num_;
     /// The occupation of the beta orbitals
     SharedVector bocc_num_;

     // Information about the excited states
     /// The alpha eigenvalues for each electronic state
     std::vector<SharedVector> state_epsilon_a;
     /// The alpha MO coefficients for each electronic state
     std::vector<SharedMatrix> state_Ca;
     /// The beta MO coefficients for each electronic state
     std::vector<SharedMatrix> state_Cb;
     /// The alpha density matrix for each electronic state
     std::vector<Dimension> state_nalphapi;
     /// The beta MO coefficients for each electronic state
     std::vector<Dimension> state_nbetapi;
     /// Details of the excited states
     SharedExcitedState current_excited_state;
     /// Details of the other excited states
     std::vector<SharedExcitedState> excited_states;

     /// The alpha density matrix for each electronic state
     std::vector<SharedMatrix> state_Da;
     /// The beta density matrix for each electronic state
     std::vector<SharedMatrix> state_Db;
     /// The ground state scf object
     boost::shared_ptr<UCKS> ref_scf_;
     /// The alpha Fock matrix projected onto the occupied space
     SharedMatrix PoFaPo_;
     /// The alpha Fock matrix projected onto the virtual space
     SharedMatrix PvFaPv_;
     /// The eigenvectors of PoFaPo
     SharedMatrix Uo;
     /// The eigenvectors of PvFaPv
     SharedMatrix Uv;
     /// The eigenvalues of PoFaPo
     SharedVector lambda_o;
     /// The eigenvalues of PvFaPv
     SharedVector lambda_v;
     /// The beta Fock matrix projected onto the occupied space
     SharedMatrix PoFbPo_;
     /// The beta Fock matrix projected onto the virtual space
     SharedMatrix PvFbPv_;
     /// The eigenvectors of PoFbPo
     SharedMatrix Uob;
     /// The eigenvectors of PvFbPv
     SharedMatrix Uvb;
     /// The eigenvalues of PoFbPo
     SharedVector lambda_ob;
     /// The eigenvalues of PvFbPv
     SharedVector lambda_vb;

     /// The alpha penalty function
     SharedMatrix Pa;
     /// The alpha unitary transformation <phi'|phi>
     SharedMatrix Ua;
     /// The beta unitary transformation <phi'|phi>
     SharedMatrix Ub;
     /// The constraint objects
     std::vector<SharedConstraint> constraints;


     /// A temporary matrix
     SharedMatrix TempMatrix;
     /// A temporary matrix
     SharedMatrix TempMatrix2;
     /// A temporary vector
     SharedVector TempVector;
     /// SVD temporary matrix V
     SharedMatrix svdV;
     /// SVD temporary matrix U
     SharedMatrix svdU;
     /// SVD temporary vector sigma
     SharedVector svds;

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
     /// Flag to save the one-electron part of the Hamiltonian
     bool save_H_;

     int nW_opt;

     // UKS specific functions
     /// Class initializer
     void init();
     /// Build the fragment constrain matrices in the SO basis
     void build_W_frag();
     /// Build the excitation constraint matrices in the SO basis
     void gradient_of_W();
     /// Compute the hessian of W with respect to the Lagrange multiplier
     void hessian_of_W();
     /// Update the hessian using the BFGS formula
     void hessian_update(SharedMatrix h, SharedVector dx, SharedVector dg);
     /// Optimize the constraint
     void constraint_optimization();
     /// Compute the overlap of this solution to the n-th state
     double compute_overlap(int n);
     /// Compute a correction for the mixed excited states
     double compute_triplet_correction();
     /// Compute the corresponding orbitals for the alpha-beta MOs
     void corresponding_ab_mos();

     // Overloaded UKS function
     /// Form the Fock matrix augmented with the constraints and/or projected
     virtual void form_F();
     /// Diagonalize the Fock matrix to get the MO coefficients
     virtual void form_C();
     /// Computes the density matrix using the occupation numbers
     virtual void form_D();
     /// Compute the value of the Lagrangian, at convergence it yields the energy
     virtual double compute_E();
     /// Test the convergence of the CKS procedure
     virtual bool test_convergency();
     /// Guess the starting MO
     virtual void guess();
};

}} // Namespaces

#endif // Header guard
