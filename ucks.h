#ifndef SRC_LIB_UCKS_H
#define SRC_LIB_UCKS_H

#include "boost/tuple/tuple.hpp"
#include <libscf_solver/ks.h>
#include <constraint.h>
#include <determinant.h>

namespace psi{
class Options;
namespace scf{

class ERIComputer
{
public:
    ERIComputer(double* Ci,double* Cj,double* Ck,double* Cl) {
        integral = 0.0;
        Ci_ = Ci;
        Cj_ = Cj;
        Ck_ = Ck;
        Cl_ = Cl;
    }
    bool k_required() const {return true;}
    void initialize(){}
    void finalize(){}
    void operator()(boost::shared_ptr<PKIntegrals> pk_integrals) {}
    void operator() (int pabs, int qabs, int rabs, int sabs,
                     int pirrep, int pso,
                     int qirrep, int qso,
                     int rirrep, int rso,
                     int sirrep, int sso,
                     double value)
    {
        integral += value * Ci_[pabs] * Cj_[qabs] * Ck_[rabs] * Cl_[sabs];
    }
    double integral;
    double* Ci_;
    double* Cj_;
    double* Ck_;
    double* Cl_;
};

/// A class for unrestricted constrained Kohn-Sham theory
class UCKS : public UKS {
public:
    explicit UCKS(Options &options, boost::shared_ptr<PSIO> psio);
    explicit UCKS(Options &options, boost::shared_ptr<PSIO> psio, boost::shared_ptr<Wavefunction> ref_scf, int state);
    explicit UCKS(Options &options, boost::shared_ptr<PSIO> psio, boost::shared_ptr<Wavefunction> ref_scf, int state, int symmetry);
    virtual ~UCKS();
protected:
    /// The fragment constraint matrices in the SO basis
    std::vector<SharedMatrix> W_frag;

    // Excitation energy
    /// Compute an excited state?
    bool do_excitation;
    /// Compute a state of a given symmetry?
    bool do_symmetry;
    /// Ground state energy
    double ground_state_energy;
    /// Ground state symmetry
    int ground_state_symmetry_;
    /// Symmetry of the excited state, if specified
    int excited_state_symmetry_;
    /// Excited state number, starting from one
    int state_;

    // Information about the excited states
    /// Determinant information for each electronic state
    std::vector<SharedDeterminant> dets;

    /// The Fock matrix projected onto the occupied space
    SharedMatrix PoFPo_;
    /// The Fock matrix projected onto the virtual space
    SharedMatrix PvFPv_;
    /// The Fock matrix projected onto the spectator space
    SharedMatrix QFQ_;
    /// The holes
    SharedMatrix Ch_;
    /// The particles
    SharedMatrix Cp_;
    /// The effective alpha Fock matrix in the MO basis
    SharedMatrix moFeffa_;
    /// The effective beta Fock matrix in the MO basis
    SharedMatrix moFeffb_;
    /// The eigenvectors of PoFPo
    SharedMatrix Uo_;
    /// The eigenvectors of PvFPv
    SharedMatrix Uv_;
    /// The eigenvalues of PoFPo
    SharedVector lambda_o_;
    /// The eigenvalues of PvFPv
    SharedVector lambda_v_;


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

    /// The old alpha density matrix
    SharedMatrix Dolda_;
    /// The old beta density matrix
    SharedMatrix Doldb_;

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
    /// A shift to apply to the virtual orbitals to improve convergence
    double level_shift_;

    int nW_opt;

    // UKS specific functions
    /// Class initializer
    void init();
    /// Initialize the exctiation functions
    void init_excitation( boost::shared_ptr<Wavefunction> ref_scf);
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
    /// The constrained hole algorithm for computing the orbitals
    void form_C_CH_algorithm();
    /// The constrained particle algorithm for computing the orbitals
    void form_C_CP_algorithm();
    /// The constrained hole/particle algorithm for computing the orbitals
    void form_C_CHP_algorithm();
    /// Compute a correction for the mixed excited states
    double compute_triplet_correction();
    /// Compute the singlet and triplet energy of a mixed excited state
    void spin_adapt_mixed_excitation();
    /// Compute the corresponding orbitals for a pair of MO sets
    boost::tuple<SharedMatrix,SharedMatrix,SharedVector,double> corresponding_orbitals(SharedMatrix A, SharedMatrix B, Dimension dima, Dimension dimb);

    // Helper functions
    /// Extract a block from matrix A and copies it to B
    void extract_square_subblock(SharedMatrix A, SharedMatrix B, bool occupied, Dimension npi, double diagonal_shift);
    /// Copy a subblock of dimension rowspi x colspi from matrix A into B.  If desired, it can copy the complementary subblock
    void copy_subblock(SharedMatrix A, SharedMatrix B, Dimension rowspi, Dimension colspi,bool occupied);
    /// Copy a subblock of dimension rowspi x colspi from matrix A into B.  If desired, it can copy the complementary subblock
    void copy_block(SharedMatrix A, SharedMatrix B, Dimension rowspi, Dimension colspi,
                    Dimension A_rows_offsetpi = Dimension(8), Dimension A_cols_offsetpi = Dimension(8),
                    Dimension B_rows_offsetpi = Dimension(8), Dimension B_cols_offsetpi = Dimension(8));
    // ROKS functions and variables
    /// Do ROKS?
    bool do_roks;

    std::pair<double,double> matrix_element(SharedDeterminant A, SharedDeterminant B);

    // Overloaded UKS function
    virtual void save_density_and_energy();
    virtual void form_F();
    virtual void form_C();
    virtual double compute_E();
    virtual void damp_update();
    virtual bool test_convergency();
    virtual void guess();
    virtual void save_information();
    virtual void save_fock();
};

}} // Namespaces

#endif // Header guard
