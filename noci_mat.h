#ifndef _noci_mat_h_
#define _noci_mat_h_

#include "boost/tuple/tuple.hpp"

#include <libscf_solver/hf.h>

#include "determinant.h"

namespace psi{
class Options;
namespace scf{


class NOCI_mat : public UHF {
public:
    explicit NOCI_mat(Options &options, boost::shared_ptr<PSIO> psio,std::vector<SharedDeterminant> dets);
    virtual ~NOCI_mat();
    void print();
protected:

    SharedMatrix Ca_gs_;
    SharedMatrix Cb_gs_;
    std::vector<SharedDeterminant> dets_;
    void init();
};

}} // Namespaces

#endif // Header guard
