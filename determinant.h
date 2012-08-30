#ifndef DETERMINANT_H
#define DETERMINANT_H

#include <libscf_solver/ks.h>

namespace psi{
namespace scf{

class Determinant
{
public:
    Determinant(double energy, SharedMatrix Ca, SharedMatrix Cb, Dimension nalphapi, Dimension nbetapi);
    Determinant(const Determinant& det);
    ~Determinant();
    double energy() {return energy_;}
    SharedMatrix Ca() {return Ca_;}
    SharedMatrix Cb() {return Cb_;}
    const Dimension nalphapi() {return nalphapi_;}
    const Dimension nbetapi() {return nbetapi_;}
private:
    double energy_;
    Dimension nalphapi_;
    Dimension nbetapi_;
    SharedMatrix Ca_;
    SharedMatrix Cb_;
};

}
typedef boost::shared_ptr<psi::scf::Determinant> SharedDeterminant;
} // Namespaces

#endif // Header guard
