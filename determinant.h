#ifndef DETERMINANT_H
#define DETERMINANT_H

#include <libscf_solver/ks.h>

namespace psi{
namespace scf{

class Determinant
{
public:
    Determinant(SharedMatrix Ca, SharedMatrix Cb, Dimension nalphapi, Dimension nbetapi);
    ~Determinant();
    SharedMatrix Ca() {return Ca_;}
    SharedMatrix Cb() {return Cb_;}
    const Dimension& nalphapi() {return nalphapi_;}
    const Dimension& nbetapi() {return nbetapi_;}
    void spin_flip();
private:
    Dimension nalphapi_;
    Dimension nbetapi_;
    Dimension nsopi_;
    SharedMatrix Ca_;
    SharedMatrix Cb_;
};

}
typedef boost::shared_ptr<psi::scf::Determinant> SharedDeterminant;
} // Namespaces

#endif // Header guard
