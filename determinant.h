#ifndef DETERMINANT_H
#define DETERMINANT_H

#include <libscf_solver/ks.h>

namespace psi{
namespace scf{

class Determinant
{
public:
    Determinant(SharedMatrix Ca, SharedMatrix Cb, Dimension nalphapi, Dimension nbetapi);
private:
    Dimension nalphapi_;
    Dimension nbetapi_;
    SharedMatrix Ca_;
    SharedMatrix Cb_;
};

}} // Namespaces

#endif // Header guard
