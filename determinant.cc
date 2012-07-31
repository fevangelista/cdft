#include "determinant.h"


namespace psi{
namespace scf{

Determinant::Determinant(SharedMatrix Ca, SharedMatrix Cb, Dimension nalphapi, Dimension nbetapi)
    : Ca_(Ca), Cb_(Cb), nalphapi_(nalphapi), nbetapi_(nbetapi)
{
}

}} // namespaces
