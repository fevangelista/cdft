#include <libmints/mints.h>

#include "determinant.h"

using namespace psi;

namespace psi{
namespace scf{

Determinant::Determinant(SharedMatrix Ca, SharedMatrix Cb, Dimension nalphapi, Dimension nbetapi)
    : nalphapi_(nalphapi), nbetapi_(nbetapi), Ca_(Ca->clone()), Cb_(Cb->clone())
{
}

Determinant::~Determinant()
{}

}} // namespaces
