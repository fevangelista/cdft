#include <libmints/mints.h>

#include "determinant.h"

using namespace psi;

namespace psi{
namespace scf{

Determinant::Determinant(double energy, SharedMatrix Ca, SharedMatrix Cb, Dimension nalphapi, Dimension nbetapi)
    : energy_(energy),nalphapi_(nalphapi), nbetapi_(nbetapi), Ca_(Ca->clone()), Cb_(Cb->clone())
{
}

Determinant::Determinant(const Determinant& det)
{
    energy_ = det.energy_;
    Ca_ = det.Ca_;
    Cb_ = det.Cb_;
    nalphapi_ = det.nalphapi_;
    nbetapi_ = det.nbetapi_;
}

Determinant::~Determinant()
{}

}} // namespaces
