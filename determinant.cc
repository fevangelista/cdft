#include <libmints/mints.h>

#include "determinant.h"

using namespace psi;

namespace psi{
namespace scf{

Determinant::Determinant(double energy, SharedMatrix Ca, SharedMatrix Cb, const Dimension& nalphapi, const Dimension& nbetapi)
    : energy_(energy),nalphapi_(nalphapi), nbetapi_(nbetapi), Ca_(Ca->clone()), Cb_(Cb->clone())
{
}

Determinant::Determinant(const Determinant& det)
{
    energy_ = det.energy_;
    Ca_ = det.Ca_->clone();
    Cb_ = det.Cb_->clone();
    nalphapi_ = det.nalphapi_;
    nbetapi_ = det.nbetapi_;
}

Determinant::~Determinant()
{}

int Determinant::symmetry()
{
    int symm = 0;
    int nirrep = nalphapi_.n();
    for (int h = 0; h < nirrep; ++h){
        // Check if there is an odd number of electrons in h
        if( std::abs(nalphapi_[h] - nbetapi_[h]) % 2 == 1){
            symm ^= h;
        }
    }
    return symm;
}

void Determinant::print()
{
    nalphapi_.print();
    nbetapi_.print();
}

}} // namespaces
