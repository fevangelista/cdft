#ifndef CUCKS_H
#define CUCKS_H

#include <libscf_solver/ks.h>

namespace psi{
    class Options;
namespace scf{

/// A class for unrestricted constrained Kohn-Sham theory
class CUCKS : public UKS {
public:
     CUCKS(Options &options, boost::shared_ptr<PSIO> psio);
     virtual ~CUCKS();
};

}} // Namespaces
#endif // CUCKS_H
