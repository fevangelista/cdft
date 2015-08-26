#include <physconst.h>
#include <psifiles.h>
#include <libmints/mints.h>
#include <libfock/apps.h>
#include <libfock/v.h>
#include <libfock/jk.h>
#include <liboptions/liboptions.h>

#include "noci_mat.h"

#define DEBUG_NOCI 0


using namespace psi;

namespace psi{ namespace scf{

NOCI_mat::NOCI_mat(Options &options, boost::shared_ptr<PSIO> psio,std::vector<SharedDeterminant> dets): UHF(options, psio),dets_(dets)
{
    init();
}


void NOCI_mat::init()
{

       Ca_gs_ =  SharedMatrix(new Matrix("Ca_gs_",nsopi_,nmopi_));
       Cb_gs_ =  SharedMatrix(new Matrix("Cb_gs_",nsopi_,nmopi_));
}

NOCI_mat::~NOCI_mat()
{
}

void NOCI_mat::print()
{
    for(int i=0; i < 2; ++i){
           outfile->Printf("what is this iiiiiii %d \n", i);
           Ca_gs_->copy(dets_[i]->Ca());
           Ca_gs_->print();
           outfile->Printf("\n");
           Cb_gs_->copy(dets_[i]->Cb());
           Cb_gs_->print();
           outfile->Printf("\n");

       }


}


}} // Namespaces

