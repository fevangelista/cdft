#ifndef JUNK_H
#define JUNK_H

void UCKS::form_C_CP_algorithm()
{

    // Excited state: use specialized code
    int nstate = static_cast<int>(state_Ca.size());
    fprintf(outfile,"  Computing %d optimal particle orbitals\n",nstate);

    for (int m = 0; m < nstate; ++m){

    }
    // Transform Fa to the ground state MO basis
    TempMatrix->transform(Fa_,dets[0]->Ca());

    // Set the orbital transformation matrices for the occ and vir blocks
    // equal to the identity so that if we decide to optimize only the hole
    // or the particle all is ok
    Uo_->identity();
    Uv_->identity();
    boost::tuple<double,int,int> hole;
    boost::tuple<double,int,int> particle;
    // Grab the occ and vir blocks
    // |--------|--------|
    // |        |        |
    // | PoFaPo |        |
    // |        |        |
    // |--------|--------|
    // |        |        |
    // |        | PvFaPv |
    // |        |        |
    // |--------|--------|
    if(do_constrained_hole){
        PoFPo_->identity();
        PoFPo_->scale(1.0e9);
        for (int h = 0; h < nirrep_; ++h){
            int nocc = dets[0]->nalphapi()[h];
            if (nocc != 0){
                double** Temp_h = TempMatrix->pointer(h);
                double** PoFaPo_h = PoFPo_->pointer(h);
                for (int i = 0; i < nocc; ++i){
                    for (int j = 0; j < nocc; ++j){
                        PoFaPo_h[i][j] = Temp_h[i][j];
                    }
                }
            }
        }
        PoFPo_->diagonalize(Uo_,lambda_o_);
        // Sort the orbitals according to the eigenvalues of PoFaPo
        std::vector<boost::tuple<double,int,int> > sorted_occ;
        for (int h = 0; h < nirrep_; ++h){
            int nocc = dets[0]->nalphapi()[h];
            for (int i = 0; i < nocc; ++i){
                sorted_occ.push_back(boost::make_tuple(lambda_o_->get(h,i),h,i));
            }
        }
        std::sort(sorted_occ.begin(),sorted_occ.end());
        // Extract the hole alpha orbital according to an energy criteria (this needs a generalization)
        if (KS::options_.get_str("CDFT_EXC_HOLE") == "VALENCE"){
            // For valence excitations select the highest lying orbital (HOMO-like)
            hole = sorted_occ.back();
        }else if(KS::options_.get_str("CDFT_EXC_HOLE") == "CORE"){
            // For core excitations select the lowest lying orbital (1s-like)
            hole = sorted_occ.front();
        }

    }

    if(do_constrained_part){
        PvFPv_->identity();
        PvFPv_->scale(1.0e9);
        for (int h = 0; h < nirrep_; ++h){
            int nocc = dets[0]->nalphapi()[h];
            int nvir = nmopi_[h] - nocc;
            if (nvir != 0){
                double** Temp_h = TempMatrix->pointer(h);
                double** PvFaPv_h = PvFPv_->pointer(h);
                for (int i = 0; i < nvir; ++i){
                    for (int j = 0; j < nvir; ++j){
                        PvFaPv_h[i][j] = Temp_h[i + nocc][j + nocc];
                    }
                }
            }
        }
        PvFPv_->diagonalize(Uv_,lambda_v_);
        // Sort the orbitals according to the eigenvalues of PvFaPv
        std::vector<boost::tuple<double,int,int> > sorted_vir;
        for (int h = 0; h < nirrep_; ++h){
            int nocc = dets[0]->nalphapi()[h];
            int nvir = nmopi_[h] - nocc;
            for (int i = 0; i < nvir; ++i){
                sorted_vir.push_back(boost::make_tuple(lambda_v_->get(h,i),h,i + nocc));  // N.B. shifted to full indexing
            }
        }
        std::sort(sorted_vir.begin(),sorted_vir.end());
        // In the case of particle, we assume that we are always interested in the lowest lying orbitals
        particle = sorted_vir.front();
    }

    // Form the transformation matrix that diagonalizes the PoFaPo and PvFaPv blocks
    // |----|----|
    // | Uo | 0  |
    // |----|----|
    // | 0  | Uv |
    // |----|----|
    TempMatrix->zero();
    for (int h = 0; h < nirrep_; ++h){
        int nocc = dets[0]->nalphapi()[h];
        int nvir = nmopi_[h] - nocc;
        if (nocc != 0){
            double** Temp_h = TempMatrix->pointer(h);
            double** Uo_h = Uo_->pointer(h);
            for (int i = 0; i < nocc; ++i){
                epsilon_a_->set(h,i,lambda_o_->get(h,i));
                for (int j = 0; j < nocc; ++j){
                    Temp_h[i][j] = Uo_h[i][j];
                }
            }
        }
        if (nvir != 0){
            double** Temp_h = TempMatrix->pointer(h);
            double** Uv_h = Uv_->pointer(h);
            for (int i = 0; i < nvir; ++i){
                epsilon_a_->set(h,i + nocc,lambda_v_->get(h,i));
                for (int j = 0; j < nvir; ++j){
                    Temp_h[i + nocc][j + nocc] = Uv_h[i][j];
                }
            }
        }
    }

    // Get the excited state orbitals: Ca(ex) = Ca(gs) * (Uo | Uv)
    Ca_->gemm(false,false,1.0,dets[0]->Ca(),TempMatrix,0.0);
    if(do_constrained_hole and do_constrained_part){
        fprintf(outfile,"   constrained hole/particle pair :(irrep = %d,mo = %d,energy = %.6f) -> (irrep = %d,mo = %d,energy = %.6f)\n",
                hole.get<1>(),hole.get<2>(),hole.get<0>(),
                particle.get<1>(),particle.get<2>(),particle.get<0>());
    }else if(do_constrained_hole and not do_constrained_part){
        fprintf(outfile,"   constrained hole :(irrep = %d,mo = %d,energy = %.6f)\n",
                hole.get<1>(),hole.get<2>(),hole.get<0>());
    }else if(not do_constrained_hole and do_constrained_part){
        fprintf(outfile,"   constrained particle :(irrep = %d,mo = %d,energy = %.6f)\n",
                particle.get<1>(),particle.get<2>(),particle.get<0>());
    }

    // Save the hole and particle information and at the same time zero the columns in Ca_
    current_excited_state = SharedExcitedState(new ExcitedState);
    if(do_constrained_hole){
        SharedVector hole_mo = Ca_->get_column(hole.get<1>(),hole.get<2>());
        Ca_->zero_column(hole.get<1>(),hole.get<2>());
        epsilon_a_->set(hole.get<1>(),hole.get<2>(),0.0);
        current_excited_state->add_hole(hole.get<1>(),hole_mo,true);
    }
    if(do_constrained_part){
        SharedVector particle_mo = Ca_->get_column(particle.get<1>(),particle.get<2>());
        Ca_->zero_column(particle.get<1>(),particle.get<2>());
        epsilon_a_->set(particle.get<1>(),particle.get<2>(),0.0);
        current_excited_state->add_particle(particle.get<1>(),particle_mo,true);
    }

    // Adjust the occupation (nalphapi_,nbetapi_)
    for (int h = 0; h < nirrep_; ++h){
        nalphapi_[h] = dets[0]->nalphapi()[h];
        nbetapi_[h] = state_nbetapi[0][h];
    }
    nalphapi_[hole.get<1>()] -= 1;
    nalphapi_[particle.get<1>()] += 1;

    int old_socc[8];
    int old_docc[8];
    for(int h = 0; h < nirrep_; ++h){
        old_socc[h] = soccpi_[h];
        old_docc[h] = doccpi_[h];
    }

    for (int h = 0; h < nirrep_; ++h) {
        soccpi_[h] = std::abs(nalphapi_[h] - nbetapi_[h]);
        doccpi_[h] = std::min(nalphapi_[h] , nbetapi_[h]);
    }

    bool occ_changed = false;
    for(int h = 0; h < nirrep_; ++h){
        if( old_socc[h] != soccpi_[h] || old_docc[h] != doccpi_[h]){
            occ_changed = true;
            break;
        }
    }

    // If print > 2 (diagnostics), print always
    if((print_ > 2 || (print_ && occ_changed)) && iteration_ > 0){
        if (Communicator::world->me() == 0)
            fprintf(outfile, "\tOccupation by irrep:\n");
        print_occupation();
    }

    // Optionally, include relaxation effects
    if(do_relax_spectators){
        // Transform Fa to the excited state MO basis, this includes the hole and particle states
        TempMatrix->transform(Fa_,Ca_);

        // Zero the terms that couple the hole, particle, and the rest of the orbitals
        // |--------|--------|
        // |       0|0       |
        // |       0|0       |
        // |00000000|00000000|
        // |--------|--------|
        // |00000000|00000000|
        // |       0|0       |
        // |       0|0       |
        // |--------|--------|
        if(do_constrained_hole){
            // Zero the hole couplings
            int h = hole.get<1>();
            int i = hole.get<2>();
            int nmo = nmopi_[h];
            if (nmo != 0){
                double** Temp_h = TempMatrix->pointer(h);
                for (int p = 0; p < nmo; ++p){
                    if(p != i){
                        Temp_h[i][p] = Temp_h[p][i] = 0.0;
                    }
                }
            }

        }
        if(do_constrained_part){
            // Zero the LUMO couplings
            int h = particle.get<1>();
            int i = particle.get<2>();
            int nmo = nmopi_[h];
            if (nmo != 0){
                double** Temp_h = TempMatrix->pointer(h);
                for (int p = 0; p < nmo; ++p){
                    if(p != i){
                        Temp_h[i][p] = Temp_h[p][i] = 0.0;
                    }
                }
            }
        }

        TempMatrix->diagonalize(TempMatrix2,epsilon_a_);
        TempMatrix->copy(Ca_);
        Ca_->gemm(false,false,1.0,TempMatrix,TempMatrix2,0.0);
    }

    // At this point the orbitals are sorted according to the energy but we
    // want to make sure that the hole and the particle MO appear where they
    // should, that is the holes in the virtual space and the particles in
    // the occupied space.
    // |(1) (2) ... (hole) | (particle) ...> will become
    // |(particle) (1) (2) ...  | ... (hole)>
    std::vector<int> naholepi = current_excited_state->aholepi();
    std::vector<int> napartpi = current_excited_state->apartpi();
    TempMatrix->zero();
    TempVector->zero();
    for (int h = 0; h < nirrep_; ++h){
        int m = napartpi[h];  // Offset by the number of holes
        int nso = nsopi_[h];
        int nmo = nmopi_[h];
        double** T_h = TempMatrix->pointer(h);
        double** C_h = Ca_->pointer(h);
        for (int p = 0; p < nmo; ++p){
            // Is this MO a hole or a particle?
            if(std::fabs(epsilon_a_->get(h,p)) > 1.0e-6){
                TempVector->set(h,m,epsilon_a_->get(h,p));
                for (int q = 0; q < nso; ++q){
                    T_h[q][m] = C_h[q][p];
                }
                m += 1;
            }

        }
    }
    if(do_constrained_hole){
        // Place the hole orbital in the last MO of its irrep
        TempMatrix->set_column(hole.get<1>(),nmopi_[hole.get<1>()]-1,current_excited_state->get_hole(0,true));
        TempVector->set(hole.get<1>(),nmopi_[hole.get<1>()]-1,hole.get<0>());
    }
    if(do_constrained_part){
        // Place the particle orbital in the first MO of its irrep
        TempMatrix->set_column(particle.get<1>(),0,current_excited_state->get_particle(0,true));
        TempVector->set(particle.get<1>(),0,particle.get<0>());
    }
    Ca_->copy(TempMatrix);
    epsilon_a_->copy(TempVector.get());

    // BETA
    diagonalize_F(Fb_, Cb_, epsilon_b_);

    //find_occupation();

    if (debug_) {
        Ca_->print(outfile);
        Cb_->print(outfile);
    }
}


//void UCKS::form_D()
//{
//    for (int h = 0; h < nirrep_; ++h) {
//        int nso = nsopi_[h];
//        int nmo = nmopi_[h];
//        int na = nalphapi_[h];
//        int nb = nbetapi_[h];

//        if (nso == 0 || nmo == 0) continue;

//        double* aocc_num_h = aocc_num_->pointer(h);
//        double* bocc_num_h = bocc_num_->pointer(h);
//        double** Ca = Ca_->pointer(h);
//        double** Cb = Cb_->pointer(h);
//        double** Da = Da_->pointer(h);
//        double** Db = Db_->pointer(h);

//        if (na == 0)
//            ::memset(static_cast<void*>(Da[0]), '\0', sizeof(double)*nso*nso);
//        if (nb == 0)
//            ::memset(static_cast<void*>(Db[0]), '\0', sizeof(double)*nso*nso);
//        for (int mu = 0; mu < nso; ++mu){
//            for (int nu = 0; nu < nso; ++nu){
//                for (int p = 0; p < nmo; ++p){
//                    Da[mu][nu] += Ca[mu][p] * Ca[nu][p] * aocc_num_h[p];
//                    Db[mu][nu] += Cb[mu][p] * Cb[nu][p] * bocc_num_h[p];
//                }
//            }
//        }
//    }

//    Dt_->copy(Da_);
//    Dt_->add(Db_);

//    if (debug_) {
//        fprintf(outfile, "in UCKS::form_D:\n");
//        Da_->print();
//        Db_->print();
//    }
//}

//        if(do_penalty){
//            // Find the alpha HOMO of the ground state wave function
//            int homo_h = 0;
//            int homo_p = 0;
//            double homo_e = -1.0e9;
//            for (int h = 0; h < nirrep_; ++h) {
//                int nocc = dets[0]->nalphapi()[h] - 1;
//                if (nocc < 0) continue;
//                if(state_epsilon_a[0]->get(h,nocc) > homo_e){
//                    homo_h = h;
//                    homo_p = nocc;
//                    homo_e = state_epsilon_a[0]->get(h,nocc);
//                }
//            }
//            fprintf(outfile,"  The HOMO orbital has energy %.9f and is %d of irrep %d.\n",homo_e,homo_p,homo_h);
//            Pa = SharedMatrix(factory_->create_matrix("Penalty matrix alpha"));
//            for (int mu = 0; mu < nsopi_[homo_h]; ++mu){
//                for (int nu = 0; nu < nsopi_[homo_h]; ++nu){
//                    double P_mn = 1000000.0 * dets[0]->Ca()->get(homo_h,mu,homo_p) * dets[0]->Ca()->get(homo_h,nu,homo_p);
//                    Pa->set(homo_h,mu,nu,P_mn);
//                }
//            }
//            Pa->transform(S_);
//        }



//    if(gs_scf_ and do_excitation){
//        // Form the projected Fock matrices
//        // Po = DS
//        Temp->gemm(false,false,1.0,state_Da[0],S_,0.0);
//        // SDFDS
//        PoFaPo_->transform(Fa_,Temp);
//        // Temp = 1 - DS
//        Temp2->identity();
//        Temp2->subtract(Temp);
//        PvFaPv_->transform(Fa_,Temp2);
//    }



////// EXCITED STATES
//        Temp->copy(Fa_);
//        Temp->transform(Ca_);

//        Temp2->copy(Fb_);
//        Temp2->transform(Ca_);

//        Ub->copy(Temp);
//        Ub->add(Temp2);
//        Ub->scale(0.5);

//        for (int h = 0; h < nirrep_; ++h) {
//          // CO
//          for (int i = 0; i < doccpi_[h]; ++i) {
//            for (int j = doccpi_[h]; j < doccpi_[h] + soccpi_[h]; ++j) {
//                Ub->set(h,i,j,Temp2->get(h,i,j));
//                Ub->set(h,j,i,Temp2->get(h,j,i));
//            }
//          }
//          for (int i = doccpi_[h]; i < doccpi_[h] + soccpi_[h]; ++i) {
//            for (int j = doccpi_[h] + soccpi_[h]; j < nmopi_[h]; ++j) {
//                Ub->set(h,i,j,Temp->get(h,i,j));
//                Ub->set(h,j,i,Temp->get(h,j,i));
//            }
//          }
//        }
//        Ub->diagonalize(Ua,epsilon_a_);
//        Temp->copy(Ca_);
//        Ca_->gemm(false,false,1.0,Temp,Ua,0.0);
//        Cb_->copy(Ca_);
//        epsilon_b_->copy(epsilon_a_.get());

//        Temp->diagonalize(Ua,epsilon_a_);
//        Temp2->diagonalize(Ub,epsilon_b_);
//        Temp->copy(Ca_);
//        Temp2->copy(Cb_);
//        Ca_->gemm(false,false,1.0,Temp,Ua,0.0);
//        Cb_->gemm(false,false,1.0,Temp2,Ub,0.0);


//        diagonalize_F(Fa_, Ca_, epsilon_a_);
//        Temp->copy(Fb_);
//        Temp->transform(Ca_);
//        for (int h = 0; h < nirrep_; ++h){
//            int nocc = doccpi_[h];
//            int nvir = nmopi_[h] - nocc;
//            if (nocc != 0 and nvir!= 0){
//                double** Temp_h = Temp->pointer(h);
//                for (int i = 0; i < nocc; ++i){
////                    for (int j = 0; j < nvir; ++j){
////                        Temp_h[i][j + nocc] = Temp_h[j + nocc][i] = 0.0;
//                    for (int j = doccpi_[h] + soccpi_[h]; j < nmopi_[h]; ++j){
//                        Temp_h[i][j] = Temp_h[j][i] = 0.0;
//                    }
//                }
//            }
//        }
//        Temp->diagonalize(Temp2,epsilon_b_);
//        Temp->copy(Ca_);
//        Cb_->gemm(false,false,1.0,Temp,Temp2,0.0);

//        for (int h = 0; h < nirrep_; ++h){
//            int nso = nsopi_[h];
//            if (nso != 0){
//                double** Temp_h = Temp->pointer(h);
//                for (int p = 0; p < nso; ++p){
//                    epsilon_b_->set(h,p,Temp_h[p][p]);
//                }
//            }
//        }


//        // Transform Fa to the ground state MO basis
//        Temp->transform(Fb_,state_Cb[0]);
//        // Grab the occ and vir blocks
//        for (int h = 0; h < nirrep_; ++h){
//            int nocc = state_nbetapi[0][h];
//            int nvir = nmopi_[h] - nocc;
//            if (nocc != 0){
//                double** Temp_h = Temp->pointer(h);
//                double** PoFbPo_h = PoFbPo_->pointer(h);
//                for (int i = 0; i < nocc; ++i){
//                    for (int j = 0; j < nocc; ++j){
//                        PoFbPo_h[i][j] = Temp_h[i][j];
//                    }
//                }
//            }
//            if (nvir != 0){
//                double** Temp_h = Temp->pointer(h);
//                double** PvFbPv_h = PvFbPv_->pointer(h);
//                for (int i = 0; i < nvir; ++i){
//                    for (int j = 0; j < nvir; ++j){
//                        PvFbPv_h[i][j] = Temp_h[i + nocc][j + nocc];
//                    }
//                }
//            }
//        }
//        PoFbPo_->diagonalize(Uob,lambda_ob);
//        PvFbPv_->diagonalize(Uvb,lambda_vb);

//        Temp->zero();
//        for (int h = 0; h < nirrep_; ++h){
//            int nocc = state_nbetapi[0][h];
//            int nvir = nmopi_[h] - nocc;
//            if (nocc != 0){
//                double** Temp_h = Temp->pointer(h);
//                double** Uob_h = Uob->pointer(h);
//                for (int i = 0; i < nocc; ++i){
//                    epsilon_b_->set(h,i,lambda_ob->get(h,i));
//                    for (int j = 0; j < nocc; ++j){
//                        Temp_h[i][j] = Uob_h[i][j];
//                    }
//                }
//            }
//            if (nvir != 0){
//                double** Temp_h = Temp->pointer(h);
//                double** Uvb_h = Uvb->pointer(h);
//                for (int i = 0; i < nvir; ++i){
//                    epsilon_b_->set(h,i + nocc,lambda_vb->get(h,i));
//                    for (int j = 0; j < nvir; ++j){
//                        Temp_h[i + nocc][j + nocc] = Uvb_h[i][j];
//                    }
//                }
//            }
//        }
//        // Get the new orbitals
//        Cb_->gemm(false,false,1.0,state_Cb[0],Temp,0.0);
//        Temp->transform(Fb_,Cb_);
//        Temp->print();



//        // Grab the occ and vir blocks
//        for (int h = 0; h < nirrep_; ++h){
//            int nocc = new_occupation[h];
//            int nvir = nmopi_[h] - nocc;
//            if (nocc != 0){
//                double** Temp_h = Temp->pointer(h);
//                double** Fcan_h = Fcan->pointer(h);
//                for (int i = 0; i < nocc; ++i){
//                    for (int j = 0; j < nocc; ++j){
//                        Fcan_h[i][j] = Temp_h[i][j];
//                    }
//                }
//            }
//        }
//        Fcan->diagonalize(Ucan,lambda_can);
//        Temp->zero();
//        for (int h = 0; h < nirrep_; ++h){
//            int nocc = new_occupation[h];
//            int nvir = nmopi_[h] - nocc;
//            if (nocc != 0){
//                double** Temp_h = Temp->pointer(h);
//                double** Ucan_h = Ucan->pointer(h);
//                for (int i = 0; i < nocc; ++i){
//                    epsilon_a_->set(h,i,lambda_can->get(h,i));
//                    for (int j = 0; j < nocc; ++j){
//                        Temp_h[i][j] = Ucan_h[i][j];
//                    }
//                }
//            }
//            if (nvir != 0){
//                double** Temp_h = Temp->pointer(h);
//                for (int i = 0; i < nvir; ++i){
//                    Temp_h[i + nocc][i + nocc] = 1.0;
//                }
//            }
//        }

//        Temp->transform(Fa_,Ca_);
//        // Get the new orbitals
//        Temp2->copy(Ca_);
//        Ca_->gemm(false,false,1.0,Temp2,Temp,0.0);

//        Dimension new
//        SharedVector epsilon_ao_ = SharedVector(factory_->create_vector());
//        SharedVector epsilon_av_ = SharedVector(factory_->create_vector());
//        diagonalize_F(PoFaPo_, Temp,  epsilon_ao_);
//        diagonalize_F(PvFaPv_, Temp2, epsilon_av_);
//        int homo_h = 0;
//        int homo_p = 0;
//        double homo_energy = -1.0e10;
//        int lumo_h = 0;
//        int lumo_p = 0;
//        double lumo_energy = +1.0e10;
//        for (int h = 0; h < nirrep_; ++h){
//            int nmo  = nmopi_[h];
//            int nso  = nsopi_[h];
//            if (nmo == 0 or nso == 0) continue;
//            double** Ca_h  = Ca_->pointer(h);
//            double** Cao_h = Temp->pointer(h);
//            double** Cav_h = Temp2->pointer(h);
//            int no = 0;
//            for (int p = 0; p < nmo; ++p){
//                if(std::fabs(epsilon_ao_->get(h,p)) > 1.0e-6 ){
//                    for (int mu = 0; mu < nmo; ++mu){
//                        Ca_h[mu][no] = Cao_h[mu][p];
//                    }
//                    epsilon_a_->set(h,no,epsilon_ao_->get(h,p));
//                    if (epsilon_ao_->get(h,p) > homo_energy){
//                        homo_energy = epsilon_ao_->get(h,p);
//                        homo_h = h;
//                        homo_p = no;
//                    }
//                    no++;
//                }
//            }
//            for (int p = 0; p < nmo; ++p){
//                if(std::fabs(epsilon_av_->get(h,p)) > 1.0e-6 ){
//                    for (int mu = 0; mu < nmo; ++mu){
//                        Ca_h[mu][no] = Cav_h[mu][p];
//                    }
//                    epsilon_a_->set(h,no,epsilon_av_->get(h,p));
//                    if (epsilon_av_->get(h,p) < lumo_energy){
//                        lumo_energy = epsilon_av_->get(h,p);
//                        lumo_h = h;
//                        lumo_p = no;
//                    }
//                    no++;
//                }
//            }
//        }
//        // Shift the HOMO orbital in the occupied space


//int old_socc[8];
//int old_docc[8];
//for(int h = 0; h < nirrep_; ++h){
//    old_socc[h] = soccpi_[h];
//    old_docc[h] = doccpi_[h];
//}

//for (int h = 0; h < nirrep_; ++h) {
//    soccpi_[h] = std::abs(nalphapi_[h] - nbetapi_[h]);
//    doccpi_[h] = std::min(nalphapi_[h] , nbetapi_[h]);
//}

//bool occ_changed = false;
//for(int h = 0; h < nirrep_; ++h){
//    if( old_socc[h] != soccpi_[h] || old_docc[h] != doccpi_[h]){
//        occ_changed = true;
//        break;
//    }
//}

//// If print > 2 (diagnostics), print always
//if((print_ > 2 || (print_ && occ_changed)) && iteration_ > 0){
//    if (Communicator::world->me() == 0)
//        fprintf(outfile, "\tOccupation by irrep:\n");
//    print_occupation();
//}

//fprintf(outfile, "\tNA   [ ");
//for(int h = 0; h < nirrep_-1; ++h) fprintf(outfile, " %4d,", nalphapi_[h]);
//fprintf(outfile, " %4d ]\n", nalphapi_[nirrep_-1]);
//fprintf(outfile, "\tNB   [ ");
//for(int h = 0; h < nirrep_-1; ++h) fprintf(outfile, " %4d,", nbetapi_[h]);
//fprintf(outfile, " %4d ]\n", nbetapi_[nirrep_-1]);

//// Compute the density matrices with the new occupation
//form_D();

//// Compute the triplet energy from the density matrices
//double triplet_energy = compute_E();

//        if(nexclude_occ == 0 and nexclude_vir){
//            // Find the lowest single excitations
//            std::vector<boost::tuple<double,int,int,int,int> > sorted_exc;
//            // Loop over occupied MOs
//            for (int hi = 0; hi < nirrep_; ++hi){
//                int nocci = ref_scf_->nalphapi_[0][hi];
//                for (int i = 0; i < nocc; ++i){
//                    for (int ha = 0; ha < nirrep_; ++ha){
//                        int nocca = ref_scf_->nalphapi_[0][ha];
//                        for (int a = nocca; a < nmopi_[ha]; ++a){
//                            sorted_exc.push_back(boost::make_tuple(
//                        }
//                    }
//                    int nocc = dets[0]->nalphapi()[h];

//                int nvir = nmopi_[h] - nocc;
//                for (int i = 0; i < nocc; ++i){
//                    sorted_occ.push_back(boost::make_tuple(lambda_o->get(h,i),h,i));
//                }
//                for (int i = 0; i < nvir; ++i){
//                    sorted_vir.push_back(boost::make_tuple(lambda_v->get(h,i),h,i));
//                }
//            }
//            std::sort(sorted_occ.begin(),sorted_occ.end());
//            std::sort(sorted_vir.begin(),sorted_vir.end());
//        }

//    if(do_roks){
//        moFa_->transform(Fa_, Ca_);
//        moFb_->transform(Fb_, Ca_);
//        /*
//        * Fo = open-shell fock matrix = 0.5 Fa
//        * Fc = closed-shell fock matrix = 0.5 (Fa + Fb)
//        *
//        * Therefore
//        *
//        * 2(Fc-Fo) = Fb
//        * 2Fo = Fa
//        *
//        * Form the effective Fock matrix, too
//        * The effective Fock matrix has the following structure
//        *          |  closed     open    virtual
//        *  ----------------------------------------
//        *  closed  |    Fc     2(Fc-Fo)    Fc
//        *  open    | 2(Fc-Fo)     Fc      2Fo
//        *  virtual |    Fc       2Fo       Fc
//        */
//        Feff_->copy(moFa_);
//        Feff_->add(moFb_);
//        Feff_->scale(0.5);
//        for (int h = 0; h < nirrep_; ++h) {
//            for (int i = doccpi_[h]; i < doccpi_[h] + soccpi_[h]; ++i) {
//                // Set the open/closed portion
//                for (int j = 0; j < doccpi_[h]; ++j) {
//                    double val = moFb_->get(h, i, j);
//                    Feff_->set(h, i, j, val);
//                    Feff_->set(h, j, i, val);
//                }
//                // Set the open/virtual portion
//                for (int j = doccpi_[h] + soccpi_[h]; j < nmopi_[h]; ++j) {
//                    double val = moFa_->get(h, i, j);
//                    Feff_->set(h, i, j, val);
//                    Feff_->set(h, j, i, val);
//                }
//            }
//        }

//        // Form the orthogonalized SO basis Feff matrix, for use in DIIS
//        soFeff_->copy(Feff_);
//        soFeff_->back_transform(Ct_);
//    }


//    // SUHF Contributions
//    double suhf_weight = std::pow(0.5,std::max(10.0 - iteration_,1.0));
//    TempMatrix->transform(Db_,S_);
//    TempMatrix->scale(- 4.0 * suhf_weight * KS::options_.get_double("CDFT_SUHF_LAMBDA"));
//    Fa_->add(TempMatrix);
//    TempMatrix->transform(Da_,S_);
//    TempMatrix->scale(- 4.0 * suhf_weight * KS::options_.get_double("CDFT_SUHF_LAMBDA"));
//    Fb_->add(TempMatrix);


#endif // JUNK_H
