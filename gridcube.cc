#include "gridcube.h"

GridCube::GridCube()
{
}



//GridCube::write_file(std::string file)
//{
////    int i,j,k;
////    int atom, mo;
////    int n_per_line;
////    double step_x, step_y, step_z, x, y, z;

////    print_3d_summary();

//    grid_file = fopen(file.c_str(),"w+");

//    /*------------------------------------------------------------------------
//     * Gaussian Cube standard overhead as described in G94 programmer's manual
//     *------------------------------------------------------------------------*/
//    bool dump_mo = true;
//    fprintf(grid_file,"Gaussian Cube file created by GridCube (Psi 4)\n");
//    fprintf(grid_file,"Calculation title\n");

//    if (dump_mo){
//        fprintf(grid_file,"%5d",-1*natom);
//    }else if (grid == 6){
//        fprintf(grid_file,"%5d",natom);
//    }
//    fprintf(grid_file,"%12.6lf%12.6lf%12.6lf\n", grid_origin[0], grid_origin[1], grid_origin[2]);
//    fprintf(grid_file,"%5d%12.6lf%12.6lf%12.6lf\n", nix+1, grid_step_x[0], grid_step_x[1], grid_step_x[2]);
//    fprintf(grid_file,"%5d%12.6lf%12.6lf%12.6lf\n", niy+1, grid_step_y[0], grid_step_y[1], grid_step_y[2]);
//    fprintf(grid_file,"%5d%12.6lf%12.6lf%12.6lf\n", niz+1, grid_step_z[0], grid_step_z[1], grid_step_z[2]);
//    for(int atom = 0; atom < natom; ++atom) {
//        fprintf(grid_file,"%5d%12.6lf%12.6lf%12.6lf%12.6lf\n", (int)zvals[atom], zvals[atom],
//                geom[atom][0], geom[atom][1], geom[atom][2]);
//    }
//    if (dump_mo) {
//        fprintf(grid_file,"%5d",num_mos_to_plot);
//        for(int mo = 0; mo < num_mos_to_plot; ++mo)
//            fprintf(grid_file,"%5d",mos_to_plot[mo]);
//        fprintf(grid_file,"\n");
//    }

//    if (dump_mo) {
//        for(int i = 0; i <= nix; ++i){
//            for(int j = 0; j <= niy; ++j){
//                int n_per_line = 0;
//                for(int k = 0; k <= niz; ++k){
//                    for(int mo = 0; mo < num_mos_to_plot; ++mo) {
//                        fprintf(grid_file,"%13.5E",grid3d_pts[mo][i][j][k]);
//                        n_per_line++;
//                        if (n_per_line == 6) {
//                            fprintf(grid_file,"\n");
//                            n_per_line = 0;
//                        }
//                    }
//                }
//                if (n_per_line != 0){
//                    fprintf(grid_file,"\n");
//                }
//            }
//        }
//    }else{
//        for(int i = 0; i <= nix; ++i){
//            for(int j = 0; j <= niy; ++j){
//                n_per_line = 0;
//                for(int k = 0; k <= niz; ++k){
//                    fprintf(grid_file,"%13.5E",grid3d_pts[0][i][j][k]);
//                    n_per_line++;
//                    if (n_per_line == 6) {
//                        fprintf(grid_file,"\n");
//                        n_per_line = 0;
//                    }
//                }
//                if (n_per_line != 0){
//                    fprintf(grid_file,"\n");
//                }
//            }
//        }
//    }
//    fclose(grid_file);
//}
