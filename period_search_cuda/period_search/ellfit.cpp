/* Find the curv. fn. Laplace series for given ellipsoid
   converted from Mikko's fortran code

   8.11.2006
*/

#include <algorithm>

#include "stdafx.h"
#include <cmath>
#include <ostream>

#include "globals.h"
#include "declarations.h"

#include <iostream>
#include <cmath>
#include <cassert>
#include <cstdlib>

#include "arrayHelpers.hpp"

// v4

//void ellfit(double cg[], double a, double b, double c, int ndir, int ncoef, double at[], double af[]) {
//void ellfit(struct ellfits& ef, std::unique_ptr<double[]>& cg, double a, double b, double c, int ndir, int ncoef, double* at, double* af)
//void ellfit(struct ellfits& ef, double* cg, double a, double b, double c, int ndir, int ncoef, double* at, double* af)
void ellfit(struct ellfits& ef, std::vector<double>& cg, double a, double b, double c, int ndir, int ncoef, std::vector<double>& at, std::vector<double>& af)
{
    int i, m, l, n, j, k;
    double sum, st;

    //int* indx = new_vector_int(ncoef);
    //auto fitvec = create_vector_double(ncoef);
    //auto er     = create_vector_double(ndir);
    //auto d      = create_vector_double(1);
    //auto fmat   = create_matrix_double(ndir, ncoef);
    //auto fitmat = create_matrix_double(ncoef, ncoef);

    //assert(indx != nullptr && ef.fitvec != nullptr && ef.er != nullptr && ef.d != nullptr); // && fmat != nullptr && fitmat != nullptr);
    //assert(ef.indx != nullptr && ef.fitvec != nullptr && ef.er != nullptr && ef.d != nullptr && ef.fmat != nullptr && ef.fitmat != nullptr);
    // Check if all elements in fmat and fitmat are not null
    auto are_all_elements_non_null = [](const std::vector<std::unique_ptr<double[]>>& vec)
    {
        return std::all_of(vec.begin(), vec.end(), [](const std::unique_ptr<double[]>& ptr)
        {
            return ptr != nullptr;
        });
    };
    //assert(are_all_elements_non_null(ef.fmat) && are_all_elements_non_null(ef.fitmat)); // Use this with smart pointers

    std::cout << "Memory allocated successfully" << std::endl;

    // Compute the LOG curv. func. of the ellipsoid
    for (i = 1; i <= ndir; i++) {
        assert(i <= ndir); // Bounds check
        st = sin(at[i]);
        sum = pow(a * st * cos(af[i]), 2) + pow(b * st * sin(af[i]), 2) + pow(c * cos(at[i]), 2);
        ef.er[i] = 2 * (log(a * b * c) - log(sum));
        //std::cout << "er[" << i << "] = " << er[i] << std::endl; // Logging
    }

    // Compute the spherical harmonic values and construct fmat
    for (i = 1; i <= ndir; i++) {
        assert(i <= ndir); // Bounds check
        n = 0;
        for (m = 0; m <= m_max; m++) {
            for (l = m; l <= l_max; l++) {
                n++;
                if (m != 0) {
                    assert(n <= ncoef); // Bounds check
                    ef.fmat[i][n] = pleg[i][l][m] * cos(m * af[i]);
                    //std::cout << "fmat[" << i << "][" << n << "] = " << fmat[i][n] << std::endl; // Logging
                    n++;
                    assert(n <= ncoef); // Bounds check
                    ef.fmat[i][n] = pleg[i][l][m] * sin(m * af[i]);
                    //std::cout << "fmat[" << i << "][" << n << "] = " << fmat[i][n] << std::endl; // Logging
                }
                else {
                    assert(n <= ncoef); // Bounds check
                    ef.fmat[i][n] = pleg[i][l][m];
                    //std::cout << "fmat[" << i << "][" << n << "] = " << fmat[i][n] << std::endl; // Logging
                }
            }
        }
    }

    // Fit the coefficients r
    for (i = 1; i <= ncoef; i++) {
        assert(i <= ncoef); // Bounds check
        for (j = 1; j <= ncoef; j++) {
            assert(j <= ncoef); // Bounds check
            ef.fitmat[i][j] = 0;
            for (k = 1; k <= ndir; k++) {
                assert(k <= ndir); // Bounds check
                ef.fitmat[i][j] += ef.fmat[k][i] * ef.fmat[k][j];
                //std::cout << "fitmat[" << i << "][" << j << "] += fmat[" << k << "][" << i << "] * fmat[" << k << "][" << j << "]" << std::endl; // Logging
            }
        }

        ef.fitvec[i] = 0;
        for (j = 1; j <= ndir; j++) {
            assert(j <= ndir); // Bounds check
            ef.fitvec[i] += ef.fmat[j][i] * ef.er[j];
            //std::cout << "fitvec[" << i << "] += fmat[" << j << "][" << i << "] * er[" << j << "]" << std::endl; // Logging
        }
    }

    //ludcmp(ef.fitmat, ncoef, ef.indx, ef.d);
    ludcmp(ef, ncoef);
    //lubksb(ef.fitmat, ncoef, ef.indx, ef.fitvec);
    lubksb(ef, ncoef);

    for (i = 1; i <= ncoef; i++) {
        assert(i <= ncoef); // Bounds check
        cg[i] = ef.fitvec[i];
        //std::cout << "cg[" << i << "] = " << cg[i] << std::endl; // Logging
    }

    // Deallocate memory
    //new_deallocate_vector(fitvec);
    //new_deallocate_vector(d);
    //new_deallocate_vector(indx);
    //new_deallocate_vector(er);
    //new_deallocate_matrix_double(fitmat, ncoef);
    //new_deallocate_matrix_double(fmat, ndir);
    std::cout << "Memory deallocated successfully" << std::endl; // Debug statement
}



//int main() {
//    int ndir = 5;
//    int ncoef = 3;
//    double cg[3], at[6] = { 0, 1, 2, 3, 4, 5 }, af[6] = { 0, 1, 2, 3, 4, 5 };
//
//    ellfit(cg, 1.0, 2.0, 3.0, ndir, ncoef, at, af);
//
//    return 0;
//}

//v3
//void ellfit(double cg[], double a, double b, double c, int ndir, int ncoef, double at[], double af[]) {
//    int i, m, l, n, j, k;
//    int* indx = vector_int(ncoef);
//    double sum, st;
//    double* fitvec = vector_double(ncoef);
//    double* er = vector_double(ndir);
//    double* d = vector_double(1);
//    double** fmat = matrix_double(ndir, ncoef);
//    double** fitmat = matrix_double(ncoef, ncoef);
//
//    assert(indx != nullptr && fitvec != nullptr && er != nullptr && d != nullptr && fmat != nullptr && fitmat != nullptr);
//    std::cout << "Memory allocated successfully" << std::endl;
//
//    // Compute the LOG curv. func. of the ellipsoid
//    for (i = 1; i <= ndir; i++) {
//        assert(i <= ndir); // Bounds check
//        st = sin(at[i]);
//        sum = pow(a * st * cos(af[i]), 2) + pow(b * st * sin(af[i]), 2) + pow(c * cos(at[i]), 2);
//        er[i] = 2 * (log(a * b * c) - log(sum));
//        std::cout << "er[" << i << "] = " << er[i] << std::endl; // Logging
//    }
//
//    // Compute the spherical harmonic values and construct fmat
//    for (i = 1; i <= ndir; i++) {
//        assert(i <= ndir); // Bounds check
//        n = 0;
//        for (m = 0; m <= m_max; m++) {
//            for (l = m; l <= l_max; l++) {
//                n++;
//                if (m != 0) {
//                    assert(n <= ncoef); // Bounds check
//                    fmat[i][n] = pleg[i][l][m] * cos(m * af[i]);
//                    std::cout << "fmat[" << i << "][" << n << "] = " << fmat[i][n] << std::endl; // Logging
//                    n++;
//                    assert(n <= ncoef); // Bounds check
//                    fmat[i][n] = pleg[i][l][m] * sin(m * af[i]);
//                    std::cout << "fmat[" << i << "][" << n << "] = " << fmat[i][n] << std::endl; // Logging
//                }
//                else {
//                    assert(n <= ncoef); // Bounds check
//                    fmat[i][n] = pleg[i][l][m];
//                    std::cout << "fmat[" << i << "][" << n << "] = " << fmat[i][n] << std::endl; // Logging
//                }
//            }
//        }
//    }
//
//    // Fit the coefficients r
//    for (i = 1; i <= ncoef; i++) {
//        assert(i <= ncoef); // Bounds check
//        for (j = 1; j <= ncoef; j++) {
//            assert(j <= ncoef); // Bounds check
//            fitmat[i][j] = 0;
//            for (k = 1; k <= ndir; k++) {
//                assert(k <= ndir); // Bounds check
//                fitmat[i][j] += fmat[k][i] * fmat[k][j];
//                std::cout << "fitmat[" << i << "][" << j << "] += fmat[" << k << "][" << i << "] * fmat[" << k << "][" << j << "]" << std::endl; // Logging
//            }
//        }
//        fitvec[i] = 0;
//        for (j = 1; j <= ndir; j++) {
//            assert(j <= ndir); // Bounds check
//            fitvec[i] += fmat[j][i] * er[j];
//            std::cout << "fitvec[" << i << "] += fmat[" << j << "][" << i << "] * er[" << j << "]" << std::endl; // Logging
//        }
//    }
//
//    ludcmp(fitmat, ncoef, indx, d);
//    lubksb(fitmat, ncoef, indx, fitvec);
//
//    for (i = 1; i <= ncoef; i++) {
//        assert(i <= ncoef); // Bounds check
//        cg[i] = fitvec[i];
//        std::cout << "cg[" << i << "] = " << cg[i] << std::endl; // Logging
//    }
//
//    // Deallocate memory
//    deallocate_vector((void*)fitvec);
//    deallocate_vector((void*)d);
//    deallocate_vector((void*)indx);
//    deallocate_vector((void*)er);
//
//    deallocate_matrix_double(fmat, ndir);
//    deallocate_matrix_double(fitmat, ncoef);
//
//    std::cout << "Memory deallocated successfully" << std::endl; // Debug statement
//}



//void ellfit(double cg[], double a, double b, double c, int ndir, int ncoef, double at[], double af[])
//{
//   int i, m, l, n, j, k;
//   int *indx;
//
//   double sum, st;
//   double *fitvec, *d, *er,
//          **fmat, **fitmat;
//
//   indx = vector_int(ncoef);
//
//   fitvec = vector_double(ncoef);
//   er = vector_double(ndir);
//   d = vector_double(1);
//
//   fmat = matrix_double(ndir, ncoef);
//   fitmat = matrix_double(ncoef, ncoef);
//
//   /* Compute the LOGcurv.func. of the ellipsoid */
//   for (i = 1; i <= ndir; i++)
//   {
//      st = sin(at[i]);
//      sum = pow(a * st * cos(af[i]), 2) + pow(b * st * sin(af[i]), 2) + pow(c * cos(at[i]), 2);
//      er[i] = 2 * (log(a * b * c) - log(sum));
//   }
//   /* Compute the sph. harm. values at each direction and
//      construct the matrix fmat from them */
//   for (i = 1; i <= ndir; i++)
//   {
//      n = 0;
//      for (m = 0; m <= m_max; m++)
//         for (l = m; l <= l_max; l++)
//	 {
//            n++;
//            if (m != 0)
//	    {
//               fmat[i][n] = pleg[i][l][m] * cos(m * af[i]);
//               n++;
//               fmat[i][n] = pleg[i][l][m] * sin(m * af[i]);
//            }
//            else
//               fmat[i][n] = pleg[i][l][m];
//         }
//   }
//
//   /* Fit the coefficients r from fmat[ndir,ncoef]*r[ncoef]=er[ndir] */
//   for (i = 1; i <= ncoef; i++)
//   {
//      for (j = 1; j <= ncoef; j++)
//      {
//         fitmat[i][j]=0;
//
//         for (k = 1; k <= ndir; k++)
//            fitmat[i][j] = fitmat[i][j] + fmat[k][i] * fmat[k][j];
//
//      }
//      fitvec[i]=0;
//
//      for (j = 1; j <= ndir; j++)
//         fitvec[i] = fitvec[i] + fmat[j][i] * er[j];
//   }
//
//
//   ludcmp(fitmat,ncoef,indx,d);
//   lubksb(fitmat,ncoef,indx,fitvec);
//
//   for (i = 1; i <= ncoef; i++)
//      cg[i] = fitvec[i];
//
//   deallocate_matrix_double(fitmat, ncoef);
//   deallocate_matrix_double(fmat, ndir);
//   deallocate_vector((void *) fitvec);
//   deallocate_vector((void *) d);
//   deallocate_vector((void *) indx);
//   deallocate_vector((void *) er);
//
//}

