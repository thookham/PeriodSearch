/* Numerical Recipes */

//#include "stdafx.h"
//#include <memory>
//#include <vector>
#include "arrayHelpers.hpp"


//void lubksb(double **ef.fitmat, int ncoef, int indx[], double ef.fitvec[])  // ef.fitmat = ef.fitmat, indx = ef.indx, ef.fitvec = ef.fitvec
//void lubksb(std::vector<std::unique_ptr<double[]>>& ef.fitmat, int ncoef, int* indx, std::unique_ptr<double[]>& ef.fitvec)
void lubksb(struct ellfits& ef, int ncoef)
{
   int i, ii=0, ip, j;
   double sum;

   for (i = 1; i <= ncoef; i++)
   {
      ip = ef.indx[i];
      sum = ef.fitvec[ip];
      ef.fitvec[ip] = ef.fitvec[i];
      if (ii)
         for (j = ii; j <= i-1; j++) sum -= ef.fitmat[i][j] * ef.fitvec[j];
      else if (sum) ii = i;
      ef.fitvec[i] = sum;
   }
   for (i = ncoef; i >= 1; i--)
   {
      sum = ef.fitvec[i];
      for (j = i + 1;j <= ncoef; j ++) sum -= ef.fitmat[i][j] * ef.fitvec[j];
      ef.fitvec[i] = sum / ef.fitmat[i][i];
   }
}
