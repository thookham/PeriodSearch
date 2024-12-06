/* Numerical Recipes */

#include <memory>
#include <vector>

#include "stdafx.h"

//void lubksb(double **a, int n, int indx[], double b[])
void lubksb(std::vector<std::unique_ptr<double[]>>& a, int n, int* indx, std::unique_ptr<double[]>& b)
{
   int i, ii=0, ip, j;
   double sum;

   for (i = 1; i <= n; i++)
   {
      ip = indx[i];
      sum = b[ip];
      b[ip] = b[i];
      if (ii)
         for (j = ii; j <= i-1; j++) sum -= a[i][j] * b[j];
      else if (sum) ii = i;
      b[i] = sum;
   }
   for (i = n; i >= 1; i--)
   {
      sum = b[i];
      for (j = i + 1;j <= n; j ++) sum -= a[i][j] * b[j];
      b[i] = sum / a[i][i];
   }
}
