/* Numerical Recipes */
#include <vector>

void lubksb(const std::vector<std::vector<double>>& a, const int n, const std::vector<int>& indx, std::vector<double>& b)
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
