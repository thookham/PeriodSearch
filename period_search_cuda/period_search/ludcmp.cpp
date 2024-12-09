/* Numerical Recipes */

//#include "stdafx.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "arrayHelpers.hpp"

constexpr auto tiny = 1.0e-20;

//void ludcmp(double** ef.fitmat, int ncoef, int indx[], double d[])  // ef.fitmat = ef.fitmat, indx = ef.indx, d = ef.d
//void ludcmp(std::vector<std::unique_ptr<double[]>>& ef.fitmat, int ncoef, int* indx, std::unique_ptr<double[]>& d)
void ludcmp(struct ellfits& ef, int ncoef)
{
	int i, imax = -999, j, k;
	double big, dum, sum, temp;
	//auto v = vector_double(ncoef);
	//auto v = new_vector_double(ncoef);

	//*d = 1.0;
	ef.d[0] = 1.0;
	for (i = 1; i <= ncoef; i++)
	{
		big = 0.0;
		for (j = 1; j <= ncoef; j++)
		{
			if ((temp = fabs(ef.fitmat[i][j])) > big)
			{
				big = temp;
			}
		}
		if (big == 0.0)
		{
			fprintf(stderr, "Singular matrix in routine ludcmp\ncoef");
			fflush(stderr);
			exit(4);
		}
	    ef.v[i] = 1.0 / big;
	}

	for (j = 1; j <= ncoef; j++)
	{
		for (i = 1; i < j; i++)
		{
			sum = ef.fitmat[i][j];
			for (k = 1; k < i; k++)
			{
				sum -= ef.fitmat[i][k] * ef.fitmat[k][j];
			}

			ef.fitmat[i][j] = sum;
		}

		big = 0.0;
		for (i = j; i <= ncoef; i++)
		{
			sum = ef.fitmat[i][j];
			for (k = 1; k < j; k++)
			{
				sum -= ef.fitmat[i][k] * ef.fitmat[k][j];
			}

			ef.fitmat[i][j] = sum;
			if ((dum = ef.v[i] * fabs(sum)) >= big)
			{
				big = dum;
				imax = i;
			}
		}

		if (j != imax)
		{
			for (k = 1; k <= ncoef; k++)
			{
				dum = ef.fitmat[imax][k];
				ef.fitmat[imax][k] = ef.fitmat[j][k];
				ef.fitmat[j][k] = dum;
			}

			//*d = -(*d);
			ef.d[0] = -(ef.d[0]);
			ef.v[imax] = ef.v[j];
		}

		ef.indx[j] = imax;
		if (ef.fitmat[j][j] == 0.0)
		{
			ef.fitmat[j][j] = tiny;
		}

		if (j != ncoef)
		{
			dum = 1.0 / (ef.fitmat[j][j]);
			for (i = j + 1; i <= ncoef; i++)
			{
				ef.fitmat[i][j] *= dum;
			}
		}
	}

	//new_deallocate_vector(v);
}