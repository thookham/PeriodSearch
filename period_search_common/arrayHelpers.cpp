#include "pch.h"
#include <cstdio>
#include <algorithm>
#include <memory>
#include <vector>
#include "arrayHelpers.hpp"

void init3Darray(std::vector<double>& vector, int dim1, int dim2, int dim3)
{
    vector.resize(static_cast<unsigned long long>(dim1) * dim2 * dim3);
    std::fill(vector.begin(), vector.end(), 0.0);
}

void init3Darray(std::unique_ptr<std::unique_ptr<std::unique_ptr<double[]>[]>[]>& de, int dim1, int dim2, int dim3)
{
    de = std::make_unique<std::unique_ptr<std::unique_ptr<double[]>[]>[]>(dim1);
    for (int i = 0; i < dim1; ++i)
    {
        de[i] = std::make_unique<std::unique_ptr<double[]>[]>(dim2); // 4
        for (int j = 0; j < dim2; ++j)
        {
            de[i][j] = std::make_unique<double[]>(dim3); // 4
            std::fill_n(de[i][j].get(), dim3, 0.0); // Initialize each element to 0.0
        }
    }
}


void init2Darray(std::vector<std::unique_ptr<double[]>>& matrix, const int xSize, const int ySize)
{
    matrix.resize(xSize);
    for (int i = 0; i < xSize; ++i)
    {
        matrix[i] = std::make_unique<double[]>(ySize);
        std::fill_n(matrix[i].get(), ySize, 0.0); // Initialize each element to 0.0
    }
}

void init2Darray(double **&matrix, int dytemp_siszeX, int dytemp_sizeY)
{
	matrix = new double* [dytemp_siszeX];
	for (int i = 0; i < dytemp_siszeX; ++i)
	{
		matrix[i] = new double[dytemp_sizeY];
		std::fill_n(matrix[i], dytemp_sizeY, 0.0); // Initialize each element to 0.0
	}
}

void delete2Darray(double **&ary, int sizeX)
{
	for (int i = 0; i < sizeX; ++i) {
		delete[] ary[i];
	}

	delete[] ary;
}

void printArray(int array[], int iMax, char msg[])
{
    printf("\n%s[%d]:\n", msg, iMax);
    for (int i = 0; i <= iMax; i++)
    {
        printf("%d, ", array[i]);
        if (i % 20 == 0)
            printf("\n");
    }
}

void printArray(double array[], int iMax, char msg[])
{
    printf("\n%s[%d]:\n", msg, iMax);
	printf("[0] ");
    for (int i = 0; i <= iMax; i++)
    {
        printf("%.6f, ", array[i]);
        if (i > 0 && i < iMax && i % 9 == 0)
		{
            printf("\n");
			printf("[%d] ", i + 1);
		}
    }
}

void printArray(double **array, int iMax, int jMax, char msg[])
{
    printf("\n%s[%d][%d]:\n", msg, iMax, jMax);
    for (int j = 0; j <= jMax; j++)
    {
        printf("\n_%s_%d[] = { ", msg, j);
        for (int i = 0; i <= iMax; i++)
        {
            printf("% 0.6f, ", array[i][j]);
            if (i % 9 == 0)
                printf("\n");
        }
        printf("};\n");
    }
}

void printArray(double ***array, int iMax, int jMax, int kMax, char msg[])
{
    printf("\n%s[%d][%d][%d]:\n", msg, iMax, jMax, kMax);
    for(int k = 0; k <= kMax; k++)
    {
        for(int j = 0; j <= jMax; j++)
        {
            printf("\n_%s_j%d_k%d[] = {", msg, j, k);
            for(int i = 0; i <= iMax; i++)
            {

                printf("%.30f, ", array[i][j][k]);
                if (i % 9 == 0)
                    printf("\n");
            }

            printf("};\n");
        }
    }
}