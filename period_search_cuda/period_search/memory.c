/* allocation and deallocation of memory

   24.10.2005 consulted with Lada Subr

   8.11.2006
*/

#include <iostream>
#include <new>

#include "stdafx.h"
#include <stdlib.h>
#include <stdio.h>

// per v3
#include <iostream>
#include <malloc.h>
#include <exception>
#include <exception>
#include <vector>

std::unique_ptr<int[]> create_vector_int(int size) {
    return std::make_unique<int[]>(size + 1); // +1 for 1-based indexing, zero-initialized
}

std::unique_ptr<double[]> create_vector_double(int size) {
    return std::make_unique<double[]>(size + 1); // +1 for 1-based indexing, zero-initialized
}

std::vector<std::unique_ptr<int[]>> create_matrix_int(int rows, int cols) {
    std::vector<std::unique_ptr<int[]>> mat(rows + 1);
    for (int i = 0; i <= rows; ++i) {
        mat[i] = std::make_unique<int[]>(cols + 1); // +1 for  1-based indexing, zero-initialized
    }
    return mat;
}

std::vector<std::unique_ptr<double[]>> create_matrix_double(int rows, int cols) {
    std::vector<std::unique_ptr<double[]>> mat(rows + 1);
    for (int i = 0; i <= rows; ++i) {
        mat[i] = std::make_unique<double[]>(cols + 1); // +1 for 1-based indexing, zero-initialized
    }
    return mat;
}

int* new_vector_int(int size) {
    return new int[size + 1](); // +1 for 1-based indexing, zero-initialized
}

double* new_vector_double(int size) {
    return new double[size + 1](); // +1 for 1-based indexing, zero-initialized
}

int** new_matrix_int(int rows, int cols) {
    int** mat = new int* [rows + 1]; // +1 for 1-based indexing
    for (int i = 0; i <= rows; ++i) {
        mat[i] = new int[cols + 1](); // +1 for 1-based indexing, zero-initialized
    }
    return mat;
}

double** new_matrix_double(int rows, int cols) {
    double** mat = new double* [rows + 1]; // +1 for 1-based indexing
    for (int i = 0; i <= rows; ++i) {
        mat[i] = new double[cols + 1](); // +1 for 1-based indexing, zero-initialized
    }
    return mat;
}

void new_deallocate_vector(int* ptr) {
    delete[] ptr;
}

void new_deallocate_vector(double* ptr) {
    delete[] ptr;
}

void new_deallocate_matrix_int(int** p_x, int rows) {
    for (int i = 0; i <= rows; i++) {
        delete[] p_x[i];
    }
    delete[] p_x;
}

void new_deallocate_matrix_double(double** p_x, int rows) {
    for (int i = 0; i <= rows; i++) {
        delete[] p_x[i];
    }
    delete[] p_x;
}


double* vector_double_old(int length)
{
    double* p_x;
    if ((p_x = (double*)malloc((length + 1) * sizeof(double))) == NULL)
    {
        fprintf(stderr, "failure in 'vector_double()' \n");
        fflush(stderr);
        throw std::bad_alloc();
    }

    return p_x;
}

void deallocate_vector_old(void* p_x)
{
    free(p_x);
}

// old one
//double *vector_double(int length)
//{
//   double *p_x;
//
//   if ((p_x = (double *) malloc((length + 1) * sizeof(double))) == NULL)
//   {
//      fprintf(stderr, "failure in 'vector_double()' \n");
//      fflush(stderr);
//   }
//   return (p_x);
//}
//
//void deallocate_vector(void* p_x)
//{
//    free((void*)p_x);
//}

int *vector_int_old(int length)
{
   int *p_x;

   if ((p_x = (int *) malloc((length + 1) * sizeof(long int))) == NULL)
   {
      fprintf(stderr, "failure in 'vector_int()' \n");
      fflush(stderr);
    }
    return (p_x);
}


// v3
double** matrix_double_old(int rows, int cols) {
    double** mat = (double**)malloc((rows + 1) * sizeof(double*)); // +1 for 1-based indexing
    if (mat == nullptr) {
        throw std::bad_alloc();
    }
    for (int i = 0; i <= rows; ++i) {
        mat[i] = (double*)malloc((cols + 1) * sizeof(double)); // +1 for 1-based indexing
        if (mat[i] == nullptr) {
            std::cerr << "Allocation failed for row " << i << std::endl;
            // Free already allocated rows in case of failure
            for (int j = 0; j < i; ++j) {
                free(mat[j]);
            }
            free(mat);
            throw std::bad_alloc();
        }
    }
    return mat;
}

void deallocate_matrix_double_old(double** p_x, int rows) {
    for (int i = 0; i <= rows; i++) {
        std::cout << "Deallocating row " << i << std::endl; // Debug statement
        if (p_x[i] != nullptr) {
            free(p_x[i]);
        }
        else {
            std::cerr << "Null pointer detected at row " << i << std::endl;
        }
    }
    free(p_x);
}


// v2
//double** matrix_double(int rows, int cols) {
//    double** mat = (double**)malloc((rows + 1) * sizeof(double*)); // +1 for 1-based indexing
//    if (mat == nullptr) {
//        throw std::bad_alloc();
//    }
//    for (int i = 0; i <= rows; ++i) {
//        mat[i] = (double*)malloc((cols + 1) * sizeof(double)); // +1 for 1-based indexing
//        if (mat[i] == nullptr) {
//            // Free already allocated rows in case of failure
//            for (int j = 0; j < i; ++j) {
//                free(mat[j]);
//            }
//            free(mat);
//            throw std::bad_alloc();
//        }
//    }
//    return mat;
//}
//
//void deallocate_matrix_double(double** p_x, int rows) {
//    for (int i = 0; i <= rows; i++) {
//        free(p_x[i]); // Use free
//    }
//    free(p_x); // Use free
//}

// old one
//double **matrix_double(int rows, int columns)
//{
//   double **p_x;
//   int i;
//
//   p_x = (double **)malloc((rows + 1) * sizeof(double *));
//   for (i = 0; (i <= rows) && (!i || p_x[i-1]); i++)
//      p_x[i] = (double *) malloc((columns + 1) * sizeof(double));
//   if (i < rows)
//   {
//      fprintf(stderr,"failure in 'matrix_double()' \n");
//      fflush(stderr);
//   }
//   return (p_x);
//}

int **matrix_int_old(int rows, int columns)
{
   int **p_x;
   int i;

   p_x = (int **) malloc((rows + 1) * sizeof(int *));
   for (i = 0; (i <= rows) && (!i || p_x[i-1]); i++)
      p_x[i] = (int *) malloc((columns + 1) * sizeof(int));
   if (i < rows)
   {
      fprintf(stderr,"failure in 'matrix_int()' \n");
      fflush(stderr);
   }
   return (p_x);
}

double ***matrix_3_double(int n_1, int n_2, int n_3)
{
   int i, j;
   double ***p_x;

   p_x = (double ***) malloc((n_1 + 1) * sizeof(double **));
   for (i = 0; i <= n_1; i++)
   {
      p_x[i] = (double **) malloc((n_2 + 1) * sizeof(double *));
      for (j = 0; j <= n_2; j++)
         p_x[i][j] = (double *)malloc((n_3 + 1) * sizeof(double));
      if (j < n_2)
      {
         fprintf(stderr,"failure in 'matrix_3_double' \n");
         fflush(stderr);
      }
   }
   if (i < n_1)
   {
      fprintf(stderr,"failure in 'matrix_3_double' \n");
      fflush(stderr);
   }

   return (p_x);
}


// old
//void deallocate_matrix_double(double **p_x, int rows)
//{
//   int i;
//
//   for (i = 0; i <= rows; i++) free(p_x[i]);
//   free(p_x);
//}

void deallocate_matrix_int_old(int **p_x, int rows)
{
   int i;

   for (i = 0; i <= rows; i++) free(p_x[i]);
   free(p_x);
}

// old
/*void deallocate_matrix_3(void ***p_x, int n_1, int n_2)
{
   int i, j;

   for (i = 0; i <= n_1; i++)
   {
      for (j = 1; j <= n_2; j++)
      {
         free(p_x[i][j]);
         p_x[i][j] = NULL;
      }
      free(p_x[i]);
      p_x[i] = NULL;
   }
   free(p_x);
}
*/
