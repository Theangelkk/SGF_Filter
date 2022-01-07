/** 
 *  @file static_cuSolver.h
 *
 *  @author Angelo Casolaro, Pasquale De Trino, Gennaro Iannuzzo
 * 
 *  @brief The Static Library QR cuSolver gives several methods based
 *  on cuBLAS and cuSolverDN Library for solving some linear algebra problems.
 *  
 *  Steps to execute:
 *  1.  nvcc -o static_lib_cuSolver_QR.o -c static_cuSolver_QR.cpp -lcublas -lcusolver
 *  2.  ar cr static_lib_cuSolver_QR.a static_lib_cuSolver_QR.o 
 *  
 *  @date 05-01-2022
 *  
 */

#ifndef H_STATIC_CUSOLVER
#define H_STATIC_CUSOLVER

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <algorithm>   
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

void checkCUDAError (cudaError_t err, const char* msg);
void Create_Handles(cublasHandle_t* cublasH,cusolverDnHandle_t* cudenseH);
void QR_Factorization(cusolverDnHandle_t cudenseH, double* d_A, double* d_tau, int m, int n, int lda, double* d_work, int lwork, 
                        int* info_gpu, int* devInfo, bool debug);
void Compute_QT_B(cusolverDnHandle_t cudenseH, double* d_A, double* d_B, double* d_tau, int m, int n, int lda, int ldb, int nrhs, double* d_work,
                        int lwork, int* devInfo, int* info_gpu, bool debug);
void Compute_X(cublasHandle_t cublasH, int n, int nrhs, double* d_A, int lda, double* d_B, int ldb, const double one, double* XC, bool debug);
void Compute_R(double* d_A, double* A_app, int m, int n);
void Compute_Q(cusolverDnHandle_t cudenseH, double* d_A, double* d_Q, double* d_tau, double* d_work, int lwork, int* devInfo, int m, int n);
void invert(double* src_d, double* dst_d, int n);
void Product_MatrixMatrix(const double* A, cublasOperation_t operation_A, const double* B, cublasOperation_t operation_B, double* C, const int m, const int k, const int n);
void Product_MatrixVector(const double* A, cublasOperation_t operation_A, const double* Vec, double* C, const int m, const int n);
void Lin_Solve_QR(cublasHandle_t cublasH, cusolverDnHandle_t cudenseH, double* XC, double* d_A, double* d_B, double* d_tau, int m, int n,
                        int lda, int ldb, int nrhs, double* d_work, int lwork, int* devInfo, int* info_gpu, const double one, bool debug);
#endif // H_STATIC_CUSOLVER