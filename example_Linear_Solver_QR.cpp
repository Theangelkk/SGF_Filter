
/* 
 *  QR Factorization Dense Linear Solver
 *
 *  filename: test_cusolver_cuda6d5.cpp
 *
 *  module add cudatoolkit
 *  compile:  nvcc -o test_cusolver_cuda6d5 test_cusolver_cuda6d5.cpp -lcublas -lcusolver
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <cuda_runtime.h>

#include <cublas_v2.h>
#include <cusolverDn.h>


void printMatrix(int m, int n, const double*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            double Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
        }
    } 
}

/*
    cuSolver è composto da 3 parti fondamentali:

        1.  cuSolverDN = lavora con matrici dense ed effettua svariare fattorizzazioni
        2.  cuSolverSP = lavora con matrici sparse, tipicamente basata sulla fattorizzazione QR
        3.  cuSolverRF = effettua una rifattorizzazione delle matrici per ottenere delle performance maggiori

    Ricorda che questa libreria si occupa automaticamente del trasferimento, allocazione delle memoria sulla GPU, quindi
    incorpora le routine come: cudaFree, cudaMemcpy e cudaMemcpyAsync.
*/

/*
    cuSolverDN è una libreria progettata per risolvere un sistema lineare denso della forma: Ax = b

    Tipicamente questa libreria provede all'utilizzare la fattorizzazione QR e la fattorizzazione LU con pivoting parziale.

    Quando la matrice dei coefficienti A è simmetria, si può anche utilizzare la fattorizzazione di Cholesky.

    Per tutte le altre matrici simmetriche indefinite, utilizza la fattorizzazione LDL --> Bunch-Kaufman.
*/

/*
    Naming Convertions

    La libreria cuSolver fornisce due specifiche APIs:

        1.  legagy = Questa API è disponibile per i tipi di dato: Float, Double, cuComplex e cuDoubleComplex.

        Tipicamente per la specificazioone del tipo, si utilizza il templete:

                cusolverDn<t><operation>

        Dove i tipi di dato possono essere:

            -   S = Float
            -   D = Double
            -   C = cuComplex
            -   Z = cuDoubleComplex
            -   X = generic type

        Invece, l'operation può essere:

            -   potrf = Cholesky factorization
            -   getrf = LU with partial pivoting
            -   geqrf = QR factorization
            -   sytrf = Bunch-Kaufman

        Abbiamo anche la possibilità di lavorare con gli int64, dove semplicemente non bisogna specificare il tipo:

                cusolverDn<operation>


        2.  generic
*/

/*
    Asynchronous Execution

    Generalmente, cuSolver preferisce tenere una esecuzione asincrona in modo da massimizzare le performance.

    Di fatto, questa libreria fa un largo uso della funzione cudaDeviceSynchronize per verificare l'eventuale completamento
    delle funzioni di cuSolver richiamate.

    Ricordiamo che questa operazione viene spesso richiamata durante l'invocazione del trasferimento dei dati da Host --> Device.
*/

/*
    Si tiene a precisare che la libreria cuSolver è Thread-Safe.
    
    Tutti i parametri devono essere semnpre passati per riferimento dall'host.
*/

/*
    Ricordiamo che un tipo errore è quando l'handle di cuSolver non viene inizializzato:

        CUSOLVER_STATUS_NOT_INITIALIZED

*/





int main(int argc, char*argv[])
{
    cusolverDnHandle_t cudenseH = NULL;

    cublasHandle_t cublasH = NULL;

    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;

    const int m = 3;
    const int lda = m;
    const int ldb = m;
    const int nrhs = 1; // number of right hand side vectors

/*       | 1 2 3 |
 *   A = | 4 5 6 | 
 *       | 2 1 1 |
 *   x = ( 1 1 1 )'
 *   b = ( 6 15 4)'
 */
    double A[lda*m] = { 1.0, 4.0, 2.0, 2.0, 5.0, 1.0, 3.0, 6.0, 1.0};
//    double X[ldb*nrhs] = { 1.0, 1.0, 1.0}; // exact solution
    double B[ldb*nrhs] = { 6.0, 15.0, 4.0};
    double XC[ldb*nrhs]; // solution matrix from GPU

    double *d_A = NULL; // linear memory of GPU
    double *d_tau = NULL; // linear memory of GPU
    double *d_B  = NULL;
    int *devInfo = NULL; // info in gpu (device copy)
    double *d_work = NULL;
    int  lwork = 0;

    int info_gpu = 0;

    const double one = 1;
    printf("A = (matlab base-1)\n");
    printMatrix(m, m, A, lda, "A");
    printf("=====\n");
    printf("B = (matlab base-1)\n");
    printMatrix(m, nrhs, B, ldb, "B");
    printf("=====\n");

// step 1: create cudense/cublas handle
    cusolver_status = cusolverDnCreate(&cudenseH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

// step 2: copy A and B to device
    cudaStat1 = cudaMalloc ((void**)&d_A  , sizeof(double) * lda * m);
    cudaStat2 = cudaMalloc ((void**)&d_tau, sizeof(double) * m);
    cudaStat3 = cudaMalloc ((void**)&d_B  , sizeof(double) * ldb * nrhs);
    cudaStat4 = cudaMalloc ((void**)&devInfo, sizeof(int));
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);
    cudaStat1 = cudaMemcpy(d_A, A, sizeof(double) * lda * m   , cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_B, B, sizeof(double) * ldb * nrhs, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

// step 3: query working space of geqrf and ormqr
    cusolver_status = cusolverDnDgeqrf_bufferSize(
        cudenseH,
        m,
        m,
        d_A,
        lda,
        &lwork);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
    assert(cudaSuccess == cudaStat1);

// step 4: compute QR factorization
    cusolver_status = cusolverDnDgeqrf(
        cudenseH,
        m,
        m,
        d_A,
        lda,
        d_tau,
        d_work,
        lwork,
        devInfo);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    // check if QR is good or not
    cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    printf("after geqrf: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);

// step 5: compute Q^T*B
    cusolver_status= cusolverDnDormqr(
        cudenseH,
        CUBLAS_SIDE_LEFT,
        CUBLAS_OP_T,
        m,
        nrhs,
        m,
        d_A,
        lda,
        d_tau,
        d_B,
        ldb,
        d_work,
        lwork,
        devInfo);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    // check if QR is good or not
    cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    printf("after ormqr: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);


// step 6: compute x = R \ Q^T*B

    cublas_status = cublasDtrsm(
         cublasH,
         CUBLAS_SIDE_LEFT,
         CUBLAS_FILL_MODE_UPPER,
         CUBLAS_OP_N,
         CUBLAS_DIAG_NON_UNIT,
         m,
         nrhs,
         &one,
         d_A,
         lda,
         d_B,
         ldb);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(XC, d_B, sizeof(double)*ldb*nrhs, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    printf("X = (matlab base-1)\n");
    printMatrix(m, nrhs, XC, ldb, "X");

// free resources
    if (d_A    ) cudaFree(d_A);
    if (d_tau  ) cudaFree(d_tau);
    if (d_B    ) cudaFree(d_B);
    if (devInfo) cudaFree(devInfo);
    if (d_work ) cudaFree(d_work);


    if (cublasH ) cublasDestroy(cublasH);
    if (cudenseH) cusolverDnDestroy(cudenseH);

    cudaDeviceReset();

    return 0; 
}
