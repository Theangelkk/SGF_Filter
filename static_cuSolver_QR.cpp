/** 
 *  @file static_cuSolver.cpp
 *
 *  @author Angelo Casolaro, Pasquale De Trino, Gennaro Iannuzzo
 * 
 *  @date 05-01-2022
 *  
 */

#include "static_cuSolver.h"

void checkCUDAError (cudaError_t err, const char* msg)
{	       
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Cuda error: %s %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void Create_Handles(cublasHandle_t* cublasH, cusolverDnHandle_t* cudenseH)
{
    // Creation of Handle cuBLAS
    cublasStatus_t cublas_status = cublasCreate(cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    // Creation of Handle cuSolverDN
    cusolverStatus_t cusolver_status = cusolverDnCreate(cudenseH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
}

/*
    The QR_Factorization method computes the QR Factorization of the matrix d_A using
    cuSolver Library.
*/
void QR_Factorization(cusolverDnHandle_t cudenseH, double* d_A, double* d_tau, int m, int n, int lda, double* d_work, int lwork, 
                        int* info_gpu, int* devInfo, bool debug)
{
    /*
        Paramters of cusolverDnDgeqrf:
            1.  cublasHandle_t  --> cuSolver Handle
            2.  Int             --> Number of Rows of the Matrix (M)
            3.  Int             --> Number of Colums of the Matrix (N)
            4.  pointer*        --> Vector Pointer of the Linearized Matrix (ex. double*)
            5.  Int             --> Leading Dimension
            6.  pointer*        --> Tau vector
            7.  pointer*        --> Dwork vector
            8.  Int             --> lwork
            9.  Int*            --> Info GPU
    */

    /*      
     *      NOTE THAT: CUDA GEQRF saves the matrices R and Q overwriting the 
     *      original matrix d_A.
     *      In particular, the matrx R is overwritten in upper triangular 
     *      part of d_A, including diagonal elements. 
     *      
     *      The matrix Q is not formed explicitly, instead, a sequence of householder 
     *      vectors are stored in lower triangular part of A.
     * 
     */

    cusolverStatus_t cusolver_status = cusolverDnDgeqrf(
                                            cudenseH,
                                            m,
                                            n,
                                            d_A,
                                            lda,
                                            d_tau,
                                            d_work,
                                            lwork,
                                            devInfo
                                    );
    
    // Synchronization of all CUDA threads
    checkCUDAError(cudaDeviceSynchronize(), "cudaDeviceSynchronize cusolverDnDgeqrf");

    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    // Check if the factorization QR is good or not
    checkCUDAError(cudaMemcpy(info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost), "QR_Factorization cudaMemcpy devInfo --> info_gpu");
}

/*
    The Compute_QT_B method computes the product between the matrix Q and the
    vector b using cuSolver Library.
*/
void Compute_QT_B(cusolverDnHandle_t cudenseH, double* d_A, double* d_B, double* d_tau, int m, int n, int lda, int ldb, int nrhs, double* d_work,
                        int lwork, int* devInfo, int* info_gpu, bool debug)
{
    /*
        Paramters of cusolverDnDormqr:
            1.  cublasHandle_t  --> cuSolver Handle
            2.  operation
            3.  operation       --> Transposition of matrix Q
            4.  Int             --> Number of Rows of the Matrix (M)
            5.  Int             --> Number of Colums of vector b (so 1)
            6.  Int             --> Number of Colums of the Matrix (N)
            7.  pointer*        --> Vector Pointer of the Linearized Matrix(ex. double*)
            8.  Int             --> Leading Dimension
            9.  pointer*        --> Tau vector
            10. pointer*        --> Vector b
            11. Int             --> Leading Dimension vector b
            12.  pointer*       --> Dwork vector
            13.  Int            --> lwork
            14.  Int*           --> Info GPU
    */

    cusolverStatus_t cusolver_status = cusolverDnDormqr(
                                        cudenseH,
                                        CUBLAS_SIDE_LEFT,
                                        CUBLAS_OP_T,
                                        m,
                                        nrhs,
                                        n,
                                        d_A,
                                        lda,
                                        d_tau,
                                        d_B,
                                        ldb,
                                        d_work,
                                        lwork,
                                        devInfo
                                );

    // Synchronization of all CUDA threads
    checkCUDAError(cudaDeviceSynchronize(), "cudaDeviceSynchronize cusolverDnDormqr");

    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    // Check if the product Q^t * b is good or not
    checkCUDAError(cudaMemcpy(info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost), "Compute_QT_B cudaMemcpy devInfo --> info_gpu");

    if(debug)
    {
        printf("After ormqr (Q^T * B): info_gpu = %d\n", info_gpu);
        assert(0 == info_gpu); 
    }
}

/*
    The Compute_X method solves the triangular linear system x = R \ Q^t * b using 
    cuBLAS Library.
*/
void Compute_X(cublasHandle_t cublasH, int n, int nrhs, double* d_A, int lda, double* d_B, int ldb, const double one, double* XC, bool debug)
{
    /*
        Parameters of cusolverDnDormqr:
            1.  cublasHandle_t  --> cuBlas Handle
            2.  operation
            3.  operation       
            4.  operation       --> No Trasposition
            5.  operation    
            6.  Int             --> Number of Rows of the Matrix (M)
            7.  Int             --> Number of Colums of vector b (so 1)
            8.  Double          --> One 
            9.  pointer*        --> Vector Pointer of the Linearized Matrix(ex. double*)
            10. Int             --> Leading Dimension
            11. pointer*        --> Vector b (Note that it will be the results of x)
            12. Int             --> Leading Dimension Vector b
    */

    cublasStatus_t cublas_status = cublasDtrsm(
                                    cublasH,
                                    CUBLAS_SIDE_LEFT,
                                    CUBLAS_FILL_MODE_UPPER,
                                    CUBLAS_OP_N,
                                    CUBLAS_DIAG_NON_UNIT,
                                    n,
                                    nrhs,
                                    &one,
                                    d_A,
                                    lda,
                                    d_B,
                                    ldb
                            );

    // Synchronization of all CUDA threads
    checkCUDAError(cudaDeviceSynchronize(), "cudaDeviceSynchronize cublasDtrsm");

    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    // Copy the result d_B --> XC
    checkCUDAError(cudaMemcpy(XC, d_B, sizeof(double)*n*nrhs, cudaMemcpyDeviceToHost), "cudaMemcpy d_B --> XC");
}

/*
    The Compute_R method extracts the upper triangular matrix R from d_A.
*/
void Compute_R(double* d_A, double* R, int m, int n)
{   
    double* A_app = (double*)calloc(m * n, sizeof(double));

    checkCUDAError(cudaMemcpy(A_app, d_A, sizeof(double) * m * n, cudaMemcpyDeviceToHost), "cudaMemcpy d_A --> A_app");

    // Extraction of all elements stored in upper triangular part of d_A
    for(int i = 0 ; i < n; i++)
        for(int j = i; j < n; j++)
            R[j*n + i] = A_app[j*m + i];

    free(A_app);
}

/*
    The Compute_Q method computes the matrix Q of QR factorization of d_A
    starting to the Householder vectors stored in lower triangular part of d_A. 
*/
void Compute_Q(cusolverDnHandle_t cudenseH, double* d_A, double* d_Q, double* d_tau, double* d_work, int lwork, int* devInfo, int m, int n)
{
    double* h_Q = (double*)malloc(m * m * sizeof(double));

    // Initializing the output Q matrix 
    for (int j = 0; j < m; j++)
    {
        for (int i = 0; i < m; i++)
        {
            if (j == i) 
                h_Q[j + i*m] = 1.;
            else        
                h_Q[j + i*m] = 0.;
        }
    }

    double* square_Q_d = NULL;

    checkCUDAError(cudaMalloc(&square_Q_d, m * m * sizeof(double)), "cudaMalloc square_Q_d");

    checkCUDAError(cudaMemcpy(square_Q_d, h_Q, m * m * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy h_Q --> square_Q_d");

    /*
        Paramters of cusolverDnDormqr:
            1.  cublasHandle_t  --> cuSolver Handle
            2.  operation
            3.  operation       --> Operation on matrix 
            4.  Int             --> Number of Rows of the Matrix (M)
            5.  Int             --> Number of Colums of the Matrix (N)
            6.  Int             --> Min dimension between Rows and Colums
            7.  pointer*        --> Vector Pointer of the Linearized Matrix(ex. double*)
            8.  Int             --> Leading Dimension
            9.  pointer*        --> Tau vector
            10. pointer*        --> Matrix
            11. Int             --> Leading Dimension
            12. pointer*        --> Dwork vector
            13. Int             --> lwork
            14. Int*            --> Info GPU
    */
    cusolverStatus_t cusolver_status = cusolverDnDormqr(
                                                        cudenseH, 
                                                        CUBLAS_SIDE_LEFT, 
                                                        CUBLAS_OP_N, 
                                                        m, 
                                                        n, 
                                                        std::min(m, n), 
                                                        d_A, 
                                                        m, 
                                                        d_tau, 
                                                        square_Q_d, 
                                                        m, 
                                                        d_work, 
                                                        lwork, 
                                                        devInfo
                                                    );
    
    // Synchronization of all CUDA threads
    checkCUDAError(cudaDeviceSynchronize(), "cudaDeviceSynchronize cusolverDnDormqr");

    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    checkCUDAError(cudaMemcpy(d_Q, square_Q_d, m * n * sizeof(double), cudaMemcpyDeviceToDevice), "cudaMemcpy square_Q_d --> d_Q");
    
    free(h_Q);
    cudaFree(square_Q_d);
}

/*
    The invert method computes the inverse of a matrix using the LU Factorization
    available by the cuBLAS Library. 
*/
void invert(double* src_d, double* dst_d, int n)
{   
    // Creation of Handle cuBLAS
    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    int batchSize = 1;

    int *P, *INFO;
    int bug_dim = 17;

    /*
        Problem of cublasDgetrfBatched when the dimension of a matrix are:
                n >= 3 and n <= 16
        
        For more details see this link:
        https://coderedirect.com/questions/165993/cublas-incorrect-inversion-for-matrix-with-zero-pivot
    */
    if(n >= 17)
        bug_dim = n;

    cudaMalloc<int>(&P,bug_dim * batchSize * sizeof(int));
    cudaMalloc<int>(&INFO,batchSize * sizeof(int));

    int lda = bug_dim;

    double *A[] = { src_d };
    double** A_d;

    cudaMalloc<double*>(&A_d,sizeof(A));
    cudaMemcpy(A_d,A,sizeof(A),cudaMemcpyHostToDevice);

    /*
        Parameters of cublasDgetrfBatched:
            1.  cublasHandle_t  --> cuBLAS Handle
            2.  Int             --> Dimension of Square matrix
            3.  double*         --> Pointer Matrix A
            4.  int             --> Leading Dimension of A
            5.  int*         
            6.  int*            
            7.  int         
    */
    cublasDgetrfBatched(handle,bug_dim,A_d,lda,P,INFO,batchSize);

    int INFOh = 0;
    cudaMemcpy(&INFOh,INFO,sizeof(int),cudaMemcpyDeviceToHost);

    if(INFOh == 17)
    {
        fprintf(stderr, "Factorization Failed: Matrix is singularn");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }

    double* C[] = { dst_d };
    double** C_d;

    cudaMalloc<double*>(&C_d,sizeof(C));
    cudaMemcpy(C_d,C,sizeof(C),cudaMemcpyHostToDevice);

    /*
        Parameters of cublasDgetriBatched:
            1.  cublasHandle_t  --> cuBLAS Handle
            2.  int             --> Dimension of Square matrix
            3.  double*         --> Pointer Matrix A
            4.  int             --> Leading Dimension of A
            5.  int*         
            6.  double*         --> Pointer Output Matrix C 
            7.  int             --> Dimension of Square matrix C
            8.  int*    
            9.  int
    */
    cublasDgetriBatched(handle,n,A_d,lda,P,C_d,n,INFO,batchSize);

    cudaMemcpy(&INFOh,INFO,sizeof(int),cudaMemcpyDeviceToHost);

    if(INFOh != 0)
    {
        fprintf(stderr, "Inversion Failed: Matrix is singularn");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }

    cudaFree(P), cudaFree(INFO), cublasDestroy_v2(handle);
}

/*
    The Product_MatrixMatrix method compute the product between two matrices
    using cuBLAS Library. 

    The operation exectued by this function is:
        C = alpha * op(A) * op(B) + beta * C

*/
void Product_MatrixMatrix(const double* A, cublasOperation_t operation_A, const double* B, cublasOperation_t operation_B, double* C, const int m, const int n, const int k) 
{
    // Leading Dimension
    int lda=m;
    int ldb=n;
    int ldc=k;

    const double alf = 1;
    const double bet = 0;
    const double* alpha = &alf;
    const double* beta = &bet;
    
    // Create a handle for cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    /*
        Parameters of cublasDgemm:
            1.  cublasHandle_t  --> cuSolver Handle
            2.  operation       --> Operation on matrix A
            3.  operation       --> Operation on matrix B
            4.  Int             --> Number of Rows of the Output Matrix C
            5.  Int             --> Number of Colums of the Output Matrix C
            6.  Int             --> Internal dimension of Matrices (Number of Colums N)
            7.  double*         --> Constant Value alpha
            8.  double*         --> Pointer Matrix A
            9.  int             --> Leading Dimension of A
            10. double*         --> Pointer Matrix B
            11. int             --> Leading Dimension of B
            12. double*         --> Constant Value beta
            13. double*         --> Pointer Matrix C
            14. int             --> Leading Dimension of C
    */
    cublasStatus_t cublas_status = cublasDgemm(handle, operation_A, operation_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    // Destroy the handle
    cublasDestroy(handle);
}

/*
    The Product_MatrixVector method compute the product between matrix and vector
    using cuBLAS Library. 

    The operation exectued by this function is:
        r = alpha * op(A) * y + beta * r

*/
void Product_MatrixVector(const double* A, cublasOperation_t operation_A, const double* Vec, double* C, const int m, const int n) 
{   
    int lda = m;

    const double alf = 1;
    const double bet = 0;
    const double* alpha = &alf;
    const double* beta = &bet;

    // Create a handle for cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    /*
        Parameters of cublasDgemv:
            1.  cublasHandle_t  --> cuSolver Handle
            2.  operation       --> Operation on matrix A
            3.  operation       --> Operation on matrix B
            4.  Int             --> Number of Rows of the Output Matrix C
            5.  Int             --> Number of Colums of the Output Matrix C
            6.  double*         --> Constant Value alpha
            7.  double*         --> Pointer Matrix A
            8.  int             --> Leading Dimension of A
            9. double*          --> Pointer Vector Vec
            10. int             --> Leading Dimension of Vec
            11. double*         --> Pointer Vector C
            12. int             --> Leading Dimension of C
    */
    cublasStatus_t cublas_status = cublasDgemv(handle, operation_A, m, n, alpha, A, lda, Vec, 1, beta, C, 1);
    
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    // Destroy the handle
    cublasDestroy(handle);
}

void Lin_Solve_QR(cublasHandle_t cublasH, cusolverDnHandle_t cudenseH, double* XC, double* d_A, double* d_B, double* d_tau, int m, 
                            int n, int lda, int ldb, int nrhs, double* d_work, int lwork, int* devInfo, int* info_gpu, const double one, bool debug)
{
    // ------------------ STEP 1: Compute QR Factorization  ------------------------
    QR_Factorization(cudenseH, d_A, d_tau, m, n, lda, d_work, lwork, info_gpu, devInfo, debug);

    // ------------------ STEP 2: Compute operation Q^T * B ------------------------- 
    Compute_QT_B(cudenseH, d_A, d_B, d_tau, m, n, lda, ldb, nrhs, d_work, lwork, devInfo, info_gpu, debug);

    // ------------------ STEP 3: Compute vector x = R \ Q^T * B --------------------
    Compute_X(cublasH, n, nrhs, d_A, lda, d_B, ldb, one, XC, debug);
}

