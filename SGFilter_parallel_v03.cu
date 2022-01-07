/**
 * @file SGFilter_parallel_v03.cpp
 * 
 * @author Angelo Casolaro, Pasquale De Trino, Gennaro Iannuzzo
 * 
 * @brief A Savitzky–Golay filter is a digital filter that can be applied to a
 * set of digital data points for the purpose of smoothing the data, that is,
 * to increase the precision of the data without distorting the signal tendency.
 * 
 * This is achieved, in a process known as convolution, by fitting successive
 * sub-sets of adjacent data points with a low-degree polynomial by the method
 * of linear least squares. 
 * 
 * When the data points are equally spaced, an analytical solution to the 
 * least-squares  equations can be found, in the form of a single set of 
 * "convolution coefficients" that can be applied to all data sub-sets, 
 * to give estimates of the smoothed signal, (or derivatives of the smoothed signal) 
 * at the central point of each sub-set. 
 * 
 * The method, based on established mathematical procedures, was popularized by 
 * A. Savitzky and M. J. E. Golay, who published tables of convolution coefficients
 * for various polynomials and sub-set sizes in 1964. 
 * 
 * Steps to execute:
 * 1.   nvcc -o SGFilter_parallel_v03.o -c SGFilter_parallel_v03.cu -lcublas -lcusolver
 * 2.   nvcc SGFilter_parallel_v03.o static_lib_cuSolver_QR.a -o SGFilter_parallel_v03.out -lcublas -lcusolver
 *
 * @date 05-01-2022
 * 
 */

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include<cuda.h>

#include "static_cuSolver.h"

using namespace std;

__global__ void compute_matrix_frames_samples(double*, double*, int, int);
void printMatrix(int, int, const double*, int, const char*);

int main(int argc, char const *argv[])
{   
    // Handle cuSolver Library
	cublasHandle_t cublasH = NULL;
    
	// Handle cuSolverDN Librart
	cusolverDnHandle_t cudenseH = NULL;

    // Status cuSolver
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

    // Vandermonde matrix --> N+1 x 2M+1
    double* A = NULL;

    // Linear Vector (column-stored) of Matrix A
    double* d_A = NULL;

    // Leading Dimension of matrix A --> LDA >= M
    int lda = 0;

    // Number of samples of all windows
    int window_size = 0;

    // Polynomial Order + 1
    int nPlusOne = 0;

    // Linear Vector (column-stored) of Matrix tau
    double* d_tau = NULL;

    // Matrix R of Factorization QR
    double* R = NULL;

    // Inverse Matrix of R 
    double* R_1 = NULL;
    
    // Matrix R_1_padding (bug of cuBLAS Library)
    double* R_1_padding = NULL;

    double* inverse_app_d = NULL;
    
    // Linear Vector (column-stored) of Matrix R_1
    double* R_1_d = NULL;

    // Linear Vector (column-stored) of Matrix Q
    double* d_Q = NULL;

    // PseudoInverse of A
    double* PseudoA = NULL;

    // Linear Vector (column-stored) of Matrix PseudoA
    double* PseudoA_d = NULL;

    /*
        Matrix of Windows --> N_Sample * 2M + 1
        In each row are contained the samples of i-th window
    */
    double* matrix_frames_samples = NULL;

    // Linear Vector (column-stored) of Matrix matrix_frames_samples
    double* matrix_frames_samples_d = NULL;

    // Impulse Response of Savitzky–Golay filter
    double* H = NULL;

    // Linear Vector (column-stored) of Matrix H
    double* H_d = NULL;

    // Matrix word of cuSolver Library
    double* d_work = NULL;

    // Linear Vector (column-stored) of Matrix Y
    double* Y_d = NULL;

    // Info in GPU (Device copy)
    int* devInfo = NULL;

    double* d = NULL;
    
    int lwork = 0;

    int info_gpu = 0;

    // Print some structures to debug
    int debug = 0;
    
    // Event of Cuda Timer
    cudaEvent_t start, stop;

    // Final execution time
    float time;

    // Input File object 
    ifstream inFile;
    
    // Path Input File
    string filename;

    // Input File object 
    ofstream outFile("results.txt");

    // Vector Input X samples
    vector<double> input_x;

    // Number of Input samples
    int x_size = 0;

    // Vector Padding X samples
    vector<double> padding_x;

    // Vector Output Y samples
    vector<double> y;
    
    // Left Bound of Window
    int ML = 0;

    // Right Bound of Window
    int MR = 0; 
    
    // Polynomial Order
    int pol_order = 1;

    // Block number and threads number per block
    dim3 nBlocks, nThreadForBlock;
    
    double* padding_x_d = NULL;

    // Input Parameters
    if(argc < 6)
    {
        cout << "Usage: ./SGFilter_parallel\t<filename>\t<ml>\t<mr>\t<polynomial order>\t<debug>" << endl;
        
        filename = "ECG_signal.txt";
        ML = -5;
        MR = 5;
        pol_order = 3;
        debug = 0;
    }
    else
    {
        filename = argv[1];
        ML = atoi(argv[2]);
        MR = atoi(argv[3]);
        pol_order = atoi(argv[4]);
        debug = atoi(argv[5]);
    }

    // ---------------------- Step 0: Initial Checks -----------------------

    // Check if ML < MR
    if (ML >= MR) {
        cout << "ML must be less and not equal than MR" << endl;
        exit(-2);
    }

    // Check if polynomial order is less than window size
    if(pol_order > abs(ML) + MR + 1) {
        cout << "Polynomial order must be less than window size" << endl;
        exit(-3);
    }

    // ***********************************************************************

    // --------------------- Step 1: Read file -------------------------------
    inFile.open(filename.c_str());
    if (!inFile) 
    {
        cout << "Unable to open file";

        // Terminate with error
        exit(-1);
    }
    
    double x = 0.0;

    while (inFile >> x) 
        input_x.push_back(x);
    
    // Check if window size is less than input signal size
    if(abs(ML) + MR + 1 > input_x.size()) {
        cout << "Window size must be less than input signal size" << endl;
        exit(-4);
    }

    // ----------- Step 2: Initialization of execution times -----------------
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // ------------------ Step 3: Padding Input X ----------------------------
    x_size = input_x.size();
    
    padding_x = input_x;

    // Add zeros before signal values
    for(int i = 0; i < abs(ML); i++)
        padding_x.insert(padding_x.begin(), 0.0);
    
    // Add zeros after signal values
    for(int i = 0; i < MR; i++)
        padding_x.push_back(0.0);
    
    y = vector<double>(x_size, 0.0);
    
    // ------------------ Step 4: Definition Matrix A ------------------------

    /*
        2M + 1 = abs(ML) + MR + 1
        window_size = 2M+1
    */
    window_size = abs(ML) + MR + 1;

    /* 
        N + 1 = order + 1
        nPlusOne = N + 1
    */
    nPlusOne = pol_order + 1;

    // Leading dimension of matrix A
    lda = window_size;

    // Vector to define each row of matrix A
    d = (double*)calloc(window_size, sizeof(double));

    /*
        Example ML = -3:
            d = [-3, -2, -1, 0, 1, 2, 3]
    */
    for(int i = 0, j = ML; i < window_size; i++, j++)
        d[i] = (double)j;
    
    if(debug != 0)
        printMatrix(1, window_size, &d[0], 1, "d");

    /*
        Define Matrix A
        Example with window size = 7 and polynomial order = 3:

            Matrix size A = 11 x 4

                |1  -5  25  -125|            
                |1  -4  16   -64|
                |1  -3   9   -27|
                |1  -2   4    -8|
                |1  -1   1     1|
            A = |1   0   0     0|
                |1   1   1     1|
                |1   2   4     8|
                |1   3   9    27|
                |1   4  16    64|
                |1   5  25   125|
    */
    A = (double*)calloc(window_size * nPlusOne, sizeof(double));

    for(int j = 0; j < nPlusOne; j++)
        for(int i = 0; i < window_size; i++)
            A[j * window_size + i] = pow(d[i], (double)j);

    if(debug != 0)
        printMatrix(window_size, nPlusOne, A, window_size, "A");

    // **********************************************************************

    // ****************** QR Factorization of matrix A **********************

    // ----------------- STEP 1: Creation of the Handles --------------------
    Create_Handles(&cublasH,&cudenseH);
    
    // --- STEP 2: Allocation and Copy Data Structures from Host to Device ---
    checkCUDAError(cudaMalloc((void**)&d_A, sizeof(double) * lda * nPlusOne), "cudaMalloc d_A");
    
    checkCUDAError(cudaMemcpy(d_A, A, sizeof(double) * lda * nPlusOne, cudaMemcpyHostToDevice), "cudaMemcpy A --> d_A");

    cusolver_status = cusolverDnDgeqrf_bufferSize(
        cudenseH,
        window_size,
        nPlusOne,
        d_A,
        lda,
        &lwork
    );

    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    // Allocate elements for cuSolver library
    checkCUDAError(cudaMalloc((void**)&d_tau, sizeof(double) * window_size), "cudaMalloc d_Tau");
    checkCUDAError(cudaMalloc((void**)&devInfo, sizeof(int)), "cudaMalloc devInfo");
    checkCUDAError(cudaMalloc((void**)&d_work, sizeof(double)*lwork), "cudaMalloc d_work");

    // matrix A -> QR factorization
    QR_Factorization(cudenseH, d_A, d_tau, window_size, nPlusOne, lda, d_work, lwork, &info_gpu, devInfo, debug);

    // ------------------ STEP 3: Compute matrix R --------------------------
    R = (double*)calloc(nPlusOne * nPlusOne, sizeof(double));

    Compute_R(d_A, R, window_size, nPlusOne);

    if(debug)
        printMatrix(nPlusOne, nPlusOne, R, nPlusOne, "R");

    // ------------------ STEP 4: Compute matrix Q ---------------------------
    checkCUDAError(cudaMalloc(&d_Q, window_size * window_size * sizeof(double)), "cudaMalloc d_Q");

    Compute_Q(cudenseH, d_A, d_Q, d_tau, d_work, lwork, devInfo, window_size, nPlusOne);

    if(debug != 0)
    {
        double* app_Q = (double*)malloc(window_size * nPlusOne * sizeof(double));
        checkCUDAError(cudaMemcpy(app_Q, d_Q, window_size * nPlusOne * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy d_Q --> app_Q");

        printMatrix(window_size, nPlusOne, app_Q, window_size, "Q");

        free(app_Q);
    }

    // ---------------- STEP 5: Compute Inverse of R -------------------------
    /**
     * An undefined bug in the cuBlas library forces us to use matrices of
     * size equal to or greater than 17
     */
    int bug_dim = 17;

    if(nPlusOne > 17)
        bug_dim = nPlusOne;

    // Allocate R^(-1) padding matrix
    R_1_padding = (double*)calloc(bug_dim * bug_dim, sizeof(double));

    // Insert ones on main diagonal
    for(int i=nPlusOne; i<bug_dim; i++)
        R_1_padding[i*17 + i]  = 1;
    
    // Insert data in padded matrix
    for(int i=0; i<nPlusOne; i++)
        for(int j=0; j<nPlusOne; j++)
            R_1_padding[i*bug_dim + j] = R[i*nPlusOne + j];
    
    if(debug != 0)
        printMatrix(bug_dim, bug_dim, R_1_padding, bug_dim,"R_1_padding");

    R_1 = (double*)calloc(nPlusOne * nPlusOne, sizeof(double));

    // Allocate and move data to compute operations
    checkCUDAError(cudaMalloc((void**)&inverse_app_d, bug_dim * bug_dim * sizeof(double)), "cudaMalloc inverse_app_d");
    checkCUDAError(cudaMemcpy(inverse_app_d, R_1_padding, bug_dim * bug_dim * sizeof(double),cudaMemcpyHostToDevice), "cudaMemcpy src  --> inverse_app_d");
    checkCUDAError(cudaMalloc((void**)&R_1_d, nPlusOne * nPlusOne * sizeof(double)), "cudaMalloc dst_d");

    // Invert matrix R
    invert(inverse_app_d, R_1_d, nPlusOne);
    
    // Destroy and free space
    cusolverDnDestroy(cudenseH);
    cublasDestroy(cublasH);

    cudaFree(devInfo);

    if(debug != 0) {
        checkCUDAError(cudaMemcpy(R_1, R_1_d, nPlusOne * nPlusOne * sizeof(double),cudaMemcpyDeviceToHost), "cudaMemcpy dst_d --> dst");    
        printMatrix(nPlusOne, nPlusOne, R_1, nPlusOne, "R_1");
    }

    // ---------------- STEP 6: Compute Pseudoinverse of A -------------------
    // Allocate space for pseudo-inverse of A
    checkCUDAError(cudaMalloc((void**)&PseudoA_d, nPlusOne * window_size * sizeof(double)), "cudaMalloc PseudoA");
    
    // Compute Matrix x Matrix product using cuSolver
    Product_MatrixMatrix(R_1_d, CUBLAS_OP_N, d_Q, CUBLAS_OP_T, PseudoA_d, nPlusOne, window_size, nPlusOne);

    // Transfer pseudo-inverse from Device to Host
    PseudoA = (double*)calloc(nPlusOne * window_size, sizeof(double));
    checkCUDAError(cudaMemcpy(PseudoA, PseudoA_d, nPlusOne * window_size * sizeof(double),cudaMemcpyDeviceToHost), "cudaMemcpy PseudoA --> app_PseudoA");    

    if(debug != 0)
        printMatrix(nPlusOne, window_size, PseudoA, nPlusOne, "PseudoA");

    // ---------------- STEP 7: Compute Output Samples Y ----------------------
    matrix_frames_samples = (double*)calloc(x_size * window_size, sizeof(double));

    nBlocks.x = 512;
    nThreadForBlock.x = 1024;

    // Correct determination of the blocks number
    nBlocks.x = x_size/nThreadForBlock.x+((x_size%nThreadForBlock.x)==0? 0:1);

    checkCUDAError(cudaMalloc((void**)&matrix_frames_samples_d, x_size * window_size * sizeof(double)), "cudaMalloc matrix_frames_samples_d_parallel");
    checkCUDAError(cudaMalloc((void**)&padding_x_d, padding_x.size() * sizeof(double)), "cudaMalloc padding_x_d");
    checkCUDAError(cudaMemcpy(padding_x_d, &padding_x[0], padding_x.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy padding_x --> padding_x_d");
    
    // Compute matrix x matrix product on device
    compute_matrix_frames_samples<<<nBlocks, nThreadForBlock>>>(matrix_frames_samples_d, padding_x_d, x_size, window_size);

    cudaDeviceSynchronize();

    // Allocate space and compute H coefficients vector 
    H = (double*)calloc(window_size, sizeof(double));

    // H dimension 1 x window_size
    checkCUDAError(cudaMalloc((void**)&H_d, window_size * sizeof(double)), "cudaMalloc H_d");

    for(int j=0; j < window_size; j++)
        H[j] = PseudoA[j * nPlusOne];
    
    if(debug != 0)
        printMatrix(1, window_size, H, 1, "H");

    checkCUDAError(cudaMemcpy(H_d, H, window_size * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy H --> H_d");

    checkCUDAError(cudaMalloc((void**)&Y_d, x_size * sizeof(double)), "cudaMalloc Y_d");

    // Compute Matrix x vector product obtaining final results
    Product_MatrixVector(matrix_frames_samples_d, CUBLAS_OP_N, H_d, Y_d, x_size, window_size);

    checkCUDAError(cudaMemcpy(&y[0], Y_d, x_size * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy Y_d --> Y");
    
    // ------------------- Step 8: Stop execution times -----------------------
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    // **********************************************************************

    // ------------------- Step 9: Save Output Y on file ----------------------
    for(vector<double>::iterator it = y.begin(); it != y.end(); ++it)
        outFile << *it << endl;

    // Close file streams
    inFile.close();
    outFile.close();

    printf("Exectuion time %8.2f ms\n", time);

    // ---------- Step 10: De-allocation of tge data structures used ----------

    // Clear vectors
    y.clear();
    input_x.clear();
    padding_x.clear();

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_tau);
    cudaFree(d_Q);
    cudaFree(R_1_d);
    cudaFree(R_1_padding);
    cudaFree(inverse_app_d);
    cudaFree(d_work);
    cudaFree(H_d);\
    cudaFree(Y_d);
    cudaFree(PseudoA_d);
    cudaFree(matrix_frames_samples_d);

    // Distruct time variables
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free Host memory
    free(A);
    free(d);
    free(H);
    free(R_1);
    free(PseudoA);
    free(matrix_frames_samples);
    exit(0);
}

__global__ void compute_matrix_frames_samples(double* matrix_frames_samples_d, double* padding_x_d, int x_size, int window_size)
{
	// Coalescence formula
	int Global_idx_i = (blockIdx.x*blockDim.x) + threadIdx.x;

	if (Global_idx_i < x_size)
	    for(int j=0; j<window_size; j++)
            matrix_frames_samples_d[j*x_size + Global_idx_i] = padding_x_d[Global_idx_i + j];
}

void printMatrix(int m, int n, const double* A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++)
        {
            double Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
        }
    } 
}