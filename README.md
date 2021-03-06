# Savitzky-Golay Filter Parallel CUDA

This is a simple C++ parallel implementation about Savitzky-Golay Filter. Our work is based on these papers:

* [What Is a Savitzky-Golay Filter?](https://c.mql5.com/forextsd/forum/147/sgfilter.pdf);
* [Savitzky-Golay Smoothing Filters](https://aip.scitation.org/doi/pdf/10.1063/1.4822961);

Savitzky-Golay Smoothing Filters in parallel C++ version runs in less than 1 second, to be more precisely in 0.3 ms.
We performed experiments on ECG text files. The largest on which we relied is about 400,000 samples.

To compile the code we used the NVCC compiler ([Here the documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)) and used libraries such as cuBlas and cuSolver.

**Here are some results plotted with MATLAB and execution times:**

An image of the signal after using the filter. The blue signal is the original one while the green is the filtered one

![signalExample](https://user-images.githubusercontent.com/47244184/148555280-f096ae4e-8c88-4633-9dea-5d84f5ab2440.jpg)

Instead here we have some run times on a file of about 400.000 samples

![Execution_time](https://user-images.githubusercontent.com/47244184/148555298-f0c1f321-224a-480e-ba8c-7f8a3ba7f927.jpg)

## Installation and Execution
* Clone this repository and enter it:

```bash
git clone https://github.com/Theangelkk/SGF_Filter.git
cd SGF_Filter
```
* Execute it using `exectuion.sh`. This is an example:

```bash
./execution.sh SGFilter_parallel_v03 ECG_signal.txt -5 5 3 0
```

* These are the parameters the program needs:

```bash
./execution.sh SGFilter_parallel_v03 <txt file> <ML> <MR> <Polynomial Order> <Debug>
```
The first parameter is the text file from which the signal values are taken. The `ML` and `MR` parameters are used to indicate the window size (ML + MR + 1). The following parameter indicates the order of the interpolation polynomial. And finally, the `debug` parameter tells if you want to print the data structures that are created as you go through the code.
There are some constraints to be respected for execution:

1. `ML` must be less then `MR`;
2. Window size must be less then `Polynomial Order`;
3. Window size must be less then input signal size;

## Evaluation

Below are some evaluations of signals of different sizes using the following input configuration:
```
ML = -5
MR = 5
Pol_Ord = 3
Debug = 0
```

| Signal size | Time (milliseconds) |
| :---: | :---: | 
| 34 values | 296.26 ms |
| 4.001 values | 354.50 ms |
| 46.064 values | 345.23 ms |
| 73.114 values | 350.27 ms |
| 399.150 values | 351.83 ms |

The performance has been measured on a machine that has the following characteristics:

* **CPU** Intel Core i7 860 2.80 GHz
* **RAM** 8 GB DDR3
* **GPU** Nvidia Quadro K5000
* **HDD** 500 GB SATA

The times shown consider the data transfers from the host to the device and vicersa and the allocation of the various data structures. In addition, performance is affected by the number of users using the machine.

The following shows the time of the main parts of the code. Allocation and transfer times are not considered.

|  | QR Factorization | Inverse R | Pseudoinverse A | Compute M | Compute Y | Total |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| 34 values | 0.11 ms | 0.30 ms | 0.12 ms | 0.03 ms | 0.11 ms | 0.68 ms |
| 4.001 values | 0.11 ms | 0.32 ms | 0.12 ms | 0.03 ms | 0.12 ms | 0.69 ms |
| 46.064 values | 0.11 ms | 0.31 ms | 0.13 ms | 0.05 ms | 0.20 ms | 0.78 ms |
| 73.114 values | 0.11 ms | 0.30 ms | 0.11 ms | 0.07 ms | 0.26 ms | 0.84 ms |
| 399.150 values | 0.11 ms | 0.31 ms | 0.11 ms | 0.28 ms | 0.92 ms | 1.73 ms |

From the table it is easy to notice that the times that change are those for the calculation of the **matrix M** and the **final values Y**.

## Bibliography
Below are the links that helped us in the development of the code:

* **cuBLAS**: https://docs.nvidia.com/cuda/cublas/index.html
* **cuSOLVER**: https://docs.nvidia.com/cuda/cusolver/index.html
* **SGFilter MATLAB Implementation**: https://it.mathworks.com/matlabcentral/answers/335433-how-to-implement-savitzky-golay-filter-without-using-inbuilt-functions#answer_322261

## Contact
For any questions about code, please contact us.

### Github Accounts
[Theangelkk](https://github.com/Theangelkk), [pasqualedetrino](https://github.com/pasqualedetrino) or [GennaroIannuzzo](https://github.com/GennaroIannuzzo).

### Linkedin Accounts
[Theangelkk](https://www.linkedin.com/in/angelo-casolaro/), [pasuqaledetrino](https://www.linkedin.com/in/pasquale-de-trino-65467216b/) or [GennaroIannuzzo](https://www.linkedin.com/in/gennaro-iannuzzo-93ab1a217/).
