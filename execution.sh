#!/bin/bash
if [[ $# < 6 ]]; then
    echo -e "Correct usage: ./exectuion.sh\t<execution_file>\t<file.txt>\t<ML>\t<MR>\t<pol_order>\t<debug>"
else
    if [[ ! -d /Execution_files ]]; then
        mkdir Execution_files
    fi

    echo -e "\nCompilazione...\n"

    nvcc -o Execution_files/static_lib_cuSolver_QR.o -c static_cuSolver_QR.cpp -lcublas -lcusolver
    ar cr Execution_files/static_lib_cuSolver_QR.a Execution_files/static_lib_cuSolver_QR.o

    nvcc -o Execution_files/$1.o -c $1.cu -lcublas -lcusolver
    nvcc Execution_files/$1.o Execution_files/static_lib_cuSolver_QR.a -o Execution_files/$1.out -lcublas -lcusolver
    
    echo -e "\nCode Compiled!\n"

    echo -e "\nComputing...\n"
    
    Execution_files/$1.out $2 $3 $4 $5 $6
    
    echo -e "\n\nExecution finished!"
fi