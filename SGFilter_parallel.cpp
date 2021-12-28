/**
 * @file SGFilter_parallel.cpp
 * @author Iannuzzo, De Trino, Casolaro
 * @brief SGFilter on a punch of data
 * @version 0.1
 * @date 2021-12-28
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <time.h>
#include <cmath>
#include <iostream>
#include <vector>

#define ROW 4
#define COL 5

using namespace std;

void SGFilter_function();

int main(int argc, char const *argv[])
{
    ifstream inFile;
    vector<double> vect;

    // Read file
    inFile.open("data.txt");
    if (!inFile) {
        cout << "Unable to open file";
        exit(1); // terminate with error
    }
    
    double x = 0.0;

    while (inFile >> x) {
        vect.push_back(x);
    }

    // Input data
    int ML = -5;
    int MR = 5;
    int pol_order = 3;

    int x_size = vect.size();

    // xn=[zeros(1,ML),x,zeros(1,MR)];
    for(int i = 0; i < abs(ML); i++)
        vect.insert(vect.begin(), 0.0);
    
    for(int i = 0; i < MR; i++)
        vect.push_back(0.0);

    for(vector<double>::iterator it = vect.begin(); it != vect.end(); ++it){
        cout << *it << endl;
    }

    vector<double> y(x_size, 0.0);

    cout << endl << endl;

    for(vector<double>::iterator it = y.begin(); it != y.end(); ++it){
        cout << *it << endl;
    }

    // create d vector [-ML:MR]


    // Matrix size A[N+1][2M+1]
    // N + 1 = order + 1;
    // 2M + 1 = abs(ML) + MR + 1

    // column major order
    // vector<vector<double> > matrix_A;

    vector<double> matrix_A(pol_order+1*abs(ML) + MR + 1, 0.0);
    cout << matrix_A.size() << endl << endl;

    

    /*
    // Elements to insert in column
    double num = 10.0;
  
    // Inserting elements into vector
    for (int i = 0; i < ROW; i++) {
        // Vector to store column elements
        vector<double> v1;
  
        for (int j = 0; j < COL; j++) {
            v1.push_back(num);
            num += 5;
        }
  
        // Pushing back above 1D vector
        // to create the 2D vector
        matrix_A.push_back(v1);
    }

    // Displaying the 2D vector
    for (int i = 0; i < matrix_A.size(); i++) {
        for (int j = 0; j < matrix_A[i].size(); j++)
            cout << matrix_A[i][j] << " ";
        cout << endl;
    }
    */

    return 0;
}
