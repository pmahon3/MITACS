#ifndef DATA_H
#define DATA_H

#include <stdlib.h>

// Structure to hold the data
typedef struct {
    size_t rows;
    size_t cols;
    char **timestamps;
    double *values; // 1D array for data values
} Data;

// Function to load data from a CSV file
Data* load_data(const char* filename);

// Function to free the data structure
void free_data(Data* data);

// Function to allocate a matrix as a 1D array
double* allocate_matrix(size_t rows, size_t cols);

// Function to free a matrix allocated as a 1D array
void free_matrix(double* matrix);

#endif // DATA_H
