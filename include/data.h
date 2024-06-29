#ifndef DATA_H
#define DATA_H

#include <stdlib.h>

// Structure to hold the data
typedef struct {
    size_t rows;
    size_t cols;
    char **timestamps;
    double **values;
} Data;

// Function to load data from a CSV file
Data* load_data(const char* filename);

// Function to free the data structure
void free_data(Data* data);

// Function to allocate a matrix
double** allocate_matrix(size_t rows, size_t cols);

// Function to free a matrix
void free_matrix(double** matrix, size_t rows);

// Function to transform data using a transformation matrix
double** embed_series(Data* data, size_t column, double** D, size_t m, size_t n);

#endif // DATA_H
