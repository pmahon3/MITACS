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

// Function to perform lag embedding on a specified column
Data* lag_embed(const Data* data, size_t column, size_t lag, size_t embedding_dim);

// Function to allocate a matrix
double** allocate_matrix(size_t rows, size_t cols);

// Function to free a matrix
void free_matrix(double** matrix, size_t rows);

#endif // DATA_H
