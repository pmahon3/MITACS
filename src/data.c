#include "data.h"
#include <stdio.h>
#include <string.h>

#define MAX_LINE_LENGTH 1024

// Function to allocate a matrix as a 1D array
double* allocate_matrix(size_t rows, size_t cols) {
    return (double*)malloc(rows * cols * sizeof(double));
}

// Function to free a matrix allocated as a 1D array
void free_matrix(double* matrix) {
    free(matrix);
}

// Function to load data from a CSV file
Data* load_data(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file");
        return NULL;
    }

    // Count rows and columns
    char line[MAX_LINE_LENGTH];
    size_t rows = 0, cols = 0;
    if (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ",");
        while (token) {
            cols++;
            token = strtok(NULL, ",");
        }
    }
    while (fgets(line, sizeof(line), file)) {
        rows++;
    }
    rewind(file);

    // Allocate memory for data structure
    Data* data = (Data*)malloc(sizeof(Data));
    data->rows = rows;
    data->cols = cols;
    data->timestamps = (char**)malloc(rows * sizeof(char*));
    data->values = allocate_matrix(rows, data->cols);

    // Read data from file
    size_t row = 0;
    fgets(line, sizeof(line), file); // Skip header
    while (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ",");
        data->timestamps[row] = strdup(token);
        for (size_t col = 0; col < data->cols; ++col) {
            token = strtok(NULL, ",");
            data->values[row * data->cols + col] = atof(token);
        }
        row++;
    }

    fclose(file);
    return data;
}

// Function to free the data structure
void free_data(Data* data) {
    for (size_t i = 0; i < data->rows; ++i) {
        free(data->timestamps[i]);
    }
    free(data->timestamps);
    free_matrix(data->values);
    free(data);
}

// Function to transform data using a transformation matrix
double* embed_series(Data* data, size_t column, double* D, size_t m, size_t n) {
    size_t L = data->rows - n;
    double* transformed_data = allocate_matrix(L, m);

    for (size_t l = 0; l < L; l++) {
        for (size_t i = 0; i < m; i++) {
            transformed_data[l * m + i] = 0;
            for (size_t j = 0; j < n; j++) {
                transformed_data[l * m + i] += D[i * n + j] * data->values[(l + j) * data->cols + column];
            }
        }
    }

    return transformed_data;
}
