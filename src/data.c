#include "data.h"
#include <stdio.h>
#include <string.h>

#define MAX_LINE_LENGTH 1024

// Function to allocate a matrix
double** allocate_matrix(size_t rows, size_t cols) {
    double** matrix = (double**)malloc(rows * sizeof(double*));
    for (size_t i = 0; i < rows; ++i) {
        matrix[i] = (double*)malloc(cols * sizeof(double));
    }
    return matrix;
}

// Function to free a matrix
void free_matrix(double** matrix, size_t rows) {
    for (size_t i = 0; i < rows; ++i) {
        free(matrix[i]);
    }
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
    data->cols = cols; // First column is timestamp
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
            data->values[row][col] = atof(token);
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
    free_matrix(data->values, data->rows);
    free(data);
}

// Function to transform data using a transformation matrix
double** embed_series(Data* data, size_t column, double** D, size_t m, size_t n) {
    size_t L = data->rows - n;
    double** transformed_data = allocate_matrix(L, n);

    for (size_t l = 0; l < L; l++) {
        for (size_t i = 0; i < m; i++) {
            transformed_data[l][i] = 0;
            for (size_t j = 0; j < n; j++) {
                transformed_data[l][i] += D[i][j] * data->values[l + j][column];
            }
        }
    }

    return transformed_data;
}