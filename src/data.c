#include "data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LENGTH 1024

double** allocate_matrix(size_t rows, size_t cols) {
    double **matrix = (double**)malloc(rows * sizeof(double*));
    for (size_t i = 0; i < rows; ++i) {
        matrix[i] = (double*)malloc(cols * sizeof(double));
    }
    return matrix;
}

void free_matrix(double** matrix, size_t rows) {
    for (size_t i = 0; i < rows; ++i) {
        free(matrix[i]);
    }
    free(matrix);
}

char** allocate_timestamps(size_t rows) {
    char **timestamps = (char**)malloc(rows * sizeof(char*));
    for (size_t i = 0; i < rows; ++i) {
        timestamps[i] = (char*)malloc(MAX_LINE_LENGTH * sizeof(char));
    }
    return timestamps;
}

void free_timestamps(char** timestamps, size_t rows) {
    for (size_t i = 0; i < rows; ++i) {
        free(timestamps[i]);
    }
    free(timestamps);
}

size_t count_lines(const char* filename) {
    FILE *file = fopen(filename, "r");
    if (!file) return 0;
    size_t lines = 0;
    char line[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), file)) {
        lines++;
    }
    fclose(file);
    return lines;
}

size_t count_columns(const char* filename) {
    FILE *file = fopen(filename, "r");
    if (!file) return 0;
    size_t cols = 0;
    char line[MAX_LINE_LENGTH];
    if (fgets(line, sizeof(line), file)) {
        char *token = strtok(line, ",");
        while (token) {
            cols++;
            token = strtok(NULL, ",");
        }
    }
    fclose(file);
    return cols; // Do not subtract 1, since the CSV does not include a title for the timestamp column
}

Data* load_data(const char* filename) {
    size_t rows = count_lines(filename) - 1; // Subtract one for the header
    size_t cols = count_columns(filename);
    char line[MAX_LINE_LENGTH];

    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        return NULL;
    }

    Data* data = (Data*)malloc(sizeof(Data));
    data->rows = rows;
    data->cols = cols;

    data->timestamps = allocate_timestamps(rows);
    data->values = allocate_matrix(rows, cols - 1); // Subtract one for the timestamp column

    // Skip header
    fgets(line, sizeof(line), file);

    for (size_t i = 0; i < rows; ++i) {
        fgets(line, sizeof(line), file);
        char *token = strtok(line, ",");
        if (token != NULL) {
            strcpy(data->timestamps[i], token);
        }

        for (size_t j = 0; j < cols - 1; ++j) { // Adjust to exclude the timestamp column
            token = strtok(NULL, ",");
            if (token != NULL) {
                data->values[i][j] = atof(token);
            } else {
                data->values[i][j] = 0.0; // Handle missing values
            }
        }
    }

    fclose(file);

    return data;
}

void free_data(Data* data) {
    free_timestamps(data->timestamps, data->rows);
    free_matrix(data->values, data->rows);
    free(data);
}

Data* lag_embed(const Data* data, size_t column, size_t lag, size_t embedding_dim) {
    size_t embedded_length = data->rows - lag * (embedding_dim - 1);
    Data* embedded_data = (Data*)malloc(sizeof(Data));
    embedded_data->rows = embedded_length;
    embedded_data->cols = embedding_dim;
    embedded_data->timestamps = allocate_timestamps(embedded_length);
    embedded_data->values = allocate_matrix(embedded_length, embedding_dim);

    // Copy timestamps (for simplicity, we use the timestamp of the earliest point in the window)
    for (size_t i = 0; i < embedded_length; ++i) {
        strcpy(embedded_data->timestamps[i], data->timestamps[i + lag * (embedding_dim - 1)]);
    }

    // Embed the data
    for (size_t i = 0; i < embedded_length; ++i) {
        for (size_t j = 0; j < embedding_dim; ++j) {
            embedded_data->values[i][j] = data->values[i + j * lag][column];
        }
    }

    return embedded_data;
}
