#include "model.h"
#include "pseudo_inverse.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function to initialize the model
Model* init_model(size_t size, double theta) {
    Model* model = (Model*)malloc(sizeof(Model));
    model->m = size;
    model->n = size;
    model->theta = theta;
    model->D = (double*)malloc(size * size * sizeof(double));

    // Initialize D to the identity matrix
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            model->D[i * size + j] = (i == j) ? 1.0 : 0.0;
        }
    }
    return model;
}

// Function to initialize the model with a custom D
Model* init_model_custom(double theta, double* custom_D, size_t m, size_t n) {
    Model* model = (Model*)malloc(sizeof(Model));
    model->m = m;
    model->n = n;
    model->theta = theta;
    model->D = (double*)malloc(m * n * sizeof(double));

    // Initialize D to the custom matrix
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            model->D[i * n + j] = custom_D[i * n + j];
        }
    }
    return model;
}

// Function to calculate the distance between two points
double calculate_distance(double* x1, double* x2, size_t n) {
    double distance = 0.0;
    for (size_t i = 0; i < n; ++i) {
        distance += (x1[i] - x2[i]) * (x1[i] - x2[i]);
    }
    return sqrt(distance);
}

// Function to compute the weighting matrix
void compute_weighting_matrix(const Model* model, const Data* data, double* x_star, double* W) {
    size_t rows = data->rows;
    size_t cols = data->cols;

    // Calculate the average distance
    double avg_distance = 0.0;
    for (size_t i = 0; i < rows; ++i) {
        avg_distance += calculate_distance(x_star, &data->values[i * cols], cols);
    }
    avg_distance /= rows;

    // Compute the weights
    for (size_t i = 0; i < rows; ++i) {
        double distance = calculate_distance(x_star, &data->values[i * cols], cols);
        W[i] = exp(-model->theta * distance / avg_distance);
    }
}

// Function to compute the matrix C directly using data->values
void compute_C(const Data* data, double* W, double* C) {
    size_t rows = data->rows;
    size_t cols = data->cols;

    // Compute the pseudo-inverse and solve for C directly using data->values
    pseudo_inverse(data->values, W, C, rows - 1, cols);
}

// Function to apply the model for prediction
void predict(const Model* model, const Data* data, size_t start_row, int steps, int step_size, double *result, double *W) {
    size_t m = model->m;
    size_t n = model->n;
    double* x_star = &data->values[start_row * data->cols];

    // Compute the weighting matrix
    compute_weighting_matrix(model, data, x_star, W);

    // Compute the matrix C
    double* C = (double*)malloc(data->cols * data->cols * sizeof(double));
    compute_C(model, data, W, C); // Exclude future values

    // Predict the next state using C
    for (size_t i = 0; i < data->cols; ++i) {
        result[i] = 0.0;
        for (size_t j = 0; j < data->cols; ++j) {
            result[i] += C[i * data->cols + j] * x_star[j];
        }
    }

    // Free allocated memory
    free(C);
}

// Function to apply the model for leave-one-out prediction
void predict_leave_one_out(const Model* model, const Data* data, size_t exclude_row, double *result, double *W) {
    size_t m = model->m;
    size_t n = model->n;
    double* x_star = &data->values[exclude_row * data->cols];

    // Compute the weighting matrix
    compute_weighting_matrix(model, data, x_star, W);

    // Compute the matrix C
    double* C = (double*)malloc(data->cols * data->cols * sizeof(double));
    compute_C(model, data, W, C); // Leave-one-out

    // Predict the next state using C
    for (size_t i = 0; i < data->cols; ++i) {
        result[i] = 0.0;
        for (size_t j = 0; j < data->cols; ++j) {
            result[i] += C[i * data->cols + j] * x_star[j];
        }
    }

    // Free allocated memory
    free(C);
}

// Function to free the model structure
void free_model(Model* model) {
    free(model->D);
    free(model);
}
