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
Model* init_model_custom(const double theta, const double* custom_D, size_t m, size_t n) {
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
double calculate_distance(const double* x1, const double* x2, const size_t n) {
    double distance = 0.0;
    for (size_t i = 0; i < n; ++i) {
        distance += (x1[i] - x2[i]) * (x1[i] - x2[i]);
    }
    return sqrt(distance);
}

// Function to compute the weighting matrix
void compute_weighting_matrix(const Model* model, const Data* data, const double* x_star, double* W, const size_t max_idx) {
    size_t cols = data->cols;
    double sum_distances = 0.0;

    // First pass: compute distances, accumulate them for the average, and compute the weights
    for (size_t i = 0; i < max_idx; ++i) {
        double distance = calculate_distance(x_star, &data->values[i * cols], cols);
        sum_distances += distance;
        W[i] = distance;  // Store the distance temporarily in W
    }

    // Calculate the average distance
    const double avg_distance = sum_distances / max_idx;

    // Second pass: compute the weights using the average distance
    for (size_t i = 0; i < max_idx; ++i) {
        W[i] = exp(-model->theta * W[i] / avg_distance);
    }
}

// Function to compute the matrix C directly using data->values
void compute_C(const Data* data, size_t max_idx, double* W, double* weighted_data, double* Y_t1, double* pseudo_inverse_matrix, double* C, double* u, double* vt, double* s, double* superb) {
    const size_t rows = max_idx - 1;
    size_t cols = data->cols;

    // Compute the weighted data values
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            weighted_data[i * cols + j] = data->values[i * cols + j] * W[i];
            Y_t1[i * cols + j] = data->values[(i + 1) * cols + j];
        }
    }

    // Compute the pseudo-inverse
    compute_pseudo_inverse(weighted_data, rows, cols, pseudo_inverse_matrix, u, vt, s, superb);

    // Apply the pseudo-inverse to solve for C
    apply_pseudo_inverse(pseudo_inverse_matrix, Y_t1, C, rows, cols, W);
}

// Function to apply the model for multiple predictions
void predict(const Model* model, const Data* data, const size_t* start_indices, const size_t num_predictions, double** results) {
    const size_t rows = data->rows;
    const size_t cols = data->cols;

    // Preallocate memory for SVD components
    double* u = (double*)malloc((rows - 1) * (rows - 1) * sizeof(double));
    double* vt = (double*)malloc(cols * cols * sizeof(double));
    double* s = (double*)malloc((rows - 1 < cols ? rows - 1 : cols) * sizeof(double));
    double* superb = (double*)malloc((rows - 1 < cols ? rows - 1 : cols - 1) * sizeof(double));

    // Preallocate memory for weighted data, Y_t1, and pseudo-inverse matrix
    double* weighted_data = (double*)malloc((rows - 1) * cols * sizeof(double));
    double* Y_t1 = (double*)malloc((rows - 1) * cols * sizeof(double));
    double* pseudo_inverse_matrix = (double*)malloc(cols * rows * sizeof(double));

    // Allocate memory for weights
    double* W = (double*)malloc(rows * sizeof(double));

    // Allocate memory for C
    double* C = (double*)malloc(cols * cols * sizeof(double));

    // Loop over each start index to compute the prediction
    for (size_t p = 0; p < num_predictions; ++p) {
        size_t start_row = start_indices[p];
        double* x_star = &data->values[start_row * cols];

        // Compute the weighting matrix
        compute_weighting_matrix(model, data, x_star, W, start_row);

        // Compute C
        compute_C(data, start_row, W, weighted_data, Y_t1, pseudo_inverse_matrix, C, u, vt, s, superb);

        // Predict the next state using C
        for (size_t i = 0; i < cols; ++i) {
            results[p][i] = 0.0;
            for (size_t j = 0; j < cols; ++j) {
                results[p][i] += C[i * cols + j] * x_star[j];
            }
        }
    }

    // Free allocated memory
    free(C);
    free(u);
    free(vt);
    free(s);
    free(superb);
    free(weighted_data);
    free(Y_t1);
    free(pseudo_inverse_matrix);
    free(W);
}

// Function to free the model structure
void free_model(Model* model) {
    free(model->D);
    free(model);
}
