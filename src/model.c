#include "model.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <lapacke.h>
#include <cblas.h>
#include <omp.h>
#include <string.h>

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
Model* init_model_custom(const double theta, const double* custom_D, const size_t m, const size_t n) {
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

void compute_C(const double* pseudo_inverse_matrix, const double* data_values, double* C, const size_t rows, const size_t cols) {
    // Parallelize the computation of C
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < cols; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            C[i * cols + j] = 0.0;
            for (size_t k = 0; k < rows; ++k) {
                C[i * cols + j] += pseudo_inverse_matrix[i * rows + k] * data_values[(k + 1) * cols + j];
            }
        }
    }
}

double calculate_distance(const double* x1, const double* x2, size_t n) {
    double distance = 0.0;
    for (size_t i = 0; i < n; ++i) {
        distance += pow(x1[i] - x2[i], 2);
    }
    return sqrt(distance);
}

void compute_weighting_matrix(const Model* model, const Data* data, const double* x_star, double* W, const size_t max_idx) {
    const size_t cols = data->cols;
    double avg_distance = 0.0;

    // Parallelize distance calculation for average distance
    #pragma omp parallel for reduction(+:avg_distance)
    for (size_t i = 0; i < max_idx; ++i) {
        avg_distance += calculate_distance(x_star, &data->values[i * cols], cols);
    }
    avg_distance /= max_idx;

    // Parallelize weight computation
    #pragma omp parallel for
    for (size_t i = 0; i < max_idx; ++i) {
        const double distance = calculate_distance(x_star, &data->values[i * cols], cols);
        W[i] = exp(-model->theta * distance / avg_distance);
    }
}

void compute_pseudo_inverse(double* data_values, size_t rows, size_t cols, double* pseudo_inverse_matrix, double* u, double* vt, double* s, double* superb) {
    // Compute SVD of weighted data_values directly
    LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', rows, cols, data_values, cols, s, u, rows, vt, cols, superb);

    // Compute the pseudo-inverse of weighted data
    double tolerance = 1e-15;  // Tolerance for small singular values
    for (size_t i = 0; i < cols; ++i) {
        if (s[i] > tolerance) {
            s[i] = 1.0 / s[i];
        } else {
            s[i] = 0.0;
        }
    }

    // Create a diagonal matrix for S^+
    double* S_inv = (double*)calloc(rows * cols, sizeof(double));
    for (size_t i = 0; i < rows && i < cols; ++i) {
        S_inv[i * cols + i] = s[i];
    }

    // Allocate intermediate matrix for V * S^+
    double* VS_inv = (double*)malloc(cols * rows * sizeof(double));

    // Compute V * S^+
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, cols, rows, cols, 1.0, vt, cols, S_inv, rows, 0.0, VS_inv, rows);

    // Compute pseudo-inverse matrix = (V * S^+) * U^T
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, cols, rows, rows, 1.0, VS_inv, rows, u, rows, 0.0, pseudo_inverse_matrix, rows);

    // Free allocated memory
    free(S_inv);
    free(VS_inv);
}

// Function to apply the model for prediction
void predict(const Model* model, const Data* data, const size_t* start_indices, size_t num_indices, int step_size, double *result) {
    const size_t rows = data->rows;
    const size_t cols = data->cols;

    // Allocate memory for SVD components and weighting matrix
    double* u = (double*)malloc(rows * rows * sizeof(double));
    double* vt = (double*)malloc(cols * cols * sizeof(double));
    double* s = (double*)malloc((rows < cols ? rows : cols) * sizeof(double));
    double* superb = (double*)malloc((rows < cols ? rows : cols - 1) * sizeof(double));
    double* W = (double*)malloc(rows * sizeof(double));
    double* pseudo_inverse_matrix = (double*)malloc(rows * cols * sizeof(double));
    double* C = (double*)malloc(cols * cols * sizeof(double));

    const size_t progress_step = num_indices / 10; // Calculate step size for 10% progress

    // Parallelize the prediction over multiple start indices
    #pragma omp parallel for
    for (size_t idx = 0; idx < num_indices; ++idx) {
        const size_t start_idx = start_indices[idx];
        const double* x_star = &data->values[start_idx * cols];

        // Compute the weighting matrix
        compute_weighting_matrix(model, data, x_star, W, start_idx);

        // Compute the pseudo-inverse matrix using data up to start_idx
        double* data_subset = (double*)malloc((start_idx + 1) * cols * sizeof(double));
        memcpy(data_subset, data->values, (start_idx + 1) * cols * sizeof(double)); // Copy data up to start_idx

        compute_pseudo_inverse(data_subset, start_idx + 1, cols, pseudo_inverse_matrix, u, vt, s, superb);

        free(data_subset); // Free the allocated memory for data_subset


        // Compute C
        compute_C(pseudo_inverse_matrix, data->values, C, rows, cols);

        // Predict the next state
        for (size_t i = 0; i < cols; ++i) {
            result[idx * cols + i] = 0.0;
            for (size_t j = 0; j < cols; ++j) {
                result[idx * cols + i] += C[i * cols + j] * x_star[j];
            }
        }

        // Print progress every 10%
        #pragma omp critical
        {
            if (idx % progress_step == 0) {
                printf("Progress: %zu%%\n", (idx / progress_step) * 10);
            }
        }
    }
    printf("Progress: %d%%\n", 100);
    // Free allocated memory
    free(u);
    free(vt);
    free(s);
    free(superb);
    free(W);
    free(pseudo_inverse_matrix);
    free(C);
}



void free_model(Model* model) {
    free(model->D);
    free(model);
}
