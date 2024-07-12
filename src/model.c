#include "model.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <lapacke.h>
#include <cblas.h>
#include <omp.h>
#include <string.h>

#include "../../../../../../opt/homebrew/opt/openblas/include/cblas.h"

// Function to initialize the model
Model* init_model(const size_t size, const double theta) {
    Model* model = (Model*)malloc(sizeof(Model));
    model->n_lags = size;
    model->max_lag = size;
    model->l = size * size;
    model->theta = theta;
    model->D = (double*)malloc(size * size * sizeof(double));
    model->C = (double*)malloc(size * size * sizeof(double));

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
    model->n_lags = m;
    model->max_lag = n;
    model->l = m * n;
    model->theta = theta;
    model->D = (double*)malloc(m * n * sizeof(double));
    model->C = (double*)malloc(m * m * sizeof(double));

    // Initialize D to the custom matrix
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            model->D[i * n + j] = custom_D[i * n + j];
        }
    }
    return model;
}

double calculate_distance(const double* x1, const double* x2, const size_t n) {
    double distance = 0.0;
    for (size_t i = 0; i < n; ++i) {
        distance += pow(x1[i] - x2[i], 2);
    }
    return sqrt(distance);
}

void compute_weighting_vector(const Model* model, const Data* embedding, const double* x_star, double* W, const size_t max_idx) {
    const size_t cols = embedding->cols;
    double avg_distance = 0.0;

    // Parallelize distance calculation for average distance
    //#pragma omp parallel for reduction(+:avg_distance)
    for (size_t i = 0; i < max_idx; ++i) {
        avg_distance += calculate_distance(x_star, &embedding->values[i * cols], cols);
    }
    avg_distance /= max_idx;

    // Parallelize weight computation
    #pragma omp parallel for
    for (size_t i = 0; i < max_idx; ++i) {
        const double distance = calculate_distance(x_star, &embedding->values[i * cols], cols);
        W[i] = exp(-model->theta * distance / avg_distance);
    }
}

void compute_pseudo_inverse(double* data_values, size_t max_idx, size_t cols, double* pseudo_inverse_matrix, double* u, double* vt, double* s, double* superb) {
    // Compute SVD of weighted data_values directly
    LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', max_idx, cols, data_values, cols, s, u, max_idx, vt, cols, superb);

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
    double* S_inv = (double*)calloc(max_idx * cols, sizeof(double));
    for (size_t i = 0; i < max_idx && i < cols; ++i) {
        S_inv[i * cols + i] = s[i];
    }

    // Allocate intermediate matrix for V * S^+
    double* VS_inv = (double*)malloc(cols * max_idx * sizeof(double));

    // Compute V * S^+
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, cols, max_idx, cols, 1.0, vt, cols, S_inv, max_idx, 0.0, VS_inv, max_idx);

    // Compute pseudo-inverse matrix = (V * S^+) * U^T
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, cols, max_idx, max_idx, 1.0, VS_inv, max_idx, u, max_idx, 0.0, pseudo_inverse_matrix, max_idx);

    // Free allocated memory
    free(S_inv);
    free(VS_inv);
}

// Function to apply the model for prediction
void predict(const Model* model, const Data* embedding, const size_t* start_indices, const size_t num_indices, int step_size, double* result) {
    const size_t rows = embedding->rows;
    const size_t cols = embedding->cols;

    // Allocate memory for SVD components and weighting matrix
    double* u = (double*)malloc(rows * rows * sizeof(double) * num_indices);
    double* vt = (double*)malloc(cols * cols * sizeof(double) * num_indices);
    double* s = (double*)malloc((rows < cols ? rows : cols) * sizeof(double) * num_indices);
    double* superb = (double*)malloc((rows < cols ? rows : cols - 1) * sizeof(double) * num_indices);
    double* W = (double*)malloc(rows * sizeof(double) * num_indices);
    double* wY_t = (double*)malloc(rows * cols * sizeof(double) * num_indices);
    double* wY_t1 = (double*)malloc(rows * cols * sizeof(double) * num_indices);
    double* wY_t_pseudo_inverse = (double*)malloc(rows * cols * sizeof(double) * num_indices);
    double* C = (double*)malloc(cols * cols * sizeof(double) * num_indices);
    double** x_star = (double**)malloc( num_indices * sizeof(double*));

    const size_t progress_step = num_indices / 10; // Calculate step size for 10% progress

    // Parallelize the prediction over multiple start indices
    //#pragma omp parallel for
    for (size_t idx = 0; idx < num_indices; ++idx) {
        const size_t start_idx = start_indices[idx];
        x_star[idx] = &embedding->values[start_idx * cols];

        // We are solving for C, w_t * Y_t+1 = w_t * Y_t * C

        // Compute the weighting vector, w_t = w(X, t)
        compute_weighting_vector(model, embedding, x_star[idx], &W[idx], start_idx);

        // Compute weighted data, wY_t = w_t * Y_t = w(X, t) * Y_t
        cblas_dgemv(CblasRowMajor, CblasTrans, start_idx, cols, 1.0, embedding->values, cols, W, 1, 0.0, &wY_t[idx], 1);

        // Compute the weigted successor(?) data wY_t+1 =  w_t * Y_t+1 = w_t(X, t) * Y_t+1
        cblas_dgemv(CblasRowMajor, CblasTrans, start_idx + 1, cols, 1.0, &embedding->values[1], cols, W, 1, 0.0, &wY_t1[idx], 1);

        // Compute Moore Penrose psuedo inverse  of weighted data (w_t Y_t)^-1
        compute_pseudo_inverse(&wY_t[idx], start_idx, cols, &wY_t_pseudo_inverse[idx], &u[idx], &vt[idx], &s[idx], &superb[idx]);

        // Compute C as (w_t Y_t)^-1 * w_t * Y_t+1 = C
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, start_idx, cols, cols, 1.0, &wY_t1[idx], cols, &wY_t_pseudo_inverse[idx], 1, 0.0, &C[idx], 1);

        // Predict the next state, C * Y_t
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, start_idx, cols, cols, 1.0, &C[idx], cols, &wY_t[idx], 1, 0.0, &result[idx], 1);

        // Print progress every 10%
        //#pragma omp critical
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
    free(wY_t);
    free(wY_t1);
    free(x_star);
    free(wY_t_pseudo_inverse);
    free(C);
}


double* embed_series(const double* data, const size_t rows, const size_t cols, double* D, const size_t n_lags, const size_t max_lag) {
    const size_t n_points = rows * n_lags;
    double* embedding = (double*)malloc(n_points);

    double* x_i = data;
    // Loop over embedding array filling in embedding vectors
    for (double* y = embedding; y < embedding + n_points - max_lag; y) {
        *y = 0.0;
        for (size_t j = 0; j < n_lags; j++, y++) {
            for (double* phi = D + j * max_lag; phi < D + max_lag * (j+1); phi++, x_i += cols ) {
                *y += *x_i * *phi;
            }
            // Reset X pointer to start of current data vector
            x_i -= max_lag * cols;
        }
        x_i += cols;
    }

    // Deal with the last few where x_i + max_lag exceeds data length
    for (double* y = embedding + n_points - max_lag; y < embedding + n_points; y){
        *y = 0.0;
        for (size_t j = 0; j < n_lags; j++, y++) {
            for (double* phi = D + j * max_lag; phi < D + max_lag * (j+1); phi++, x_i += cols ) {
                *y += *x_i * *phi;
            }
            // Reset X pointer to start of current data vector
            x_i -= max_lag * cols;
        }
        x_i += cols;
    }
    return embedding;
}

void free_model(Model* model) {
    free(model->D);
    free(model);
}
