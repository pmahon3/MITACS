#include "pseudo_inverse.h"
#include <lapacke.h>
#include <stdlib.h>

// Function to compute the pseudo-inverse using SVD
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

    // Compute pseudo-inverse matrix = V * S^+ * U^T
    for (size_t i = 0; i < cols; ++i) {
        for (size_t j = 0; j < rows; ++j) {
            pseudo_inverse_matrix[i * rows + j] = 0.0;
            for (size_t k = 0; k < rows; ++k) {
                pseudo_inverse_matrix[i * rows + j] += vt[i * cols + k] * s[k] * u[k * rows + j];
            }
        }
    }
}

// Function to apply the pseudo-inverse to solve for C
void apply_pseudo_inverse(double* pseudo_inverse_matrix, double* Y_t1, double* C, size_t rows, size_t cols, double* W) {
    // Compute C = pseudo_inverse_matrix * Y_t1
    for (size_t i = 0; i < cols; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            C[i * cols + j] = 0.0;
            for (size_t k = 0; k < rows; ++k) {
                C[i * cols + j] += pseudo_inverse_matrix[i * rows + k] * Y_t1[k * cols + j] * W[k];
            }
        }
    }
}
