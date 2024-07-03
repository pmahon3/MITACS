#include "pseudo_inverse.h"
#include <lapacke.h>
#include <stdlib.h>

// Function to compute the pseudo-inverse using SVD
void pseudo_inverse(double* data_values, double* W, double* C, size_t rows, size_t cols) {
    // Allocate memory for SVD components
    double* u = (double*)malloc(rows * rows * sizeof(double));
    double* vt = (double*)malloc(cols * cols * sizeof(double));
    double* s = (double*)malloc((rows < cols ? rows : cols) * sizeof(double));
    double* superb = (double*)malloc((rows < cols ? rows : cols - 1) * sizeof(double));

    // Compute SVD of weighted data_values directly
    LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', rows, cols, data_values, cols, s, u, rows, vt, cols, superb);

    // Compute the pseudo-inverse of weighted data and directly compute C
    double tolerance = 1e-15;  // Tolerance for small singular values
    for (size_t i = 0; i < cols; ++i) {
        if (s[i] > tolerance) {
            s[i] = 1.0 / s[i];
        } else {
            s[i] = 0.0;
        }
    }

    // Compute C = V * S^+ * U^T * data_values (next state)
    for (size_t i = 0; i < cols; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            C[i * cols + j] = 0.0;
            for (size_t k = 0; k < rows; ++k) {
                double sum = 0.0;
                for (size_t l = 0; l < rows; ++l) {
                    sum += vt[i * cols + l] * s[l] * u[l * rows + k];
                }
                C[i * cols + j] += sum * data_values[(k + 1) * cols + j] * W[k + 1];
            }
        }
    }

    // Free allocated memory
    free(u);
    free(vt);
    free(s);
    free(superb);
}
