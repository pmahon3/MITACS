#ifndef PSEUDO_INVERSE_H
#define PSEUDO_INVERSE_H

#include <stddef.h>

// Function to compute the pseudo-inverse using SVD
void compute_pseudo_inverse(double* data_values, size_t rows, size_t cols, double* pseudo_inverse_matrix, double* u, double* vt, double* s, double* superb);

// Function to apply the pseudo-inverse to solve for C
void apply_pseudo_inverse(double* pseudo_inverse_matrix, double* Y_t1, double* C, size_t rows, size_t cols, double* W);

#endif // PSEUDO_INVERSE_H
