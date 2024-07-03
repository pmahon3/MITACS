#ifndef PSEUDO_INVERSE_H
#define PSEUDO_INVERSE_H

#include <stddef.h>

// Function to compute the pseudo-inverse using SVD
void pseudo_inverse(double* data_values, double* W, double* C, size_t rows, size_t cols);

#endif // PSEUDO_INVERSE_H
