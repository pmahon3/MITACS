#ifndef MODEL_H
#define MODEL_H

#include "data.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <lapacke.h>
#include <cblas.h>
#include <omp.h>

// Define a struct for the model
typedef struct {
    size_t m;
    size_t n;
    size_t l;
    double theta;
    double* D;  // Pointer to the model's matrix D
} Model;

// Function to initialize the model with identity matrix
Model* init_model(size_t size, double theta);

// Function to initialize the model with a custom matrix
Model* init_model_custom(const double theta, const double* custom_D, const size_t m, const size_t n);

// Function to compute the weighting matrix W
void compute_weighting_matrix(const Model* model, const Data* data, const double* x_star, double* W, const size_t max_idx);

// Function to compute the pseudo-inverse matrix
void compute_pseudo_inverse(double* data_values, size_t rows, size_t cols, double* pseudo_inverse_matrix, double* u, double* vt, double* s, double* superb);

// Function to compute matrix C
void compute_C(const double* pseudo_inverse_matrix, const double* data_values, double* C, const size_t rows, const size_t cols);

// Function to predict the next state
void predict(const Model* model, const Data* data, const size_t* start_indices, size_t num_indices, int step_size, double* result);

// Function to calculate distance between vectors x1 and x2
double calculate_distance(const double* x1, const double* x2, size_t n);

// Function to free memory allocated for the model
void free_model(Model* model);

// Function to transform data using a transformation matrix
double* embed_series(const double* data, size_t rows, size_t cols, double* D, size_t n_lags, size_t max_lag);


#endif // MODEL_H
