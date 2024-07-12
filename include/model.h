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
    // Dimension of transformation
    size_t n_lags;
    // Maximum lag of original space, X, to inform points in transformed space, Y
    size_t max_lag;
    // Number of variables in the transfomration (i.e: n_lags * max_lag)
    size_t l;
    // Locality parameter, Θ,  for local linear fit of transformed dynamics
    double theta;
    // The transformation matrix D (n_lags x max_lag) where DX = Y
    double* D;
    // The current local linear dynamics of the transformed space, Y_{t+1} = C * Y_t, weigted by a kernal w(y_t, Θ)
    double* C;
} Model;

// Function to initialize the model with identity matrix
Model* init_model(size_t size, double theta);

// Function to initialize the model with a custom matrix
Model* init_model_custom(const double theta, const double* custom_D, const size_t m, const size_t n);

// Function to compute the weighting vector W, of the embedding based on distances from x_star
void compute_weighting_vector(const Model* model, const Data* embedding, const double* x_star, double* W, const size_t max_idx);

// Function to compute the pseudo-inverse matrix
void compute_pseudo_inverse(double* data_values, size_t max_idx, size_t cols, double* pseudo_inverse_matrix, double* u, double* vt, double* s, double* superb);

// Function to compute matrix C
void compute_C(const double* pseudo_inverse_matrix, const double* data_values, double* C, size_t max_idx, size_t cols);

// Function to predict the next state
void predict(const Model* model, const Data* embedding, const size_t* start_indices, size_t num_indices, int step_size, double* result);

// Function to calculate distance between vectors x1 and x2
double calculate_distance(const double* x1, const double* x2, size_t n);

// Function to free memory allocated for the model
void free_model(Model* model);

// Function to transform data using a transformation matrix
double* embed_series(const double* data, size_t rows, size_t cols, double* D, size_t n_lags, size_t max_lag);


#endif // MODEL_H
