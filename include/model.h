#ifndef MODEL_H
#define MODEL_H

#include "data.h"

// Structure to hold the model parameters
typedef struct {
    double* D;   // Now a 1D array
    double theta;
    size_t m;    // Number of rows
    size_t n;    // Number of columns
} Model;

// Function to initialize the model
Model* init_model(size_t size, double theta);
Model* init_model_custom(double theta, const double* custom_D, size_t m, size_t n);

// Function to apply the model for multiple predictions
void predict(const Model* model, const Data* data, const size_t* start_indices, size_t num_predictions, double** results);

// Function to compute the weighting matrix
void compute_weighting_matrix(const Model* model, const Data* data, const double* x_star, double* W, size_t max_idx);

// Function to compute the matrix C directly using data->values
void compute_C(const Data* data, size_t max_idx, double* W, double* weighted_data, double* Y_t1, double* pseudo_inverse_matrix, double* C, double* u, double* vt, double* s, double* superb);

// Function to free the model structure
void free_model(Model* model);

#endif // MODEL_H
