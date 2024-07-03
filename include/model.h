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
Model* init_model_custom(double theta, double* custom_D, size_t m, size_t n);

// Function to apply the model for prediction
void predict(const Model* model, const Data* data, size_t start_row, int steps, int step_size, double* result, double* W);

// Function to apply the model for leave-one-out prediction
void predict_leave_one_out(const Model* model, const Data* data, size_t exclude_row, double* result, double* W);

// Function to free the model structure
void free_model(Model* model);

// Sub-functions
double calculate_distance(double* x1, double* x2, size_t n);
void compute_weighting_matrix(const Model* model, const Data* data, double* x_star, double* W);
void compute_C(const Data* data, double* W, double* C);
void pseudo_inverse(double* data_values, double* W, double* C, size_t rows, size_t cols);

#endif // MODEL_H
