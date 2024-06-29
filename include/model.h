#ifndef MODEL_H
#define MODEL_H

#include "data.h"

// Structure to hold the model parameters
typedef struct {
    double **D;
    double theta;
    size_t m;
    size_t n;
} Model;

// Function to initialize the model
Model* init_model(size_t size, double theta);

Model* init_model_custom(double theta, double** custom_D, size_t m, size_t n);

// Function to apply the model
void predict(const Model* model, const Data* data, double start_time, int steps, int step_size, double **result);

// Function to free the model structure
void free_model(Model* model);

#endif // MODEL_H
