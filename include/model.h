#ifndef MODEL_H
#define MODEL_H

#include "data.h"

// Structure to hold the model parameters
typedef struct {
    double **D;
    double theta;
    size_t size;
} Model;

// Function to initialize the model
Model* init_model(size_t size, double theta);

// Function to apply the model
void apply_model(const Model* model, const Data* data, double **result);

// Function to free the model structure
void free_model(Model* model);

#endif // MODEL_H
