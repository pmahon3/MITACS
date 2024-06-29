#include "model.h"
#include <stdlib.h>

// Function to initialize the model
Model* init_model(size_t size, double theta) {
    Model* model = (Model*)malloc(sizeof(Model));
    model->m = size;
    model->n = size;
    model->theta = theta;
    model->D = allocate_matrix(size, size);

    // Initialize D to the identity matrix
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            model->D[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
    return model;
}

// Function to initialize the model with a custom D
Model* init_model_custom(double theta, double** custom_D, size_t m, size_t n) {
    Model* model = (Model*)malloc(sizeof(Model));
    model->m = m;
    model->n = n;
    model->theta = theta;
    model->D = allocate_matrix(m, n);

    // Initialize D to the custom matrix
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            model->D[i][j] = custom_D[i][j];
        }
    }
    return model;
}


// Function to apply the model for prediction
void predict(const Model* model, const Data* data, double start_time, int steps, int step_size, double **result) {
    size_t num_steps = (size_t)steps;
    size_t m = model->m;
    size_t n = model->n;
}

// Function to free the model structure
void free_model(Model* model) {
    free_matrix(model->D, model->m);
    free(model);
}
