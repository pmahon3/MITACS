#include "data.h"
#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Function to check if the matrix D is close to the identity matrix
int is_close_to_identity(double** D, size_t size, double tolerance) {
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            if (i == j) {
                if (fabs(D[i][j] - 1.0) > tolerance) {
                    return 0;
                }
            } else {
                if (fabs(D[i][j]) > tolerance) {
                    return 0;
                }
            }
        }
    }
    return 1;
}

int main() {
    // Seed the random number generator
    srand(time(NULL));

    // Load data
    Data* data = load_data("./resources/test_data/lorenz.csv");
    if (!data) {
        fprintf(stderr, "Failed to load data\n");
        return 1;
    }

    // Perform lag embedding on the first column (assuming it represents X)
    size_t lag = 1;
    size_t embedding_dim = 3;
    Data* embedded_data = lag_embed(data, 0, lag, embedding_dim);

    // Print the lag-embedded data for verification (first 10 lines)
    printf("Lag-embedded data (first 10 lines):\n");
    size_t lines_to_print = embedded_data->rows < 10 ? embedded_data->rows : 10;
    for (size_t i = 0; i < lines_to_print; ++i) {
        for (size_t j = 0; j < embedding_dim; ++j) {
            printf("%f ", embedded_data->values[i][j]);
        }
        printf("\n");
    }

    // Initialize the model
    double initial_theta = 1.0;
    Model* model = init_model(embedding_dim, initial_theta);

    // Print the initialized model parameters for verification
    printf("\nInitialized model parameters:\n");
    printf("Theta: %f\n", model->theta);
    printf("Matrix D:\n");
    for (size_t i = 0; i < model->size; ++i) {
        for (size_t j = 0; j < model->size; ++j) {
            printf("%f ", model->D[i][j]);
        }
        printf("\n");
    }

    // Allocate memory for the result
    double** result = allocate_matrix(embedded_data->rows, embedding_dim);

    // Apply the model
    apply_model(model, embedded_data, result);

    // Print the transformed data
    printf("\nTransformed data (first 10 lines):\n");
    for (size_t i = 0; i < lines_to_print; ++i) {
        for (size_t j = 0; j < embedding_dim; ++j) {
            printf("%f ", result[i][j]);
        }
        printf("\n");
    }

    // Check if the current D matrix is close to the identity matrix
    double tolerance = 0.1;
    if (is_close_to_identity(model->D, model->size, tolerance)) {
        printf("\nThe matrix D is close to the identity matrix.\n");
    } else {
        printf("\nThe matrix D is not close to the identity matrix.\n");
    }

    // Free resources
    free_matrix(result, embedded_data->rows);
    free_model(model);
    free_data(embedded_data);
    free_data(data);

    return 0;
}
