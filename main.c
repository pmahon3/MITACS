#include <stdio.h>
#include <stdlib.h>
#include "data.h"
#include "model.h"
#include "pseudo_inverse.h"

int main() {
    // Load data
    const char* filename = "resources/test_data/lorenz.csv";
    Data* data = load_data(filename);
    if (!data) {
        return 1;
    }

    // Print the number of rows and columns
    printf("Data loaded: %zu rows, %zu columns\n", data->rows, data->cols);

    // Print the first 10 rows of data
    size_t rows_to_print = data->rows < 10 ? data->rows : 10;
    for (size_t i = 0; i < rows_to_print; ++i) {
        printf("%s", data->timestamps[i]);
        for (size_t j = 0; j < data->cols; ++j) {
            printf(", %f", data->values[i * data->cols + j]);
        }
        printf("\n");
    }

    // Initialize custom D matrix with lags at 1, 3, 5
    size_t m = data->cols; // Assume the embedding dimension is equal to the number of columns in data
    size_t n = 5; // Maximum lag to consider
    double theta = 1.0; // Example theta value
    double* custom_D = (double*)malloc(m * n * sizeof(double));

    Model* model = init_model_custom(theta, custom_D, m, n);
    if (!model) {
        free_data(data);
        free(custom_D);
        return 1;
    }

    // Initialize the custom D matrix
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            custom_D[i * n + j] = 0.0;
        }
    }
    custom_D[0 * n + 0] = 1.0; // Lag 1
    custom_D[1 * n + 2] = 1.0; // Lag 3
    custom_D[2 * n + 4] = 1.0; // Lag 5

    model->D = custom_D;

    // Print the transformation matrix D
    printf("Transformation matrix D:\n");
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            printf("%f ", model->D[i * n + j]);
        }
        printf("\n");
    }

    // Define the start indices for prediction
    size_t start_indices[] = {7500, 8000, 8500, 9000, 9500, 10000};
    size_t num_predictions = sizeof(start_indices) / sizeof(start_indices[0]);

    // Allocate memory for results
    double** results = (double**)malloc(num_predictions * sizeof(double*));
    for (size_t i = 0; i < num_predictions; ++i) {
        results[i] = (double*)malloc(data->cols * sizeof(double));
    }

    // Make predictions
    predict(model, data, start_indices, num_predictions, results);

    // Print predictions
    for (size_t i = 0; i < num_predictions; ++i) {
        printf("Prediction at index %zu: ", start_indices[i]);
        for (size_t j = 0; j < data->cols; ++j) {
            printf("%f ", results[i][j]);
        }
        printf("\n");
    }

    // Free allocated memory
    for (size_t i = 0; i < num_predictions; ++i) {
        free(results[i]);
    }
    free(results);
    free_model(model);
    free_data(data);

    return 0;
}
