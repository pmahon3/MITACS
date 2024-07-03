#include <stdio.h>
#include <stdlib.h>
#include "data.h"
#include "model.h"

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

    // Define start indices for predictions (e.g., from 7500 to 10000)
    size_t start = 2001;
    size_t end = 3000;
    size_t num_indices = end - start + 1;
    size_t* start_indices = (size_t*)malloc(num_indices * sizeof(size_t));
    for (size_t i = 0; i < num_indices; ++i) {
        start_indices[i] = start + i;
    }

    // Allocate memory for the predictions
    double* predictions = (double*)malloc(num_indices * data->cols * sizeof(double));

    // Perform predictions
    predict(model, data, start_indices, num_indices, 1, predictions);

    // Free allocated memory
    free(start_indices);
    free(predictions);
    free_data(data);
    free_model(model);

    return 0;
}
