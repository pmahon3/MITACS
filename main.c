#include <stdio.h>
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
            printf(", %f", data->values[i][j]);
        }
        printf("\n");
    }

    // Initialize custom D matrix with lags at 1, 3, 5
    size_t m = data->cols; // Assume the embedding dimension is equal to the number of columns in data
    size_t n = 5; // Maximum lag to consider
    double theta = 1.0; // Example theta value
    double** custom_D = allocate_matrix(m, n);

    Model* model = init_model_custom(theta, custom_D, m, n);
    if (!model) {
        free_data(data);
        free_matrix(custom_D, m);
        return 1;
    }


    // Initialize the custom D matrix
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            custom_D[i][j] = 0.0;
        }
    }
    custom_D[0][0] = 1.0; // Lag 1
    custom_D[1][2] = 1.0; // Lag 3
    custom_D[2][4] = 1.0; // Lag 5

    model->D = custom_D;

    // Print the transformation matrix D
    printf("Transformation matrix D:\n");
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            printf("%f ", model->D[i][j]);
        }
        printf("\n");
    }

    // Apply transformation matrix D to the X series
    size_t column = 1;
    double** embedding = embed_series(data, column, model->D, m, n);

    // Print the first 10 rows of transformed data
    printf("Embedded data:\n");
    for (size_t i = 0; i < 10; ++i) {
        for (size_t j = 0; j < m; ++j) {
            printf("%f ", embedding[i][j]);
        }
        printf("\n");
    }

    // Free resources
    free_data(data);
    free_matrix(embedding, data->rows - n);
    free_model(model);
    free_matrix(custom_D, n);

    return 0;
}
