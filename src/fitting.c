#include "fitting.h"

// Placeholder function to compute the error
double compute_error(const Data* predicted, const Data* actual) {
    double error = 0.0;
    size_t rows = predicted->rows;
    size_t cols = predicted->cols;

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double diff = predicted->values[i][j] - actual->values[i][j];
            error += diff * diff;
        }
    }

    return error / (rows * cols);
}

// Placeholder function to fit the model (to be implemented)
void fit_model(Model* model, const Data* data, ErrorFunction error_fn, size_t iterations) {
    // Placeholder implementation
}
