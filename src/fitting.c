#include "fitting.h"

// Placeholder function to compute the error
// Error function implementation
double compute_error(const Data* predicted, const Data* actual) {
    double error = 0.0;
    for (size_t i = 0; i < predicted->rows; ++i) {
        for (size_t j = 0; j < predicted->cols; ++j) {
            double diff = predicted->values[i * predicted->cols + j] - actual->values[i * actual->cols + j];
            error += diff * diff;
        }
    }
    return error;
}

// Placeholder function to fit the model (to be implemented)
void fit_model(Model* model, const Data* data, ErrorFunction error_fn, size_t iterations) {
    // Placeholder implementation
}
