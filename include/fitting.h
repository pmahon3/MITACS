#ifndef FITTING_H
#define FITTING_H

#include "data.h"
#include "model.h"

// Error function type definition
typedef double (*ErrorFunction)(const Data* predicted, const Data* actual);

// Function to compute the error
double compute_error(const Data* predicted, const Data* actual);

// Function to fit the model
void fit_model(Model* model, const Data* data, ErrorFunction error_fn, size_t iterations);

#endif // FITTING_H
