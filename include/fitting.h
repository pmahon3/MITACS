#ifndef FITTING_H
#define FITTING_H

#include "data.h"
#include "model.h"

// Error function type definition
typedef double (*ErrorFunction)(const Model* model, const Data* data);

// Function to fit the model parameters
void fit_model(Model* model, const Data* data, ErrorFunction error_fn, size_t iterations);

#endif // FITTING_H
