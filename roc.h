#ifndef MROC_ROC_H
#define MROC_ROC_H

#include <math.h>
#include <stdlib.h>

#define EPS 1e-10

#define EPS_EQUAL(a, b) (fabs((a) - (b)) < EPS)

double roc_auc(const int* actual, const double* pred, size_t n);
double mean_roc_auc(const int* labels, const int* actual, const double* pred, size_t n);

#endif //MROC_ROC_H
