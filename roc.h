#ifndef MROC_ROC_H
#define MROC_ROC_H

#include <math.h>
#include <stdlib.h>

//#define EPS 1e-10
#define EPS 0.0 // like in sklearn.metrics.roc_auc_score

#define EPS_EQUAL(a, b) (fabs((a) - (b)) <= EPS)

double mean_roc_auc(const int* labels, const int* actual, const double* pred, size_t n);

#endif //MROC_ROC_H
