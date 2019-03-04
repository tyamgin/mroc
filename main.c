#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "roc.h"

double roc_brute(const int* actual, const double* pred, size_t n) {
    double num = 0;
    int denum = 0;
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            int f = actual[i] < actual[j];
            double g = EPS_EQUAL(pred[i], pred[j]) ? 0.5 : (pred[i] < pred[j] ? 1.0 : 0.0);
            num += f * g;
            denum += f;
        }
    }
    return num / denum;
}

void run_test(const int* label, const int* actual, const double* pred, size_t n, double expected) {
    double res = mean_roc_auc(label, actual, pred, n);
    if (fabs(res - expected) > 1e-8) {
        int i;
        printf("Assert failed\nActual:");
        for (i = 0; i < n; i++) {
            printf("\t%d", actual[i]);
        }
        printf("\n");
        printf("Pred:");
        for (i = 0; i < n; i++) {
            printf("\t%f", pred[i]);
        }
        printf("\n");
        printf("EXPECTED %f TO EQUAL %f\n", res, expected);
        printf("\n");
    }
}

void run_tests() {
    {
        int label[] = {0, 0, 0, 0};
        int actual[] = {0, 1, 1, 0};
        double pred[] = {0.1, 0.2, 0.3, 0.4};
        run_test(label, actual, pred, sizeof(actual) / sizeof(actual[0]), 0.5);
    }
    {
        int label[] = {0, 0, 0};
        int actual[] = {0, 1, 1};
        double pred[] = {0.0, 0.1, 0.1};
        run_test(label, actual, pred, sizeof(actual) / sizeof(actual[0]), 1.0);
    }
    {
        int label[] = {0, 0, 0, 0, 0};
        int actual[] = {0, 1, 0, 1, 0};
        double pred[] = {0.1, 0.1, 0.0, -0.4, 0.7};
        run_test(label, actual, pred, sizeof(actual) / sizeof(actual[0]), 0.25);
    }
    {
        int label[] = {0, 0, 0, 0, 0, 0};
        int actual[] = {0, 1, 0, 1, 1, 1};
        double pred[] = {0.1, 0.5, 0.1, 0.2, 0.2, -7};
        run_test(label, actual, pred, sizeof(actual) / sizeof(actual[0]), 0.75);
    }
    {
        int label[] = {0, 0, 0, 0, 0, 0, 0};
        int actual[] = {0, 0, 0, 0, 0, 1, 0};
        double pred[] = {1.20648799e-30, 1.48942199e-01, 5.61005734e-01, 0.00000000e+00,
                5.84090818e-01, 0.00000000e+00, 3.83774804e-01};
        run_test(label, actual, pred, sizeof(actual) / sizeof(actual[0]), 0.08333333333333331);
    }
    {
        int label[] = {7,  7, 13, 19,  7, 16,  1, 13, 13, 13, 13, 13,  1, 13,  1};
        int actual[] = {1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        double pred[] = {1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
        run_test(label, actual, pred, sizeof(actual) / sizeof(actual[0]), 1.0);
    }
}

int main() {
    run_tests();
    return 0;
}