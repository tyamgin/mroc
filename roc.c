#include "roc.h"

#include <stdlib.h>
#include <stdint.h>

// formula described here: https://dyakonov.org/2017/07/28/auc-roc-площадь-под-кривой-ошибок/

typedef struct {
    int label;
    int actual;
    double pred;
} RocItem;

int items_compare(const void* va, const void* vb) {
    const RocItem* a = (const RocItem*) va;
    const RocItem* b = (const RocItem*) vb;
    if (a->label != b->label) {
        return a->label < b->label ? -1 : 1;
    }
    if (!EPS_EQUAL(a->pred, b->pred)) {
        return a->pred < b->pred ? -1 : 1;
    }
    return a->actual < b->actual ? -1 : 1;
}

int _roc_auc(const RocItem* arr, size_t n, double* result) {
    size_t i, next_num_it;
    size_t prev_num_zeroes_count = 0;
    size_t cur_num_zeroes_count = 0;
    uint64_t num_wholes = 0, num_halfs = 0;
    for (i = 0; i < n; i++) {
        double cur = arr[i].pred;
        if (i == 0 || !EPS_EQUAL(cur, arr[i - 1].pred)) {
            prev_num_zeroes_count += cur_num_zeroes_count;

            cur_num_zeroes_count = 0;
            for (next_num_it = i; next_num_it < n && EPS_EQUAL(arr[next_num_it].pred, cur); next_num_it++) {
                cur_num_zeroes_count += arr[next_num_it].actual == 0;
            }
        }

        if (arr[i].actual) {
            num_wholes += prev_num_zeroes_count;
            num_halfs += cur_num_zeroes_count;
        }
    }
    prev_num_zeroes_count += cur_num_zeroes_count;
    if (prev_num_zeroes_count == 0 || prev_num_zeroes_count == n) {
        return 0;
    }
    *result = (num_wholes + num_halfs * 0.5) / ((uint64_t)prev_num_zeroes_count * (n - prev_num_zeroes_count));
    return 1;
}

double mean_roc_auc(const int* labels, const int* actual, const double* pred, size_t n) {
    RocItem* arr = malloc(sizeof(RocItem) * n);
    size_t i;
    for (i = 0; i < n; i++) {
        arr[i].label = labels[i];
        arr[i].actual = actual[i];
        arr[i].pred = pred[i];
    }
    qsort(arr, n, sizeof(RocItem), items_compare);
    double sum = 0;
    size_t count = 0;
    size_t prev_start = 0;
    for (i = 1; i <= n; i++) {
        if (i == n || arr[i].label != arr[i - 1].label) {
            double val;
            if (_roc_auc(arr + prev_start, i - prev_start, &val)) {
                sum += val;
                count++;
            }
            prev_start = i;
        }
    }
    free(arr);
    return sum / count;
}