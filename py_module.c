#include <Python.h>
#include <numpy/arrayobject.h>
//#include "/usr/local/lib/python3.7/site-packages/numpy/core/include/numpy/arrayobject.h"
#include "roc.h"

static char module_docstring[] = "This module provides an interface for calculating mean ROC-AUC";
static char mroc_docstring[] = "Calculate mean of ROC-AUC by labels.";

static PyObject *mroc_mean_roc_auc(PyObject *self, PyObject *args);
static PyMethodDef module_methods[] = {
    {"mean_roc_auc", mroc_mean_roc_auc, METH_VARARGS, mroc_docstring},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC PyInit__mroc(void) {
    PyObject *module;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_mroc",
        module_docstring,
        -1,
        module_methods,
        NULL,
        NULL,
        NULL,
        NULL
    };
    module = PyModule_Create(&moduledef);
    if (!module) {
        return NULL;
    }

    // Load `numpy` functionality.
    import_array();
    return module;
}

static PyObject *mroc_mean_roc_auc(PyObject *self, PyObject *args) {
    PyObject *labels_obj, *actuals_obj, *preds_obj;

    // Parse the input tuple
    if (!PyArg_ParseTuple(args, "OOO", &labels_obj, &actuals_obj, &preds_obj)) {
        return NULL;
    }

    // Interpret the input objects as numpy arrays.
    PyObject *labels_array = PyArray_FROM_OTF(labels_obj, NPY_INT, NPY_IN_ARRAY);
    PyObject *actuals_array = PyArray_FROM_OTF(actuals_obj, NPY_INT, NPY_IN_ARRAY);
    PyObject *preds_array = PyArray_FROM_OTF(preds_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    // If that didn't work, throw an exception.
    if (labels_array == NULL || actuals_array == NULL || preds_array == NULL) {
        Py_XDECREF(labels_array);
        Py_XDECREF(actuals_array);
        Py_XDECREF(preds_array);
        return NULL;
    }

    size_t n = (size_t) PyArray_DIM(labels_array, 0);

    int* labels = (int*) PyArray_DATA(labels_array);
    int* actuals = (int*) PyArray_DATA(actuals_array);
    double* preds = (double*) PyArray_DATA(preds_array);

    double value = mean_roc_auc(labels, actuals, preds, n);

    // Clean up.
    Py_DECREF(labels_array);
    Py_DECREF(actuals_array);
    Py_DECREF(preds_array);

    // Build the output tuple
    PyObject *ret = Py_BuildValue("d", value);
    return ret;
}

// Article: https://dfm.io/posts/python-c-extensions/