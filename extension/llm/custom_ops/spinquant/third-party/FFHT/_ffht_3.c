#include <Python.h>
#include <numpy/arrayobject.h>
#include "fht.h"

#define UNUSED(x) (void)(x)

static char module_docstring[] =
    "A C extension that computes the Fast Hadamard Transform";
static char fht_docstring[] =
    "Compute the Fast Hadamard Transform (FHT) for a given "
    "one-dimensional NumPy array.\n\n"
    "The Hadamard Transform is a linear orthogonal map defined on real vectors "
    "whose length is a _power of two_. For the precise definition, see the "
    "[Wikipedia entry](https://en.wikipedia.org/wiki/Hadamard_transform). The "
    "Hadamard Transform has been recently used a lot in various machine "
    "learning "
    "and numerical algorithms.\n\n"
    "The implementation uses "
    "[AVX](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions) "
    "to speed up the computation. If AVX is not supported on your machine, "
    "a simpler implementation without (explicit) vectorization is used.\n\n"
    "The function takes two parameters:\n\n"
    "* `buffer` is a NumPy array which is being transformed. It must be a "
    "one-dimensional array with `dtype` equal to `float32` or `float64` (the "
    "former is recommended unless you need high accuracy) and of size being a "
    "power "
    "of two. If your CPU supports AVX, then `buffer` must be aligned to 32 "
    "bytes. "
    "To allocate such an aligned buffer, use the function `created_aligned` "
    "from this "
    "module.\n"
    "* `chunk` is a positive integer that controls when the implementation "
    "switches "
    "from recursive to iterative algorithm. The overall algorithm is "
    "recursive, but as "
    "soon as the vector becomes no longer than `chunk`, the iterative "
    "algorithm is "
    "invoked. For technical reasons, `chunk` must be at least 8. A good choice "
    "is to "
    "set `chunk` to 1024. But to fine-tune the performance one should use a "
    "program "
    "`best_chunk` supplied with the library.\n";

static PyObject *ffht_fht(PyObject *self, PyObject *args) {
  UNUSED(self);

  PyObject *buffer_obj;

  if (!PyArg_ParseTuple(args, "O", &buffer_obj)) {
    return NULL;
  }

  PyArray_Descr *dtype;
  int ndim;
  npy_intp dims[NPY_MAXDIMS];
  PyArrayObject *arr = NULL;

  if (PyArray_GetArrayParamsFromObject(buffer_obj, NULL, 1, &dtype, &ndim, dims,
                                       &arr, NULL) < 0) {
    return NULL;
  }

  if (arr == NULL) {
    PyErr_SetString(PyExc_TypeError, "not a numpy array");
    return NULL;
  }

  dtype = PyArray_DESCR(arr);

  if (dtype->type_num != NPY_FLOAT && dtype->type_num != NPY_DOUBLE) {
    PyErr_SetString(PyExc_TypeError, "array must consist of floats or doubles");
    Py_DECREF(arr);
    return NULL;
  }

  if (PyArray_NDIM(arr) != 1) {
    PyErr_SetString(PyExc_TypeError, "array must be one-dimensional");
    Py_DECREF(arr);
    return NULL;
  }

  int n = PyArray_DIM(arr, 0);

  if (n == 0 || (n & (n - 1))) {
    PyErr_SetString(PyExc_ValueError, "array's length must be a power of two");
    Py_DECREF(arr);
    return NULL;
  }

  int log_n = 0;
  while ((1 << log_n) < n) {
    ++log_n;
  }

  void *raw_buffer = PyArray_DATA(arr);
  int res;
  if (dtype->type_num == NPY_FLOAT) {
    float *buffer = (float *)raw_buffer;
    res = fht_float(buffer, log_n);
  } else {
    double *buffer = (double *)raw_buffer;
    res = fht_double(buffer, log_n);
  }

  if (res) {
    PyErr_SetString(PyExc_RuntimeError, "FHT did not work properly");
    Py_DECREF(arr);
    return NULL;
  }

  Py_DECREF(arr);

  return Py_BuildValue("");
}

static PyMethodDef module_methods[] = {
    {"fht", ffht_fht, METH_VARARGS, fht_docstring},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef ffhtmodule = {
    PyModuleDef_HEAD_INIT,
    "ffht",
    module_docstring,
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit_ffht(void) {
    PyObject *module = PyModule_Create(&ffhtmodule);

    if (module == NULL) {
        printf("NULL");
        return NULL;
    }

    import_array();
    return module;
}


