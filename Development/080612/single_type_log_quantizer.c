#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

/*
 * single_type_log_quantizer.c
 * This is the C code for creating your own
 * Numpy ufunc for a log_quantizer function.
 *
 * In this code we only define the ufunc for
 * a single dtype. The computations that must
 * be replaced to create a ufunc for
 * a different funciton are marked with BEGIN
 * and END.
 *
 * Details explaining the Python-C API can be found under
 * 'Extending and Embedding' and 'Python/C API' at
 * docs.python.org .
 */

static PyMethodDef LogQuantizerMethods[] = {
        {NULL, NULL, 0, NULL}
};

/* define the global lookup table */

float log_lookup_table[100] = {-13.815510558 ,
			       -4.60517018599 ,
			       -3.91202300543 ,
			       -3.50655789732 ,
			       -3.21887582487 ,
			       -2.99573227355 ,
			       -2.81341071676 ,
			       -2.65926003693 ,
			       -2.52572864431 ,
			       -2.40794560865 ,
			       -2.30258509299 ,
			       -2.20727491319 ,
			       -2.1202635362 ,
			       -2.04022082853 ,
			       -1.96611285637 ,
			       -1.89711998489 ,
			       -1.83258146375 ,
			       -1.77195684193 ,
			       -1.71479842809 ,
			       -1.66073120682 ,
			       -1.60943791243 ,
			       -1.56064774826 ,
			       -1.51412773263 ,
			       -1.46967597006 ,
			       -1.42711635564 ,
			       -1.38629436112 ,
			       -1.34707364797 ,
			       -1.30933331998 ,
			       -1.27296567581 ,
			       -1.237874356 ,
			       -1.20397280433 ,
			       -1.1711829815 ,
			       -1.13943428319 ,
			       -1.10866262452 ,
			       -1.07880966137 ,
			       -1.0498221245 ,
			       -1.02165124753 ,
			       -0.994252273344 ,
			       -0.967584026262 ,
			       -0.941608539858 ,
			       -0.916290731874 ,
			       -0.891598119284 ,
			       -0.867500567705 ,
			       -0.843970070295 ,
			       -0.82098055207 ,
			       -0.798507696218 ,
			       -0.776528789499 ,
			       -0.755022584278 ,
			       -0.73396917508 ,
			       -0.713349887877 ,
			       -0.69314718056 ,
			       -0.673344553264 ,
			       -0.653926467407 ,
			       -0.634878272436 ,
			       -0.616186139424 ,
			       -0.597837000756 ,
			       -0.579818495253 ,
			       -0.562118918154 ,
			       -0.544727175442 ,
			       -0.527632742082 ,
			       -0.510825623766 ,
			       -0.494296321815 ,
			       -0.478035800943 ,
			       -0.462035459597 ,
			       -0.446287102628 ,
			       -0.430782916092 ,
			       -0.415515443962 ,
			       -0.400477566597 ,
			       -0.385662480812 ,
			       -0.371063681391 ,
			       -0.356674943939 ,
			       -0.342490308947 ,
			       -0.328504066972 ,
			       -0.31471074484 ,
			       -0.301105092784 ,
			       -0.287682072452 ,
			       -0.274436845702 ,
			       -0.261364764134 ,
			       -0.248461359298 ,
			       -0.235722333521 ,
			       -0.223143551314 ,
			       -0.210721031316 ,
			       -0.198450938724 ,
			       -0.186329578191 ,
			       -0.174353387145 ,
			       -0.162518929498 ,
			       -0.150822889735 ,
			       -0.139262067334 ,
			       -0.12783337151 ,
			       -0.116533816256 ,
			       -0.105360515658 ,
			       -0.0943106794712 ,
			       -0.0833816089391 ,
			       -0.0725706928348 ,
			       -0.0618754037181 ,
			       -0.0512932943876 ,
			       -0.0408219945203 ,
			       -0.0304592074847 ,
			       -0.0202027073175 ,
			       -0.0100503358535};

/* The loop definition must precede the PyMODINIT_FUNC. */

static void float_log_quantizer(char **args, npy_intp *dimensions,
                            npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    int tmp;

    for (i = 0; i < n; i++) {
      /*BEGIN main ufunc computation*/
      tmp = (int) ((*(float *)in) * 100 + .5);
      
      *((float *)out) = log_lookup_table[tmp];
      /*END main ufunc computation*/
      
      in += in_step;
      out += out_step;
    }
}

/*This a pointer to the above function*/
PyUFuncGenericFunction funcs[1] = {&float_log_quantizer};

/* These are the input and return dtypes of log_quantizer.*/
static char types[2] = {NPY_FLOAT, NPY_FLOAT};

static void *data[1] = {NULL};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "npufunc_log_quantizer",
    NULL,
    -1,
    LogQuantizerMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyObject *PyInit_npufunc_log_quantizer(void)
{
    PyObject *m, *log_quantizer, *d;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    import_array();
    import_umath();

    log_quantizer = PyUFunc_FromFuncAndData(funcs, data, types, 1, 1, 1,
                                    PyUFunc_None, "log_quantizer",
                                    "log_quantizer_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "log_quantizer", log_quantizer);
    Py_DECREF(log_quantizer);

    return m;
}
#else
PyMODINIT_FUNC initnpufunc_log_quantizer(void)
{
    PyObject *m, *log_quantizer, *d;


    m = Py_InitModule("npufunc_log_quantizer", LogQuantizerMethods);
    if (m == NULL) {
        return;
    }

    import_array();
    import_umath();

    log_quantizer = PyUFunc_FromFuncAndData(funcs, data, types, 1, 1, 1,
                                    PyUFunc_None, "log_quantizer",
                                    "log_quantizer_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "log_quantizer", log_quantizer);
    Py_DECREF(log_quantizer);
}
#endif
