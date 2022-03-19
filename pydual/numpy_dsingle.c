#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <numpy/ufuncobject.h>
#include "structmember.h"
#include "numpy/npy_3kcompat.h"

#include "dsingle.h"

typedef struct {
        PyObject_HEAD
        dsingle obval;
} PydsingleScalarObject;

PyMemberDef PydsingleArrType_members[] = {
    {"real", T_FLOAT, offsetof(PydsingleScalarObject, obval.real), READONLY,
        "The real component of the dsingle"},
    {"dual", T_FLOAT, offsetof(PydsingleScalarObject, obval.dual), READONLY,
        "The dual component of the dsingle"},
    {NULL}
};

static PyObject *
PydsingleArrType_get_components(PyObject *self, void *closure)
{
    dsingle *q = &((PydsingleScalarObject *)self)->obval;
    PyObject *tuple = PyTuple_New(2);
    PyTuple_SET_ITEM(tuple, 0, PyFloat_FromDouble(q->real));
    PyTuple_SET_ITEM(tuple, 1, PyFloat_FromDouble(q->dual));
    return tuple;
}

PyGetSetDef PydsingleArrType_getset[] = {
    {"components", PydsingleArrType_get_components, NULL,
        "The components of the dsingle as a (real, dual) tuple", NULL},
    {NULL}
};

PyTypeObject PydsingleArrType_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "pydual.dsingle",                    /* tp_name*/
    sizeof(PydsingleScalarObject),           /* tp_basicsize*/
    0,                                          /* tp_itemsize */
    0,                                          /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
#if defined(NPY_PY3K)
    0,                                          /* tp_reserved */
#else
    0,                                          /* tp_compare */
#endif
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    0,                                          /* tp_flags */
    0,                                          /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    0,                                          /* tp_methods */
    PydsingleArrType_members,                /* tp_members */
    PydsingleArrType_getset,                 /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    0,                                          /* tp_new */
    0,                                          /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
#if PY_VERSION_HEX >= 0x02060000
    0,                                          /* tp_version_tag */
#endif
};

static PyArray_ArrFuncs _Pydsingle_ArrFuncs;
PyArray_Descr *dsingle_descr;

static PyObject *
dsingle_getitem(char *ip, PyArrayObject *ap)
{
    dsingle q;
    PyObject *tuple;

    if ((ap == NULL) || PyArray_ISBEHAVED_RO(ap)) {
        q = *((dsingle *)ip);
    }
    else {
        PyArray_Descr *descr;
        descr = PyArray_DescrFromType(NPY_FLOAT);
        descr->f->copyswap(&q.real, ip, !PyArray_ISNOTSWAPPED(ap), NULL);
        descr->f->copyswap(&q.dual, ip+4, !PyArray_ISNOTSWAPPED(ap), NULL);
        Py_DECREF(descr);
    }

    tuple = PyTuple_New(2);
    PyTuple_SET_ITEM(tuple, 0, PyFloat_FromDouble(q.real));
    PyTuple_SET_ITEM(tuple, 1, PyFloat_FromDouble(q.dual));

    return tuple;
}

static int dsingle_setitem(PyObject *op, char *ov, PyArrayObject *ap)
{
    dsingle q;

    if (PyArray_IsScalar(op, dsingle)) {
        q = ((PydsingleScalarObject *)op)->obval;
    }
    else {
        q.real = PyFloat_AsDouble(PyTuple_GetItem(op, 0));
        q.dual = PyFloat_AsDouble(PyTuple_GetItem(op, 1));
    }
    if (PyErr_Occurred()) {
        if (PySequence_Check(op)) {
            PyErr_Clear();
            PyErr_SetString(PyExc_ValueError,
                    "setting an array element with a sequence.");
        }
        return -1;
    }
    if (ap == NULL || PyArray_ISBEHAVED(ap))
        *((dsingle *)ov)=q;
    else {
        PyArray_Descr *descr;
        descr = PyArray_DescrFromType(NPY_FLOAT);
        descr->f->copyswap(ov, &q.real, !PyArray_ISNOTSWAPPED(ap), NULL);
        descr->f->copyswap(ov+4, &q.dual, !PyArray_ISNOTSWAPPED(ap), NULL);
        Py_DECREF(descr);
    }

    return 0;
}

static void
dsingle_copyswap(dsingle *dst, dsingle *src,
        int swap, void *NPY_UNUSED(arr))
{
    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(NPY_FLOAT);
    descr->f->copyswapn(dst, sizeof(float), src, sizeof(float), 2, swap, NULL);
    Py_DECREF(descr);
}

static void
dsingle_copyswapn(dsingle *dst, npy_intp dstride,
        dsingle *src, npy_intp sstride,
        npy_intp n, int swap, void *NPY_UNUSED(arr))
{
    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(NPY_FLOAT);
    descr->f->copyswapn(&dst->real, dstride, &src->real, sstride, n, swap, NULL);
    descr->f->copyswapn(&dst->dual, dstride, &src->dual, sstride, n, swap, NULL);
    Py_DECREF(descr);    
}

static void
dsingle_dotfunc(char *ip1, npy_intp is1, char *ip2, npy_intp is2, dsingle *op, npy_intp n, void *NPY_UNUSED(arr))
{       
    float sumr = 0;
    float sumi = 0;
    npy_intp i;
    for (i = 0; i < n; i++, ip1 += is1, ip2 += is2) {
        const float ip1r = ((dsingle *)ip1)->real;
        const float ip1i = ((dsingle *)ip1)->dual;
        const float ip2r = ((dsingle *)ip2)->real;
        const float ip2i = ((dsingle *)ip2)->dual;
        sumr += ip1r * ip2r;
        sumi += ip1r * ip2i + ip1i * ip2r;
    }
    op->real = sumr;
    op->dual = sumi;
}

static int
dsingle_compare (dsingle *pa, dsingle *pb, PyArrayObject *NPY_UNUSED(ap))
{
    dsingle a = *pa, b = *pb;
    npy_bool anan, bnan;
    int ret;

    anan = dsingle_isnan(a);
    bnan = dsingle_isnan(b);

    if (anan) {
        ret = bnan ? 0 : -1;
    } else if (bnan) {
        ret = 1;
    } else if(dsingle_less(a, b)) {
        ret = -1;
    } else if(dsingle_less(b, a)) {
        ret = 1;
    } else {
        ret = 0;
    }

    return ret;
}

static int
dsingle_argmax(dsingle *ip, npy_intp n, npy_intp *max_ind, PyArrayObject *NPY_UNUSED(aip))
{
    npy_intp i;
    dsingle mp = *ip;

    *max_ind = 0;

    if (dsingle_isnan(mp)) {
        /* nan encountered; it's maximal */
        return 0;
    }

    for (i = 1; i < n; i++) {
        ip++;
        /*
         * Propagate nans, similarly as max() and min()
         */
        if (!(dsingle_less_equal(*ip, mp))) {  /* negated, for correct nan handling */
            mp = *ip;
            *max_ind = i;
            if (dsingle_isnan(mp)) {
                /* nan encountered, it's maximal */
                break;
            }
        }
    }
    return 0;
}

static npy_bool
dsingle_nonzero (char *ip, PyArrayObject *ap)
{
    dsingle q;
    if (ap == NULL || PyArray_ISBEHAVED_RO(ap)) {
        q = *(dsingle *)ip;
    }
    else {
        PyArray_Descr *descr;
        descr = PyArray_DescrFromType(NPY_FLOAT);
        descr->f->copyswap(&q.real, ip, !PyArray_ISNOTSWAPPED(ap), NULL);
        descr->f->copyswap(&q.dual, ip+4, !PyArray_ISNOTSWAPPED(ap), NULL);
        Py_DECREF(descr);
    }
    return (npy_bool) !dsingle_equal(q, (dsingle) {0,0});
}

static void
dsingle_fillwithscalar(dsingle *buffer, npy_intp length, dsingle *value, void *NPY_UNUSED(ignored))
{
    npy_intp i;
    dsingle val = *value;

    for (i = 0; i < length; ++i) {
        buffer[i] = val;
    }
}

#define MAKE_T_TO_dsingle(TYPE, type)                                       \
static void                                                                    \
TYPE ## _to_dsingle(type *ip, dsingle *op, npy_intp n,                   \
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop)) \
{                                                                              \
    while (n--) {                                                              \
        op->real = (float)(*ip++);                                               \
        op->dual = 0;                                                             \
    }                                                                          \
}

MAKE_T_TO_dsingle(FLOAT, npy_float);
MAKE_T_TO_dsingle(DOUBLE, npy_double);
MAKE_T_TO_dsingle(LONGDOUBLE, npy_longdouble);
MAKE_T_TO_dsingle(BOOL, npy_bool);
MAKE_T_TO_dsingle(BYTE, npy_byte);
MAKE_T_TO_dsingle(UBYTE, npy_ubyte);
MAKE_T_TO_dsingle(SHORT, npy_short);
MAKE_T_TO_dsingle(USHORT, npy_ushort);
MAKE_T_TO_dsingle(INT, npy_int);
MAKE_T_TO_dsingle(UINT, npy_uint);
MAKE_T_TO_dsingle(LONG, npy_long);
MAKE_T_TO_dsingle(ULONG, npy_ulong);
MAKE_T_TO_dsingle(LONGLONG, npy_longlong);
MAKE_T_TO_dsingle(ULONGLONG, npy_ulonglong);

#define MAKE_CT_TO_dsingle(TYPE, type)                                      \
static void                                                                    \
TYPE ## _to_dsingle(type *ip, dsingle *op, npy_intp n,                   \
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop)) \
{                                                                              \
    while (n--) {                                                              \
        op->real = (float)(*ip++);                                               \
        op->dual =  (float)(*ip++);                                               \
    }                                                                          \
}

MAKE_CT_TO_dsingle(CFLOAT, npy_float);
MAKE_CT_TO_dsingle(CDOUBLE, npy_double);
MAKE_CT_TO_dsingle(CLONGDOUBLE, npy_longdouble);

static void register_cast_function(int sourceType, int destType, PyArray_VectorUnaryFunc *castfunc)
{
    PyArray_Descr *descr = PyArray_DescrFromType(sourceType);
    PyArray_RegisterCastFunc(descr, destType, castfunc);
    PyArray_RegisterCanCast(descr, destType, NPY_NOSCALAR);
    Py_DECREF(descr);
}

static PyObject *
dsingle_arrtype_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsingle q;

    if (!PyArg_ParseTuple(args, "ff", &q.real, &q.dual))
        return NULL;

    return PyArray_Scalar(&q, dsingle_descr, NULL);
}

static PyObject *
gentype_richcompare(PyObject *self, PyObject *other, int cmp_op)
{
    PyObject *arr, *ret;

    arr = PyArray_FromScalar(self, NULL);
    if (arr == NULL) {
        return NULL;
    }
    ret = Py_TYPE(arr)->tp_richcompare(arr, other, cmp_op);
    Py_DECREF(arr);
    return ret;
}

static long
dsingle_arrtype_hash(PyObject *o)
{
    dsingle q = ((PydsingleScalarObject *)o)->obval;
    long value = 0x456789;
    value = (10000004 * value) ^ _Py_HashDouble(q.real);
    value = (10000004 * value) ^ _Py_HashDouble(q.dual);
    if (value == -1)
        value = -2;
    return value;
}

static PyObject *
dsingle_arrtype_repr(PyObject *o)
{
    char str[128];
    dsingle q = ((PydsingleScalarObject *)o)->obval;
    if (q.dual >=0){sprintf(str, "%g +%ge", q.real, q.dual);}
    else {sprintf(str, "%g %ge", q.real, q.dual);}

    return PyUString_FromString(str);
}

static PyObject *
dsingle_arrtype_str(PyObject *o)
{
    char str[128];
    dsingle q = ((PydsingleScalarObject *)o)->obval;
    if (q.dual >=0){sprintf(str, "%g +%ge", q.real, q.dual);}
    else {sprintf(str, "%g %ge", q.real, q.dual);}
    return PyUString_FromString(str);
}

static PyMethodDef dsingleMethods[] = {
    {NULL, NULL, 0, NULL}
};

#define UNARY_UFUNC(name, ret_type)\
static void \
dsingle_##name##_ufunc(char** args, npy_intp* dimensions,\
    npy_intp* steps, void* data) {\
    char *ip1 = args[0], *op1 = args[1];\
    npy_intp is1 = steps[0], os1 = steps[1];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; i++, ip1 += is1, op1 += os1){\
        const dsingle in1 = *(dsingle *)ip1;\
        *((ret_type *)op1) = dsingle_##name(in1);};}

UNARY_UFUNC(isnan, npy_bool)
UNARY_UFUNC(isinf, npy_bool)
UNARY_UFUNC(isfinite, npy_bool)
UNARY_UFUNC(absolute, npy_float)
UNARY_UFUNC(log, dsingle)
UNARY_UFUNC(exp, dsingle)
UNARY_UFUNC(negative, dsingle)
UNARY_UFUNC(conjugate, dsingle)

#define BINARY_GEN_UFUNC(name, func_name, arg_type, ret_type)\
static void \
dsingle_##func_name##_ufunc(char** args, npy_intp* dimensions,\
    npy_intp* steps, void* data) {\
    char *ip1 = args[0], *ip2 = args[1], *op1 = args[2];\
    npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; i++, ip1 += is1, ip2 += is2, op1 += os1){\
        const dsingle in1 = *(dsingle *)ip1;\
        const arg_type in2 = *(arg_type *)ip2;\
        *((ret_type *)op1) = dsingle_##func_name(in1, in2);};};

#define BINARY_UFUNC(name, ret_type)\
    BINARY_GEN_UFUNC(name, name, dsingle, ret_type)
#define BINARY_SCALAR_UFUNC(name, ret_type)\
    BINARY_GEN_UFUNC(name, name##_scalar, npy_float, ret_type)

BINARY_UFUNC(add, dsingle)
BINARY_UFUNC(subtract, dsingle)
BINARY_UFUNC(multiply, dsingle)
BINARY_UFUNC(divide, dsingle)
BINARY_UFUNC(power, dsingle)
BINARY_UFUNC(copysign, dsingle)
BINARY_UFUNC(equal, npy_bool)
BINARY_UFUNC(not_equal, npy_bool)
BINARY_UFUNC(less, npy_bool)
BINARY_UFUNC(less_equal, npy_bool)

BINARY_SCALAR_UFUNC(multiply, dsingle)
BINARY_SCALAR_UFUNC(divide, dsingle)
BINARY_SCALAR_UFUNC(power, dsingle)

#if defined(NPY_PY3K)
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "numpy_dsingle",
    NULL,
    -1,
    dsingleMethods,
    NULL,
    NULL,
    NULL,
    NULL
};
#endif

#if defined(NPY_PY3K)
PyMODINIT_FUNC PyInit_numpy_dsingle(void) {
#else
PyMODINIT_FUNC initnumpy_dsingle(void) {
#endif

    PyObject *m;
    int dsingleNum;
    PyObject* numpy = PyImport_ImportModule("numpy");
    PyObject* numpy_dict = PyModule_GetDict(numpy);
    int arg_types[3];

#if defined(NPY_PY3K)
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule("numpy_dsingle", dsingleMethods);
#endif

    if (!m) {
        return NULL;
    }

    /* Make sure NumPy is initialized */
    import_array();
    import_umath();

    /* Register the dsingle array scalar type */
#if defined(NPY_PY3K)
    PydsingleArrType_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
#else
    PydsingleArrType_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_CHECKTYPES;
#endif
    PydsingleArrType_Type.tp_new = dsingle_arrtype_new;
    PydsingleArrType_Type.tp_richcompare = gentype_richcompare;
    PydsingleArrType_Type.tp_hash = dsingle_arrtype_hash;
    PydsingleArrType_Type.tp_repr = dsingle_arrtype_repr;
    PydsingleArrType_Type.tp_str = dsingle_arrtype_str;
    PydsingleArrType_Type.tp_base = &PyGenericArrType_Type;
    if (PyType_Ready(&PydsingleArrType_Type) < 0) {
        PyErr_Print();
        PyErr_SetString(PyExc_SystemError, "could not initialize PydsingleArrType_Type");
        return NULL;
    }

    /* The array functions */
    PyArray_InitArrFuncs(&_Pydsingle_ArrFuncs);
    _Pydsingle_ArrFuncs.getitem = (PyArray_GetItemFunc*)dsingle_getitem;
    _Pydsingle_ArrFuncs.setitem = (PyArray_SetItemFunc*)dsingle_setitem;
    _Pydsingle_ArrFuncs.copyswap = (PyArray_CopySwapFunc*)dsingle_copyswap;
    _Pydsingle_ArrFuncs.copyswapn = (PyArray_CopySwapNFunc*)dsingle_copyswapn;
    _Pydsingle_ArrFuncs.compare = (PyArray_CompareFunc*)dsingle_compare;
    _Pydsingle_ArrFuncs.argmax = (PyArray_ArgFunc*)dsingle_argmax;
    _Pydsingle_ArrFuncs.nonzero = (PyArray_NonzeroFunc*)dsingle_nonzero;
    _Pydsingle_ArrFuncs.fillwithscalar = (PyArray_FillWithScalarFunc*)dsingle_fillwithscalar;
    _Pydsingle_ArrFuncs.dotfunc = (PyArray_DotFunc*)dsingle_dotfunc;

    /* The dsingle array descr */
    dsingle_descr = PyObject_New(PyArray_Descr, &PyArrayDescr_Type);
    dsingle_descr->typeobj = &PydsingleArrType_Type;
    dsingle_descr->kind = 'q';
    dsingle_descr->type = 'e';
    dsingle_descr->byteorder = '=';
    dsingle_descr->flags = 0;
    dsingle_descr->type_num = 0; /* assigned at registration */
    dsingle_descr->elsize = 4*2;
    dsingle_descr->alignment = 4;
    dsingle_descr->subarray = NULL;
    dsingle_descr->fields = NULL;
    dsingle_descr->names = NULL;
    dsingle_descr->f = &_Pydsingle_ArrFuncs;

    Py_INCREF(&PydsingleArrType_Type);
    dsingleNum = PyArray_RegisterDataType(dsingle_descr);

    if (dsingleNum < 0)
        return NULL;

    register_cast_function(NPY_BOOL, dsingleNum, (PyArray_VectorUnaryFunc*)BOOL_to_dsingle);
    register_cast_function(NPY_BYTE, dsingleNum, (PyArray_VectorUnaryFunc*)BYTE_to_dsingle);
    register_cast_function(NPY_UBYTE, dsingleNum, (PyArray_VectorUnaryFunc*)UBYTE_to_dsingle);
    register_cast_function(NPY_SHORT, dsingleNum, (PyArray_VectorUnaryFunc*)SHORT_to_dsingle);
    register_cast_function(NPY_USHORT, dsingleNum, (PyArray_VectorUnaryFunc*)USHORT_to_dsingle);
    register_cast_function(NPY_INT, dsingleNum, (PyArray_VectorUnaryFunc*)INT_to_dsingle);
    register_cast_function(NPY_UINT, dsingleNum, (PyArray_VectorUnaryFunc*)UINT_to_dsingle);
    register_cast_function(NPY_LONG, dsingleNum, (PyArray_VectorUnaryFunc*)LONG_to_dsingle);
    register_cast_function(NPY_ULONG, dsingleNum, (PyArray_VectorUnaryFunc*)ULONG_to_dsingle);
    register_cast_function(NPY_LONGLONG, dsingleNum, (PyArray_VectorUnaryFunc*)LONGLONG_to_dsingle);
    register_cast_function(NPY_ULONGLONG, dsingleNum, (PyArray_VectorUnaryFunc*)ULONGLONG_to_dsingle);
    register_cast_function(NPY_FLOAT, dsingleNum, (PyArray_VectorUnaryFunc*)FLOAT_to_dsingle);
    register_cast_function(NPY_DOUBLE, dsingleNum, (PyArray_VectorUnaryFunc*)DOUBLE_to_dsingle);
    register_cast_function(NPY_LONGDOUBLE, dsingleNum, (PyArray_VectorUnaryFunc*)LONGDOUBLE_to_dsingle);
    register_cast_function(NPY_CFLOAT, dsingleNum, (PyArray_VectorUnaryFunc*)CFLOAT_to_dsingle);
    register_cast_function(NPY_CDOUBLE, dsingleNum, (PyArray_VectorUnaryFunc*)CDOUBLE_to_dsingle);
    register_cast_function(NPY_CLONGDOUBLE, dsingleNum, (PyArray_VectorUnaryFunc*)CLONGDOUBLE_to_dsingle);

#define REGISTER_UFUNC(name)\
    PyUFunc_RegisterLoopForType((PyUFuncObject *)PyDict_GetItemString(numpy_dict, #name),\
            dsingle_descr->type_num, dsingle_##name##_ufunc, arg_types, NULL)

#define REGISTER_SCALAR_UFUNC(name)\
    PyUFunc_RegisterLoopForType((PyUFuncObject *)PyDict_GetItemString(numpy_dict, #name),\
            dsingle_descr->type_num, dsingle_##name##_scalar_ufunc, arg_types, NULL)

    /* dual -> bool */
    arg_types[0] = dsingle_descr->type_num;
    arg_types[1] = NPY_BOOL;

    REGISTER_UFUNC(isnan);
    REGISTER_UFUNC(isinf);
    REGISTER_UFUNC(isfinite);
    /* dual -> float */
    arg_types[1] = NPY_DOUBLE;

    REGISTER_UFUNC(absolute);

    /* dual -> dual */
    arg_types[1] = dsingle_descr->type_num;

    REGISTER_UFUNC(log);
    REGISTER_UFUNC(exp);
    REGISTER_UFUNC(negative);
    REGISTER_UFUNC(conjugate);

    /* dual, dual -> bool */

    arg_types[2] = NPY_BOOL;

    REGISTER_UFUNC(equal);
    REGISTER_UFUNC(not_equal);
    REGISTER_UFUNC(less);
    REGISTER_UFUNC(less_equal);

    /* dual, float -> dual */

    arg_types[1] = NPY_DOUBLE;
    arg_types[2] = dsingle_descr->type_num;

    REGISTER_SCALAR_UFUNC(multiply);
    REGISTER_SCALAR_UFUNC(divide);
    REGISTER_SCALAR_UFUNC(power);

    /* dual, dual -> dual */

    arg_types[1] = dsingle_descr->type_num;

    REGISTER_UFUNC(add);
    REGISTER_UFUNC(subtract);
    REGISTER_UFUNC(multiply);
    REGISTER_UFUNC(divide);
    REGISTER_UFUNC(power);
    REGISTER_UFUNC(copysign);

    PyModule_AddObject(m, "dsingle", (PyObject *)&PydsingleArrType_Type);

    return m;
}
