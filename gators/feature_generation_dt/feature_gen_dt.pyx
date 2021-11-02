import cython
import numpy as np

cimport numpy as np
from libc.math cimport cos, isnan, pi, sin



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] ordinal_datetime(
        np.ndarray[object, ndim=2] X, np.ndarray[np.int64_t, ndim=1] bounds):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = X.shape[1]
    cdef int start_idx = bounds[0]
    cdef int end_idx = bounds[1]
    cdef np.ndarray[np.float64_t, ndim=2] X_new = np.empty((n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            if X[i, j] is None:
                X_new[i, j] = np.nan
            else:
                X_new[i, j] = float(X[i, j][start_idx: end_idx])
    return X_new


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] ordinal_day_of_week(
        np.ndarray[object, ndim=2] X,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = X.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] X_new = (X.astype('datetime64[D]').astype('float64') - 4) % 7
    for i in range(n_rows):
        for j in range(n_cols):
            if X[i, j] is None:
                X_new[i, j] = np.nan
    return X_new


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] cyclic_minute_of_hour(
        np.ndarray[object, ndim=2] X,
        double PREFACTOR,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = X.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] X_new = np.empty((n_rows, 2 * n_cols))
    cdef double val
    for i in range(n_rows):
        for j in range(n_cols):
            if X[i, j] is None:
                X_new[i, 2*j] = np.nan
                X_new[i, 2*j+1] = np.nan
            else:
                val = PREFACTOR * float(X[i, j][14: 16])
                X_new[i, 2*j] = cos(val)
                X_new[i, 2*j+1] = sin(val)
    return X_new


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] cyclic_hour_of_day(
        np.ndarray[object, ndim=2] X,
        double PREFACTOR,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = X.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] X_new = np.empty((n_rows, 2 * n_cols))
    cdef double val
    for i in range(n_rows):
        for j in range(n_cols):
            if X[i, j] is None:
                X_new[i, 2*j] = np.nan
                X_new[i, 2*j+1] = np.nan
            else:
                val = PREFACTOR * float(X[i, j][11: 13])
                X_new[i, 2*j] = cos(val)
                X_new[i, 2*j+1] = sin(val)
    return X_new


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] cyclic_day_of_week(
        np.ndarray[object, ndim=2] X,
        double PREFACTOR,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = X.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] X_new = np.empty((n_rows, 2 * n_cols))
    cdef np.ndarray[np.float64_t, ndim=2] days_of_week = (X.astype('datetime64[D]').astype('float64') - 4) % 7
    cdef double val
    for i in range(n_rows):
        for j in range(n_cols):
            if X[i, j] is None:
                X_new[i, 2*j] = np.nan
                X_new[i, 2*j+1] = np.nan
            else:
                val = PREFACTOR * float(days_of_week[i, j])
                X_new[i, 2*j] = cos(val)
                X_new[i, 2*j+1] = sin(val)
    return X_new


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] cyclic_month_of_year(
        np.ndarray[object, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] bounds,
        double PREFACTOR,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = X.shape[1]
    cdef int start_idx = bounds[0]
    cdef int end_idx = bounds[1]
    cdef np.ndarray[np.float64_t, ndim=2] X_new = np.empty((n_rows, 2 * n_cols))
    cdef double val
    for i in range(n_rows):
        for j in range(n_cols):
            if X[i, j] is None:
                X_new[i, 2*j] = np.nan
                X_new[i, 2*j+1] = np.nan
            else:
                val = PREFACTOR * (float(X[i, j][start_idx: end_idx]) -1.)
                X_new[i, 2*j] = cos(val)
                X_new[i, 2*j+1] = sin(val)
    
    return X_new


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] cyclic_day_of_month(
        np.ndarray[object, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] bounds,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = X.shape[1]
    cdef int start_idx = bounds[0]
    cdef int end_idx = bounds[1]
    cdef np.ndarray[np.float64_t, ndim=2] X_new = np.empty((n_rows, 2 * n_cols))
    cdef np.ndarray[np.int64_t, ndim=2] days_in_month = np.empty((n_rows, n_cols), dtype=np.int64)
    days_in_month = ((X.astype('datetime64[M]')+1).astype('datetime64[D]') - X.astype('datetime64[M]')) // np.timedelta64(1, 'D') - 1

    for i in range(n_rows):
        for j in range(n_cols):
            if X[i, j] is None:
                X_new[i, 2*j] = np.nan
                X_new[i, 2*j+1] = np.nan
            else:
                val = 2 * pi * (float(X[i, j][start_idx: end_idx]) -1.) / days_in_month[i, j]
                X_new[i, 2*j] = cos(val)
                X_new[i, 2*j+1] = sin(val)
    return X_new


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] deltatime(
        np.ndarray[np.float64_t, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] idx_columns_a,
        np.ndarray[np.int64_t, ndim=1] idx_columns_b,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = idx_columns_a.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] X_new = np.empty((n_rows, n_cols))
    cdef float val_a
    cdef float val_b
    for i in range(n_rows):
        for j in range(n_cols):
            if (X[i, idx_columns_a[j]]  < 0) or (X[i, idx_columns_b[j]] < 0):
                X_new[i, j] = np.nan
            else:
                val_a = X[i, idx_columns_a[j]]
                val_b = X[i, idx_columns_b[j]]
                X_new[i, j] = (val_a - val_b)
    return X_new



# @cython.boundscheck(False)
# @cython.wraparound(False)
# cpdef np.ndarray[np.float64_t, ndim=2] ordinal_hour_of_day(np.ndarray[object, ndim=2] X):
#     cdef int i
#     cdef int j
#     cdef int n_rows = X.shape[0]
#     cdef int n_cols = X.shape[1]
#     cdef np.ndarray[np.float64_t, ndim=2] X_new = np.empty((n_rows, n_cols))
#     for i in range(n_rows):
#         for j in range(n_cols):
#             if X[i, j] is None:
#                 X_new[i, j] = np.nan
#             else:
#                 X_new[i, j] = float(X[i, j][11: 13])
#     return X_new

   



# @cython.boundscheck(False)
# @cython.wraparound(False)
# cpdef np.ndarray[np.float64_t, ndim=2] ordinal_day_of_month(
#         np.ndarray[object, ndim=2] X,
# ):
#     cdef int i
#     cdef int j
#     cdef int n_rows = X.shape[0]
#     cdef int n_cols = X.shape[1]
#     cdef np.ndarray[np.float64_t, ndim=2] X_new = np.empty((n_rows, n_cols))
#     cdef np.ndarray[object, ndim=1] dummy = np.empty(n_rows, object)
#     for j in range(n_cols):
#         for i in range(n_rows):
#             X_new[i, j] = float(X[i, j].day)
#     return X_new


# @cython.boundscheck(False)
# @cython.wraparound(False)
# cpdef np.ndarray[np.float64_t, ndim=2] ordinal_month_of_year(
#         np.ndarray[object, ndim=2] X,
# ):
#     cdef int i
#     cdef int j
#     cdef int n_rows = X.shape[0]
#     cdef int n_cols = X.shape[1]
#     cdef np.ndarray[np.float64_t, ndim=2] X_new = np.empty((n_rows, n_cols))
#     cdef np.ndarray[object, ndim=1] dummy = np.empty(n_rows, object)
#     for i in range(n_rows):
#         for j in range(n_cols):
#             X_new[i, j] = float(X[i, j].month)
#     return X_new
