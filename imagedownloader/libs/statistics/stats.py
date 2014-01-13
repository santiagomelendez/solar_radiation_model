# This file contains a fraction of the scipy library to simplify a complex
# instrallation process inside the Makefile. The entire source code of this
# file can be downloaded from the next URL:
# https://github.com/scipy/scipy/blob/v0.13.0/scipy/stats/stats.py

import numpy as np
from numpy import array

def scoreatpercentile(a, per, limit=(), interpolation_method='fraction',
        axis=None):
    """
    Calculate the score at a given percentile of the input sequence.

    For example, the score at `per=50` is the median. If the desired quantile
    lies between two data points, we interpolate between them, according to
    the value of `interpolation`. If the parameter `limit` is provided, it
    should be a tuple (lower, upper) of two values.

    Parameters
    ----------
    a : array_like
        A 1-D array of values from which to extract score.
    per : array_like
        Percentile(s) at which to extract score.  Values should be in range
        [0,100].
    limit : tuple, optional
        Tuple of two scalars, the lower and upper limits within which to
        compute the percentile. Values of `a` outside
        this (closed) interval will be ignored.
    interpolation : {'fraction', 'lower', 'higher'}, optional
        This optional parameter specifies the interpolation method to use,
        when the desired quantile lies between two data points `i` and `j`

          - fraction: ``i + (j - i) * fraction`` where ``fraction`` is the
            fractional part of the index surrounded by ``i`` and ``j``.
          - lower: ``i``.
          - higher: ``j``.

    axis : int, optional
        Axis along which the percentiles are computed. The default (None)
        is to compute the median along a flattened version of the array.

    Returns
    -------
    score : float (or sequence of floats)
        Score at percentile.

    See Also
    --------
    percentileofscore

    Examples
    --------
    >>> from scipy import stats
    >>> a = np.arange(100)
    >>> stats.scoreatpercentile(a, 50)
    49.5

    """
    # adapted from NumPy's percentile function
    a = np.asarray(a)

    if limit:
        a = a[(limit[0] <= a) & (a <= limit[1])]

    if per == 0:
        return a.min(axis=axis)
    elif per == 100:
        return a.max(axis=axis)

    sorted = np.sort(a, axis=axis)
    if axis is None:
        axis = 0

    return _compute_qth_percentile(sorted, per, interpolation_method, axis)


# handle sequence of per's without calling sort multiple times
def _compute_qth_percentile(sorted, per, interpolation_method, axis):
    if not np.isscalar(per):
        return [_compute_qth_percentile(sorted, i, interpolation_method, axis)
             for i in per]

    if (per < 0) or (per > 100):
        raise ValueError("percentile must be in the range [0, 100]")

    indexer = [slice(None)] * sorted.ndim
    idx = per / 100. * (sorted.shape[axis] - 1)

    if int(idx) != idx:
        # round fractional indices according to interpolation method
        if interpolation_method == 'lower':
            idx = int(np.floor(idx))
        elif interpolation_method == 'higher':
            idx = int(np.ceil(idx))
        elif interpolation_method == 'fraction':
            pass  # keep idx as fraction and interpolate
        else:
            raise ValueError("interpolation_method can only be 'fraction', "
                             "'lower' or 'higher'")

    i = int(idx)
    if i == idx:
        indexer[axis] = slice(i, i + 1)
        weights = array(1)
        sumval = 1.0
    else:
        indexer[axis] = slice(i, i + 2)
        j = i + 1
        weights = array([(j - idx), (idx - i)], float)
        wshape = [1] * sorted.ndim
        wshape[axis] = 2
        weights.shape = wshape
        sumval = weights.sum()

    # Use np.add.reduce to coerce data type
    return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval