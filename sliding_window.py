

import numpy as np
from numpy.lib.stride_tricks import as_strided as ast

def normalize_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.
    Parameters
        shape - an int, or a tuple of ints
    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass

    raise TypeError('shape must be an int, or a tuple of ints')

def sliding_window(arr,win,sl = None,flatten = True):
    '''
    Return a sliding window over a in any number of dimensions
    Parameters:
        arr  - an n-dimensional numpy array
        win - an int (arr is 1D) or tuple (arr is 2D or greater) representing the size
             of each dimension of the window
        sl - an int (arr is 1D) or tuple (arr is 2D or greater) representing the
             amount to slide the window in each dimension. If not specified, it
             defaults to win.
        flatten - if True, all slices are flattened, otherwise, there is an
                  extra dimension for each dimension of the input.
    Returns
        an array containing each n-dimensional window from arr
    '''

    if None is sl:
        # sl was not provided. the windows will not overlap in any direction.
        sl = win
    win = normalize_shape(win)
    sl = normalize_shape(sl)

    # convert win, sl, and arr.shape to numpy arrays so that we can do math in every
    # dimension at once.
    win = np.array(win)
    sl = np.array(sl)
    shape = np.array(arr.shape)


    # ensure that win, sl, and arr.shape all have the same number of dimensions
    ls = [len(shape),len(win),len(sl)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'arr.shape, win and sl must all have the same length. They were %s' % str(ls))

    # ensure that win is smaller than arr in every dimension
    if np.any(win > shape):
        raise ValueError(\
        'win cannot be larger than arr in any dimension.\
 arr.shape was %s and win was %s' % (str(arr.shape),str(win)))

    # how many slices will there be in each dimension?
    n_shape = normalize_shape(((shape - win) // sl) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    n_shape += normalize_shape(win)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstr = normalize_shape(np.array(arr.strides) * sl) + arr.strides
    strided = ast(arr,shape = n_shape,strides = newstr)
    if not flatten:
        return strided

    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    m = len(win) if win.shape else 0
    fdim = (np.product(n_shape[:-m]),) if win.shape else ()
    dim = fdim + (n_shape[-m:])
    # remove any dimensions with size 1
    #dim = filter(lambda i : i != 1,dim)
    return strided.reshape(dim)
