"""
Helper functions for envelop detection
"""
import numpy as np
from scipy.interpolate import interp1d

def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s)
        # pre-sorting of locals min based on relative position with respect to s_mid
        lmin = lmin[s[lmin] < s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid
        lmax = lmax[s[lmax] > s_mid]

    # global min of dmin-chunks of locals min
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]])
                 for i in range(0, len(lmin), dmin)]]
    # global max of dmax-chunks of locals max
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]])
                 for i in range(0, len(lmax), dmax)]]
    return lmin, lmax

def get_envelope(x, y):
    x_list, y_list = list(x), list(y)
    assert len(x_list) == len(y_list)

    # First data
    ui, ux, uy = [0], [x_list[0]], [y_list[0]]
    li, lx, ly = [0], [x_list[0]], [y_list[0]]

    # Find upper peaks and lower peaks
    for i in range(1, len(x_list)-1):
        if y_list[i] >= y_list[i-1] and y_list[i] >= y_list[i+1]:
            ui.append(i)
            ux.append(x_list[i])
            uy.append(y_list[i])
        if y_list[i] <= y_list[i-1] and y_list[i] <= y_list[i+1]:
            li.append(i)
            lx.append(x_list[i])
            ly.append(y_list[i])

    # Last data
    ui.append(len(x_list)-1)
    ux.append(x_list[-1])
    uy.append(y_list[-1])
    li.append(len(y_list)-1)
    lx.append(x_list[-1])
    ly.append(y_list[-1])

    if len(ux) == 2 or len(lx) == 2:
        return [], []

    else:
        func_ub = interp1d(ux, uy, kind='cubic', bounds_error=False)
        func_lb = interp1d(lx, ly, kind='cubic', bounds_error=False)

        ub, lb = [], []
        for i in x_list:
            ub = func_ub(x_list)
            lb = func_lb(x_list)

        ub = np.array([y, ub]).max(axis=0)
        lb = np.array([y, lb]).min(axis=0)

        return ub, lb
