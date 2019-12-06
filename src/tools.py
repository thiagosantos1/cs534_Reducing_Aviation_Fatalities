import numpy as np

ep = 0.0001

def slope(val, fn):
    vec = np.empty(4)
    vec[0] = 0.0
    vec[1] = 1.0
    vec[2] = 2.0*val
    vec[3] = 3.0*val*val
    return np.dot(vec, fn)


def delta(val, fn):
    vec1 = np.empty(4)
    vec1[0] = 1.0
    vec1[1] = (val+ep)
    vec1[2] = vec1[1]*(val+ep)
    vec1[3] = vec1[2]*(val+ep)
    y1 = np.dot(vec1, fn)
    vec2 = np.empty(4)
    vec2[0] = 1.0
    vec2[1] = (val-ep)
    vec2[2] = vec2[1]*(val-ep)
    vec2[3] = vec2[2]*(val-ep)
    y2 = np.dot(vec2, fn)
    return (y1-y2)/(2.0*ep)


def int_cubic(in_vec, in_ts, out_ts):
    # interpolation using a local 3rd degree polynomial
    in_len=in_ts.shape[0]
    out_len=out_ts.shape[0]
    in_pow = np.empty((in_len, 4))
    out_pow = np.empty((out_len, 4))
    in_pow[:,0] = np.ones(in_len)
    out_pow[:,0] = np.ones(out_len)
    for i in range(1,4):
        in_pow[:,i] = in_pow[:,i-1]*in_ts
        out_pow[:,i] = out_pow[:,i-1]*out_ts
    lo = in_vec.min()
    hi = in_vec.max()
    out_vec = np.empty(out_len)
    win_mat = in_pow[0:4,:]
    win_fn = np.linalg.solve(win_mat, in_vec[0:4])
                       
    wix = 0
    for i in range(out_len):
        if wix+4 < in_len and out_ts[i] > in_ts[wix+2]:
            while wix+4 < in_len and out_ts[i] > in_ts[wix+2]:
                wix += 1
            win_mat = in_pow[wix:wix+4,:]
            win_fn = np.linalg.solve(win_mat, in_vec[wix:wix+4])
        v = np.dot(out_pow[i,:], win_fn)
        if v<lo:
            out_vec[i] = lo
        elif v>hi:
            out_vec[i] = hi
        else:
            out_vec[i] = v
    return out_vec


def int_spline(in_vec, in_ts, out_ts):
    # interpolation using a cubic spline
    in_len=in_ts.shape[0]
    out_len=out_ts.shape[0]

    din_vec = np.empty(in_len)
    din_vec[0] = (in_vec[1]-in_vec[0])/(in_ts[1]-in_ts[0])
    din_vec[1:in_len-1] = (in_vec[2:]-in_vec[:in_len-2])/(in_ts[2:]-in_ts[:in_len-2])
    din_vec[in_len-1] = (in_vec[in_len-1]-in_vec[in_len-2])/(in_ts[in_len-1]-in_ts[in_len-2])
    in_pow = np.empty((in_len, 4))
    out_pow = np.empty((out_len, 4))
    in_pow[:,0] = np.ones(in_len)
    out_pow[:,0] = np.ones(out_len)
    for i in range(1,4):
        in_pow[:,i] = in_pow[:,i-1]*in_ts
        out_pow[:,i] = out_pow[:,i-1]*out_ts

    din_pow = np.empty((in_len, 4))
    din_pow[:,0] = np.zeros(in_len)
    din_pow[:,1] = np.ones(in_len)
    din_pow[:,2] = 2*in_ts
    din_pow[:,3] = 3*in_ts*in_ts

    lo = in_vec.min()
    hi = in_vec.max()
    out_vec = np.empty(out_len)
    win_mat = np.empty((4, 4))
    win_mat[0:2,:] = in_pow[0:2,:]
    win_mat[2:4,:] = din_pow[0:2,:]
    win_vec = np.empty((4,))
    win_vec[0:2] = in_vec[0:2]
    win_vec[2:4] = din_vec[0:2]
    win_fn = np.linalg.solve(win_mat, win_vec)
          
    wix = 0
    for i in range(out_len):
        if wix+2 < in_len and out_ts[i] > in_ts[wix+1]:
            while wix+2 < in_len and out_ts[i] > in_ts[wix+1]:
                wix += 1
            win_mat[0:2,:] = in_pow[wix:wix+2,:]
            win_mat[2:4,:] = din_pow[wix:wix+2,:]
            win_vec[0:2] = in_vec[wix:wix+2]
            win_vec[2:4] = din_vec[wix:wix+2]
            win_fn = np.linalg.solve(win_mat, win_vec)
        v = np.dot(out_pow[i,:], win_fn)
        if v<lo:
            out_vec[i] = lo
        elif v>hi:
            out_vec[i] = hi
        else:
            out_vec[i] = v
    return out_vec


def lognorm2norm(v):
    # Take waht is assumed to be lognormal distributed data and
    # convert it to a normal distribution with a mean of 0 and a
    # variance of 1
    norm = np.log(v)
    norm = norm-np.mean(norm)
    return norm/np.sqrt(np.var(norm, ddof=1))
