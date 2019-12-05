import numpy as np

def interpolate(in_vec, in_ts, out_ts):
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
    win_inv = np.linalg.inv(win_mat)
    win_fn = np.dot(win_inv, in_vec[0:4])
    for i in range(4):
        vec = in_pow[i,:]
        # print(in_vec[i], np.dot(vec, win_fn))
                       
    wix = 0
    for i in range(out_len):
        if wix+4 < in_len and out_ts[i] > in_ts[wix+2]:
            while wix+4 < in_len and out_ts[i] > in_ts[wix+2]:
                wix += 1
            win_mat = in_pow[wix:wix+4,:]
            win_inv = np.linalg.inv(win_mat)
            win_fn = np.dot(win_inv, in_vec[wix:wix+4])
        v = np.dot(out_pow[i,:], win_fn)
        if v<lo:
            out_vec[i] = lo
        elif v>hi:
            out_vec[i] = hi
        else:
            out_vec[i] = v
    return out_vec
