#!/usr/bin/python3

import numpy as np


def interpolate(v0, n1):
    xb = np.linalg.inv(np.array([[1, -1, 1], [1, 0, 0], [1, 1, 1]]))
    xa = np.linalg.inv(np.array([[1, 0, 0], [1, 1, 1], [1, 2, 4]]))
    # print(xa)
    # print(xb)
    n0 = v0.shape[0]
    r = n0/n1
    v1 = np.empty(n1)
    old = -1
    vf = np.dot(xa, np.transpose(v0[0:3]))
    vl = np.dot(xb, np.transpose(v0[n0-3:n0]))
    for i in range(n1):
        ix = (i+0.5)*r-0.5
        new = int(np.floor(ix))
        if new < 1:
            vx = np.array([1.0, ix, ix*ix])
            v1[i] = np.dot(vx, vf)
        elif new > n0-3:
            s = ix-n0+2
            vx = np.array([1.0, s, s*s])
            v1[i] = np.dot(vx, vl)
        else:
            s = ix-new
            vx = np.array([1.0, s, s])
            if new != old:
                vb = np.dot(xb, np.transpose(v0[new-1:new+2]))
                va = np.dot(xa, np.transpose(v0[new:new+3]))
            # print(np.dot(v, vb), np.dot(v, va))
            v1[i] = np.dot(vx, vb)*(ix-new)+np.dot(vx, va)*(new+1.0-ix)
        # print(new, i, ix, v1[i])
    return v1


v = np.array(range(5))
print(interpolate(v, 13))
