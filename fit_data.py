#!/usr/bin/env python
# coding: utf-8

import os
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import scipy.stats
from scipy.optimize import curve_fit

import import_data as imd


eps = 10**-9
def inv_saturation(y, a, b, n):
    y_ = 1- (y-n)/a
    y_ = np.where(y_ > eps, y_, eps)
    return -b*np.log(y_)

def saturation(x, a, b, n):
    """
        a: max concentration
        b: linear range (more or less!)
        n: mean noise
    """
    return n+a*(1-np.exp(-x/b))

def saturation_sig(x, a, b, n):
    """
        a: max concentration
        b: linear range (more or less!)
        n: mean noise
    """
    return n+a/(1+np.exp((c-x)/b))

def inv_saturation_sig2(y, a, b, n):
    """
        a: max concentration
        b: linear range (more or less!)
        n: mean noise
    """
    return b/(1 - (y-n)/a) - b

def saturation_sig2(x, a, b, n):
    """
        a: max concentration
        b: linear range (more or less!)
        n: mean noise
    """
    return n+a*(1-b/(b+x))

# def inv_saturation(y, a, b, n):
#     y_ = 1- (y-n)/a
#     y_ = np.where(y_ > eps, y_, eps)
#     return -b*np.log(y_)

# def saturation(x, a, b, n):
#     """
#         a: max concentration
#         b: linear range (more or less!)
#         n: mean noise
#     """
#     return a/(1+np.exp(-x/b))



def make_fit(x_, y):
    X = x_.reshape(-1, 1)
    #ly = -np.log(np.where(y > eps, y, eps))
    # reg = LinearRegression().fit(X, y, sample_weight=1/(x_+1))
    samps_w = 1/(x_+1)
    samps_w[x_ > 20] = 0 #10**-6
    samps_w[x_ < 0.1] = 0 #10**-6
    reg = LinearRegression().fit(X, y, sample_weight=samps_w)
    # commenting sklearn's own r2 as it doesn't seem to correspond to the usual squared pearson correlation coefficient.
    # TODO: to be further investigated...
    y_l = reg.predict(X)
    # score = reg.score(X, y)
    # score = np.corrcoef(y_l, y)[1, 0]**2
    score = scipy.stats.pearsonr(y_l, y)[0]**2
    # keep n around 0 value
    y0 = y[x_ < eps]
    if len(y0 > 0):
        n0 = max(np.mean(y0), 1)
    else:
        n0 = max(min(y), 1)
    a0 = max(y)
    b0 = 20
#     print(x_)
#     print(y)
#     print(saturation(x_, 10**7, 2000, 10**4))
#     print((saturation(x_, 10**7, 2000, 10**4)-y)**2)
    #coefs, _ = curve_fit(saturation, x_, y, p0=(10**7, 200, 10**3),
    #                     bounds=([10**5, 10, 100], [10**12, 10**5, 10**5]),

    try:
        coefs, _ = curve_fit(saturation, x_, y, p0=(a0, b0, n0),
                             bounds=([a0/3, 10, n0/4], [a0*10**5, 10**4, n0*4]),
                             #sigma=(x_+0.1)
                             sigma=(y+1)/2, #np.sqrt(np.abs(y))+10,#(x_+1)*100,#np.log(y**2),#np.sqrt(y+eps),#(y+eps)/100,
                            )
        # coefs, _ = curve_fit(saturation_sig2, x_, y,
        #                      p0=(a0, b0, n0),
        #                      bounds=([10, 10, 10], [10**9,10**9, 10**9]),
        #                      sigma=(np.abs(y)+1)/100
        #                     )
    except:
        coefs = np.array([a0, b0, n0])
    e_R2, _ = scipy.stats.pearsonr(y, saturation(x_, *coefs))
    llod = np.nan
    hlod = np.nan
    sn_ratio = np.nan
    Drange = np.nan
    try:
        if reg.coef_[0] > 0:
            noise = coefs[2]
            llod = inv_saturation(noise*2, *coefs)
            llod = min(llod, max(x_))
            llod = max(llod, 0.001)
            hlod = inv_saturation(min(max(y),coefs[0])*0.8, *coefs)
            hlod = min(hlod, max(x_))
            hlod = max(hlod, 10)
            Drange = np.log10(hlod) - np.log10(llod)
            # no point in having a negative range
            Drange = max(0, Drange)
            sn_ratio = np.log((saturation(10, *coefs)) / noise)
#             # get the points at 0
#             y0 = max(eps, y[x_ < 10**-3].mean())
#             # compute an initial LLOD: the points you expect to be around the 0 level, based on the model
#             ini_llod = inv_saturation(y0, *coefs)
#             # get the average and std for all points below that
#             y0 = max(eps, y[x_ < ini_llod].mean())
#             y0_std = max(eps, y[x_ < ini_llod].std())
#             llod = inv_saturation(y0+y0_std, *coefs)
#             #llod = y0 / reg.coef_[0]
#             #sn_ratio = saturation(20, *coefs) / y0
    except:
        pass
    return reg.coef_[0], reg.intercept_, score, llod, hlod, coefs[0], coefs[1], coefs[2], e_R2, sn_ratio, Drange

def make_auto_fit(x, y):
    yf = pd.notnull(y)
    yf = yf & (y != np.inf)
    y = y[yf]
    ys = y.sum()
    if ys <= 0 or len(y) < 3:
        raise ValueError
        #return None
    x_ = x[yf]
    return make_fit(x_, y)

def make_fits(sum_data):
    fits = list()
    idxs = list()
    for gn, g in sum_data.groupby(level=(1)):
        x_ = np.array(g.index.get_level_values(2))
        for col in sum_data.filter(like="Area"):
            y = g[col]
            try:
                fit_data = make_auto_fit(x_, y)
            except:
                fit_data = [np.nan, ]*11
            fits.append((gn, col, *fit_data))
    fits = pd.DataFrame(data=fits, columns=["molecule", "method", "coef", "intercept", "R2", "LLOD", "HLOD", "e_sat", "e_coef", "e_n", "e_R2", "SN_ratio", "Drange"])
    #fits["Drange"] = fits["e_coef"].apply(lambda x: np.log10(x))
    fits.set_index(["molecule", "method"], inplace=True)
    return fits

def make_all_fits(sum_data):
    fits = dict()
    for m, d in sum_data.items():
        fit = make_fits(d)
        fits[m] = fit
    return fits, pd.concat(fits, names=["Mode", ])


