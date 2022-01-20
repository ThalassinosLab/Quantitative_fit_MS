#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import scipy.stats
from scipy.optimize import curve_fit


prec_list = ["precursor", "precursor [M+1]", "precursor [M+2]"]
iso_corrs = {
        # metabolites
        "1-Butyl-3-(1-naphthoyl)indole": dict(zip(prec_list, [np.nan, 0.256, 0.0334])),
        "1-Butyl-3-(1-naphthoyl)indole-D5": dict(zip(prec_list, [np.nan, 0.2554, 0.0332])),
        "2-(2-Methoxyphenyl)-1-(1-pentylindol-3-yl)ethanone": dict(zip(prec_list, [np.nan, 0.246, 0.0331])),
        "2-(2-Methoxyphenyl)-1-(1-pentylindol-3-yl)ethanone-D5": dict(zip(prec_list, [np.nan, 0.245, 0.0329])),
        "3-(4-Methylnaphthalene-1-carbonyl)-1-pentylindole": dict(zip(prec_list, [np.nan, 0.278, 0.0392])),
        "3-(4-Methylnaphthalene-1-carbonyl)-1-pentylindole-D5": dict(zip(prec_list, [np.nan, 0.2775, 0.0391])),
        "N-(1-Adamantyl)-1-pentyl-1H-indazole-3-carboxamide": dict(zip(prec_list, [np.nan, 0.2643, 0.03545796])),
        "N-(1-Adamantyl)-1-pentyl-1H-indazole-3-carboxamide-D4": dict(zip(prec_list, [np.nan, 0.2649, 0.03560])),
        # peptides
        "AEFVEVTK": dict(zip(prec_list, [1, 0.5, 0.151])),
        "AVDDFLISLDGTANK": dict(zip(prec_list, [1, 0.829, 0.392])),
        "DIVGAVLK": dict(zip(prec_list, [1, 0.444, 0.119])),
        "EALDFFAR": dict(zip(prec_list, [1, 0.540, 0.170])),
        "HLVDEPQNLIK": dict(zip(prec_list, [1, 0.705, 0.280])),
        "IGDYAGIK": dict(zip(prec_list, [1, 0.455, 0.126])),
        "NLAENISR": dict(zip(prec_list, [1, 0.461, 0.133])),
        "LLSYVDDEAFIR": dict(zip(prec_list, [1, 0.789, 0.340])),
        "VIFLENYR": dict(zip(prec_list, [1, 0.598, 0.203])),
        "NVNDVIAPAFVK": dict(zip(prec_list, [1, 0.710, 0.284])),
        "VNQIGTLSESIK": dict(zip(prec_list, [1, 0.668, 0.261])),
        "LVNELTEFAK": dict(zip(prec_list, [1, 0.634, 0.232])),
         # XL
        "DTHK[1514.77]SEIAHR_FK[1458.75]DLGEEHFK": dict(zip(prec_list, [0.759, 1, 0.735])),
        "LAK[1266.74]EYEATLEEC[160.03]C[160.03]AK_ALK[1965.94]AWSVAR": dict(zip(prec_list, [0.709, 1, 0.815])),
        "C[160.03]C[160.03]TK[1082.64]PESERM[147.03]PC[160.13]TEDYLSLILNR_SLGK[2966.40]VGTR": dict(zip(prec_list, [0.498, 0.912, 1])),
}
def get_icorr(x, y):
    m = iso_corrs.get(x)
    if m is not None:
        return m.get(y, np.nan)
    return np.nan


def get_ppb(s):
    i = s.find("ppb")
    # if not found
    if i == -1:
        return 0
    f = s[i-5:i].split()[-1]
    return float(f)


def is_precursor(s):
    return "precursor" in s.lower()


def correct_vals(t, corrs):
    # add check for saturation
    oc = ["precursor", "precursor [M+1]", "precursor [M+2]"]
    #filt = t["ppb"] > 100
    for c, cor in zip(oc, corrs):
        filt = t["Fragment Ion"] == c
        ci = "Area_isocorr"
        # multiply value based on relative isotope abundance (m+1 m+2)
        d.loc[filt, ci] = t.loc[filt, "Area"] / cor
        cb = "Area_blanksub"
        # what does this do?
        t.loc[filt, cb] = t[ci] - t[ci][t["ppb"] < 10**-6].mean()
    return t


def import_data_met(methods=["DDA", "MSe", "SONAR", "MS", "TofMRM",]):
    ori_data = dict()
    for method in methods:
        #a = pd.read_csv(f"Results_Waters_skyline/met_{method}.csv")
        a = pd.read_csv(f"Results_Waters_skyline/met_{method}.csv")
        #a = pd.read_csv(f"/Users/ct001/Downloads/transition results acquistion project - 170619/metabolites/metabolites - {method} - Transition Results.csv")
        a.drop(["Precursor Mz", "Precursor Charge", "Product Charge", "Product Mz", "Retention Time"], axis=1, inplace=True)
        a = a[a["Replicate"].apply(lambda x: "1a" not in x)]
        a = a[a["Replicate"].apply(lambda x: "1b" not in x)]
        a = a[a["Replicate"].apply(lambda x: "2a" not in x)]
        a = a[a["Replicate"].apply(lambda x: "2b" not in x)]
        a = a[a["Replicate"].apply(lambda x: "4a" not in x)]
        a = a[a["Replicate"].apply(lambda x: "4b" not in x)]
        a = a[a["Replicate"].apply(lambda x: "5a" not in x)]
        a = a[a["Replicate"].apply(lambda x: "5b" not in x)]

        if method == "MSe":# or method == "DDA" or method == "HDDDA":
            a = a[a["Replicate"].apply(lambda x: " 001" not in x)]
            a = a[a["Replicate"].apply(lambda x: " 002" not in x)]

        a["ppb"] = a["Replicate"].apply(get_ppb)
        a["Replicate"] = a["Replicate"].apply(lambda x: x.split()[-1])
        a["precursor"] = a["Fragment Ion"].apply(is_precursor)
        for ai, ar in a.iterrows():
            a.loc[ai, "isocorr"] = get_icorr(ar["Peptide"], ar["Fragment Ion"])

        # compute blank (corresponding 0 ppb value)
        for an, ag in a.groupby(["Peptide", "Replicate", "Fragment Ion"]):
            blank = ag["Area"][ag["ppb"] < 10**-3].min()
            blank_bg = ag["Background"][ag["ppb"] < 10**-3].min()
            #if len(blank) != 1:
            #    continue
            #blank = float(blank)
            for ai, ar in ag.iterrows():
                a.loc[ai, "blank"] =  blank
                a.loc[ai, "blank_bg"] =  blank-blank_bg

        #a.fillna(0, inplace=True)

        a["Area_bg_corrected"] = a["Area"] - a["Background"]
        a.loc[a["Area_bg_corrected"] < 0, "Area_bg_corrected"] = 0
        a["Area_blank_corrected"] = a["Area"] - a["blank"]
        a.loc[a["Area_blank_corrected"] < 0, "Area_blank_corrected"] = 0
        a["Area_both_corrected"] = a["Area"] - a["Background"] - a["blank_bg"]
        a.loc[a["Area_both_corrected"] < 0, "Area_both_corrected"] = 0
        a["Area_isocorr"] = a["Area"] / a["isocorr"]
        ori_data[method] = a
    return ori_data

def import_data_pep(methods=["DDA", "MSe", "SONAR", "MS",]):
    ori_data = dict()
    for method in methods:
        #a = pd.read_csv(f"Results_Waters_skyline/met_{method}.csv")
        a = pd.read_csv(f"Results_Waters_skyline/pep_{method}.csv")
        #a = pd.read_csv(f"/Users/ct001/Downloads/transition results acquistion project - 170619/peptides/peptides - {method} - Transition Results.csv")
        a.drop(["Precursor Mz", "Precursor Charge", "Product Charge", "Product Mz", "Retention Time"], axis=1, inplace=True)
        # 10 amol to 1 pmol
        base_c = [0.045, 0.45, 4.5, 45, 450, 4500]
        # ppbs = [0, 0, ] + base_c + [0, 0] + base_c
        if method == "MS":# or method == "DDA" or method == "HDDDA":
            a = a[a["Replicate"].apply(lambda x: "008" not in x)]
            ppbs = [0, ] + base_c + [0, 0] + base_c
        else:
            a = a[a["Replicate"].apply(lambda x: "001" not in x)]
            a = a[a["Replicate"].apply(lambda x: "009" not in x)]
            ppbs = [0, 0, ] + base_c + [0, 0, ] + base_c
        # get number from 1 to 15, get corresponding concentration

        a["ppb"] = a["Replicate"].apply(lambda x: ppbs[int(x[-3:])-1])
        a["Replicate"] = a["Replicate"].apply(lambda x: x.split()[-1])
        a["precursor"] = a["Fragment Ion"].apply(is_precursor)
        for ai, ar in a.iterrows():
            a.loc[ai, "isocorr"] = get_icorr(ar["Peptide"], ar["Fragment Ion"])
        # compute blank (corresponding 0 ppb value)
        for an, ag in a.groupby(["Peptide", "Fragment Ion"]):
            blank = ag["Area"][ag["ppb"] < 10**-3].min()
            blank_bg = ag["Background"][ag["ppb"] < 10**-3].min()
            #if len(blank) < 1:
            #    continue
            #blank = float(blank.iloc[0])
            for ai, ar in ag.iterrows():
                a.loc[ai, "blank"] =  blank
                a.loc[ai, "blank_bg"] =  blank-blank_bg

        #a.fillna(0, inplace=True)

        a["Area_bg_corrected"] = a["Area"] - a["Background"]
        a.loc[a["Area_bg_corrected"] < 0, "Area_bg_corrected"] = 0
        a["Area_blank_corrected"] = a["Area"] - a["blank"]
        a.loc[a["Area_blank_corrected"] < 0, "Area_blank_corrected"] = 0
        a["Area_both_corrected"] = a["Area"] - a["blank"] - a["Background"]
        a.loc[a["Area_both_corrected"] < 0, "Area_both_corrected"] = 0
        a["Area_isocorr"] = a["Area"] / a["isocorr"]
        ori_data[method] = a
    return ori_data

def import_data_XL(methods=["TofMRM", "MSe", "HDMSe"]):
    ori_data = dict()
    for method in methods:
        #a = pd.read_csv(f"Results_Waters_skyline/met_{method}.csv")
        a = pd.read_csv(f"Results_Waters_skyline/XL_{method}.csv")
        #a = pd.read_csv(f"/Users/ct001/Downloads/transition results acquistion project - 170619/XLBSA/XLBSA - {method} - Transition Results.csv")
        a.drop(["Precursor Mz", "Precursor Charge", "Product Charge", "Product Mz", "Retention Time"], axis=1, inplace=True)
        # 10 amol to 1 pmol
        ppbs = [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0]
        if method == "MS":# or method == "DDA" or method == "HDDDA":
            ppbs = [0, ] + base_c + [0, 0] + base_c
        # get number from 1 to 15, get corresponding concentration
        a["ppb"] = a["Replicate"].apply(lambda x: ppbs[int(x[-6:-4])-1])
        a["Replicate"] = a["Replicate"].apply(lambda x: x.split()[-1])
        a["precursor"] = a["Fragment Ion"].apply(is_precursor)
        for ai, ar in a.iterrows():
            a.loc[ai, "isocorr"] = get_icorr(ar["Peptide"], ar["Fragment Ion"])
        # compute blank (corresponding 0 ppb value)
        for an, ag in a.groupby(["Peptide", "Fragment Ion"]):
            blank = ag["Area"][ag["ppb"] < 10**-3].min()
            blank_bg = ag["Background"][ag["ppb"] < 10**-3].min()
            #if len(blank) < 1:
            #    continue
            #blank = float(blank.iloc[0])
            for ai, ar in ag.iterrows():
                a.loc[ai, "blank"] =  blank
                a.loc[ai, "blank_bg"] =  blank-blank_bg

        #a.fillna(0, inplace=True)

        a["Area_bg_corrected"] = a["Area"] - a["Background"]
        a.loc[a["Area_bg_corrected"] < 0, "Area_bg_corrected"] = 0
        a["Area_blank_corrected"] = a["Area"] - a["blank"]
        a.loc[a["Area_blank_corrected"] < 0, "Area_blank_corrected"] = 0
        a["Area_both_corrected"] = a["Area"] - a["blank"] - a["Background"]
        a.loc[a["Area_both_corrected"] < 0, "Area_both_corrected"] = 0
        a["Area_isocorr"] = a["Area"] / a["isocorr"]
        ori_data[method] = a
    return ori_data


def summarise_data(ori_data):
    sum_data = dict()
    for m, a in ori_data.items():
        d = a.loc[a["Fragment Ion"] == "precursor", ["Replicate", "Peptide", "Fragment Ion", "ppb", "Area", "Background", "blank", "isocorr", "Area_isocorr", "Area_bg_corrected", "Area_blank_corrected", "Area_both_corrected"]].copy()
        d.set_index("Replicate", inplace=True)
        d.set_index("Peptide", append=True, inplace=True)
        d.set_index("ppb", append=True, inplace=True)

        for c in ["Area", "Area_bg_corrected", "Area_blank_corrected", "Area_both_corrected"]:
            # get the sum of fragments
            for gn, g in a.groupby(["Replicate", "Peptide", "ppb"]):
                precs = g.loc[g["precursor"] == True, :]
                frags = g.loc[g["precursor"] == False, :]
                if len(frags) > 0:
                    fsum = frags[c].sum()
                else:
                    fsum = 0 #np.nan
                #print(fsum)
                d.loc[gn, c+"_sum_frags"] = fsum
                try:
                    p0 = precs.loc[precs["Fragment Ion"] == prec_list[0], c].iloc[0]
                    i1 = get_icorr(gn[1], prec_list[1])
                    p1 = precs.loc[precs["Fragment Ion"] == prec_list[1], c].iloc[0]
                    i2 = get_icorr(gn[1], prec_list[2])
                    p2 = precs.loc[precs["Fragment Ion"] == prec_list[2], c].iloc[0]
                    d.loc[gn, c+"_isocorr"] = p0
                    # if (M+1)/M ratio is ok, use M
                    if p1 < 1.3*i1 * p0:
                        d.loc[gn, c+"_isocorr"] = p
                    else:
                        # if (M+2)/(M+1) ratio is ok, use M1
                        if p2 < 1.3*i2/i1 * p1:
                            d.loc[gn, c+"_isocorr"] = p1 / i1
                        else:
                            # use M2
                            d.loc[gn, c+"_isocorr"] = p2 / i2
                    d.loc[gn, c+"_isocorr_p1"] = p1 / i1
                    d.loc[gn, c+"_isocorr_p2"] = p2 / i2
                except:
                    pass
        d["Area_tot"] = d["Area"] + d["Area_sum_frags"]
        for c in ["Area", "Area_bg_corrected", "Area_blank_corrected", "Area_both_corrected"]:
            d[c+"_MS2_corr"] = d[c+"_sum_frags"]
            filt = (d[c+"_sum_frags"] / d[c]) < 3
            d.loc[filt, c+"_MS2_corr"] = d.loc[filt, c]
        sum_data[m] = d
    return sum_data



