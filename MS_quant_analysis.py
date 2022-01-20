#!/usr/bin/env python
# coding: utf-8

import os
import pathlib
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import scipy.stats
from scipy.optimize import curve_fit

import import_data as imd
import fit_data as fid


import argparse

parser = argparse.ArgumentParser(description='Parse arguments.')
parser.add_argument('-f', '--file', type=str, default="")
parser.add_argument('-o', '--out', type=str, default=None)
parser.add_argument('--ffmt', type=str, default="png")

args = parser.parse_args()

if args.out is None:
    print("No out folder")
    import sys; sys.exit()


def attempt_load(fname):
    p = pathlib.Path(fname)
    if not p.exists():
        return None
    with open(fname, 'rb') as f:
        return pickle.load(f)

ffmt = args.ffmt

eps = 10**-9


final_cols = ['r2 (MS1)',
 'r2 (MS2)',
 'r2 (MS1/MS2)',
 'r2 (MS1/iso)',
 '#orders LDR (MS1)',
 '#orders LDR (MS2)',
 '#orders LDR (MS1/MS2)',
 '#orders LDR (MS1/iso)',
 'S/N@ 10 ppb (MS1)',
 'S/N@ 10 ppb (MS2)',
 'S/N@ 10 ppb (MS1/MS2)',
 'S/N@ 10 ppb (MS1/iso)',
 'LLOD LDR (MS1)',
 'LLOD LDR (MS2)',
 'LLOD LDR (MS1/MS2)',
 'LLOD LDR (MS1/iso)']

midx = ["MS", "HDMS", "DDA", "HDDDA", "TofMRM", "HDMRM", "MSe", "HDMSe", "SONAR"]
def write_fits(cfits, msw, corr=""):
    ms1_fits = cfits.loc[(slice(None), slice(None), "Area"+corr), ["R2", "LLOD", "Drange", "SN_ratio"]].reset_index("method", drop=True)
    ms_fits = ms1_fits.copy()
    ms_fits.columns = ['r2 (MS1)', 'LLOD LDR (MS1)', "#orders LDR (MS1)", 'S/N@ 10 ppb (MS1)']

    ms12_fits = cfits.loc[(slice(None), slice(None), "Area"+corr+"_MS2_corr"), ["R2", "LLOD", "Drange", "SN_ratio"]].reset_index("method", drop=True)
    ms_fits['r2 (MS1/MS2)'] = ms12_fits["R2"]
    ms_fits['LLOD LDR (MS1/MS2)'] = ms12_fits["LLOD"]
    ms_fits["#orders LDR (MS1/MS2)"] = ms12_fits["Drange"]
    ms_fits['S/N@ 10 ppb (MS1/MS2)'] = ms12_fits["SN_ratio"]

    msiso_fits = cfits.loc[(slice(None), slice(None), "Area"+corr+"_isocorr"), ["R2", "LLOD", "Drange", "SN_ratio"]].reset_index("method", drop=True)
    ms_fits['r2 (MS1/iso)'] = msiso_fits["R2"]
    ms_fits['LLOD LDR (MS1/iso)'] = msiso_fits["LLOD"]
    ms_fits["#orders LDR (MS1/iso)"] = msiso_fits["Drange"]
    ms_fits['S/N@ 10 ppb (MS1/iso)'] = msiso_fits["SN_ratio"]

    ms2_fits = cfits.loc[(slice(None), slice(None), "Area"+corr+"_sum_frags"), ["R2", "LLOD", "Drange", "SN_ratio"]].reset_index("method", drop=True)
    ms_fits['r2 (MS2)'] = ms2_fits["R2"]
    ms_fits['LLOD LDR (MS2)'] = ms2_fits["LLOD"]
    ms_fits["#orders LDR (MS2)"] = ms2_fits["Drange"]
    ms_fits['S/N@ 10 ppb (MS2)'] = ms2_fits["SN_ratio"]

    for c in final_cols:
        if c not in ms_fits.columns:
            ms_fits[c] = np.nan
    #ms_fits.fillna('-', inplace=True)
    ms_fits = ms_fits[final_cols]
    ms_fits = ms_fits.reindex(midx, axis='index', level=0)

    if corr == "":
        sheetname = "No correction"
    else:
        sheetname = corr
    ms_fits.to_excel(msw, sheet_name=sheetname)
    ms_fits.groupby(level=0).median().to_excel(msw, sheet_name=f"{sheetname}_summary")

def write_multi_fits(fname, cfits):
    msw = pd.ExcelWriter(fname, mode='w')

    write_fits(cfits, msw, corr="")
    write_fits(cfits, msw, corr="_bg_corrected")
    write_fits(cfits, msw, corr="_blank_corrected")
    write_fits(cfits, msw, corr="_both_corrected")
    msw.close()
    return

def write_sum_data(fname, sdata):
    msw = pd.ExcelWriter(fname, mode='w')

    for n, s in sdata.items():
        s.to_excel(msw, sheet_name=n)
    msw.close()
    return


resf = args.out
os.mkdir(resf)


odata_met = attempt_load("odata_met.dump")
if odata_met is None:
    # for metabolites
    odata_met = imd.import_data_met(["MS", "HDMS", "MSe", "HDMSe", "SONAR", "TofMRM", "HDMRM", "DDA", "HDDDA"])
    with open("odata_met.dump", "wb") as f:
        pickle.dump(odata_met, f)

odata_pep = attempt_load("odata_pep.dump")
if odata_pep is None:
    # for peptide
    odata_pep = imd.import_data_pep(["MS", "HDMS", "MSe", "HDMSe", "SONAR", "TofMRM", "HDMRM", "DDA", "HDDDA"])
    # removed HDMRM for now
    #odata_pep = imd.import_data_pep(["MS", "HDMS", "MSe", "HDMSe", "SONAR", "TofMRM", "DDA", "HDDDA"])
    with open("odata_pep.dump", "wb") as f:
        pickle.dump(odata_pep, f)

odata_xl = attempt_load("odata_xl.dump")
if odata_xl is None:
    # for crosslinks
    odata_xl = imd.import_data_XL(["MSe", "HDMSe", "TofMRM"])
    with open("odata_xl.dump", "wb") as f:
        pickle.dump(odata_xl, f)

sdata_met = attempt_load("sdata_met.dump")
if sdata_met is None:
    sdata_met = imd.summarise_data(odata_met)
    with open("sdata_met.dump", "wb") as f:
        pickle.dump(sdata_met, f)

sdata_pep = attempt_load("sdata_pep.dump")
if sdata_pep is None:
    sdata_pep = imd.summarise_data(odata_pep)
    with open("sdata_pep.dump", "wb") as f:
        pickle.dump(sdata_pep, f)

sdata_xl = attempt_load("sdata_xl.dump")
if sdata_xl is None:
    sdata_xl = imd.summarise_data(odata_xl)
    with open("sdata_xl.dump", "wb") as f:
        pickle.dump(sdata_xl, f)



t = attempt_load("fits_met.dump")
if t is None:
    fits_met, cfits_met = fid.make_all_fits(sdata_met)
    with open("fits_met.dump", "wb") as f:
        pickle.dump((fits_met, cfits_met), f)
else:
    fits_met, cfits_met = t

t = attempt_load("fits_pep.dump")
if t is None:
    fits_pep, cfits_pep = fid.make_all_fits(sdata_pep)
    with open("fits_pep.dump", "wb") as f:
        pickle.dump((fits_pep, cfits_pep), f)
else:
    fits_pep, cfits_pep = t

t = attempt_load("fits_xl.dump")
if t is None:
    fits_xl, cfits_xl = fid.make_all_fits(sdata_xl)
    with open("fits_xl.dump", "wb") as f:
        pickle.dump((fits_xl, cfits_xl), f)
else:
    fits_xl, cfits_xl = t

for cfits in [cfits_met, cfits_pep, cfits_xl]:
    cfits[cfits == np.inf] = np.nan


write_multi_fits(fname=resf+"ms_fit_met.xls", cfits=cfits_met)
write_multi_fits(fname=resf+"ms_fit_pep.xls", cfits=cfits_pep)
write_multi_fits(fname=resf+"ms_fit_xl.xls", cfits=cfits_xl)

write_sum_data(fname=resf+"data_met.xls", sdata=sdata_met)
write_sum_data(fname=resf+"data_pep.xls", sdata=sdata_pep)
write_sum_data(fname=resf+"data_xl.xls", sdata=sdata_xl)


pcols = ["R2", "Drange", "SN_ratio", "LLOD"]
cfits = cfits_pep
ms1f = cfits.loc[(slice(None), slice(None), "Area"), pcols].reset_index("method", drop=True).groupby(level=0).mean()
ms2f = cfits.loc[(slice(None), slice(None), "Area_sum_frags"), pcols].reset_index("method", drop=True).groupby(level=0).mean()
ms1f_iso = cfits.loc[(slice(None), slice(None), "Area_both_corrected_isocorr"), pcols].reset_index("method", drop=True).groupby(level=0).mean()
ms2f_corr = cfits.loc[(slice(None), slice(None), "Area_both_corrected_MS2_corr"), pcols].reset_index("method", drop=True).groupby(level=0).mean()
(ms2f_corr - ms1f).reindex(midx)


print("MS1")
print(ms1f.reindex(midx))
print("MS2")
print(ms2f.reindex(midx))
print("MS2 - MS1")
print((ms2f - ms1f).reindex(midx))
print("MS1_iso")
print(ms1f_iso.reindex(midx))
print("MS1_iso - MS1")
print((ms1f_iso - ms1f).reindex(midx))
print("MS2_corr")
print(ms2f_corr.reindex(midx))
print("MS2_corr - MS1")
print((ms2f_corr - ms1f).reindex(midx))

tr = {"Area": "MS1",
      "Area_sum_frags": "MS2",
      "Area_isocorr": "isotope-corrected MS1",
      "Area_MS2_corr": "MS2-corrected MS1",
      "Area_both_corrected": "bg-corrected MS1",
      "Area_both_corrected_sum_frags": "bg-corrected MS2",
      "Area_both_corrected_MS2_corr": "MS2+bg-corrected MS1",
      "Area_both_corrected_isocorr": "isotope+bg-corrected MS1",
        }



fig, axes = plt.subplots(1, 2, figsize=(10, 7))#, sharex=True, sharey=True)

#sns.scatterplot(x="LLOD", y="Drange", size="SN_ratio", data=fd.loc[(slice(None), "Area_both_corrected"), :], hue="Mode", ax=axes[0])
#sns.scatterplot(x="LLOD", y="Drange", size="SN_ratio", data=fd.loc[(slice(None), ["Area", "Area_sum_frags"]), :], hue="Mode", ax=axes[0])
fd = cfits_met.reset_index(level=0)
fd = fd.loc[(slice(None), ["Area", "Area_sum_frags"]), :].reset_index(level=1).copy()
fd["method"] = fd["method"].apply(lambda x: tr[x])
sns.scatterplot(x="LLOD", y="Drange", size="SN_ratio", data=fd, style="method", hue="Mode", ax=axes[0])

fd = cfits_pep.reset_index(level=0)
fd = fd.loc[(slice(None), ["Area_both_corrected", "Area_both_corrected_sum_frags"]), :].reset_index(level=1).copy()
fd["method"] = fd["method"].apply(lambda x: tr[x])
sns.scatterplot(x="LLOD", y="Drange", size="SN_ratio", sizes=(1,100), data=fd, style="method", hue="Mode", ax=axes[1])
axes[0].set_xlabel("LLOD (fmol)")
axes[0].set_ylabel("Dynamic range (#orders)")
axes[1].set_ylabel("Dynamic range (#orders)")
axes[0].set_title("Metabolites")
axes[1].set_title("Peptides")
axes[1].set_xlabel("LLOD (fmol)")
axes[0].set_xlim([5*10**4, 5*10**-2])
axes[0].set_ylim([0, 4.5])
axes[1].set_xlim([5*10**1, 5*10**-4])
axes[1].set_ylim([0, 7.5])
axes[0].set_xscale("log")
axes[1].set_xscale("log")

fig.tight_layout()
plt.savefig(f"{resf}fits_both.{ffmt}", dpi=300)
plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(14, 7))#, sharex=True, sharey=True)

#Areas = ["Area", "Area_isocorr", "Area_MS2_corr"]
Areas = ["Area", "Area_isocorr"]
#Areas = ["Area", "Area_bg_corrected"]
#Areas = ["Area", "Area_blank_corrected"]
Areas_met = ["Area", "Area_MS2_corr", "Area_isocorr"]
#Areas = ["Area", "Area_sum_frags"]
Areas = ["Area_both_corrected", "Area_both_corrected_MS2_corr", "Area_both_corrected_isocorr"]

#ar = "Area_both_corrected"
#Areas = ["", "_bg_corrected", "_blank_corrected", "_both_corrected"]
#Areas = [i+"_MS2_corr" for i in Areas]

#Areas = ["", "_isocorr", "_MS2_corr"]
#Areas = [ar+i for i in Areas]

fd = cfits_met.loc[(slice(None), slice(None), Areas_met), :].reset_index(level=0)
fd = fd.reset_index(level=1)
dat = fd.loc[slice(None), :].dropna().groupby(by=["Mode", "method"]).median().reset_index().copy()
dat["method"] = dat["method"].apply(lambda x: tr[x])
sns.scatterplot(x="LLOD", y="Drange", style="method", size="SN_ratio", data=dat, hue="Mode", ax=axes[0])

fd = cfits_pep.loc[(slice(None), slice(None), Areas), :].reset_index(level=0)
fd = fd.reset_index(level=1)
dat = fd.loc[slice(None), :].dropna().groupby(by=["Mode", "method"]).median().reset_index()
dat["method"] = dat["method"].apply(lambda x: tr[x])
sns.scatterplot(x="LLOD", y="Drange", style="method", size="SN_ratio", data=dat, hue="Mode", ax=axes[1])

axes[0].set_xlabel("LLOD (fmol)")
axes[0].set_ylabel("Dynamic range (#orders)")
axes[1].set_ylabel("Dynamic range (#orders)")
axes[0].set_title("Metabolites")
axes[1].set_title("Peptides")
axes[1].set_xlabel("LLOD (fmol)")
#axes[0].set_xlim([10**2, 5*10**-3])
axes[0].set_xlim([5*10**4, 5*10**-2])
#axes[0].set_ylim([2, 3.25])
axes[0].set_ylim([0, 4.5])
axes[1].set_xlim([5*10**1, 5*10**-4])
#axes[1].set_ylim([1, 6.25])
axes[1].set_ylim([0, 4.5])
axes[0].set_xscale("log")
axes[1].set_xscale("log")

fig.tight_layout()
plt.savefig(f"{resf}fits_both_corrected.{ffmt}", dpi=300)
plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(10, 7), sharey=True)#, sharex=True, sharey=True)
#sns.scatterplot(x="LLOD", y="Drange", size="SN_ratio", data=fd.loc[(slice(None), "Area_both_corrected"), :], hue="Mode", ax=axes[0])
#sns.scatterplot(x="LLOD", y="Drange", size="SN_ratio", data=fd.loc[(slice(None), ["Area", "Area_sum_frags"]), :], hue="Mode", ax=axes[0])
fd = cfits_met.reset_index(level=1)
fd = fd.loc[(slice(None), ["Area", "Area_sum_frags"]), :].groupby(level=(0,1)).median().reset_index(level=(0,1))
fd["method"] = fd["method"].apply(lambda x: tr[x])
sns.scatterplot(x="LLOD", y="Drange", size="SN_ratio", sizes=(50,200), data=fd, style="method", hue="Mode", ax=axes[0])

fd = cfits_pep.reset_index(level=1)
fd = fd.loc[(slice(None), ["Area", "Area_sum_frags"]), :].groupby(level=(0,1)).median().reset_index(level=(0,1))
fd["method"] = fd["method"].apply(lambda x: tr[x])
sns.scatterplot(x="LLOD", y="Drange", size="SN_ratio", sizes=(50,200), data=fd, style="method", hue="Mode", ax=axes[1])
axes[0].set_xlabel("LLOD (fmol)")
axes[0].set_ylabel("Dynamic range (#orders)")
axes[1].set_ylabel("Dynamic range (#orders)")
axes[0].set_title("Metabolites")
axes[1].set_title("Peptides")
axes[1].set_xlabel("LLOD (fmol)")
axes[0].set_xlim([1*10**2, 5*10**-2])
axes[0].set_ylim([1.5, 4.0])
axes[1].set_xlim([1*10**2, 5*10**-3])
axes[1].set_ylim([1.5, 5.0])
axes[0].set_xscale("log")
axes[1].set_xscale("log")

fig.tight_layout()
plt.savefig(f"{resf}fits_both_mean.{ffmt}", dpi=300)
plt.close(fig)

#

fig, axes = plt.subplots(1, 1, figsize=(10, 7))#, sharex=True, sharey=True)

#Areas = ["Area", "Area_isocorr", "Area_MS2_corr"]
Areas = ["Area", "Area_isocorr"]
#Areas = ["Area", "Area_bg_corrected"]
#Areas = ["Area", "Area_blank_corrected", "Area_both_corrected"]
#Areas = ["Area", "Area_MS2_corr"]
#Areas = ["Area", "Area_sum_frags"]
Areas = ["Area_both_corrected", "Area_both_corrected_MS2_corr", "Area_both_corrected_isocorr"]

#ar = "Area_both_corrected"
#Areas = ["", "_bg_corrected", "_blank_corrected", "_both_corrected"]
#Areas = [i+"_MS2_corr" for i in Areas]

#Areas = ["", "_isocorr", "_MS2_corr"]
#Areas = [ar+i for i in Areas]

fd = cfits_pep.loc[(slice(None), slice(None), Areas), :].reset_index(level=0)
fd = fd.reset_index(level=1)
dat = fd.loc[slice(None), :].dropna().groupby(by=["Mode", "method"]).median().reset_index().copy()
dat["method"] = dat["method"].apply(lambda x: tr[x])

sns.scatterplot(x="LLOD", y="Drange", style="method", size="SN_ratio", sizes=(40, 400), data=dat, hue="Mode", ax=axes)

axes.set_xlabel("LLOD (fmol)")
axes.set_ylabel("Dynamic range (#orders)")
axes.set_title("Peptides")
axes.set_xlim([5*10**3, 5*10**-4])
axes.set_ylim([1, 6.25])
axes.set_xscale("log")
axes.set_xscale("log")

fig.tight_layout()
plt.savefig(f"{resf}fits_pep_corrected.{ffmt}", dpi=300)
plt.close(fig)


# In[237]:


fig, axes = plt.subplots(1, 1, figsize=(10, 7))#, sharex=True, sharey=True)

#Areas = ["Area", "Area_isocorr", "Area_MS2_corr"]
Areas = ["Area", "Area_isocorr"]
#Areas = ["Area", "Area_bg_corrected"]
#Areas = ["Area", "Area_blank_corrected", "Area_both_corrected"]
#Areas = ["Area", "Area_MS2_corr"]
#Areas = ["Area", "Area_sum_frags"]
Areas = ["Area_both_corrected", "Area_both_corrected_MS2_corr", "Area_both_corrected_isocorr"]

#ar = "Area_both_corrected"
#Areas = ["", "_bg_corrected", "_blank_corrected", "_both_corrected"]
#Areas = [i+"_MS2_corr" for i in Areas]

#Areas = ["", "_isocorr", "_MS2_corr"]
#Areas = [ar+i for i in Areas]

fd = cfits_xl.loc[(slice(None), slice(None), Areas), :].reset_index(level=0)
fd = fd.reset_index(level=1).copy()
fd["method"] = fd["method"].apply(lambda x: tr[x])

dat = fd.loc[slice(None), :].dropna().groupby(by=["Mode", "method"]).median().reset_index()
sns.scatterplot(x="LLOD", y="Drange", style="method", size="SN_ratio", sizes=(40, 200), data=dat, hue="Mode", ax=axes)

axes.set_xlabel("LLOD (fmol)")
axes.set_ylabel("Dynamic range (#orders)")
axes.set_title("Peptides")
axes.set_xlim([5*10**0, 5*10**-4])
axes.set_ylim([3, 6.5])
axes.set_xscale("log")
axes.set_xscale("log")

fig.tight_layout()
plt.savefig(f"{resf}fits_xl_corrected.{ffmt}", dpi=300)
plt.close(fig)


fig, axes = plt.subplots(1, 1, figsize=(10, 7))#, sharex=True, sharey=True)
fd = cfits_xl.loc[(slice(None), slice(None), "Area_both_corrected_MS2_corr"), :].reset_index(level=0)
fd = fd.reset_index(level=0)
dat = fd.loc[slice(None), :].dropna().groupby(by=["Mode", "molecule"]).median().reset_index()
sns.scatterplot(x="LLOD", y="Drange", style="Mode", size="SN_ratio", sizes=(40, 200), data=dat, hue="molecule", ax=axes)

axes.set_xlabel("LLOD (fmol)")
axes.set_ylabel("Dynamic range (#orders)")
axes.set_title("Peptides")
axes.set_xlim([5*10**0, 5*10**-4])
axes.set_ylim([3, 6.5])
axes.set_xscale("log")
axes.set_xscale("log")

fig.tight_layout()
plt.savefig(f"{resf}fits_xl_peps_corrected.{ffmt}", dpi=300)
plt.close(fig)


ms2c_dir = pathlib.Path(resf+"MS2_corr")
os.mkdir(ms2c_dir)
for method, sdata, nrows in zip(["met", "pep", "xl"], [sdata_met, sdata_pep, sdata_xl], [2, 3, 1]):
    for m, sd in sdata.items():
        fig, axes = plt.subplots(2, 4, figsize=(10, 5), sharex=True, sharey=True)
        legs = [None, ]*4
        for ax, (gn, g) in zip(axes.flatten(), sd.groupby(level=(1))):
            ax.set_title(gn, fontsize=5)
            x = np.array(g.index.get_level_values(2))
            for ci, col in enumerate(["Area", "Area_sum_frags", "Area_MS2_corr"]):
                try:
                    co = ["C0", "C1", "C2"][ci] #f"C{ci}"
                    s = "o+.."[ci]
                    y = g[col]
                    la, lb, lr2, llod, hlod, eC, eR, e_n, e_R2, sn_ratio, drange = fid.make_auto_fit(x, y)
                    p = ax.scatter(x, y, c=co, marker=s)
                    legs[ci] = p
                    # xs = np.linspace(x.min(), x.max(), num=200)
                    xs = np.logspace(-5, 3.5, num=200)
                    p = ax.plot(xs, la*xs+lb, c=co)
                    ax.plot(xs, fid.saturation_sig2(xs, eC, eR, e_n), "--", c=co)
                    #ax.axvline(llod, linestyle=":", c=co)
                    #ax.axvline(hlod, linestyle=":", c=co)
                    ax.set_xscale("symlog", linthreshx=0.01)
                    ax.set_yscale("symlog", linthreshy=1)
                    ax.set_xlim(left=0)
                    ax.set_ylim(bottom=10)
                except:
                    pass
        fig.legend(legs, ["MS1", "MS2", "corrected"], loc=5)
        fig.suptitle(m)
        fig.tight_layout()
        plt.subplots_adjust(right=0.9)
        plt.savefig(ms2c_dir / f"{method}_{m}_ms2corr.{ffmt}", dpi=300)
        plt.close(fig)


msisoc_dir = pathlib.Path(resf+"MSiso_corr")
os.mkdir(msisoc_dir)

for method, sdata, nrows in zip(["met", "pep", "xl"], [sdata_met, sdata_pep, sdata_xl], [2, 3, 1]):
    for m, sd in sdata.items(): 
        fig, axes = plt.subplots(3, 4, figsize=(10, 5), sharex=True, sharey=True)
        legs = ["M", "M+1", "M+2", "corrected M"]
        for ax, (gn, g) in zip(axes.flatten(), sd.groupby(level=(1))):
            title = gn
            if len(gn) > 55:
                title = gn[:15]+"..."+gn[-4:]
            ax.set_title(title, fontsize=5)
            x = np.array(g.index.get_level_values(2))
            for ci, col in enumerate(["Area", "Area_isocorr_p1", "Area_isocorr_p2", "Area_isocorr"]):
            #for ci, col in enumerate(["Area_blank_corrected", "Area_blank_corrected_isocorr"]):
    #             try:
                co = ["black", "grey", "lightgrey", "C1"][ci] #f"C{ci}"
                s = "ooo."[ci]
                try:
                    y = g[col]
                    p = ax.scatter(x, y, c=co, marker=s)
                    la, lb, lr2, llod, hlod, eC, eR, e_n, e_R2, sn_ratio, drange = fid.make_auto_fit(x, y)
                except:# ValueError:
                    legs[ci] = "_"
                    continue
                # xs = np.linspace(x.min(), x.max(), num=200)
                xs = np.logspace(-5, 3.5, num=200)
                p = ax.plot(xs, la*xs+lb, c=co)
                ax.plot(xs, fid.saturation_sig2(xs, eC, eR, e_n), "--", c=co)
                ax.axvline(llod, linestyle=":", c=co)
                ax.axvline(hlod, linestyle=":", c=co)
                #ax.set_xscale("log")
                #ax.set_yscale("log")
                ax.set_xscale("symlog", linthreshx=0.001)
                ax.set_yscale("symlog", linthreshy=0.001)
                ax.set_xlim(left=0)
                ax.set_ylim(bottom=10)
                #ax.set_xlim(left=-0.002)
                #ax.set_ylim(bottom=10)
    #             except:
    #                 pass
        fig.legend(legs, loc=5)
        fig.suptitle(m)
        fig.tight_layout()
        plt.subplots_adjust(top=0.9, right=0.85)
        plt.savefig(msisoc_dir / f"{method}_{m}_isocorr.{ffmt}", dpi=300)
        plt.close(fig)
    



msfit_dir = pathlib.Path(resf+"MS_fits")
os.mkdir(msfit_dir)

for method, sdata, nrows in zip(["met", "pep", "xl"], [sdata_met, sdata_pep, sdata_xl], [2, 3, 2]):
    for m, sd in sdata.items(): 
        for corr in ["", "_sum_frags", "_isocorr", "_MS2_corr"]:
            #for ci, col in enumerate(["Area", "Area_bg_corrected", "Area_blank_corrected", "Area_both_corrected", ]):
            for ci, col in enumerate(["Area", ]):#"Area_both_corrected", ]):
                fig, axes = plt.subplots(nrows, 4, figsize=(10, 5), sharex=True, sharey=True)
                for ax, (gn, g) in zip(axes.flatten(), sd.groupby(level=(1))):
                    title = gn
                    if len(gn) > 55:
                        title = gn[:15]+"..."+gn[-4:]
                    ax.set_title(title, fontsize=5)
                    x = np.array(g.index.get_level_values(2))
                    s = "o"#"ooo."[ci]
                    try:
                        y = g[col+corr]
                        params = fid.make_auto_fit(x, y)
                    except:
                        continue
                    la, lb, lr2, llod, hlod, eC, eR, e_n, e_R2, sn_ratio, drange = params
                    x += 10**-4
                    y += 10**-4
                    p = ax.plot(x, y, "o")
                    # xs = np.linspace(x.min(), x.max(), num=200)
                    xs = np.logspace(-5, 3.5, num=200)
                    lf = ax.plot(xs, la*xs+lb)
                    ef = ax.plot(xs, fid.saturation_sig2(xs, eC, eR, e_n), "--")
                    ax.axvline(llod, linestyle=":")
                    ax.axvline(hlod, linestyle=":")
                    ax.text(0.1, 0.8, f"{la:6.2f} x + {lb:6.2f}\n {e_n:6.2e} + {eC:6.2e} e(-x/{eR:6.2f})", fontsize=5, transform=ax.transAxes)
                    #ax.set_xscale("symlog", linthreshx=10**0)
                    #ax.set_yscale("symlog", linthreshy=10**0)
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    ax.set_xlim(left=0.00001)
                    ax.set_ylim(10, min(10**8, max(max(y), ax.get_ylim()[1]))*1.1)
                    #ax.set_xlim(left=-0.002)
                    legs = [p, lf, ef]
                fig.legend(["data", "linear fit", "full model"], loc=5)
                fig.suptitle(m+" "+col+corr)
                fig.tight_layout()
                plt.subplots_adjust(top=0.9, right=0.85)
                plt.savefig(msfit_dir / f"{method}_{m}_{col}{corr}.{ffmt}", dpi=300)
                plt.close(fig)


mscomp_dir = pathlib.Path(resf+"MS12_comp")
os.mkdir(mscomp_dir)

for method, sdata, nrows in zip(["met", "pep", "xl"], [sdata_met, sdata_pep, sdata_xl], [2, 3, 1]):
    for m, sd in sdata.items(): 
        fig, axes = plt.subplots(3, 4, figsize=(10, 5), sharex=True, sharey=True)
        legs = [None, ]*2
        for ax, (gn, g) in zip(axes.flatten(), sd.groupby(level=(1))):
            title = gn
            if len(gn) > 55:
                title = gn[:15]+"..."+gn[-4:]
            ax.set_title(title, fontsize=5)
            x = np.array(g.index.get_level_values(2))
#            for ci, col in enumerate(["Area", "Area_sum_frags"]):
            ci = 0
            co = f"C{ci}"
            s = "ooo."[ci]
            y = g["Area_sum_frags"] / g["Area"]
            p = ax.scatter(x, y, c=co, marker=s)
            legs[ci] = p
            ax.set_xscale("symlog", linthreshx=0.001)
            #ax.set_yscale("symlog", linthreshy=0.001)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
        fig.legend(legs, ["MS1", "MS2"], loc=5)
        fig.suptitle(m)
        fig.tight_layout()
        plt.subplots_adjust(top=0.9, right=0.85)
        plt.savefig(mscomp_dir / f"{method}_{m}_comp.{ffmt}", dpi=300)
        plt.close(fig)

mstot_dir = pathlib.Path(resf+"MS_tot")
os.mkdir(mstot_dir)

for method, sdata, nrows in zip(["met", "pep", "xl"], [sdata_met, sdata_pep, sdata_xl], [2, 3, 1]):
    for m, sd in sdata.items(): 
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        g = sd.groupby(level=(2)).mean()
        fig.suptitle(m)
        x = np.array(g.index)
        y = g["Area_tot"]
        print(m)
        p = axes[0].scatter(x, y, marker="o")
        p = axes[1].scatter(x, y, marker="o")
        try:
            params = fid.make_auto_fit(x, y)
            la, lb, lr2, llod, hlod, eC, eR, e_n, e_R2, sn_ratio, drange = params
            # xs = np.linspace(x.min(), x.max(), num=200)
            xs = np.logspace(-5, 3.5, num=200)
            p = axes[0].plot(xs, la*xs+lb, c="C1")
            p = axes[1].plot(xs, la*xs+lb, c="C1")
            axes[0].plot(xs, fid.saturation_sig2(xs, eC, eR, e_n), "--", c="C2")
            axes[1].plot(xs, fid.saturation_sig2(xs, eC, eR, e_n), "--", c="C2")
            axes[0].axvline(llod, linestyle=":")
            axes[1].axvline(llod, linestyle=":")
            axes[0].axvline(hlod, linestyle=":")
            axes[1].axvline(hlod, linestyle=":")
            axes[0].text(0.1, 0.8, f"{la:6.2f} x + {lb:6.2f}\n {e_n:6.2e} + {eC:6.2e} e(-x/{eR:6.2f})", fontsize=5, transform=axes[0].transAxes)
            axes[1].text(0.1, 0.8, f"{la:6.2f} x + {lb:6.2f}\n {e_n:6.2e} + {eC:6.2e} e(-x/{eR:6.2f})", fontsize=5, transform=axes[1].transAxes)
            print(f"Dynamic range: {drange} - LLOD: {llod} - HLOD: {hlod}")
        except ValueError:
            print("Error during fit!")
            pass
        axes[0].set_xscale("symlog", linthreshx=0.001)
        axes[0].set_yscale("symlog", linthreshy=0.001)
        axes[0].set_xlim(left=0)
        axes[1].set_xlim(left=0)
        axes[0].set_ylim(bottom=10)
        axes[1].set_ylim(bottom=10)
        axes[1].set_ylim(top=max(y)*1.2)
        fig.tight_layout()
        plt.subplots_adjust(top=0.9, right=0.85)
        plt.savefig(mstot_dir / f"{method}_{m}_tot.{ffmt}", dpi=300)
        plt.close(fig)
