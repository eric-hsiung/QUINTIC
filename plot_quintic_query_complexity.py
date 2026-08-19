import pandas
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tempfile
import os
import itertools
from scipy.optimize import curve_fit

def read_csv(file_path, sep="#"):
    data = pandas.read_csv(file_path, sep=sep)
    return data

def NlnN(x,C):
    return C*x*np.log(x)

def linear_line(x, a, b):
    return a*x + b

def get_data(csvs, Yvals, Xvals):
    for csv in csvs:
        df = read_csv(csv, sep="#")
        Y = df["'Num Pref Q'"].tolist()
        X = df["'Num Unique Sequences'"].tolist()
        if "'Num Pref Q'" in Y:
            continue
        Yvals.extend(Y)
        Xvals.extend(X)

def compute_R_squared(y_vals, x_vals, params):
    y_d = np.array(y_vals)
    x_d = np.array(x_vals)

    residuals = y_d - NlnN(x_d, *params)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_d - np.mean(y_d))**2)

    R_squared = 1 - (ss_res / ss_tot)
    return R_squared

## Generates plots that do not use Type 3 fonts.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 16})


## File data source files with exact REMAP:
domain_csvs = [
    "craft-world-abr-3-t105.csv",
    "craft-world-abr-3-t106.csv",
    "craft-world-abr-3-t107.csv",
    "craft-world-abr-3-t108.csv",
    "craft-world-abr-3-t109.csv",
    "craft-world-abr-3-t110.csv",
    "office-world-abr-3-t1.csv",
    "office-world-abr-3-t2.csv",
    "office-world-abr-3-t3.csv",
    "office-world-abr-3-t4.csv",
]

moore_machine_names = "test_zero_A test_zero_B test_one_A test_one_A_extra test_one_B test_two_C testB testB2 testB3 testB4 testB5 testB6 testC_linear simple_rm_A simple_rm_B1 simple_rm_B2 simple_rm_B3 simple_rm_B4 simple_rm_B5 simple_rm_B6 simple_rm_B7 simple_rm_B8 simple_office_t4_A simple_office_t4_B simple_office_t4_C simple_craft_t105_A"

#mode = "classification"
mode = "product"
suffix = ""
if mode == "product":
    suffix = "_prod"


lstar_csvs = list(f"concrete_{x}.csv.0" for x in moore_machine_names.split(" "))
remap_csvs = list(f"quintic_remap_baseline/baseline/abstract_{x}.csv.0" for x in moore_machine_names.split(" "))
csvs_strong_cc_ve =  list(f"{mode}/yes-cex-expansion/yes-b-cex-yes-ids/strong-closed-consistent-variable-equality/abstract_{x}{suffix}.csv.0" for x in moore_machine_names.split(" "))
csvs_weak_cc_ve =  list(f"{mode}/yes-cex-expansion/yes-b-cex-yes-ids/weak-closed-consistent-variable-equality/abstract_{x}{suffix}.csv.0" for x in moore_machine_names.split(" "))
csvs_strong_ve =  list(f"{mode}/yes-cex-expansion/yes-b-cex-yes-ids/strong-variable-equality/abstract_{x}{suffix}.csv.0" for x in moore_machine_names.split(" "))
csvs_weak_ve =  list(f"{mode}/yes-cex-expansion/yes-b-cex-yes-ids/weak-variable-equality/abstract_{x}{suffix}.csv.0" for x in moore_machine_names.split(" "))
## Read in all the data points from each CSV:

strong_cc_ve = list()
weak_cc_ve = list()
strong_ve = list()
weak_ve = list()
strong_cc_ve_X = list()
weak_cc_ve_X = list()
strong_ve_X = list()
weak_ve_X = list()
remap_Y = list()
remap_X = list()

get_data(csvs_strong_cc_ve, strong_cc_ve, strong_cc_ve_X)
get_data(csvs_weak_cc_ve, weak_cc_ve, weak_cc_ve_X)
get_data(csvs_strong_ve, strong_ve, strong_ve_X)
get_data(csvs_weak_ve, weak_ve, weak_ve_X)
get_data(remap_csvs, remap_Y, remap_X)


#membQs = list()
#lstar_X = list()
#for csv in lstar_csvs:
#    df = read_csv(csv, sep="#")
#    Y = df["'Num Pref Q'"].tolist()
#    X = df["'Num Unique Sequences'"].tolist()
#    membQs.extend(Y)
#    lstar_X.extend(X)
#lstar_X.append(265)
#membQs.append(265)

#lstar_X.append(1371)
#membQs.append(1371)
#lstar_X.append(841)
#membQs.append(841)
#lstar_X.append(3931)
#membQs.append(3931)
#lstar_X.append(113)
#membQs.append(113)
#lstar_X.append(689)
#membQs.append(689)
#lstar_X.append(1097)
#membQs.append(1097)

print(strong_cc_ve)
print(len(strong_cc_ve))

params_sccve, pcov_sccve = curve_fit(NlnN, strong_cc_ve_X, strong_cc_ve, p0=[0.2121])
params_wccve, pcov_wccve = curve_fit(NlnN, weak_cc_ve_X, weak_cc_ve, p0=[0.2121])
params_sve, pcov_sve = curve_fit(NlnN, strong_ve_X, strong_ve, p0=[0.2121])
params_wve, pcov_wve = curve_fit(NlnN, weak_ve_X, weak_ve, p0=[0.2121])
params_remap, pcov_remap = curve_fit(NlnN, remap_X, remap_Y, p0=[0.2121])

R2_sccve = compute_R_squared(strong_cc_ve, strong_cc_ve_X, params_sccve)
R2_wccve = compute_R_squared(weak_cc_ve, weak_cc_ve_X, params_wccve)
R2_sve = compute_R_squared(strong_ve, strong_ve_X, params_sve)
R2_wve = compute_R_squared(weak_ve, weak_ve_X, params_wve)
R2_remap = compute_R_squared(remap_Y, remap_X, params_remap)

#print(membQs)
#print(lstar_X)

#print(params_remap)
#print(pcov_remap)
cm = 1/2.54
plt.figure(figsize=(2*6.1*cm, 2*8.296*cm))

plt.scatter(strong_cc_ve_X, strong_cc_ve, c="tab:green", s=10, label="S-CC-VE")
plt.scatter(weak_cc_ve_X, weak_cc_ve, c="tab:red", s=10, label="W-CC-VE")
plt.scatter(strong_ve_X, strong_ve, c="tab:blue", s=10, label="S-VE")
plt.scatter(weak_ve_X, weak_ve, c="tab:orange", s=10, label="W-VE")
plt.scatter(remap_X, remap_Y, c="tab:purple", s=10, label="REMAP")


#plt.scatter(lstar_X, membQs, c="tab:red", s=10, label="Lstar Memb Qs")
#plt.scatter(remap_X, prefQs, c="tab:blue", s=10, label="REMAP Pref Qs")
#plt.scatter(lstar_X, membQs, c="tab:red", s=10, label="Lstar Memb Qs")

max_x = max(max(strong_cc_ve_X), max(weak_cc_ve_X), max(strong_ve_X), max(weak_ve_X), max(remap_X))
x = np.linspace(0, max_x+5, 100)

plt.plot(x, NlnN(x, params_remap[0]), color="tab:purple")
plt.plot(x, NlnN(x, params_sccve[0]), color="tab:green")
plt.plot(x, NlnN(x, params_wccve[0]), color="tab:red")
plt.plot(x, NlnN(x, params_sve[0]), color="tab:blue")
plt.plot(x, NlnN(x, params_wve[0]), color="tab:orange")
plt.plot(x, x, color="black", linestyle="--")

plt.xlabel("Num Unique Sequences")
plt.ylabel("Num Queries")
plt.title("Query Complexity")
plt.grid()
plt.gca().set_box_aspect(3/4)
plt.legend(loc="upper left")
plt.savefig("remade_quintic_test.pdf", bbox_inches="tight")
plt.close()



def plot_scatter(x, y, xlabel="X Axis", ylabel="Y Axis", title="Default Title", output_file=None):
    #max_x = list(max(eval(e)) for e in x)
    #plt.scatter(max_x,y)
    plt.scatter(x,y)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.gca().set_box_aspect(3/4)

    if output_file is None:
        with tempfile.NamedTemporaryFile(mode="wb",delete=False) as f:
            plt.savefig(f, bbox_inches="tight")
            plt.close()
    else:
        plt.savefig(output_file, bbox_inches="tight")
        plt.close()
