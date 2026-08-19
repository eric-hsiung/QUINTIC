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

def count_events(event, event_list):
    count = 0
    for e in event_list:
        if e[1] == event:
            count += 1
    return count

def NlnN(x,C):
    return C*x*np.log(x)

def linear_line(x, a, b):
    return a*x + b

def get_data(csvs, Yvals, Xvals):

    Y_key = "'Num Equiv Q'"
    X_key = 'backtrack'

    for csv in csvs:
        df = read_csv(csv, sep="#")
        Y = df[Y_key].to_numpy()
        if Y_key in Y:
            continue
        event_list = list(eval(ev) for ev in df["'Events'"])
        total = 0
        counts = list()
        for tup in event_list:
            count = count_events(X_key, tup)
            total += count
            counts.append(count)

        #Y_div = df["'SAT Solves'"].to_numpy()
        #X = df["'MaxSAT Size'"].tolist()
        X = counts
        #Y = Y/Y_div
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
#mode = "strong-closed-consistent-variable-equality" ## Y-scale is 2000
#mode = "weak-closed-consistent-variable-equality" # Y-scale is 2000
#mode = "strong-variable-equality" #Y-scale is 2000
mode = "weak-variable-equality"
#suffix = ""
#if mode == "product":
#    suffix = "_prod"


lstar_csvs = list(f"concrete_{x}.csv.0" for x in moore_machine_names.split(" "))
remap_csvs = list(f"quintic_remap_baseline/baseline/abstract_{x}.csv.0" for x in moore_machine_names.split(" "))
csvs_summation =  list(f"summation/yes-cex-expansion/yes-b-cex-yes-ids/{mode}/abstract_{x}.csv.0" for x in moore_machine_names.split(" "))
csvs_discount_sum =  list(f"discount_sum/yes-cex-expansion/yes-b-cex-yes-ids/{mode}/abstract_{x}.csv.0" for x in moore_machine_names.split(" "))
csvs_product =  list(f"product/yes-cex-expansion/yes-b-cex-yes-ids/{mode}/abstract_{x}_prod.csv.0" for x in moore_machine_names.split(" "))
csvs_classification =  list(f"classification/yes-cex-expansion/yes-b-cex-yes-ids/{mode}/abstract_{x}.csv.0" for x in moore_machine_names.split(" "))
## Read in all the data points from each CSV:

summation = list()
discount_sum = list()
product = list()
classification = list()
summation_X = list()
discount_sum_X = list()
product_X = list()
classification_X = list()
remap_Y = list()
remap_X = list()

get_data(csvs_summation, summation, summation_X)
get_data(csvs_discount_sum, discount_sum, discount_sum_X)
get_data(csvs_product, product, product_X)
get_data(csvs_classification, classification, classification_X)
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

print(summation)
print(len(summation))

params_sccve, pcov_sccve = curve_fit(NlnN, summation_X, summation, p0=[0.2121])
params_wccve, pcov_wccve = curve_fit(NlnN, discount_sum_X, discount_sum, p0=[0.2121])
params_sve, pcov_sve = curve_fit(NlnN, product_X, product, p0=[0.2121])
params_wve, pcov_wve = curve_fit(NlnN, classification_X, classification, p0=[0.2121])
params_remap, pcov_remap = curve_fit(NlnN, remap_X, remap_Y, p0=[0.2121])

R2_sccve = compute_R_squared(summation, summation_X, params_sccve)
R2_wccve = compute_R_squared(discount_sum, discount_sum_X, params_wccve)
R2_sve = compute_R_squared(product, product_X, params_sve)
R2_wve = compute_R_squared(classification, classification_X, params_wve)
R2_remap = compute_R_squared(remap_Y, remap_X, params_remap)

#print(membQs)
#print(lstar_X)

#print(params_remap)
#print(pcov_remap)
cm = 1/2.54
plt.figure(figsize=(2*6.1*cm, 2*8.296*cm))
plt.grid()
plt.scatter(discount_sum_X, discount_sum, c="tab:red", s=5, label=r"$\gamma\Sigma$")
plt.scatter(product_X, product, c="tab:blue", s=5, label=r"$\Pi$")
plt.scatter(summation_X, summation, c="tab:green", s=5, label=r"$\Sigma$")
plt.scatter(classification_X, classification, c="tab:orange", s=5, label=r"$\mathbb{N}$")
plt.scatter(remap_X, remap_Y, c="tab:purple", s=5, label="REMAP")


#plt.scatter(lstar_X, membQs, c="tab:red", s=10, label="Lstar Memb Qs")
#plt.scatter(remap_X, prefQs, c="tab:blue", s=10, label="REMAP Pref Qs")
#plt.scatter(lstar_X, membQs, c="tab:red", s=10, label="Lstar Memb Qs")

#max_x = max(max(summation_X), max(discount_sum_X), max(product_X), max(classification_X), max(remap_X))
#x = np.linspace(0, max_x+5, 100)

#plt.plot(x, NlnN(x, params_remap[0]), color="tab:purple")
#plt.plot(x, NlnN(x, params_sccve[0]), color="tab:green")
#plt.plot(x, NlnN(x, params_wccve[0]), color="tab:red")
#plt.plot(x, NlnN(x, params_sve[0]), color="tab:blue")
#plt.plot(x, NlnN(x, params_wve[0]), color="tab:orange")
#plt.plot(x, x, color="black", linestyle="--")

## Pref Q vs Unique Sequences
## Average Solver Time vs Objective Size
## Equivalence Q vs Number of Backtracks

plt.xlabel("Num Backtracks")
plt.ylabel("Num Equiv Qs")
plt.title("W-VE Num EQs")
plt.ylim(0,8)
plt.xlim(0,60)
plt.gca().set_box_aspect(3/4)
#plt.legend(loc="upper left")
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
