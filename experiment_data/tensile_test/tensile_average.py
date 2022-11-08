import numpy as np
import pandas as pd

import glob
from numpy import genfromtxt
from scipy import stats


def read_analysed_file(filename):
    print("\n!!!reading: ", filename)
    print("\n")

    li = []
    df = pd.read_csv(filename, header=0)

    if "_90_" in filename:
        df1 = df[(df["y_strain"] >= 7e-06)]

    if "_0_" in filename:

        df1 = df[(df["x_strain"] >= 7e-06)]
    if "_45_" in filename:

        df1 = df[(df["x_strain"] >= 7e-06)]

    df_all = df1.reset_index(drop=True)
    li.append(len(df_all))

    rename = filename.split(".")

    df_all.to_csv(f"./{rename[1]}_temp.csv")

    return li


def read_reduced_file(filename, min_len):

    df = pd.read_csv(filename, header=0)

    df_all = df.iloc[:min_len, 2:]
    x = filename.split(".")

    df_all.to_csv(f"./{x[1]}.csv")

    return df_all


def calc_min_len(csv_files):
    li = 0
    for filename in csv_files:

        li = read_analysed_file(filename)

    min_len = min(li)

    return min_len


def gen_avg(min_len, temp_files):

    df2 = pd.DataFrame(np.zeros((min_len, 5)))
    print("temp_files: ",temp_files)
    df2.columns = ["x_strain", "y_strain", "x_stress", "y_stress", "r_value"]
    # filename = None
    files_counter = 0
    for filename in temp_files:

        df_all = read_reduced_file(filename, min_len)
        print(filename)
        # print(df_all)

        df2 = df_all.add(df2)
        files_counter += 1
    df2 = df2 / (files_counter)

    # remove blank cells
    df2 = df2.dropna()

    # remove outlier of r_value
    r_mean = stats.trim_mean(df2["r_value"], 0.1)
    # r_mean = df2["r_value"].mean()
    

    df2 = df2.iloc[:, :4]

    if "_0_" in temp_files[0]:

        print("\n\noutput:", f"./ex_data_raw/00deg.csv done!")
        print("\n\nr_value_0 mean:", r_mean)
        print("\n")

        df2.to_csv(f"./ex_data_raw/00deg.csv", header=False, index=False)
    elif "_90_" in temp_files[0]:
        print("\n\noutput:", f"./ex_data_raw/90deg.csv done!")

        df2.to_csv(f"./ex_data_raw/90deg.csv", header=False, index=False)
        print("\n\nr_value_90 mean:", r_mean)
        print("\n")
    elif "_45_" in temp_files[0]:
        print("\n\noutput:", f"./ex_data_raw/45deg.csv done!")

        df2.to_csv(f"./ex_data_raw/45deg.csv", header=False, index=False)
        print("\n\nr_value_45 mean:", r_mean)
        print("\n")


def main():

    tensile_deg = "DC06_l_90_"

    all_files = glob.glob(f"./experiment_data/tensile_test/{tensile_deg}*analysed.csv")
    print("\nall_files: ", all_files)

    csv_files_deg = []

    for filename in all_files:
        if tensile_deg in filename:
            csv_files_deg.append(filename)

    temp_files_deg = []

    temp_files = glob.glob(f"./experiment_data/tensile_test/{tensile_deg}*temp.csv")
    for filename in temp_files:
        if tensile_deg in filename:
            temp_files_deg.append(filename)

    print("\n\ncsv_files:", csv_files_deg)

    min_len = calc_min_len(csv_files_deg)
    gen_avg(min_len, temp_files_deg)


if __name__ == "__main__":
    main()
