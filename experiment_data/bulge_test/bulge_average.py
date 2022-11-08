import numpy as np
import pandas as pd

import glob
import os


def read_analysed_file(filename):

    li = []
    df = pd.read_csv(filename, header=0)
    df1 = df[(df["x_strain"] >= 0.0003) & (df["x_stress"] > 1)]
    df_all = df1.reset_index(drop=True)
    li.append(len(df_all))

    x = filename.split(".")

    df_all.to_csv(f"./{x[1]}_temp.csv")

    return li


def read_reduced_file(filename, min_len):

    df = pd.read_csv(filename, header=0)

    df_all = df.iloc[:min_len, 2:]
    return df_all


def main():

    csv_files = glob.glob("./experiment_data/bulge_test/*analysed.csv")

    print(csv_files)
    for filename in csv_files:

        li = read_analysed_file(filename)

    min_len = min(li)

    print("\nmin_len: ", min_len)
    print("\n")

    temp_file = glob.glob("./experiment_data/bulge_test/*temp.csv")
    df2 = pd.DataFrame(np.zeros((min_len, 4)))
    df2.columns = ["x_strain", "y_strain", "x_stress", "y_stress"]
    for filename in temp_file:
        df_all = read_reduced_file(filename, min_len)
        print(filename)
        print(df_all)

        df2 = df_all.add(df2)

    df2 = df2 / len(temp_file)
    df2 = df2.dropna()

    # print(len(csv_files))
    df2.to_csv("./ex_data_raw/1vs1.csv", header=False, index=False)


if __name__ == "__main__":
    main()
