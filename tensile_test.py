import glob
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from FEM_model_modify import read_subfolders, plot_3_strains


def read_csv_files(filename):

    data = pd.read_csv(
        filename,
        header=0,
    )
    column_headers = list(data.columns.values)
    # print(column_headers)

    length = data[column_headers[0]]
    x = data[column_headers[1]]
    y = data[column_headers[2]]
    eps_x = data[column_headers[3]]
    eps_y = data[column_headers[4]]
    phiM = data[column_headers[5]]

    return length, x, y, eps_x, eps_y, phiM


def calib_x(eps_x, phiM, length):
    phiM = np.array(phiM).reshape(-1, 1)
    phiM_newarr = np.array_split(phiM, 2)

    # print(newarr)
    # loccal_min_index1 = phiM_newarr[0].argmin(axis=0)[0]
    # loccal_min_index2 = phiM_newarr[1].argmin(axis=0)[0]

    eps_x = np.array(eps_x).reshape(-1, 1)
    eps_x_newarr = np.array_split(eps_x, 2)

    loccal_min_index1 = eps_x_newarr[0].argmax(axis=0)[0]
    loccal_min_index2 = eps_x_newarr[1].argmax(axis=0)[0]

    length_array = np.array(length).reshape(-1, 1)
    length_newarr = np.array_split(length_array, 2)
    length_min1 = length_newarr[0][loccal_min_index1][0]
    length_min2 = length_newarr[1][loccal_min_index2][0]
    # print(length_max1,length_max2)
    middle = (length_min1 + length_min2) / 2
    print(middle)
    calib_length = length - middle
    # print(length)
    return calib_length


def plot_diagrams(calib_length, eps_x, eps_y, phiM, label):
    # length = length - 0.5*max(length)
    # plt.plot(calib_length, eps_x, "-", label="x_strain")
    # plt.plot(calib_length, eps_y, "-", label="y_strain")
    plt.plot(calib_length, phiM, ".", label=label)
    # plt.ylim(-0.1, 0.25)
    plt.legend()

    plt.grid(True, color="grey", linewidth="1.4", linestyle="-.")
    # plt.show()


def main():

    path = "./YLD_2d_Investigation/experiment_data/"

    group = "7"
    csv_files = glob.glob(f"{path}{group}*.csv")
    
    csv_files = sorted(csv_files)
    
    for filename in csv_files:
        print(filename)
        label = filename.split("/")[3].split(".")[0]
        # filename = "YLD_2d_Investigation/experiment_data/7p1_3.csv"
        length, x, y, eps_x, eps_y, phiM = read_csv_files(filename)
        calib_length = calib_x(eps_x, phiM, length)
        plot_diagrams(calib_length, eps_x, eps_y, phiM, label)
    plt.savefig(f"{path}/eq_strain_{group}.png")
    
    foldername = "test/"

    path = f"./{foldername}"

    # sub_folders = read_subfolders(path)
    # plot_3_strains(path, sub_folders)
    
    plt.show()


if __name__ == "__main__":
    main()
