import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import glob



def get_subfolders(folderpath):

    sub_folders = [
        name
        for name in os.listdir(folderpath)
        if os.path.isdir(os.path.join(folderpath, name))
    ]
    return sub_folders


def read_bulge(source):
    
    data = pd.read_csv(source, sep=";", header=0, index_col=False)
    data = data.fillna(0)
    # column_headers = list(data.columns.values)
    # print(column_headers)
    data.columns = ["Stage", "Time", "Radius", "Pressure", "true_strain", "true_stress"]
    # print(data.columns)
    print("reading file...")
    true_strain = np.array(data.true_strain)
    true_stress = np.array(data.true_stress)
    return true_strain, true_stress


def read_flow(file):
    
    data = pd.read_csv(file, sep=";", header=0, index_col=False)
    data = data.fillna(0)
    # column_headers = list(data.columns.values)
    # print(column_headers)
    data.columns = [
        "thickness_strain_avg",
        "thickness_strain_sig",
        "thickness_strain_sig_elas",
    ]
    # print(data.columns)
    print("reading file: ", file)
    true_strain = np.array(data.thickness_strain_avg)
    true_stress = np.array(data.thickness_strain_sig)

    return true_strain, true_stress


def calc_params(true_strain, true_stress):

    thickness = 1.5 * np.exp(-true_strain)

    x_strain = true_strain / 2

    y_strain = x_strain

    sigma_x = true_stress

    sigma_y = true_stress

    plastic_x = x_strain - sigma_x / (206000 * (1 + 0.3))

    plastic_y = plastic_x

    return thickness, sigma_x, plastic_x, sigma_y, plastic_y, x_strain, y_strain


def plot_t(true_strain, true_stress, pw_total):
    plt.plot(true_strain, true_stress)
    plt.xlabel("True strain in %")
    plt.ylabel("True stress in MPa")
    plt.show()

    plt.plot(pw_total, true_stress, "-")
    plt.xlabel("Plastic Work in MPa")
    plt.ylabel("True stress in MPa")
    plt.show()


def plastic_work(sigma, plastic):
    list_pw = []
    list_pw.append(0)
    pw = 0
    for i in range(1, len(plastic) - 1):
        pw = (
            list_pw[i - 1]
            + np.multiply((sigma[i] + sigma[i + 1]), (plastic[i + 1] - plastic[i]))
            * 0.5
        )
        list_pw.append(pw)
    pw_end = 0
    list_pw.append(pw_end)
    # print("\nsigma: ",sigma)

    # print("\nlist_pw: ",list_pw)
    # print("\nlen(list_pw): ", len(sigma))

    return list_pw


def plasticwork_total(sigma_x, plastic_x, sigma_y, plastic_y):
    px = plastic_work(sigma_x, plastic_x)
    py = plastic_work(sigma_y, plastic_y)
    pw_total = np.add(px, py)
    # print("\npx: ",px)

    return pw_total, px, py

def write_csv(sigma_x, sigma_y, x_strain, y_strain, filename):

    x_stress = sigma_x
    y_stress = sigma_y

    df_Model = pd.DataFrame(
        {
            "x_strain": x_strain,
            "y_strain": y_strain,
            "x_stress": x_stress,

            "y_stress": y_stress,
        }
    )
    x = filename.split(".")
    df_Model.to_csv(f"./{x[1]}_analysed.csv")
    # df_Model.to_csv(f"/{filename}_analysed.csv")
    print("success")

def main():
    # sub_folders = get_subfolders("./experiment_data/bulge_test")
    # for i in sub_folders:

    # filename = "DC06_utg_1k5mm_400kN_P3_additional_Bulge_Test_data.csv"

    csv_files = glob.glob("./experiment_data/bulge_test/*Test_data.csv")



    li = []

    for filename in csv_files:
       
       
    

        true_strain, true_stress = read_bulge(filename)

        
        
        thickness, sigma_x, plastic_x, sigma_y, plastic_y, x_strain, y_strain = calc_params(
            true_strain, true_stress
        )
        pw_total, px, py = plasticwork_total(sigma_x, plastic_x, sigma_y, plastic_y)
        # print(plasticwork)
        # print(len(pw_total))
        # print(len(true_stress))
        # print("\ndimension: ", np.shape(pw_total))

        # plot_t(true_strain, true_stress, pw_total)
        write_csv(sigma_x, sigma_y, x_strain, y_strain, filename)


if __name__ == "__main__":
    main()
