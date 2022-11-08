import csv
import glob

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def read_filename(filename):

    txt_file = filename

    with open(txt_file, "rb") as fp:
        a = [x.rstrip() for x in fp]
        for item in a:
            if item.startswith(b"Specimen thickness"):
                l1 = item.split()
                # print(l1)
                thickness = float(l1[2])
                print(f"thickness: {thickness} mm")

            if item.startswith(b"Marked initial gage length"):
                l2 = item.split()
                # print(l2)
                gage_length = float(l2[4])
                print(f"gage_length: {gage_length} mm")

            if item.startswith(b"Specimen width"):
                l3 = item.split()
                # print(l3)
                Specimen_width = float(l3[2])
                print(f"Specimen_width: {Specimen_width} mm")

    data = pd.read_csv(txt_file, sep=";", header=6, encoding="latin-1", index_col=False)
    # column_headers = list(data.columns.values)
    # print(column_headers)

    data.columns = [
        "Standard_force",
        "Standard_travel",
        "Test_time",
        "Transverse_strain",
    ]
    # print(column_headers[0])
    Standard_force = np.array(data.Standard_force)
    Standard_travel = np.array(data.Standard_travel)
    Transverse_strain = np.array(data.Transverse_strain)

    return (
        Standard_force,
        Standard_travel,
        Transverse_strain,
        Specimen_width,
        gage_length,
        thickness,
    )


def calc_params(
    Standard_force,
    Standard_travel,
    Transverse_strain,
    Specimen_width,
    gage_length,
    thickness,
):
    E_Modul = 206000
    p_r = 0.3  # Poisson's ratio
    transverse_true_strain = np.log(1 - np.divide(Transverse_strain, (Specimen_width)))

    # print("transverse_true_strain: ", transverse_true_strain)

    eng_strain = np.divide(Standard_travel, gage_length)

    eng_stress = np.divide(Standard_force, (thickness * Specimen_width))

    true_stress = eng_stress * (1 + eng_strain)

    true_strain = np.log(1 + eng_strain)

    longitudinal = true_strain - np.divide(eng_stress, E_Modul)

    transverse = transverse_true_strain + p_r * np.subtract(true_strain, longitudinal)

    return (
        transverse_true_strain,
        transverse,
        true_stress,
        true_strain,
        longitudinal,
        eng_strain,
        eng_stress,
    )


def calc_plastic_work(true_stress, longitudinal):
    list_pw = []
    list_pw.append(0)
    pw = 0
    for i in range(1, len(true_stress) - 1):
        pw = (
            list_pw[i - 1]
            + np.multiply(
                (true_stress[i] + true_stress[i + 1]),
                (longitudinal[i + 1] - longitudinal[i]),
            )
            * 0.5
        )
        list_pw.append(pw)
    pw_end = 0
    list_pw.append(pw_end)
    return list_pw


def cal_r_value(transverse, longitudinal):
    r_value = transverse / (-longitudinal - transverse)
    return r_value


def plot_eng(eng_strain, eng_stress):
    neck_stress = eng_stress.max()
    x_max = eng_stress.argmax()
    neck_strain = eng_strain[x_max]
    plt.plot(eng_strain, eng_stress, "-", neck_strain, neck_stress, "ro")

    plt.xlabel("Eng. strain in %")
    plt.ylabel("Eng. stress in MPa")
    # plt.show()
    return x_max


def plot_true(true_strain, true_stress):

    plt.plot(true_strain, true_stress, "-")
    plt.xlabel("True strain in %")
    plt.ylabel("True stress in MPa")
    plt.show()


def plot_r_value(true_strain, r_value):
    plt.plot(true_strain, r_value)
    plt.xlabel("True stress in %")
    plt.ylabel("r-value")
    plt.ylim([0, 5])
    plt.show()


def plot_PlasticWork(list_pw, true_stress):
    plt.plot(list_pw, true_stress, "-")
    plt.xlabel("Plastic Work in MPa")
    plt.ylabel("True stress in MPa")
    plt.show()


def write_00deg_csv(
    strain, stress, eng_strain, eng_stress, transverse_true_strain, r_value, filename
):

    x_max = plot_eng(eng_strain, eng_stress)
    # plt.show()

    x_strain = strain[0:x_max]
    y_strain = transverse_true_strain[0:x_max]

    x_stress = stress[0:x_max]
    y_stress = [0] * len(x_stress)
    r_value = r_value[0:x_max]

    df_Model = pd.DataFrame(
        {
            "x_strain": x_strain,
            "y_strain": y_strain,
            "x_stress": x_stress,
            "y_stress": y_stress,
            "r_value": r_value,
        }
    )
    x = filename.split(".")
    print(x[0])
    df_Model.to_csv(f"./{x[1]}_analysed.csv")

    return x_strain, x_stress


# def write_45deg_csv(strain, stress, transverse_true_strain, r_value, filename):
#     # remove necking
#     y_strain = transverse_true_strain
#     a_index = list(range(0, len(transverse_true_strain)))

#     x_max = y_strain.argmax()

#     y_strain_cut = y_strain[0:x_max]
#     y = y_strain_cut.min()
#     x = y_strain_cut.argmin()

#     # plt.plot(a_index, y_strain, "-", x, y, "ro")

#     # plt.show()

#     x_strain = strain[0:x]
#     y_strain = transverse_true_strain[0:x]

#     x_stress = stress[0:x]
#     y_stress = 0
#     r_value = r_value[0:x]

#     df_Model = pd.DataFrame(
#         {
#             "x_strain": x_strain,
#             "y_strain": y_strain,
#             "x_stress": x_stress,
#             "y_stress": y_stress,
#             "r_value": r_value,
#         }
#     )
#     x = filename.split(".")
#     print(x[0])
#     df_Model.to_csv(f"./{x[1]}_analysed.csv")

#     return x_strain, x_stress


def write_90deg_csv(
    strain, stress, eng_strain, eng_stress, transverse_true_strain, r_value, filename
):

    x_max = plot_eng(eng_strain, eng_stress)

    # plt.show()

    y_strain = strain[0:x_max]
    x_strain_90 = transverse_true_strain[0:x_max]

    y_stress = stress[0:x_max]
    x_stress = [0] * len(y_stress)
    r_value = r_value[0:x_max]

    df_Model = pd.DataFrame(
        {
            "x_strain": x_strain_90,
            "y_strain": y_strain,
            "x_stress": x_stress,
            "y_stress": y_stress,
            "r_value": r_value,
        }
    )
    x = filename.split(".")
    print(x[0])
    df_Model.to_csv(f"./{x[1]}_analysed.csv")

    return x_strain_90, x_stress


def plot_eng_strain(x_strain, x_stress):
    max_stress = x_stress.max()
    strain = x_strain[x_stress.argmax()]

    print("max_stress: ", max_stress)
    print("strain: ", strain)

    plt.plot(x_strain, x_stress, "-", strain, max_stress, "ro")
    plt.xlabel("x_strain in %")
    plt.ylabel("x_stress in MPa")

    plt.show()


def gen_csv(filename):
    (
        Standard_force,
        Standard_travel,
        Transverse_strain,
        Specimen_width,
        gage_length,
        thickness,
    ) = read_filename(filename)
    (
        transverse_true_strain,
        transverse,
        true_stress,
        true_strain,
        longitudinal,
        eng_strain,
        eng_stress,
    ) = calc_params(
        Standard_force,
        Standard_travel,
        Transverse_strain,
        Specimen_width,
        gage_length,
        thickness,
    )
    r_value = cal_r_value(transverse, longitudinal)
    if "_90_" in filename:
        write_90deg_csv(
            true_strain,
            true_stress,
            eng_strain,
            eng_stress,
            transverse_true_strain,
            r_value,
            filename,
        )
    if "_0_" in filename:
        write_00deg_csv(
            true_strain,
            true_stress,
            eng_strain,
            eng_stress,
            transverse_true_strain,
            r_value,
            filename,
        )
    if "_45_" in filename:
        write_00deg_csv(
            true_strain,
            true_stress,
            eng_strain,
            eng_stress,
            transverse_true_strain,
            r_value,
            filename,
        )


def main():

    csv_files = glob.glob("./experiment_data/tensile_test/DC06_l_*.txt")

    for filename in csv_files:

        if "_90_" in filename:
            print(filename)
            gen_csv(filename)
            print("done!\n")

        # plot_eng_strain(x_strain, x_stress)


if __name__ == "__main__":
    main()
