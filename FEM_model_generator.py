import os
import re
import time

import numpy as np
import pandas as pd
from scipy import interpolate

from os.path import exists as file_exists
from traceback import print_tb
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

# import YDisplacement


def read_subfolders(path):
    # find all sub folders under target directory
    sub_folders = [
        name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))
    ]
    # print(sub_folders)

    return sub_folders


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def find_strain(TriaPath, StrainPath, strain_value):
    # calcuate the triaxial strain, plan strain and uniaxial strain
    # strain_value = 2/3, 1 / np.sqrt(3), 1/3

    df_var = pd.read_csv(TriaPath, header=1)

    column_headers_Tria = list(df_var.columns.values)

    x_var = df_var[column_headers_Tria[0]]

    y_var = df_var[column_headers_Tria[1]]

    f_tri = interpolate.interp1d(x_var, y_var, fill_value="extrapolate")

    y_lim_idx = y_var.argmax()
    x_lim = x_var[y_lim_idx]

    if x_lim > int(0.5 * x_var.max()):
        x_lim = x_var.max() - x_lim

    x_lim = x_lim - 0.8

    xnew = np.arange(x_lim, int(0.5 * x_var.max()), 0.0001)
    # print("xnew: ", xnew)
    ynew = f_tri(xnew)  # use interpolation function returned by `interp1d`
    y_plane_strain = find_nearest(ynew, value=strain_value)

    for i in np.arange(x_lim, int(0.5 * x_var.max()), 0.0001):
        if f_tri(i) == y_plane_strain:
            break

    x_plane_strain = i

    df_strain = pd.read_csv(StrainPath, header=1)

    column_headers_strain = list(df_strain.columns.values)

    x_strain = df_strain[column_headers_strain[0]]

    y_strain = df_strain[column_headers_strain[1]]

    f_strain = interpolate.interp1d(x_strain, y_strain, fill_value="extrapolate")

    xnew_strain = np.arange(x_lim, int(0.5 * x_strain.max()), 0.0001)
    # print("xnew: ", xnew_strain)
    ynew_strain = f_strain(
        xnew_strain
    )  # use interpolation function returned by `interp1d`

    target_strain = f_strain(x_plane_strain)

    return target_strain


def remove_command_line(fem_model):
    # remove unneccessary lines in notch.k

    source = f"{fem_model}/notch.k"
    # print(source)

    with open(source, "r") as f:
        data = f.readlines()
    pos = data.index("*ELEMENT_SHELL\n")
    with open(source, "w") as f:
        f.writelines(data[pos:])

    print(f"unneccessary lines in {fem_model} removed\n")


def gen_batch_post(foldername, sub_folders):
    # generate a batch file to extract data at one time
    with open(f"./{foldername}/post_command.bat", "w") as f:
        f.write("@echo off\n\n")

        for i in sub_folders:
            i = foldername + "\\" + str(i)
            dir = str("Z:\MA\Material_Model" + "\\" + i)
            command_line = f'start /D "{dir}" post_command.bat'
            # print(command_line)
            f.write(f'\nstart /D "{dir}" post_command.bat\n\n')

        f.write("\n\necho *** FINISHED WITH POST COMMAND SCRIPT ***")


def extract_ThicknessReduction(fem_model):
    source = f"{fem_model}/%ThicknessReduction.csv"
    if not file_exists(source):
        print("File %ThicknessReduction.csv not exist in folder: ", fem_model)

    else:
        # source = str(dir + "/" + str(file))
        print("\n", fem_model)
        df = pd.read_csv(source, header=1)
        # remove all string:"nan" from thickness reduction column
        if df["A1"].astype(str).str.contains("nan").any() == True:
            df = df[~df["A1"].str.contains("nan|-nan(ind)")]
            df_A1 = df["A1"]
        else:
            df_A1 = df["A1"]

        # convert string back to numeric
        df_A1 = df_A1.apply(pd.to_numeric)

        df_A1 = df_A1[df_A1 < 30]
        # print(df_A1)
        max_index = df_A1.idxmax()
        # print(max_index)

        # maxThicknessReduction = df_A1.max()
        state = max_index + 1

        # print(f"state: {max_index+1}\n")
        # maxThicknessReduction = round(maxThicknessReduction, 3)

        # list_ThicknessReduction.append(maxThicknessReduction)

    return (
        state,
        max_index,
    )


def extract_angle_node(fem_model):
    with open(f"{fem_model}/lspost.msg", "r") as fp:
        a = [x.rstrip() for x in fp]
        is_second_occurance = 0
        for item in a:
            if item.startswith("NODE"):
                if is_second_occurance == 1:
                    l2 = item.split()
                    Tri_point = l2[2]
                    print("tri point:", Tri_point)
                    is_second_occurance += 1
                elif is_second_occurance == 2:
                    l2 = item.split()
                    angle1 = l2[2]
                    print("angle1:", angle1)
                    is_second_occurance += 1

                elif is_second_occurance == 3:
                    l2 = item.split()
                    angle2 = l2[2]
                    print("angle2:", angle2)
                    is_second_occurance += 1

                elif is_second_occurance == 4:
                    l2 = item.split()
                    angle3 = l2[2]
                    print("angle3:", angle3)
                    break

                else:
                    idx1 = a.index(item, a.index(item))
                    l1 = item.split()
                    id1 = l1[2]
                    print("center point:", id1)

                    is_second_occurance += 1

        cut_line = f"splane dep1 {id1}  0.000  1.000  0.000"
        print("center cut line: ", cut_line)

        # insert "limited Thickness Reduction state" in gen_pos.cfile

        angle_command = f"measure angle3 N{angle1}/0 N{angle2}/0 N{angle3}/0 ;"
    return cut_line, angle_command, Tri_point


def add_state(fem_model, state):
    with open(f"{fem_model}/gen_post.cfile", "r+") as f:
        # insert the limited state in gen_post.cfile
        a = [x.rstrip() for x in f]
        for item in a:
            if item.startswith(f"state {state};"):
                print(f"state = {state} inserted already")
                break
            elif item.startswith("$# Strain Curve"):
                index_state = a.index(item, a.index(item))

                a.insert(index_state, f"state {state};")
                # Inserts "Hello everyone" into `a`
                print(f"state = {state} inserted!")

                break
        f.seek(0)
        f.truncate()
        # Write each line back
        for line in a:
            f.write(line + "\n")


def add_cut_line(fem_model, cut_line):
    with open(f"{fem_model}/gen_post.cfile", "r+") as f:
        a = [x.rstrip() for x in f]

        for item in a:

            if item.startswith(cut_line):
                print("Section NODE inserted already")
                if "splane drawcut" in item:
                    idx1 = a.index(item, a.index(item))
                    print("index1 =  ", idx1)

                    idx2 = a.index(item, idx1 + 2)
                    print("index2 =  ", idx2)

                    idx3 = a.index(item, idx2 + 2)
                    print("index3 =  ", idx3)

                    idx4 = a.index(item, idx3 + 2)
                    print("index4 =  ", idx4)

                    idx5 = a.index(item, idx4 + 2)
                    print("index5 =  ", idx5)

                break

            elif item.startswith("splane drawcut"):

                idx1 = a.index(item, a.index(item))
                print("index1 =  ", idx1)
                a.insert(idx1, cut_line)

                idx2 = a.index(item, idx1 + 2)
                print("index2 =  ", idx2)
                a.insert(idx2, cut_line)

                idx3 = a.index(item, idx2 + 2)
                print("index3 =  ", idx3)
                a.insert(idx3, cut_line)

                idx4 = a.index(item, idx3 + 2)
                print("index4 =  ", idx4)
                a.insert(idx4, cut_line)

                idx5 = a.index(item, idx4 + 2)
                print("index5 =  ", idx5)
                a.insert(idx5, cut_line)

                # idx6 = a.index(item, idx5 + 3)
                # print("index6 =  ", idx6)
                # a.insert(idx6, cut_line_60)
                print("Section NODE inserted!")

                break
        f.seek(0)
        f.truncate()
        # Write each line back
        for line in a:
            f.write(line + "\n")


def add_angle_command(fem_model, angle_command):
    with open(f"{fem_model}/gen_post.cfile", "r+") as f:
        a = [x.rstrip() for x in f]
        for item in a:

            if item.startswith(angle_command):
                print("Angle NODE insert already")
                break
            elif item.startswith("measure history angle3 a"):
                idx_angle = a.index(item, a.index(item))
                print("idx_angle =  ", idx_angle)
                a.insert(idx_angle, angle_command)
                break
        f.seek(0)
        f.truncate()
        # Write each line back
        for line in a:
            f.write(line + "\n")


def calc_end_angle(fem_model, max_index):
    # measure the arm angle at state "x"
    end_angle_path = f"{fem_model}/ArmAngle2.csv"
    if not file_exists(end_angle_path):
        end_angle = 0
        # list_end_angle.append(end_angle)

        print("File ArmAngle2.csv not exist in folder: ", fem_model)
    else:

        df_end_angle = pd.read_csv(end_angle_path, skiprows=[0])

        column_headers_end_angle = list(df_end_angle.columns.values)

        end_angles = column_headers_end_angle[1]

        df_end_angle = df_end_angle[end_angles]
        end_angle = float(df_end_angle[max_index]) - 90
        print("\nend_angle: ", end_angle)

    return end_angle


def add_arm_line(fem_model, end_angle, Tri_point):
    section_angle = round(np.tan(end_angle * np.pi / 180), 3)
    arm_line = f"splane dep1 {Tri_point}  {section_angle}  1.000  0.000"
    # print(arm_line)

    with open(f"{fem_model}/gen_post.cfile", "r+") as fa:
        a = [x.rstrip() for x in fa]

        for item in a:

            if "splane drawcut" in item:
                idx1 = a.index(item, a.index(item))

                idx2 = a.index(item, idx1 + 2)

                idx3 = a.index(item, idx2 + 2)

                idx4 = a.index(item, idx3 + 2)

                idx5 = a.index(item, idx4 + 2)

                idx6 = a.index(item, idx5 + 2)
                # line = a.readlines()
                print("arm lline: ", a[idx6 - 1])
                if arm_line == a[idx6 - 1]:

                    print("NODE for arm line insert already")
                    break
                else:
                    print("index6 =  ", idx6)
                    a.insert(idx6, arm_line)
                    print("arm_line inserted!")
                    break

        # Go to start of file and clear it
        fa.seek(0)
        fa.truncate()
        # Write each line back
        for line in a:
            fa.write(line + "\n")


def calc_strains(fem_model):
    # # curve of Triaxiality
    Tria_path = f"{fem_model}/TriaxialityCurve.csv"

    # # Strain Curve
    Strain_path = f"{fem_model}/StrainCurve.csv"

    # find id: triaxial
    triaxial_strain_type = 2 / 3

    triaxial_strain = find_strain(Tria_path, Strain_path, triaxial_strain_type)

    # find id: plane strain 1/sqr(3)

    plane_strain_type = 1 / np.sqrt(3)

    plane_strain = find_strain(Tria_path, Strain_path, plane_strain_type)

    # find id: uniaxial/ centerpoint
    uniaxial_strain_type = 1 / 3

    uniaxial_strain = find_strain(Tria_path, Strain_path, uniaxial_strain_type)

    return triaxial_strain, plane_strain, uniaxial_strain


def extract_datas(path, sub_folders):
    # parameter
    parameter = []

    # ThicknessReduction = []
    list_state = []
    list_end_angle = []

    # plane strain of state "x"
    list_plane_strain = []

    # uniaxial strain
    list_uniaxial_strain = []

    # triaxial strain
    list_triaxial_strain = []

    for files in sub_folders:

        parameter.append(re.findall("_(.*)", files)[0])

        fem_model = path + "/" + str(files)
        print("\n\nfem_model: ", fem_model)
        print("\n")
        # remove_command_line(fem_model)

        state, max_index = extract_ThicknessReduction(fem_model)
        print("\nstate: ", state)
        list_state.append(state)

        cut_line, angle_command, Tri_point = extract_angle_node(fem_model)

        add_state(fem_model, state)
        add_cut_line(fem_model, cut_line)
        add_angle_command(fem_model, angle_command)
        end_angle = calc_end_angle(fem_model, max_index)
        list_end_angle.append(end_angle)

        add_arm_line(fem_model, end_angle, Tri_point)

        triaxial_strain, plane_strain, uniaxial_strain = calc_strains(fem_model)
        list_triaxial_strain.append(triaxial_strain)
        list_plane_strain.append(plane_strain)
        list_uniaxial_strain.append(uniaxial_strain)

    # print(list_triaxial_strain)
    df_Model = pd.DataFrame(
        {
            "Model_Name": sub_folders,
            "Param": parameter,
            # "ArmAngle": list_arm_deg,
            # "Angle": list_deg,
            # "Diameter": list_d,
            # "Position": list_x,
            "State": list_state,
            # "Y_Displacement": list_maxYDisplacement,
            # "Thickness_Reduction": list_ThicknessReduction,
            # "Edge_Eq_Strains": list_maxStrain,
            "EndAngle": list_end_angle,
            "Plane_Strain": list_plane_strain,
            "Uniaxial_Strain": list_uniaxial_strain,
            "Triaxial_Strain": list_triaxial_strain,
        }
    )
    df_Model.to_csv(f"{path}/test.csv")
    print("Operation done!")


def plot_strain_distribution(path,sub_folders):
    # img = mpimg.imread("specimen_center.png")
    # # imgplot = plt.imshow(img)
    fig, ax = plt.subplots()
    # im = ax.imshow(img,zorder=0)

    for files in sub_folders:


        strain_path = f"{path}/{files}"

        Strain_path = f"{strain_path}/StrainCurve.csv"
        df_strain = pd.read_csv(Strain_path, header=1)

        column_headers_strain = list(df_strain.columns.values)

        x_strain = df_strain[column_headers_strain[0]]
        x_strain = x_strain - 0.5 * x_strain.max()

        y_strain = df_strain[column_headers_strain[1]]

        # f_strain = interpolate.interp1d(x_strain, y_strain, fill_value="extrapolate")

        # xnew_strain = np.arange(0, int(x_strain.max()), 0.0001)
        # # print("xnew: ", xnew_strain)
        # ynew_strain = f_strain(xnew_strain)  # use interpolation function returned by `interp1d`

        # plane_strain = f_strain(x_plane_strain)
        lablename = files
        # print(f"{lablename}")

        # labelname = f"{file}"
        ax.plot(x_strain, y_strain,label = lablename,zorder=1)
    ax.legend()
    title = path.split("/")[2]
    ax.set_title(f"Parameter: {title}")
    ax.set_ylabel("Eq. Strain")
    ax.set_xlabel("Section along middle line/[mm]")
    plt.savefig(f"{path}/{title}.png", format='png', transparent=True)

    plt.show()


def main():
    foldername = "YLD_2d_Investigation/r00"
    # foldername = "SwiftN/YLD_iso"

    path = f"./{foldername}"

    sub_folders = read_subfolders(path)
    # gen_batch_post(foldername, sub_folders)
    # extract_datas(path, sub_folders)
 
    plot_strain_distribution(path,sub_folders)
 


if __name__ == "__main__":
    main()
