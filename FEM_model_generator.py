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
from matplotlib.pyplot import figure

# import YDisplacement


def read_subfolders(path):
    # find all sub folders under target directory
    sub_folders = [
        name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))
    ]
    remove_list = [
        "remove",
    ]
    sub_folders = [ele for ele in sub_folders if ele not in remove_list]
    # print(sub_folders)
    sub_folders = sorted(sub_folders, key=lambda s: float(s.split("_")[1]))

    return sub_folders


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def find_strain(StrainPath, eq_StrainPath, strain_value):
    # calcuate the triaxial strain, plan strain and uniaxial strain
    # strain_value = 2/3, 1 / np.sqrt(3), 1/3

    df_var = pd.read_csv(StrainPath, header=1)

    column_headers_Tria = list(df_var.columns.values)

    x_var = df_var[column_headers_Tria[0]]

    y_var = df_var[column_headers_Tria[1]]

    y_lim_idx = y_var.argmax()
    x_lim = x_var[y_lim_idx]

    if x_lim > int(0.5 * x_var.max()):
        x_lim = x_var.max() - x_lim

    x_lim = x_lim - 0.8

    y_strain = find_nearest(y_var, value=strain_value)

    plane_strain_position_idx = df_var[y_var == y_strain].index.values[0]
    plane_strain_position = x_var.iloc[plane_strain_position_idx]

    df_strain = pd.read_csv(eq_StrainPath, header=1)

    column_headers_strain = list(df_strain.columns.values)

    df_eq_strain = df_strain[column_headers_strain[1]]

    target_strain = df_eq_strain.iloc[plane_strain_position_idx]
    print("\ny_strain_0:", y_strain)

    print("\ntarget_strain:", target_strain)

    plane_strain_position = abs(plane_strain_position - 0.5 * x_var.max())
    print("\nplane_strain_position:", plane_strain_position)

    return target_strain, plane_strain_position


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
    print("\nBatch File generated!\n")


def extract_ThicknessReduction(fem_model):
    source = f"{fem_model}/%ThicknessReduction.csv"
    if not file_exists(source):
        print("File %ThicknessReduction.csv not exist in folder: ", fem_model)

    else:
        # source = str(dir + "/" + str(file))
        # print("\n", fem_model)
        df = pd.read_csv(source, header=1)
        # remove all string:"nan" from thickness reduction column
        if df["A1"].astype(str).str.contains("nan").any() == True:
            df = df[~df["A1"].str.contains("nan|-nan(ind)")]
            df_A1 = df["A1"]
        else:
            df_A1 = df["A1"]
        # print("\ndf:", df)
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

    return state, df
    # max_index,
    # )


def extract_y_displacement(fem_model, max_index):
    source = f"{fem_model}/%ThicknessReduction.csv"

    df = pd.read_csv(source, header=1)

    # remove all string:"nan" from thickness reduction column
    if df["A1"].astype(str).str.contains("nan").any() == True:
        df = df[~df["A1"].str.contains("nan|-nan(ind)")]

    df_time = df["Time"]

    maxTime = df_time.iloc[max_index]
    # print("maxTime: ", maxTime)

    Y_path = f"{fem_model}/Y-displacement.csv"

    df_Y = pd.read_csv(Y_path, skiprows=[0])

    column_headers_Y = list(df_Y.columns.values)

    # Node2 = column_headers_Y[0]
    # print(column_headers_Y)
    # print(Node2)

    time_displacement = df_Y[column_headers_Y[0]]
    df_y_displacement = df_Y[column_headers_Y[3]]

    time_displacement = time_displacement[time_displacement <= maxTime]
    index_Time = time_displacement.idxmax()
    # print("\nmax_time: ",maxTime,max_index)
    print("time_displacement: ", time_displacement.iloc[index_Time])

    maxYDisplacement = df_y_displacement.iloc[index_Time]
    print("YDisplacement: ", maxYDisplacement)

    return maxYDisplacement


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
            if item == f"state {state};":
                print(f"state = {state} inserted already")
                break
            elif item == "$# Strain Curve":
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
                print("Angle NODE inserted already")
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
        print("end_angle: ", end_angle)

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

                    print("NODE for arm line inserted already")
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

    triaxial_strain,p = find_strain(Tria_path, Strain_path, triaxial_strain_type)

    # # find id: plane strain 1/sqr(3)

    # plane_strain_type = 1 / np.sqrt(3)

    # plane_strain = find_strain(Tria_path, Strain_path, plane_strain_type)

    # # find id: uniaxial/ centerpoint
    # uniaxial_strain_type = 1 / 3

    # uniaxial_strain = find_strain(Tria_path, Strain_path, uniaxial_strain_type)

    return triaxial_strain


def find_plane_strain(fem_model):
    # # curve of Triaxiality
    y_strain_path = f"{fem_model}/y_strain.csv"

    # # Strain Curve
    Strain_path = f"{fem_model}/StrainCurve.csv"

    # plane strain occures in y_strain = 0
    y_strain = 0

    plane_strain, distance = find_strain(y_strain_path, Strain_path, y_strain)

    return plane_strain, distance


def extract_datas(path, sub_folders):
    # parameter
    parameter = []

    # ThicknessReduction = []
    list_state = []
    state_broken = []
    list_end_angle = []

    # plane strain of state "x"
    list_plane_strain = []

    # Biaxial strain
    list_Biaxial_strain = []

    # y_displacememt
    list_maxYDisplacement = []

    # plane strain distance, position
    list_distance = []
    min_state = None
    for files in sub_folders:

        parameter.append(re.findall("_(.*)", files)[0])

        fem_model = path + "/" + str(files)
        print("\n\nfem_model: ", fem_model)

        # remove_command_line(fem_model)

        state, df = extract_ThicknessReduction(fem_model)

        state_broken.append(state)
        print("\nbroken state: ", state)

        if min_state is None or state < min_state:
            min_state = state
    max_index = min_state - 1
    print("\n\nmin_state: ", min_state)

    for files in sub_folders:
        list_state.append(min_state)
        fem_model = path + "/" + str(files)
        maxYDisplacement = extract_y_displacement(fem_model, max_index)
        list_maxYDisplacement.append(maxYDisplacement)

        print("\n\nfem_model: ", files)

        # cut_line, angle_command, Tri_point = extract_angle_node(fem_model)

        add_state(fem_model, min_state)
        # add_cut_line(fem_model, cut_line)
        # add_angle_command(fem_model, angle_command)
        end_angle = calc_end_angle(fem_model, max_index)
        list_end_angle.append(end_angle)

        # add_arm_line(fem_model, end_angle, Tri_point)

        Biaxial_strain= calc_strains(fem_model)
        plane_strain, distance = find_plane_strain(fem_model)
        list_distance.append(distance)
        list_Biaxial_strain.append(Biaxial_strain)
        list_plane_strain.append(plane_strain)
        # list_uniaxial_strain.append(uniaxial_strain)
    # sort_values: descending
    # print("\nparameter:\n",parameter)
    # parameter_num = [float(elements) for elements in parameter]
    # print("\nparameter_num:\n",parameter_num)
    yy = np.std(list_maxYDisplacement)
    print("\nStandard Deviation of Y Displacement:", yy)
    # print(list_triaxial_strain)
    df_Model = pd.DataFrame(
        {
            "Model_Name": sub_folders,
            "Param": parameter,
            # "ArmAngle": list_arm_deg,
            # "Angle": list_deg,
            # "Diameter": list_d,
            # "Position": list_x,
            "state_broken": state_broken,
            "State": list_state,
            "Y_Displacement": list_maxYDisplacement,
            # "Thickness_Reduction": list_ThicknessReduction,
            # "Edge_Eq_Strains": list_maxStrain,
            "EndAngle": list_end_angle,
            "Plane_Strain": list_plane_strain,
            "Distance": list_distance,
            # "Uniaxial_Strain": list_uniaxial_strain,
            "Biaxial_Strain": list_Biaxial_strain,
        }
    )
    # df_Model["Param"] = df_Model["Param"].apply(pd.to_numeric)
    df_Model.sort_values(
        ["Param"],
        axis=0,
        # ascending=[False],
        inplace=True,
        ignore_index=True,
    )
    # df_Model.sort_values(
    #     ["Model_Name"],
    #     axis=0,
    #     # ascending=[False],
    #     inplace=True,
    # )

    df_Model.to_csv(f"{path}/test.csv")
    print("\n\nOperation done!")


def plot_strain_distribution(path, sub_folders, strain_type):
    # img = mpimg.imread("specimen_center.png")
    # # imgplot = plt.imshow(img)

    # fig, ax = plt.subplots()

    # im = ax.imshow(img,zorder=0)
    # ax = plt.axes()
    if strain_type == "eq_strain":
        strain_distribution = "StrainCurve.csv"
    elif strain_type == "x_strain":
        strain_distribution = "x_strain.csv"
    elif strain_type == "y_strain":
        strain_distribution = "y_strain.csv"

    for files in sub_folders:

        strain_path = f"{path}/{files}"

        Strain_path = f"{strain_path}/{strain_distribution}"
        df_strain = pd.read_csv(Strain_path, header=1)

        column_headers_strain = list(df_strain.columns.values)

        x_strain = df_strain[column_headers_strain[0]]
        x_strain = x_strain - 0.5 * x_strain.max()

        y_strain = df_strain[column_headers_strain[1]]

        lablename = files

        plt.plot(x_strain, y_strain, label=lablename, zorder=1)
    plt.legend()
    plt.ylabel(f"{strain_type}")
    plt.xlabel("Section along middle line/[mm]")


    # plt.show()


def plot_distance(file_path):
    parameter_name = file_path.split("/")[2]

    Labelname = f"parameter: {parameter_name}"
    # Variable = f"Distance"
    
    df_var = pd.read_csv(file_path + "/test.csv")
    df_var.sort_values(
        ["Param"],
        axis=0,
        # ascending=[False],
        inplace=True,
        ignore_index=True,
    )
    df_var.groupby("Param")["Distance"].mean().plot()
    plt.xticks(np.linspace(df_var["Param"].min(), df_var["Param"].max(), len(df_var["Param"])))

    plt.xlabel(Labelname)
    plt.ylabel("Distance to Center/[mm]")
    plt.savefig(f"{file_path}/distance_to_center.png", format="png")

    plt.show()


def plot_strain(file_path):
    # file_path: folder of variable
    # cuttingline: cutting section of the model in y = cuttingline

    parameter_name = file_path.split("/")[2]

    Labelname = f"parameter: {parameter_name}"
    Variable = f"Param"

    df_var = pd.read_csv(file_path + "/test.csv")
    df_var.sort_values(
        ["Param"],
        axis=0,
        # ascending=[False],
        inplace=True,
        ignore_index=True,
    )
    x_var = df_var[Variable]

    y_plane_strain = df_var["Plane_Strain"]

    y_Bi_strain = df_var["Biaxial_Strain"]

    y_state_var = df_var["State"]
    y_state_End_Angle = df_var["EndAngle"]
    y_state_End_Angle = round(y_state_End_Angle, 1)
    # print(x)-*

    print("Setup Complete")

    plt.xticks(np.linspace(x_var.min(), x_var.max(), len(x_var)))

    plt.plot(x_var, y_Bi_strain, "o-", label="eq. biaxial strain")
    plt.plot(x_var, y_plane_strain, "o-", label="plane strain")
    # plt.plot(x_var, y_uniaxial_strain, "o-", label="uniaxial strain")

    # plt.plot(x_var, y_pred, "r", label="fitted line")
    plt.ylabel("Eq_Strains")
    plt.xlabel(Labelname)

    plt.legend()

    for i, txt in enumerate(y_state_var):
        plt.annotate(txt, (x_var[i], y_plane_strain[i]))

    plt.savefig(f"{file_path}/{parameter_name}_plane_strain.png")
    plt.show()


def plot_3_strains(path, sub_folders):
    title = path.split("/")[2]
    plt.title(f"Parameter: {title}")

    strain_type_list = ["eq_strain", "x_strain", "y_strain"]
    for strain_type in strain_type_list:

        plot_strain_distribution(path, sub_folders, strain_type)
        plt.savefig(f"{path}/{strain_type}_strain_distribution.png", format="png")
        plt.show()

def plot_yield_curve(path, sub_folders):
    for files in sub_folders:

        filepath = f"{path}/{files}/"
   
        sig_x_path = f"{filepath}/sigma_x.csv"
        sig_y_path = f"{filepath}/sigma_y.csv"
        sig_eq_path = f"{filepath}/sigma_eq.csv"



        df_x = pd.read_csv(sig_x_path, header=1)
        df_y = pd.read_csv(sig_y_path, header=1)
        df_eq = pd.read_csv(sig_eq_path, header=1)


        column_headers_x = list(df_x.columns.values)
        column_headers_y = list(df_y.columns.values)
        column_headers_eq = list(df_eq.columns.values)





        sigma_x = df_x[column_headers_x[1]]
        sigma_y = df_y[column_headers_y[1]]
        sigma_eq = df_eq[column_headers_eq[1]]

        y = sigma_y/sigma_x.max()
        x = sigma_x/sigma_x.max()

        # plt.scatter(x, y)
        plt.plot(x, y, "-",label=f"{files}")


        plt.xlabel(r'$\sigma_x/\sigma_{eq}$')

        plt.ylabel(r'$\sigma_y/\sigma_{eq}$')
        # plt.savefig(filepath+"/sigma.svg")
    plt.legend()
    plt.show()



def main():
    # foldername = "YLD_2d_Investigation/sig90"
    foldername = "YLD_2d_Investigation/N"

    path = f"./{foldername}"

    sub_folders = read_subfolders(path)
    # gen_batch_post(foldername, sub_folders)
    # extract_datas(path, sub_folders)

    # plot_3_strains(path, sub_folders)

    # plot_strain(path)
    plot_yield_curve(path, sub_folders)
    # plot_distance(path)
   


if __name__ == "__main__":
    main()
