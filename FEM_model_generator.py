import os
import re

import numpy as np
import pandas as pd


from matplotlib import pyplot as plt

from Extract_data_from_result import extract_ThicknessReduction, extract_y_displacement
from Modify_postfile import add_state, calc_end_angle
from generate_batch_files import gen_batch_post


from yld2000.YLD2000_2d_realM_EN import export_yld_parameter
from YLD_2d_Investigation.Draw_Yield_curve import export_yield_curve
from yld2000.plot_mult_yld import get_normals, plot_yield

from Extract_data_from_result import extract_angle_node,extract_ThicknessReduction,extract_y_displacement
from Modify_postfile import add_angle_command,add_cut_line,add_arm_line

from intersect import intersection



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
    # find the nearest value in an array
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]




def find_strain(StrainPath, eq_StrainPath, strain_value):
    # calcuate the triaxial strain, plan strain and uniaxial strain
    # strain_value = 2/3, 1 / np.sqrt(3), 1/3

    df_var = pd.read_csv(StrainPath, header=1)

    column_headers= list(df_var.columns.values)

    x_var = df_var[column_headers[0]]

    y_var = df_var[column_headers[1]]

    y_lim_idx = y_var.argmax()
    x_lim = x_var[y_lim_idx]

    if x_lim > int(0.5 * x_var.max()):
        x_lim = x_var.max() - x_lim

    x_lim = x_lim - 0.8

    y_strain = find_nearest(y_var, value=strain_value)

    strain_position_idx = df_var[y_var == y_strain].index.values[0]
    strain_position = x_var.iloc[strain_position_idx]

    df_strain = pd.read_csv(eq_StrainPath, header=1)

    column_headers_strain = list(df_strain.columns.values)

    df_eq_strain = df_strain[column_headers_strain[1]]

    target_strain = df_eq_strain.iloc[strain_position_idx]
    print("\ny_strain_0:", y_strain)

    print("\ntarget_strain:", target_strain)

    strain_position = abs(strain_position - 0.5 * x_var.max())
    print("\nplane_strain_position:", strain_position)

    return target_strain, strain_position


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



def intersection_x_y_strain(fem_model):
    x_path = f"{fem_model}/x_strain.csv"
    y_path = f"{fem_model}/y_strain.csv"
    dfx = pd.read_csv(x_path, header=1)

    column_headers_x= list(dfx.columns.values)

    dfx_distance = dfx[column_headers_x[0]]

    dfx_strain = dfx[column_headers_x[1]]
    dfx_distance = dfx_distance - 0.5 * dfx_distance.max()

    x1 = np.array(dfx_distance).reshape(-1, 1)
    y1 = np.array(dfx_strain).reshape(-1, 1)


    dfy = pd.read_csv(y_path, header=1)

    column_headers_y= list(dfy.columns.values)

    # dfy_distance = dfy[column_headers_y[0]]

    dfy_strain = dfy[column_headers_y[1]]


    x2 = x1


    y2 = np.array(dfy_strain).reshape(-1, 1)

    x,y = intersection(x1,y1,x1,y2)


    # plt.plot(x1, y1, c="r")
    # plt.plot(x2, y2, c="g")
    # plt.plot(x, y, "*k")
    # plt.show()
    # print(f"\nx = {x[1]}\ny = {y}")

    biaxial_distance = x[1]

    return biaxial_distance






def calc_strains(fem_model):
    # # curve of Triaxiality
    Tria_path = f"{fem_model}/TriaxialityCurve.csv"

    # # Strain Curve
    Strain_path = f"{fem_model}/StrainCurve.csv"

    # find id: triaxial
    triaxial_strain_type = 2 / 3

    triaxial_strain, p = find_strain(Tria_path, Strain_path, triaxial_strain_type)



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
    list_plane_distance = []
    list_biaxial_distance = []
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

        # max_index = state -1
        # list_state.append(state)

        fem_model = path + "/" + str(files)
        maxYDisplacement = extract_y_displacement(fem_model, max_index)
        list_maxYDisplacement.append(maxYDisplacement)

        print("\n\nfem_model: ", files)

        cut_line, angle_command, Tri_point = extract_angle_node(fem_model)

        add_state(fem_model, min_state)
        # add_state(fem_model, state)

        add_cut_line(fem_model, cut_line)
        add_angle_command(fem_model, angle_command)
        end_angle = calc_end_angle(fem_model, max_index)
        list_end_angle.append(end_angle)

        biaxial_distance = intersection_x_y_strain(fem_model)

        # add_arm_line(fem_model, end_angle, Tri_point)

        Biaxial_strain = calc_strains(fem_model)
        plane_strain, plane_distance = find_plane_strain(fem_model)
        list_plane_distance.append(plane_distance)
        list_biaxial_distance.append(biaxial_distance)

        list_Biaxial_strain.append(Biaxial_strain)
        list_plane_strain.append(plane_strain)

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
            "Plane_Distance": list_plane_distance,
            "Biaxial_Distance": list_biaxial_distance,

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

        # plt.plot(x_strain, y_strain, label=strain_type, zorder=1)
        plt.plot(x_strain, y_strain, label=lablename, zorder=1)
        
    plt.legend()
    plt.ylabel(f"{strain_type}")
    plt.xlabel("Section along middle line/[mm]")
    # plt.xticks(np.linspace(-8, 8, 9))
    plt.ylim(-0.35,0.35)
    plt.grid(True, color="grey", linewidth="1.4", linestyle="-.")
    

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
    

    x = df_var["Param"]
    y1 = df_var["Plane_Distance"]
    y2 = df_var["Biaxial_Distance"]
    plt.plot(x, y1,"o-",label= "Plane_strain_position")
    plt.plot(x,y2, "o-",label = "Biaxial_strain_position")

    plt.xticks(
        np.linspace(df_var["Param"].min(), df_var["Param"].max(), len(df_var["Param"]))
    )
    plt.legend()
    plt.xlabel(Labelname)
    plt.ylabel("Position to Center/[mm]")
    plt.ylim(2.5,6)
    plt.savefig(f"{file_path}/Position_to_center.png", format="png")

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
    plt.ylim(0, 0.1)

    plt.legend()

    for i, txt in enumerate(y_state_var):
        plt.annotate(txt, (x_var[i], y_plane_strain[i]))

    plt.savefig(f"{file_path}/{parameter_name}_plane_strain.png")
    plt.show()


def plot_3_strains(path, sub_folders):
    title = path.split("/")[2]
    # title= "notch"
    plt.title(f"{title}")

    strain_type_list = [ "x_strain", "y_strain","eq_strain"]

    selected_folders = [
                        sub_folders[0], 
                        sub_folders[2], 
                        sub_folders[-1]
                        ]
    print(selected_folders)

    
    for strain_type in strain_type_list:

        plot_strain_distribution(path, selected_folders, strain_type)
        plt.savefig(f"{path}/{strain_type}_strain_distribution.png", transparent=True,dpi=600)
        # plt.xlim(-6,-5)
        # plt.ylim(0,0.1)
        # plt.xticks(np.linspace(-6, -5, 3))

        # plt.savefig(f"{path}/{strain_type}_zoomin_strain_distribution.png", transparent=True,dpi=600)
        plt.show()


def plot_yield_curve(path, sub_folders):

    selected_folders = [sub_folders[0], sub_folders[2], sub_folders[-1]]
    print(selected_folders)

    for i in range(len(selected_folders)):

        # extract parameter and value by reading folders
        parameter = selected_folders[i].split("_")
        parameter_value = float(parameter[1])
        print(selected_folders[i])

        ex_para = ["sig00", "sig45", "sig90", "sigb", "r00", "r45", "r90", "rb", "M"]
        ex_value = [1, 1.0251, 0.9893, 1.2023, 2.1338, 1.5367, 2.2030, 0.8932, 6]
        print(parameter[0])

        # replace the parameter value in ex value
        ex_value_index = ex_para.index(parameter[0])
        print(ex_value_index)
        ex_value = [
            parameter_value if x == ex_value[ex_value_index] else x for x in ex_value
        ]

        print("\n", ex_value)

        # export_yld_parameter(ex_value, path, selected_folders[i])
        # export_yield_curve(path, selected_folders[i])
    plot_yield(path)


def main():
    foldername = "YLD_2d_Investigation/r90"
    # foldername = "test/"

    path = f"./{foldername}"

    sub_folders = read_subfolders(path)
    gen_batch_post(foldername, sub_folders)
    extract_datas(path, sub_folders)

    plot_3_strains(path, sub_folders)

    plot_strain(path) 
    plot_distance(path)
    plot_yield_curve(path, sub_folders)


if __name__ == "__main__":
    main()
