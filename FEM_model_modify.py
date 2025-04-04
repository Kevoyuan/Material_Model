import os
import re

import numpy as np
import pandas as pd


from matplotlib import pyplot as plt

from Extract_data_from_result import (
    extract_ThicknessReduction,
    extract_y_displacement,
    extract_angle_node,
    set_state_from_y_dis,
)
from Modify_postfile import add_state, calc_end_angle, add_angle_command, add_cut_line, add_arm_line
from generate_batch_files import gen_batch_post


from yld2000.YLD2000_2d_realM_EN import export_yld_parameter
from YLD_2d_Investigation.Draw_Yield_curve import export_yield_curve
from yld2000.plot_mult_yld import plot_yield

from intersect import intersection
from read_subfolders import read_subfolders
from calc_strains import calculate_strains


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


def extract_datas(path, sub_folders):
    # Initialize data collection lists
    parameter = []
    list_state = []
    state_broken = []
    list_end_angle = []
    list_plane_strain = []
    list_Biaxial_strain = []
    list_maxYDisplacement = []
    list_plane_distance = []
    list_biaxial_distance = []
    
    for files in sub_folders:
        parameter.append(re.findall("_(.*)", files)[0])
        fem_model = os.path.join(path, files)
        print(f"\n\nfem_model: {fem_model}")
        
        # Fixed state for all models
        state = 22
        state_broken.append(state)
        print(f"\nstate: {state}")
        
        max_index = state - 1
        list_state.append(state)
        
        # Extract data
        maxYDisplacement = extract_y_displacement(fem_model, max_index)
        list_maxYDisplacement.append(maxYDisplacement)
        
        print(f"\n\nfem_model: {files}")
        
        cut_line, angle_command, Tri_point = extract_angle_node(fem_model)
        
        # Add state and lines
        add_state(fem_model, state)
        add_cut_line(fem_model, cut_line)
        add_angle_command(fem_model, angle_command)
        
        end_angle = calc_end_angle(fem_model, max_index)
        list_end_angle.append(end_angle)
        
        # Calculate strains
        biaxial_distance, biaxial_strain, plane_distance, plane_strain = calculate_strains(fem_model)
        
        list_plane_distance.append(plane_distance)
        list_biaxial_distance.append(biaxial_distance)
        list_Biaxial_strain.append(biaxial_strain)
        list_plane_strain.append(plane_strain)
    
    # Calculate statistics
    y_displacement_std = np.std(list_maxYDisplacement)
    print(f"\nStandard Deviation of Y Displacement: {y_displacement_std}")
    
    # Create and save dataframe
    df_Model = pd.DataFrame({
        "Model_Name": sub_folders,
        "Param": parameter,
        "state_broken": state_broken,
        "State": list_state,
        "Y_Displacement": list_maxYDisplacement,
        "EndAngle": list_end_angle,
        "Plane_Strain": list_plane_strain,
        "Plane_Distance": list_plane_distance,
        "Biaxial_Distance": list_biaxial_distance,
        "Biaxial_Strain": list_Biaxial_strain,
    })
    
    # Sort dataframe
    df_Model.sort_values(["Param"], axis=0, inplace=True, ignore_index=True)
    
    # Save to CSV
    output_path = os.path.join(path, "test.csv")
    df_Model.to_csv(output_path)
    print("\n\nOperation done!")
    
    return df_Model


def plot_strain_distribution(path, sub_folders, strain_type):
    # Map strain type to file name
    strain_files = {
        "eq_strain": "StrainCurve.csv",
        "x_strain": "x_strain.csv",
        "y_strain": "y_strain.csv",
        "xy_strain": "xy_strain.csv"
    }
    
    strain_distribution = strain_files.get(strain_type)
    if not strain_distribution:
        print(f"Unknown strain type: {strain_type}")
        return
    
    for files in sub_folders:
        strain_path = os.path.join(path, files, strain_distribution)
        
        try:
            df_strain = pd.read_csv(strain_path, header=1)
            
            column_headers_strain = list(df_strain.columns.values)
            
            x_strain = df_strain[column_headers_strain[0]]
            x_strain = x_strain - 0.5 * x_strain.max()
            
            y_strain = df_strain[column_headers_strain[1]]
            
            plt.plot(x_strain, y_strain, label=files, zorder=1)
            
        except Exception as e:
            print(f"Error processing {strain_path}: {e}")
    
    plt.legend()
    plt.ylabel(f"{strain_type}")
    plt.xlabel("Section along middle line/[mm]")
    plt.grid(True, color="grey", linewidth="1.4", linestyle="-.")


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
    plt.plot(x, y1, "o-", label="Plane_strain_position")
    plt.plot(x, y2, "o-", label="Biaxial_strain_position")

    plt.xticks(
        np.linspace(df_var["Param"].min(), df_var["Param"].max(), len(df_var["Param"]))
    )
    plt.legend()
    plt.xlabel(Labelname)
    plt.ylabel("Position to Center/[mm]")
    plt.ylim(2.5, 6)
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


def plot_different_strains(path, sub_folders):
    title = path.split("/")[2]
    # title= "notch"
    plt.title(f"{title}")

    strain_type_list = ["x_strain", "y_strain", "eq_strain", 
                        # "xy_strain"
                        ]
    # choose 3 folders
    # selected_folders = [sub_folders[0], sub_folders[2], sub_folders[-1]]
    
    # choose all folders
    selected_folders = sub_folders
    
    print(selected_folders)

    for strain_type in strain_type_list:

        plot_strain_distribution(path, selected_folders, strain_type)
        plt.savefig(
            f"{path}/{strain_type}_strain_distribution.png", transparent=True, dpi=600
        )
        
        
        # for zoomin
        # plt.xlim(-6, -5)
        # plt.ylim(0.05, 0.09)
        # plt.xticks(np.linspace(-6, -5, 3))

        # plt.savefig(
        #     f"{path}/{strain_type}_zoomin_strain_distribution.png",
        #     transparent=True,
        #     dpi=600,
        # )
        
    plt.show()


def plot_3_strains(path, sub_folders):
    title = path.split("/")[2]
    # title= "notch"
    plt.title(f"{title}")

    strain_type_list = [
        "x_strain",
        "y_strain",
        "eq_strain",
    ]

    selected_folders = [
        sub_folders[0],
        # sub_folders[1],
    ]
    print(selected_folders)

    for strain_type in strain_type_list:

        plot_strain_distribution(path, selected_folders, strain_type)
        # plt.savefig(f"{path}/{strain_type}_strain_distribution.png", transparent=True,dpi=600)
        # plt.xlim(-6,-5)
        # plt.ylim(0,0.1)
        # plt.xticks(np.linspace(-6, -5, 3))
        plt.ylim(-0.1, 0.2)

    plt.savefig(f"{path}/3_strain_distribution.png", transparent=True, dpi=600)
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

        export_yld_parameter(ex_value, path, selected_folders[i])
        export_yield_curve(path, selected_folders[i])
    plot_yield(path)


def main():
    # foldername = "YLD_2d_Investigation/r00"
    # foldername = "test/"
    foldername = "test_rough_mesh/rough_model"

    path = f"./{foldername}"

    sub_folders = read_subfolders(path)
    gen_batch_post(foldername, sub_folders)
    extract_datas(path, sub_folders)

    # plot_3_strains(path, sub_folders)
    plot_different_strains(path, sub_folders)

    # plot_strain(path)
    # plot_yield_curve(path, sub_folders)


if __name__ == "__main__":
    main()
