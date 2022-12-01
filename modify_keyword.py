import os

import numpy as np

from FEM_model_generator import read_subfolders


import fileinput


def replace_in_file(file_path, search_text, new_text):
    with fileinput.input(file_path, inplace=True) as file:
        for line in file:

            new_line = line.replace(search_text, new_text)
            print(new_line, end="")


def find_param(file_path, changed_parameter):
    with open(f"{file_path}", "r+") as fa:
        a = [x.rstrip() for x in fa]
        for item in a:
            if changed_parameter in item:
                print("old parameter: \n", item)
                break
    return item


def main():

    keyword_filename = "main_keyword_sigma_0.dyn"

    # modify_parameter = "SwiftN"
    # path = f"YLD_2d_Investigation/{modify_parameter}"


    modify_parameter = "SwiftN"

    path = f"YLD_2d_Investigation/N"

    if modify_parameter == "M":
        changed_parameter = " M "
    else: 
        changed_parameter = modify_parameter



    # new_text = f"R  {modify_parameter}\t{new_value}"

    sub_folders = read_subfolders(path)

    for files in sub_folders:
        # x2[i]=round(x2[i],2)

        # create new folder for models

        file_path = f"{path}/{files}/{keyword_filename}"

        val = files.split("_")

        new_value = val[1]
        # print(new_value)
        new_text = f"R  {changed_parameter}       {new_value}"
        print("\nmodified new value:\n",new_text)

        print("file_path: ", file_path)
        search_text = find_param(file_path, changed_parameter)
        replace_in_file(file_path, search_text, new_text)


if __name__ == "__main__":
    main()
