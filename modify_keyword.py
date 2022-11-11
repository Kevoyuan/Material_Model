import os

import numpy as np

from FEM_model_generator import read_subfolders


import fileinput


def replace_in_file(file_path, search_text, new_text):
    with fileinput.input(file_path, inplace=True) as file:
        for line in file:

            new_line = line.replace(search_text, new_text)
            print(new_line, end="")


def find_param(file_path, modify_parameter):
    with open(f"{file_path}", "r+") as fa:
        a = [x.rstrip() for x in fa]
        for item in a:
            if modify_parameter in item:
                print("old parameter: ", item)
                break
    return item


def main():

    path = f"SwiftN/YLD_aniso/test"
    keyword_filename = "main_keyword_sigma_0.dyn"

    modify_parameter = "r00"

    # new_text = f"R  {modify_parameter}\t{new_value}"

    sub_folders = read_subfolders(path)

    for files in sub_folders:
        # x2[i]=round(x2[i],2)

        # create new folder for models

        file_path = f"{path}/{files}/{keyword_filename}"

        val = files.split("_")

        new_value = val[1]
        print(new_value)
        new_text = f"R     {modify_parameter}\t\t{new_value}"

        print("file_path: ", file_path)
        search_text = find_param(file_path, modify_parameter)
        replace_in_file(file_path, search_text, new_text)


if __name__ == "__main__":
    main()
