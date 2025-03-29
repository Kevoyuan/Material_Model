from os.path import exists as file_exists
import pandas as pd
import numpy as np



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
                
                idx6 = a.index(item, idx5 + 2)
                print("index6 =  ", idx6)
                a.insert(idx6, cut_line)
                
                idx7 = a.index(item, idx6 + 2)
                print("index7 =  ", idx7)
                a.insert(idx7, cut_line)

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
                
                idx7 = a.index(item, idx6 + 2)
                
                idx8 = a.index(item, idx7 + 2)
                
                
                # line = a.readlines()
                print("arm lline: ", a[idx8 - 1])
                if arm_line == a[idx8 - 1]:

                    print("NODE for arm line inserted already")
                    break
                else:
                    print("index8 =  ", idx8)
                    a.insert(idx8, arm_line)
                    print("arm_line inserted!")
                    break

        # Go to start of file and clear it
        fa.seek(0)
        fa.truncate()
        # Write each line back
        for line in a:
            fa.write(line + "\n")