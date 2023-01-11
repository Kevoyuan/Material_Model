from os.path import exists as file_exists
import pandas as pd


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


def set_state_from_y_dis(fem_model):
    source = f"{fem_model}/Y-displacement.csv"
    df_Y = pd.read_csv(source, header=1)
    column_headers = list(df_Y.columns.values)
    print(column_headers)
    

    time_displacement = df_Y[column_headers[0]]
    df_y_displacement = df_Y[column_headers[3]]

    # convert string back to numeric
    df_y_displacement = df_y_displacement.apply(pd.to_numeric)
    df_y_displacement = df_y_displacement[df_y_displacement < 3]
    max_index = df_y_displacement.idxmax()
    state = max_index + 1
    return state


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
