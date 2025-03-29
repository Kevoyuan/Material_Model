from os.path import exists as file_exists
import pandas as pd
import numpy as np

# Inserts a specific 'state' declaration into the gen_post.cfile if not already present
def add_state(fem_model, state):
    filepath = f"{fem_model}/gen_post.cfile"
    with open(filepath, "r+") as f:
        lines = [line.rstrip() for line in f]

        for i, line in enumerate(lines):
            if line == f"state {state};":
                print(f"state = {state} inserted already")  # Avoid duplicate insertions
                break
            elif line == "$# Strain Curve":
                index = i  # Find insertion point before "$# Strain Curve"
                lines.insert(index, f"state {state};")
                print(f"state = {state} inserted!")
                break

        # Rewrite file with updated content
        f.seek(0)
        f.truncate()
        f.writelines(line + "\n" for line in lines)


# Inserts a 'cut_line' multiple times after finding "splane drawcut" entries
def add_cut_line(fem_model, cut_line):
    filepath = f"{fem_model}/gen_post.cfile"
    with open(filepath, "r+") as f:
        lines = [line.rstrip() for line in f]

        for i, line in enumerate(lines):
            if line.startswith(cut_line):
                print("Section NODE inserted already")
                if "splane drawcut" in line:
                    # Trace all related drawcut indices (for debugging/reference)
                    idx = i
                    for n in range(1, 6):
                        idx = lines.index(line, idx + 2)
                        print(f"index{n + 1} =  ", idx)
                break

            elif line.startswith("splane drawcut"):
                # Insert the cut_line 7 times after repeated offsets from drawcut lines
                idx = i
                for n in range(7):
                    idx = lines.index(line, idx + 2)
                    print(f"index{n + 1} =  ", idx)
                    lines.insert(idx, cut_line)
                print("Section NODE inserted!")
                break

        # Rewrite file with modifications
        f.seek(0)
        f.truncate()
        f.writelines(line + "\n" for line in lines)


# Inserts an angle measurement command if it doesn't already exist
def add_angle_command(fem_model, angle_command):
    filepath = f"{fem_model}/gen_post.cfile"
    with open(filepath, "r+") as f:
        lines = [line.rstrip() for line in f]

        for i, line in enumerate(lines):
            if line.startswith(angle_command):
                print("Angle NODE inserted already")
                break
            elif line.startswith("measure history angle3 a"):
                index = i  # Insert before the specific measurement line
                lines.insert(index, angle_command)
                print("angle_command inserted!")
                break

        # Rewrite file with changes
        f.seek(0)
        f.truncate()
        f.writelines(line + "\n" for line in lines)


# Calculates final arm angle at a given index from CSV, adjusted relative to 90 degrees
def calc_end_angle(fem_model, max_index):
    angle_file = f"{fem_model}/ArmAngle2.csv"

    if not file_exists(angle_file):
        print("File ArmAngle2.csv not exist in folder:", fem_model)
        return 0

    # Load CSV and extract relevant column
    df = pd.read_csv(angle_file, skiprows=[0])
    angle_col = list(df.columns.values)[1]
    end_angle = float(df[angle_col][max_index]) - 90  # Normalize angle

    print("end_angle:", end_angle)
    return end_angle


# Inserts an arm line based on calculated angle and triangle reference point
def add_arm_line(fem_model, end_angle, Tri_point):
    section_angle = round(np.tan(np.radians(end_angle)), 3)
    arm_line = f"splane dep1 {Tri_point}  {section_angle}  1.000  0.000"

    filepath = f"{fem_model}/gen_post.cfile"
    with open(filepath, "r+") as f:
        lines = [line.rstrip() for line in f]

        for i, line in enumerate(lines):
            if "splane drawcut" in line:
                # Find where to place the new arm_line after the 7th drawcut jump
                idx = i
                for _ in range(7):
                    idx = lines.index(line, idx + 2)

                last_check = lines[idx - 1]
                print("arm lline:", last_check)

                if arm_line == last_check:
                    print("NODE for arm line inserted already")
                else:
                    print("index8 = ", idx)
                    lines.insert(idx, arm_line)
                    print("arm_line inserted!")
                break

        # Rewrite file with added arm line
        f.seek(0)
        f.truncate()
        f.writelines(line + "\n" for line in lines)
