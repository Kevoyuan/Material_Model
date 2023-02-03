import pandas as pd
import os
from pathlib import Path



group = "7_3mm"


def filter_55(df):
    step = int(df.shape[0] / 50)
    edge_points = df.iloc[[0, -1]]
    rest_points = df.iloc[1:-1:step]
    df_reduced = pd.concat([edge_points, rest_points])*5
    print(df_reduced.shape[0])  # Output: 55
    return df_reduced


# Read the CSV file into a pandas dataframe
df = pd.read_csv(
    # f"./YLD_2d_Investigation/experiment_data/test2/experiments/avg_data_{group}.csv",
    "Z:/MA/ML/nnc/simulation/simulation.csv",
    header=0,
    index_col=None,
)
df = df[(df["calib_length"] >=  -6.55) & (df["calib_length"] <= 6.55)]


# Extract each column into separate dataframes
col1 = df["calib_length"]
col2 = df["eps_x_avg"]
col3 = df["eps_y_avg"]
# col4 = df['col4']

# print(col3)
df_reduced1 = filter_55(col1)

df_reduced2 = filter_55(col2)
df_reduced3 = filter_55(col3)

# create new folder
folder = Path(f"Z:/MA/ML/nnc/{group}")

if not folder.exists():
    folder.mkdir()

# Write each column dataframe to a separate CSV file
df_reduced2.to_csv(f"Z:/MA/ML/nnc/{group}/x_strain.csv", index=False, header=False)
df_reduced3.to_csv(f"Z:/MA/ML/nnc/{group}/y_strain.csv", index=False, header=False)

x_path = f"Z:/MA/ML/nnc/{group}/x_strain.csv"
y_path = f"Z:/MA/ML/nnc/{group}/y_strain.csv"
# col4.to_csv('col4.csv', index=False)

# Load the CSV file into a pandas DataFrame
df_path = pd.read_csv("Z:/MA/ML/nnc/experiment.csv", header=0)

df_x = df_path["x4__0:X2D_0deg"]
df_y = df_path["x4__1:Y2D_0deg"]


# Replace a cell value
# df_x.at[0:] = x_path
# df_y.at[0:] = y_path
# df_path["x4__0:X2D_0deg"] = df_path["x4__0:X2D_0deg"].replace(x_path)
# df_path["x4__1:Y2D_0deg"] = df_path["x4__1:Y2D_0deg"].replace(y_path)
df_x.loc[0:2] = x_path
df_y.loc[0:2] = y_path

print(df_path)


# Save the updated DataFrame to a CSV file
df_path.to_csv("Z:/MA/ML/nnc/experiment.csv", index=False)
