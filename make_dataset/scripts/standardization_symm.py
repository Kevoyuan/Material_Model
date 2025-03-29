import pandas as pd
import numpy as np


def standardization_symm(path):

    # Read in the CSV file
    df = pd.read_csv(f"../{path}/parameter_success.csv")

    # Calculate the new value "sig_45"
    df["sig_45_new"] = df["sig_45"] / df["sig_90"]
    df["sig_90_new"] = 1 / df["sig_90"]
    df["r_00_new"] = df["r_90"] 
    df["r_45_new"] = df["r_45"] 
    df["r_90_new"] = df["r_00"] 



    # Create a new dataframe with only the "sig_45" column
    df_sig_45 = df[["sig_45_new"]]
    df_sig_90 = df[["sig_90_new"]]
    df_r_00 = df[["r_00_new"]]
    df_r_45 = df[["r_45_new"]]
    df_r_90 = df[["r_90_new"]]



    # # Stack the column vertically with its original column
    df_sig_45_vstack = pd.concat([df["sig_45"], df["sig_45_new"]], axis=0)
    df_sig_90_vstack = pd.concat([df["sig_90"], df["sig_90_new"]], axis=0)
    df_r_00_vstack = pd.concat([df["r_00"], df["r_00_new"]], axis=0)
    df_r_45_vstack = pd.concat([df["r_45"], df["r_45_new"]], axis=0)
    df_r_90_vstack = pd.concat([df["r_90"], df["r_90_new"]], axis=0)

    def standardization(df):
        std = np.std(df, axis=0)
        mean = np.mean(df, axis=0)
        df_vstack = (df-mean)/std
        return df_vstack

    df_sig_45_vstack=standardization(df_sig_45_vstack)
    df_sig_90_vstack=standardization(df_sig_90_vstack)

    df_r_00_vstack=standardization(df_r_00_vstack)
    df_r_45_vstack=standardization(df_r_45_vstack)
    df_r_90_vstack=standardization(df_r_90_vstack)



    df_Model = pd.DataFrame(
        {
            "x7__0:sig_45": df_sig_45_vstack,
            "x7__1:sig_90": df_sig_90_vstack,

            "x7__2:r_00": df_r_00_vstack,
            "x7__3:r_45": df_r_45_vstack,
            "x7__4:r_90": df_r_90_vstack,
        }
    )
    # # Save the new dataframe to a new CSV file
    df_Model.to_csv(f"../{path}/temp.csv", index=False)
    df_merged = pd.concat([pd.read_csv(f"../{path}/temp.csv"), pd.read_csv(f"../{path}/dataset.csv")], axis=1)
    df_merged.to_csv(f"../{path}/merged_dataset.csv", index=False)


if __name__ == '__main__':
    standardization_symm()


