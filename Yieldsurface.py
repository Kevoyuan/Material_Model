# Code you have previously used to load data
import os
import re
import time
from operator import mod

import numpy as np
import pandas as pd
import scipy.interpolate
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression



variable="test"
path = f"./{variable}"

# find all sub folders under main directory
sub_folders = [
    name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))
]

remove_list = [
    ".git",
    ".vscode",
    "Archive",
    "Model_template",
    "Surface Model",
    "__pycache__",
]
sub_folders = [ele for ele in sub_folders if ele not in remove_list]

print(sub_folders)


for filename in sub_folders:
    file = f"{path}/{filename}"

    sig_x_path = f"{file}/sigma_x.csv"
    sig_y_path = f"{file}/sigma_y.csv"
    sig_eq_path = f"{file}/sigma_eq.csv"



    df_x = pd.read_csv(sig_x_path, header=1)
    df_y = pd.read_csv(sig_y_path, header=1)
    df_eq = pd.read_csv(sig_eq_path, header=1)


    column_headers_x = list(df_x.columns.values)
    column_headers_y = list(df_y.columns.values)
    column_headers_eq = list(df_eq.columns.values)





    sigma_x = df_x[column_headers_x[1]]
    sigma_y = df_y[column_headers_y[1]]
    sigma_eq = df_eq[column_headers_eq[1]]

    y = sigma_y/sigma_eq
    x = sigma_x/sigma_eq

    # plt.scatter(x, y)
    plt.plot(x, y, "--", label = filename)
    plt.legend()
    
    plt.xlabel(r'$\sigma_x/\sigma_{eq}$')

    plt.ylabel(r'$\sigma_y/\sigma_{eq}$')


# plt.ylim([-0.1,1.2])
plt.axis("equal")     
# plt.savefig(path+f"/sigma.svg")
plt.show()