
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import glob


def read_csv_files(filename):

    data = pd.read_csv(
        filename,
        header=5,
        sep='\t',
        encoding='latin-1'
    )
    column_headers = list(data.columns.values)
    print(column_headers)
    
    
    force = data[column_headers[1]]
    displacement = data[column_headers[2]]
    
    return force, displacement

def plot_diagrams(force, displacement,label):


    plt.plot(displacement, force, "-", label=label)
    plt.xlabel("Displacement/[mm]")
    plt.ylabel("Force/[N]")
    # plt.ylim(-0.1, 0.25)
    plt.legend()

    # plt.grid(True, color="grey", linewidth="1.4", linestyle="-.")
    



def main():
    path = "YLD_2d_Investigation/experiment_data/force_displacement/"
    group = "R"
    csv_files = glob.glob(f"{path}{group}*.TXT")
    
    csv_files = sorted(csv_files)
    for filename in csv_files:
        print(filename)
        label = filename.split("/")[3].split(".")[0]
    # filename = "YLD_2d_Investigation/experiment_data/force_displacement/7p1_3.TXT"
        force, displacement = read_csv_files(filename)
        plot_diagrams(force, displacement,label)
    plt.savefig(f"{path}/f_d_{group}.png")
    
    plt.show()
        
   


if __name__ == "__main__":
    main()

