import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import math
import numpy as np


def plot_yield(path):
    csv_files = glob.glob(f"{path}/*_yield.csv")

    print(csv_files)

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    
    csv_files = sorted(csv_files, key=lambda s: float(s.split("_")[-2]))
    
    # a,b = 1,0

    for file in csv_files:

        # print(file.split("/"))
        if "\\" in file:
            file = file.replace("\\", "/")
        print("\nfile: ",file)

        parameter = file.split("/")[3].split("_")[0]
        value = file.split("/")[3].split("_")[1]
        legendname = f"{parameter}_{value}"


        ### Read .csv file and append to list
        df = pd.read_csv(file, index_col=0)
        sig1_x1 = df["sig_x/sig_0"]
        sig1_y1 = df["sig_y/sig_0"]
        sig1_x2 = df["sig_b/sig_0"]
        sig1_y2 = df["sig_xy/sig_0"]

        ### Create line for every file
        ax1.plot(sig1_x1,sig1_y1,label = legendname)
        ax2.plot(sig1_x2,sig1_y2,label = legendname)
        ax1.legend()
        ax2.legend()
        # slope = get_normals(sig1_x1,sig1_y1,a,b)
        # # plot the tangent line
        # # ax1.plot(sig1_x1, slope*(sig1_x1-a) + b)

        # x = np.linspace(a,a+0.1,100)
       
        # # plot the normal line
        # ax1.plot(x, slope*(x-a) + b)

        # ax1.plot(a,b,'ro')



    ax1.set_aspect('equal', adjustable='box')
    # ax2.set_aspect('equal', adjustable='box')
    ax1.set_xlabel(r"$\sigma_x/\sigma_0$")
    ax1.set_ylabel(r"$\sigma_y/\sigma_0$")

    ax2.set_xlabel(r"$\sigma_b/\sigma_0$")
    ax2.set_ylabel(r"$\sigma_{xy}/\sigma_0$")
    

    


    fig.savefig(f"{path}/yld.png")
    ax1.set_aspect('equal', adjustable='box')
    
    plt.show()
    
    

def get_normals(x,y,a,b):
    
    
    
    # fit a polynomial to the data
    coeffs = np.polyfit(x, y, 2)
    # print(coeffs)

    # find the slope of the tangent line at (1,1)
    slope = 2*coeffs[0]*a + coeffs[1]
    # print(slope)
    return slope

