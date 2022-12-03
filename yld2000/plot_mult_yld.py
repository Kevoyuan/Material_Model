import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

def plot_yield(path):
    csv_files = glob.glob(f"{path}/*_yield.csv")
    print(csv_files)

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    
    csv_files = sorted(csv_files, key=lambda s: float(s.split("_")[4]))
    

    for file in csv_files:

        # print(file.split("/"))
        if "\\" in file:
            file = file.replace("\\", "/")
        print("\nfile: ",file)

        parameter = file.split("/")[3].split("_")[0]
        value = file.split("/")[1].split("_")[1]
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



    ax1.set_xlabel(r"$\sigma_x/\sigma_0$")
    ax1.set_ylabel(r"$\sigma_y/\sigma_0$")

    ax2.set_xlabel(r"$\sigma_b/\sigma_0$")
    ax2.set_ylabel(r"$\sigma_{xy}/\sigma_0$")


    fig.savefig(f"{path}/yld.png")

    ### Generate the plot
    plt.show()
