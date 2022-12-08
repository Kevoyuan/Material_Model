import pandas as pd
from matplotlib import pyplot as plt
import csv


def xdisplacement(filename, path):


    df = pd.read_csv(filename, header=1)


    column_headers = list(df.columns.values)


    Node3 = column_headers[3]


    ax = df.plot(x="Time", y=Node3, label="Node 3")
    # ax.get_legend().remove()
    ax.set_xlabel("Time/[s]")
    ax.set_ylabel("X-Displacement/[mm]")
    plt.savefig(path + "/" + "X-Displacement.png")
    plt.show()
