import csv

import pandas as pd
from matplotlib import pyplot as plt


def ydisplacement(filename):

    # path = filename - "Y-displacement.csv"

    df = pd.read_csv(filename, skiprows=[0])

    # df = df.head()
    # plt.show()

    column_headers = list(df.columns.values)

    Node1 = column_headers[1]
    Node2 = column_headers[2]
    # Node3 = column_headers[3]
    print(Node2)

    ax = df.plot(x="Time", y=[Node1, Node2], label=["Node 1", "Node 2"])

    # ax.get_legend().remove()
    ax.set_xlabel("Time/[s]")
    ax.set_ylabel("Y-Displacement/[mm]")
    # plt.savefig(path + "/" + "Y-Displacement.png")
    plt.show()
