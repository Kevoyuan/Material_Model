import pandas as pd
from matplotlib import pyplot as plt

def yforce(filename,path):

    # path = filename - "Y-Force.csv"


    df = pd.read_csv(filename, header=1)


    ax = df.plot(x="Time", y="mat-2")
    ax.get_legend().remove()
    ax.set_xlabel("Time/[s]")
    ax.set_ylabel("Y-Force/[N]")
    plt.savefig(path + "/" + "Y-Force.png")
    plt.show()
