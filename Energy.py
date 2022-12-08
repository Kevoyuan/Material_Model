import pandas as pd
from matplotlib import pyplot as plt


def energy(filename, path):

    df = pd.read_csv(filename, header=1)
    # df = df.iloc[1:]

    # Time,Kinetic Energy,Internal Energy,Total Energy,

    ax = df.plot(x="Time", y=["Kinetic Energy", "Internal Energy", "Total Energy"])
    ax.set_xlabel("Time/[s]")
    ax.set_ylabel("Energy")
    plt.savefig(path + "/" + "Energy.png")
    plt.show()
