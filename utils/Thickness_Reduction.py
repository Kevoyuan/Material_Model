import pandas as pd
from matplotlib import pyplot as plt


def Thickness_Reduction(filename, path):

    df = pd.read_csv(filename, header=1)

    column_headers = list(df.columns.values)

    ThiRed = column_headers[1]

    ax = df.plot(x="Distance Along Section", y=ThiRed)
    ax.get_legend().remove()
    ax.set_xlabel("Distance Along Section/[mm]")
    ax.set_ylabel("Thickness Reduction/[mm]")
    plt.savefig(path + "/" + "Thickness_Reduction.png")
    plt.show()
