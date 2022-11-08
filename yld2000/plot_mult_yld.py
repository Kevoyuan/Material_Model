import os
import pandas as pd
import matplotlib.pyplot as plt
import glob


csv_files = glob.glob("./yld2000/eq*.csv")
print(csv_files)

fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

for file in csv_files:
    print(file)
    if "001" in file:
        legendname = "eq_strain = 0.01"
    if "005" in file:
        legendname = "eq_strain = 0.05"
    if "02" in file:
        legendname = "eq_strain = 0.2"

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


fig.savefig("./yld2000/yld.svg")

### Generate the plot
plt.show()
