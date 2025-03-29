import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Read the CSV file
# df = pd.read_csv("Z:/MA\Material_Model/make_dataset/merged_dataset/parameter_success.csv")
# df = pd.read_csv("merged_dataset\merged_dataset.csv")

df = pd.read_csv("roughmodel_test\merged_dataset.csv")

# df = pd.read_csv("./roughmodel_test/parameter_success.csv")



# Select the specific columns you want to check

columns_to_check = [
                    # "alpha1",
                    # "alpha2",
                    # "alpha3",
                    # "alpha4",
                    # "alpha5",
                    # "alpha6",
                    # "alpha7", 
                    # "alpha8",

                    # 'sig_45',
                    # 'sig_90',
                    # 'sig_b',
                    # 'r_00',
                    # 'r_45',
                    # 'r_90',
                    # 'r_b',
                    # "M"
                    # "x7__0:sig_45",
                    # "x7__1:sig_90",
                    # "x7__2:r_00",
                    # "x7__3:r_45",
                    # "x7__4:r_90",
                    "y__0:M",
                    "y__1:alpha1",
                    "y__2:alpha2",
                    "y__3:alpha3",
                    "y__4:alpha4",
                    "y__5:alpha5",
                    "y__6:alpha6",
                    "y__7:alpha7",
                    "y__8:alpha8"


                    
                    ]

# columns_to_check = [
                    # 'sig_45',
                    # 'sig_90',
                    # 'sig_b',
                    # 'r_00',
                    # 'r_45',
                    # 'r_90',
                    # 'r_b',
                    # "M"
                    # ]





data = df[columns_to_check].round(2)

# print(data)

# Use the value_counts() function to view the value distribution
value_distribution = data.value_counts()

# Plot the value distribution as a histogram
# value_distribution.plot(kind='bar')
# print(value_distribution)
plt.hist(data,width=0.01)
plt.legend(data)
# plt.xticks(np.linspace(0.4, 1.7, 14))

plt.xlabel('value of alpha')
plt.ylabel('Frequency')
plt.title(f'Value Distribution of {columns_to_check}')
plt.show()

