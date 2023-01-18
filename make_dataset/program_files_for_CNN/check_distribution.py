import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Read the CSV file
# df = pd.read_csv("Z:/MA\Material_Model/make_dataset/roughmodel_1024/parameter_success.csv")
df = pd.read_csv("D:\ge24wej\Documents\makedataset/roughmodel2_1024\parameter_success.csv")



# Select the specific columns you want to check
columns_to_check = [
                    "alpha1","alpha2","alpha3","alpha4",
                    "alpha5","alpha6","alpha7","alpha8"
                    ]
data = df[columns_to_check].round(1)

# print(data)

# Use the value_counts() function to view the value distribution
value_distribution = data.value_counts()

# Plot the value distribution as a histogram
# value_distribution.plot(kind='bar')
# print(value_distribution)
plt.hist(data)
plt.legend(data)
plt.xticks(np.linspace(0.4, 1.7, 14))

plt.xlabel('value of alpha')
plt.ylabel('Frequency')
plt.title(f'Value Distribution of {columns_to_check}')
plt.show()

