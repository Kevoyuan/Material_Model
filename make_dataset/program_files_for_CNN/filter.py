import pandas as pd

# read in the CSV file
df = pd.read_csv('roughmodel_test\parameter_success.csv')

# remove the rows 
df = df.query('0.9 <= sig_45 <= 1.1')
df = df.query('0.9 <= sig_90 <= 1.1')
df = df.query('0.8 <= sig_b <= 1.25')
df = df.query('0.6 <= r_00 <= 2.5')
df = df.query('0.6 <= r_45 <= 1.6')
df = df.query('0.6 <= r_90 <= 2.5')
df = df.query('0.8 <= r_b <= 1.2')
df = df.query('5.2 <= M <= 10')










# display the filtered dataframe
print(df)

# df.to_csv('filtered_data.csv', index=False)

