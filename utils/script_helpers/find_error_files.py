import os
from tqdm import tqdm
import re

# Set the directory you want to start from
rootDir = '.\\roughmodel_1024'

# Set the file extension you want to search for (e.g., '.txt')
file_extension = 'messag'

# Initialize a counter to keep track of the number of files
file_count = 0

unfinished_files = []
unfinished_files_number = []


# Set the total number of steps for the task
total_steps = 1024

# Walk through the subfolders and count the number of files with the specified extension
for dirName, subdirList, fileList in tqdm(os.walk(rootDir), total=3*total_steps, desc='Searching files:'):



    for fname in fileList:
        if fname.endswith(file_extension):
            

            # Construct the full file path using os.path.join()
            file_path = os.path.join(dirName, fname)

            # Open the file in read mode
            with open(file_path, 'r') as f:
                # Read the contents of the file into a string
                contents = f.read()

            # Check if the string is a substring of the file contents
            if not "N o r m a l    t e r m i n a t i o n" in contents:

                # print(f"\nString not found in {file_path}.")
                unfinished_files.append(file_path.replace("\\messag",'').replace(".\\roughmodel_1024",''))
                numbers = re.findall("keyword(.*)_", file_path.split("\\")[4])
                unfinished_files_number.append(numbers[0])
                file_count += 1
# unfinished_files = sorted(unfinished_files, key = int)

# unfinished_files_number = sorted(unfinished_files_number, key = int)

# Print the total number of files
print("\nUnfinished files:", unfinished_files)
print("\nTotal number of unfinished files:", file_count)

# Open a file for writing
with open(f'{rootDir}/log/unfinished_count.csv', 'w') as f:
    f.write("number of unfinished files\n")

    # Write each element of the list on a new line
    f.write(str(file_count) + '\n')

# Open a file for writing
with open(f'{rootDir}/log/unfinished_files_nummber.csv', 'w') as f:
    # Write the header to the file
    f.write("n\n")
    # Write each element of the list on a new line
    for file in unfinished_files_number:
        f.write(file + '\n')

import pandas as pd

# Create a dataframe from the lists 'unfinished_files_number' and 'unfinished_files'
df = pd.DataFrame({'number': unfinished_files_number, 'filename': unfinished_files})

df["number"] = df["number"].apply(pd.to_numeric)
# Sort the dataframe by the 'number' column in ascending order
df.sort_values(['number'], inplace=True)

# Save the dataframe to a CSV file
df.to_csv(f'{rootDir}/log/unfinished_files.csv', index=False)