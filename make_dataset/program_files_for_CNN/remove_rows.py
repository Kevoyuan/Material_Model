import pandas as pd
import os
from tqdm import tqdm

# Specify the main directory where the CSV files are located
main_dir = 'roughmodel_test/dataset_inp'

# Get total number of files
total_files = 4096

# Initialize tqdm progress bar
with tqdm(total=total_files, unit='file') as pbar:
    # Iterate through all the files in the main directory and its subfolders
    for dirpath, dirnames, filenames in os.walk(main_dir):
        for filename in filenames:
            # Check if the file is a CSV and its name starts with "dataset_inp_58"
            if filename.endswith('.csv') and filename.startswith('dataset_inp_57_') or filename.endswith('.csv') and filename.startswith('dataset_inp_58_'):
                # Read the CSV file into a DataFrame
                df = pd.read_csv(os.path.join(dirpath, filename))
                # Check number of rows
                if df.shape[0] != 45:
                    print(f"{filename} has {df.shape[0]} rows, skipping...")
                    continue
                # Remove first 4 rows
                df = df.drop(df.index[:1])
                # Remove last 4 rows
                df = df.drop(df.index[-1:])
                # Save the modified DataFrame to the original file
                df.to_csv(os.path.join(dirpath, filename), index=False)
                pbar.update(1)
