from tqdm import tqdm
import pandas as pd
import os

# Specify the main directory where the CSV files are located
main_dir = 'roughmodel_test0/dataset_inp'

# Get total number of files
total_files = 4096

# Initialize tqdm progress bar
with tqdm(total=total_files, unit='file') as pbar:
    # Iterate through all the files in the main directory and its subfolders
    for dirpath, dirnames, filenames in os.walk(main_dir):
        for filename in filenames:
            # Check if the file is a CSV and its name starts with "dataset_inp_57/58_"
            if filename.endswith('.csv') and filename.startswith('dataset_inp_57_') or filename.endswith('.csv') and filename.startswith('dataset_inp_58_'):
                # Read the CSV file into a DataFrame
                df = pd.read_csv(os.path.join(dirpath, filename))
                
                if df.shape[1] != 11:
                    print(f"{filename} has {df.shape[1]} columns, skipping...")
                    continue

                # Delete the first and last 5 columns
                df = df.drop(df.columns[[0,1,2,3,4,-5,-4,-3,-2,-1]], axis=1)

                # Save the modified DataFrame to the original file
                df.to_csv(os.path.join(dirpath, filename), index=False)
                pbar.update(1)
