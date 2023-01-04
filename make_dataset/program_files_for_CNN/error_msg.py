import os
from tqdm import tqdm


# Set the directory you want to start from
rootDir = '.\execution_test_1024'

# Set the file extension you want to search for (e.g., '.txt')
file_extension = 'messag'

# Initialize a counter to keep track of the number of files
file_count = 0

unfinished_files = []


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
                unfinished_files.append(file_path.split("\\")[4])
                file_count += 1

# Print the total number of files
print("\nUnfinished files:", unfinished_files)
print("\nTotal number of unfinished files:", file_count)

# Open a file for writing
with open(f'{rootDir}/log/unfinished_files.txt', 'w') as f:
    # Write each element of the list on a new line
    for file in unfinished_files:
        f.write(file + '\n')
