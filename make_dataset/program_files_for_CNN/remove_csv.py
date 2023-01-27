import os
import glob
from tqdm import tqdm

root_dir = './roughmodel_test/keyword'

csv_files = glob.glob(root_dir+'/**/*.csv', recursive=True)

for file_path in tqdm(csv_files, desc='Removing csv files'):
    os.remove(file_path)
