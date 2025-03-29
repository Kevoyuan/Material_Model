import os
import shutil

from tqdm import tqdm

def copy_csv(src_folder, dest_folder):
    for foldername, subfolders, filenames in tqdm(os.walk(src_folder)):
        for filename in filenames:
            if filename == 'nodout':
                src_file = os.path.join(foldername, filename)
                dest_folder_path = os.path.join(dest_folder, foldername[len(src_folder)+1:])
                os.makedirs(dest_folder_path, exist_ok=True)
                dest_file = os.path.join(dest_folder_path, filename)
                shutil.copy2(src_file, dest_file)

copy_csv("D:/ge24wej/Documents/makedataset/roughmodel_1024/keyword","Z:/MA/Material_Model/make_dataset/roughmodel_1024/keyword") 